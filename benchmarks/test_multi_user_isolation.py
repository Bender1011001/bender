"""
Multi-user isolation test for episodic memory.

Tests that:
1. User A can recall their own facts after injection
2. User B does NOT recall User A's facts
3. User B can recall their own (different) facts
4. Neither user's recall interferes with the other
"""
import torch
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM, AutoTokenizer
from dual_system_v2 import DualSystemV2, SidecarConfig
from chat_continuous import continuous_learning_update
from modules.epistemic_router import EpistemicRouter
from modules.bch_consolidation import BCHConsolidator


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt, user_id, device, max_new=50):
    """Generate with hard recall using latched prompt embedding."""
    user_ids = torch.tensor([user_id], dtype=torch.long, device=device)
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    h_seq = None
    z0 = z0_base = None
    past_key_values = None
    initial_hr_query = None
    initial_prompt_ids = None

    for step in range(max_new):
        with torch.no_grad():
            if past_key_values is None:
                out = model.backbone(
                    input_ids=generated, output_hidden_states=True, return_dict=True
                )
            else:
                out = model.backbone(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values,
                    output_hidden_states=True, return_dict=True, use_cache=True
                )
            past_key_values = out.past_key_values
            base_logits = out.logits[:, -1:, :]

            if h_seq is None:
                h_seq = out.hidden_states[-1]
                h_pooled = h_seq[:, -1, :]
                mu, _ = model.latent_planner.encoder(h_pooled)
                z0 = mu
                z0_base = z0
                if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
                    z0 = model.fiber_bundle.lift_to_fiber(z0, user_ids)
                if h_seq.size(1) > 128:
                    h_seq = h_seq[:, -128:, :]
            else:
                h_new = out.hidden_states[-1][:, -1:, :]
                h_seq = torch.cat([h_seq, h_new], dim=1)
                if h_seq.size(1) > 128:
                    h_seq = h_seq[:, -128:, :]

            geo_logits = model.geo_processor(h_seq.detach(), z0=z0)[:, -1:, :]
            logits = (base_logits + geo_logits)[:, -1, :]

            if getattr(model, 'fiber_proj', None) is not None and z0_base is not None:
                z0_delta = z0 - z0_base
                if getattr(model, 'episodic_memory', None) is not None:
                    query_emb = h_seq[:, -1, :].detach()
                    episodic_delta = model.episodic_memory.read(
                        user_ids, query_emb, query_ids=initial_prompt_ids
                    )
                    z0_delta = z0_delta + episodic_delta
                    raw_retrieved = model.episodic_memory.get_raw_retrieved()
                    if raw_retrieved is not None:
                        embed_weight = model.backbone.model.embed_tokens.weight
                        episodic_logits = torch.matmul(
                            raw_retrieved.to(embed_weight.dtype), embed_weight.T
                        )
                        logits += model.episodic_memory.episodic_scale * episodic_logits
                fiber_logits = model.fiber_proj(z0_delta, h_seq.detach())
                logits += fiber_logits[:, -1, :]

            if getattr(model, 'sparse_bias', None) is not None:
                sparse_logits = model.sparse_bias(user_ids)
                logits += sparse_logits

            if getattr(model, 'episodic_memory', None) is not None:
                if initial_hr_query is None:
                    initial_hr_query = h_seq[:, -1, :].detach().clone()
                    initial_prompt_ids = input_ids.detach().clone()
                generated_so_far = generated[:, input_ids.shape[-1]:]
                
                dynamic_boost = max(20.0, 8.0 * logits.std().item())
                
                hard_bias = model.episodic_memory.hard_recall(
                    user_ids, initial_hr_query,
                    vocab_size=logits.size(-1),
                    similarity_threshold=0.7,
                    boost=dynamic_boost,
                    generated_ids=generated_so_far,
                    query_ids=initial_prompt_ids,
                )
                logits += hard_bias

        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = generated[0, input_ids.shape[-1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/v2_sidecar_run4_final.pt")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--backbone", default="Bender1011001/Qwen2.5-3B-Instruct-ABLITERATED")
    args = parser.parse_args()

    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[*] Loading backbone: {args.backbone}")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, dtype=dtype, device_map="auto", trust_remote_code=True
    )

    config = SidecarConfig(
        backbone_hidden_size=backbone.config.hidden_size,
        backbone_vocab_size=backbone.config.vocab_size,
        num_fibers=16,
    )
    model = DualSystemV2(backbone, config)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        target_key = "fiber_bundle.fiber_store.fiber_vectors"
        if target_key in ckpt:
            ckpt_size = ckpt[target_key].shape[0]
            model_size = model.fiber_bundle.fiber_store.fiber_vectors.shape[0]
            if ckpt_size != model_size:
                new_fv = torch.zeros_like(model.fiber_bundle.fiber_store.fiber_vectors)
                copy_n = min(ckpt_size, model_size)
                new_fv[:copy_n] = ckpt[target_key][:copy_n]
                ckpt[target_key] = new_fv
        model.load_state_dict(ckpt, strict=False)

    bb_device = backbone.model.embed_tokens.weight.device
    model.geo_processor = model.geo_processor.to(bb_device, dtype=dtype)
    model.latent_planner = model.latent_planner.to(bb_device, dtype=dtype)
    model.ebm_critic = model.ebm_critic.to(bb_device, dtype=dtype)
    model.fiber_bundle = model.fiber_bundle.to(bb_device, dtype=dtype)
    if getattr(model, 'fiber_proj', None) is not None:
        model.fiber_proj = model.fiber_proj.to(bb_device, dtype=dtype)
    if getattr(model, 'sparse_bias', None) is not None:
        model.sparse_bias = model.sparse_bias.to(bb_device, dtype=dtype)
    if getattr(model, 'episodic_memory', None) is not None:
        model.episodic_memory = model.episodic_memory.to(bb_device, dtype=dtype)

    model.requires_grad_(False)
    model.fiber_bundle.fiber_store.fiber_vectors.requires_grad = True
    if getattr(model, 'fiber_proj', None) is not None:
        model.fiber_proj.requires_grad_(True)
    if getattr(model, 'sparse_bias', None) is not None:
        model.sparse_bias.requires_grad_(True)
    if getattr(model, 'episodic_memory', None) is not None:
        model.episodic_memory.requires_grad_(True)
    model.latent_planner.requires_grad_(False)

    # Setup for BOTH users
    USER_A, USER_B = 1, 2
    with torch.no_grad():
        fv = model.fiber_bundle.fiber_store.fiber_vectors
        fv.data[USER_A] = torch.randn_like(fv.data[USER_A]) * 0.01
        fv.data[USER_B] = torch.randn_like(fv.data[USER_B]) * 0.01

    # Setup optimizers (shared)
    sidecar_params = [
        {'params': model.fiber_proj.parameters(), 'lr': 1e-2},
        {'params': [model.fiber_bundle.fiber_store.fiber_vectors], 'lr': 1e-4},
    ]
    if getattr(model, 'episodic_memory', None) is not None:
        sidecar_params.append({'params': model.episodic_memory.parameters(), 'lr': 1e-3})
    opt_sc = torch.optim.AdamW(sidecar_params, weight_decay=0.0)
    opt_base = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=0)
    opt_sparse = None
    if getattr(model, 'sparse_bias', None) is not None:
        opt_sparse = torch.optim.SparseAdam(model.sparse_bias.parameters(), lr=0.1)

    er = EpistemicRouter(energy_threshold_init=5.0, tau_e=0.5, eta_base=0.01, eta_user=0.05).to(device)
    er.train()
    bch = BCHConsolidator(eta=0.05, order=2)

    # User A's fact
    FACT_A = {
        "inject": "My dog's name is Luna and she's afraid of thunderstorms.",
        "target": "Your dog's name is Luna, and she's afraid of thunderstorms.",
        "queries": ["What's my dog's name?"],
        "keywords": ["Luna", "thunderstorm"],
    }
    # User B's fact (DIFFERENT)
    FACT_B = {
        "inject": "My cat's name is Whiskers and he loves sitting in cardboard boxes.",
        "target": "Your cat's name is Whiskers, and he loves sitting in cardboard boxes.",
        "queries": ["What's my cat's name?"],
        "keywords": ["Whiskers", "cardboard"],
    }

    print(f"\n{'='*70}")
    print("  MULTI-USER ISOLATION TEST")
    print(f"  User A (id={USER_A}): {FACT_A['inject']}")
    print(f"  User B (id={USER_B}): {FACT_B['inject']}")
    print(f"  {args.rounds} training rounds per user")
    print(f"{'='*70}")

    def inject_fact(fact, user_id, label, rounds):
        """Train a fact into a specific user using continuous_learning_update."""
        chat = [{"role": "user", "content": fact["inject"]}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        target_ids = tokenizer(fact["target"], return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        print(f"\n  --- Train {label} ({rounds} rounds) ---")
        for r in range(1, rounds + 1):
            loss, energy, kappa = continuous_learning_update(
                model, tokenizer, prompt_ids, target_ids, reward=1.0,
                optimizer_sidecar=opt_sc, optimizer_base=opt_base,
                epistemic_router=er, bch_consolidator=bch,
                user_id=user_id, dtype=dtype,
                target_ids=target_ids, optimizer_sparse=opt_sparse
            )
            mem_count = model.episodic_memory.memory_count[user_id].item() if model.episodic_memory else 0
            print(f"  [{label} R{r}] loss={loss:.4f} mem={mem_count}")

    # Phase 1: Train User A
    inject_fact(FACT_A, USER_A, "User A", args.rounds)

    # Phase 2: Train User B
    inject_fact(FACT_B, USER_B, "User B", args.rounds)

    # Phase 3: Cross-user recall test
    print(f"\n  {'='*60}")
    print("  CROSS-USER RECALL TEST")
    print(f"  {'='*60}")

    results = {"pass": 0, "fail": 0}

    def check(user_id, label, query, keywords, should_recall):
        resp = generate_greedy(model, tokenizer, query, user_id, device)
        resp_short = resp.replace('\n', ' ')[:120]
        found = [kw for kw in keywords if kw.lower() in resp.lower()]
        if should_recall:
            ok = len(found) > 0
            tag = "[OK]" if ok else "[FAIL]"
        else:
            ok = len(found) == 0
            tag = "[OK: no leak]" if ok else "[LEAK!]"
        print(f"    {label} → Q: '{query}'")
        print(f"      A: {resp_short}")
        print(f"      {tag} found={found}")
        results["pass" if ok else "fail"] += 1

    # Test 1: User A recalls A's fact
    print(f"\n  1. User A recalls own fact:")
    for q in FACT_A["queries"]:
        check(USER_A, "User A", q, FACT_A["keywords"], should_recall=True)

    # Test 2: User A does NOT recall B's fact
    print(f"\n  2. User A does NOT recall B's fact:")
    for q in FACT_B["queries"]:
        check(USER_A, "User A", q, FACT_B["keywords"], should_recall=False)

    # Test 3: User B recalls B's fact
    print(f"\n  3. User B recalls own fact:")
    for q in FACT_B["queries"]:
        check(USER_B, "User B", q, FACT_B["keywords"], should_recall=True)

    # Test 4: User B does NOT recall A's fact
    print(f"\n  4. User B does NOT recall A's fact:")
    for q in FACT_A["queries"]:
        check(USER_B, "User B", q, FACT_A["keywords"], should_recall=False)

    # Test 5: Cross-queries
    print(f"\n  5. Cross-queries (wrong domain):")
    check(USER_A, "User A", "What's my cat's name?", ["Whiskers"], should_recall=False)
    check(USER_B, "User B", "What's my dog's name?", ["Luna"], should_recall=False)

    print(f"\n  {'='*60}")
    print(f"  RESULTS: {results['pass']} passed, {results['fail']} failed")
    print(f"  {'='*60}\n")


if __name__ == "__main__":
    main()
