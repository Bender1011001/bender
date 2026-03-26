"""
Repetition Recall Micro-Test
=============================
Hypothesis: The 200-cycle test failed context-dependent recall because each
fact was injected only ONCE. With more repetitions, the fiber/sparse_bias
should accumulate enough signal to enable recall.

Protocol:
  1. Load model, setup fiber-only mode (same as long_term_memory_test)
  2. Run 20-cycle warmup
  3. Pick ONE fact ("My dog's name is Luna and she's afraid of thunderstorms")
  4. Inject it N times (5, 10, 20) using contrastive forcing
  5. After each injection, test recall with greedy generation
  6. Report at which repetition (if any) the model starts recalling "Luna"

This runs in ~10 minutes vs 1.5 hours for the full test.
"""
import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dual_system_v2 import DualSystemV2, SidecarConfig
from chat_continuous import continuous_learning_update
from modules.epistemic_router import EpistemicRouter
from modules.bch_consolidation import BCHConsolidator


@torch.no_grad()
def generate_greedy(model, tokenizer, prompt, user_id, device, max_tokens=40):
    """Short greedy generation for recall testing."""
    model.eval()
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()
    past_key_values = None
    h_seq = None
    z0 = None
    z0_base = None
    initial_hr_query = None  # Cache prompt embedding for stable hard_recall
    initial_prompt_ids = None  # Cache for Jaccard overlap
    user_ids = torch.tensor([user_id], device=device, dtype=torch.long)

    for step in range(max_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if past_key_values is not None:
                out = model.backbone(
                    input_ids=generated[:, -1:], past_key_values=past_key_values,
                    output_hidden_states=True, return_dict=True, use_cache=True
                )
            else:
                out = model.backbone(
                    input_ids=generated, output_hidden_states=True, return_dict=True, use_cache=True
                )
            past_key_values = out.past_key_values
            base_logits = out.logits[:, -1:, :]

            if h_seq is None:
                h_seq = out.hidden_states[-1]
                
                if initial_hr_query is None:
                    initial_hr_query = h_seq[:, -1, :].detach().clone()
                    initial_prompt_ids = input_ids.detach().clone()
                    
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
                # Add episodic memory retrieval for context-dependent recall
                if getattr(model, 'episodic_memory', None) is not None:
                    episodic_delta = model.episodic_memory.read(
                        user_ids, initial_hr_query, query_ids=initial_prompt_ids
                    )
                    z0_delta = z0_delta + episodic_delta

                    # Direct episodic→logit via backbone embedding matrix
                    raw_retrieved = model.episodic_memory.get_raw_retrieved()
                    if raw_retrieved is not None:
                        embed_weight = model.backbone.model.embed_tokens.weight
                        episodic_logits = torch.matmul(
                            raw_retrieved.to(embed_weight.dtype),
                            embed_weight.T
                        )
                        logits += model.episodic_memory.episodic_scale * episodic_logits

                fiber_logits = model.fiber_proj(z0_delta, h_seq.detach())
                logits += fiber_logits[:, -1, :]

            if getattr(model, 'sparse_bias', None) is not None:
                sparse_logits = model.sparse_bias(user_ids)
                logits += sparse_logits

            # Hard recall: position-aware boost of stored target tokens
            # Use INITIAL prompt embedding for similarity matching (not current
            # step's embedding which drifts and drops below threshold after step 0)
            if getattr(model, 'episodic_memory', None) is not None:
                generated_so_far = generated[:, input_ids.shape[-1]:]
                hard_bias = model.episodic_memory.hard_recall(
                    user_ids, initial_hr_query,
                    vocab_size=logits.size(-1),
                    similarity_threshold=0.7,
                    boost=20.0,
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


@torch.no_grad()
def generate_contrastive(model, tokenizer, prompt, user_id, device,
                         max_tokens=40, guidance_scale=1.5):
    """
    Contrastive decoding: amplify the DIFFERENCE between adapter-corrected
    and backbone-only logits to overcome the Adapter Gain Trap.

    final_logits = backbone_logits + guidance_scale * (corrected_logits - backbone_logits)
                 = (1 - guidance_scale) * backbone_logits + guidance_scale * corrected_logits

    At guidance_scale=1.0, this reduces to standard generation.
    At guidance_scale=1.5, the adapter's contribution is amplified 1.5x while
    the backbone's prior is reduced by 0.5x.
    At guidance_scale=2.0, the adapter's contribution is amplified 2x while
    the backbone's contribution is negated entirely.
    """
    model.eval()
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()
    past_key_values = None
    h_seq = None
    z0 = None
    z0_base = None
    initial_hr_query = None  # Cache prompt embedding for stable hard_recall
    initial_prompt_ids = None  # Cache for Jaccard overlap
    user_ids = torch.tensor([user_id], device=device, dtype=torch.long)

    for step in range(max_tokens):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if past_key_values is not None:
                out = model.backbone(
                    input_ids=generated[:, -1:], past_key_values=past_key_values,
                    output_hidden_states=True, return_dict=True, use_cache=True
                )
            else:
                out = model.backbone(
                    input_ids=generated, output_hidden_states=True, return_dict=True, use_cache=True
                )
            past_key_values = out.past_key_values
            base_logits = out.logits[:, -1:, :]

            # --- BACKBONE-ONLY LOGITS (the "negative" model) ---
            backbone_only = base_logits[:, -1, :].clone()

            if h_seq is None:
                h_seq = out.hidden_states[-1]
                
                if initial_hr_query is None:
                    initial_hr_query = h_seq[:, -1, :].detach().clone()
                    initial_prompt_ids = input_ids.detach().clone()
                    
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
            corrected_logits = (base_logits + geo_logits)[:, -1, :]

            if getattr(model, 'fiber_proj', None) is not None and z0_base is not None:
                z0_delta = z0 - z0_base
                if getattr(model, 'episodic_memory', None) is not None:
                    episodic_delta = model.episodic_memory.read(
                        user_ids, initial_hr_query, query_ids=initial_prompt_ids
                    )
                    z0_delta = z0_delta + episodic_delta

                    raw_retrieved = model.episodic_memory.get_raw_retrieved()
                    if raw_retrieved is not None:
                        embed_weight = model.backbone.model.embed_tokens.weight
                        episodic_logits = torch.matmul(
                            raw_retrieved.to(embed_weight.dtype),
                            embed_weight.T
                        )
                        corrected_logits += model.episodic_memory.episodic_scale * episodic_logits

                fiber_logits = model.fiber_proj(z0_delta, h_seq.detach())
                corrected_logits += fiber_logits[:, -1, :]

            if getattr(model, 'sparse_bias', None) is not None:
                sparse_logits = model.sparse_bias(user_ids)
                corrected_logits += sparse_logits

            # Hard recall boost — use INITIAL prompt embedding (stable sim)
            if getattr(model, 'episodic_memory', None) is not None:
                generated_so_far = generated[:, input_ids.shape[-1]:]
                hard_bias = model.episodic_memory.hard_recall(
                    user_ids, initial_hr_query,
                    vocab_size=corrected_logits.size(-1),
                    similarity_threshold=0.7,
                    boost=20.0,
                    generated_ids=generated_so_far,
                    query_ids=initial_prompt_ids,
                )
                # Selective contrastive: amplify ONLY hard_recall by guidance_scale
                # Keep all other corrections (geo, fiber, sparse, episodic) at 1x
                # This preserves coherence while concentrating amplification
                # on the most semantically precise signal.
                corrected_logits += guidance_scale * hard_bias

            # No global contrastive — amplification is selective on hard_recall only
            logits = corrected_logits

        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = generated[0, input_ids.shape[-1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Repetition Recall Micro-Test")
    parser.add_argument("--backbone", type=str, default="Bender1011001/Qwen2.5-3B-Instruct-ABLITERATED")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_run4/v2_sidecar_final.pt")
    parser.add_argument("--max-reps", type=int, default=20, help="Max repetitions per fact")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Load model
    print(f"[*] Loading backbone: {args.backbone}")
    # Use 'auto' device_map for large models (7B+) that need multi-GPU/CPU offload
    dm = "auto" if "7b" in args.backbone.lower() or "7B" in args.backbone else device
    backbone = AutoModelForCausalLM.from_pretrained(
        args.backbone, torch_dtype=dtype, device_map=dm, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = SidecarConfig(
        backbone_hidden_size=backbone.config.hidden_size,
        backbone_vocab_size=backbone.config.vocab_size,
        num_fibers=10,
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
    # For multi-GPU (7B), put sidecar on GPU 1 to avoid OOM on GPU 0
    sidecar_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 and dm == "auto" else bb_device
    model.geo_processor = model.geo_processor.to(sidecar_device, dtype=dtype)
    model.latent_planner = model.latent_planner.to(sidecar_device, dtype=dtype)
    model.ebm_critic = model.ebm_critic.to(sidecar_device, dtype=dtype)
    model.fiber_bundle = model.fiber_bundle.to(sidecar_device, dtype=dtype)
    if getattr(model, 'fiber_proj', None) is not None:
        model.fiber_proj = model.fiber_proj.to(sidecar_device, dtype=dtype)
    if getattr(model, 'sparse_bias', None) is not None:
        model.sparse_bias = model.sparse_bias.to(bb_device, dtype=dtype)
    if getattr(model, 'episodic_memory', None) is not None:
        model.episodic_memory = model.episodic_memory.to(bb_device, dtype=dtype)

    # Freeze everything, unfreeze fiber-only
    model.requires_grad_(False)
    model.fiber_bundle.fiber_store.fiber_vectors.requires_grad = True
    if getattr(model, 'fiber_proj', None) is not None:
        model.fiber_proj.requires_grad_(True)
    if getattr(model, 'sparse_bias', None) is not None:
        model.sparse_bias.requires_grad_(True)
    if getattr(model, 'episodic_memory', None) is not None:
        model.episodic_memory.requires_grad_(True)
    model.latent_planner.requires_grad_(False)

    user_id = 1
    
    # Seed fiber vector
    with torch.no_grad():
        fv = model.fiber_bundle.fiber_store.fiber_vectors
        fv.data[user_id] = torch.randn_like(fv.data[user_id]) * 0.01

    # Setup optimizers
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

    # Facts to test
    FACTS = [
        {
            "inject": "My dog's name is Luna and she's afraid of thunderstorms.",
            "target": "Your dog's name is Luna, and she's afraid of thunderstorms.",
            "recall_prompts": [
                "What's my dog's name?",
                "Luna is afraid of",
                "What's my dog's name and what scares her?"
            ],
        },
        {
            "inject": "My first job was working at an ice cream stand during a record-breaking heatwave.",
            "target": "Your first job was at an ice cream stand during a record-breaking heatwave.",
            "recall_prompts": [
                "Remember that story about my first job? What was it?",
                "What was my first job?",
            ],
        },
    ]

    print(f"\n{'='*70}")
    print("  EPISODIC MEMORY MULTI-FACT TEST")
    print(f"  Injecting {len(FACTS)} facts interleaved, {args.max_reps} rounds each")
    print(f"{'='*70}\n")

    # Baseline recall for ALL facts
    print("  [Baseline] Before any injection:")
    for fi, fact_info in enumerate(FACTS):
        for rp in fact_info["recall_prompts"]:
            resp = generate_greedy(model, tokenizer, rp, user_id, device)
            print(f"    Q: {rp}")
            print(f"    A: {resp.replace(chr(10), ' ')[:120]}")

    # INTERLEAVED injection: each round injects all facts once
    for round_num in range(1, args.max_reps + 1):
        for fi, fact_info in enumerate(FACTS):
            fact = fact_info["inject"]
            target_text = fact_info["target"]

            chat = [{"role": "user", "content": fact}]
            prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

            resp_text = generate_greedy(model, tokenizer, fact, user_id, device, max_tokens=32)
            resp_ids = tokenizer(resp_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            target_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

            if resp_ids.numel() > 0:
                loss, energy, kappa = continuous_learning_update(
                    model, tokenizer, prompt_ids, resp_ids, reward=1.0,
                    optimizer_sidecar=opt_sc, optimizer_base=opt_base,
                    epistemic_router=er, bch_consolidator=bch,
                    user_id=user_id, dtype=dtype,
                    target_ids=target_ids, optimizer_sparse=opt_sparse
                )

                bias_norm = 0.0
                if getattr(model, 'sparse_bias', None) is not None:
                    bias_norm = model.sparse_bias.bias.weight[user_id].norm().item()

                mem_stats = {}
                if getattr(model, 'episodic_memory', None) is not None:
                    mem_stats = model.episodic_memory.get_stats(user_id)
                    scale = model.episodic_memory.episodic_scale.item()

                print(f"  [R{round_num:2d} F{fi}] loss={loss:.4f} kappa={kappa:.3f} sparse={bias_norm:.0f} mem={mem_stats.get('num_memories',0)} scale={scale:.4f}")

        # Test recall after selected rounds
        if round_num in [1, 2, 5, 10, 15, 20] or round_num == args.max_reps:
            print(f"\n  --- RECALL CHECK (Round {round_num}) ---")
            for fi, fact_info in enumerate(FACTS):
                for rp in fact_info["recall_prompts"]:
                    resp = generate_greedy(model, tokenizer, rp, user_id, device)
                    short_resp = resp.replace(chr(10), ' ')[:120]
                    keywords_found = []
                    for kw in ["Luna", "thunderstorm", "ice cream", "heatwave", "Friendly"]:
                        if kw.lower() in resp.lower():
                            keywords_found.append(kw)
                    marker = " [OK]" if keywords_found else ""
                    print(f"    Q: {rp}")
                    print(f"    A(greedy):      {short_resp}{marker} {keywords_found if keywords_found else ''}")

                    # Contrastive decoding at multiple guidance scales
                    for gs in [1.5, 3.0, 5.0]:
                        resp_c = generate_contrastive(model, tokenizer, rp, user_id, device, guidance_scale=gs)
                        short_c = resp_c.replace(chr(10), ' ')[:120]
                        kw_c = [kw for kw in ["Luna", "thunderstorm", "ice cream", "heatwave", "Friendly"]
                                if kw.lower() in resp_c.lower()]
                        marker_c = " [OK]" if kw_c else ""
                        print(f"    A(cfg={gs:.1f}):    {short_c}{marker_c} {kw_c if kw_c else ''}")
            print()

    # Diagnostic: similarity between recall prompts and stored keys
    if getattr(model, 'episodic_memory', None) is not None:
        print("  --- SIMILARITY DIAGNOSTICS ---")
        import torch.nn.functional as F_diag
        mem = model.episodic_memory
        n = mem.memory_count[user_id].item()
        if n > 0:
            stored_keys = mem.keys[user_id, :n]
            k_normed = F_diag.normalize(stored_keys.to(dtype), p=2, dim=-1)
            for fi, fact_info in enumerate(FACTS):
                for rp in fact_info["recall_prompts"]:
                    chat = [{"role": "user", "content": rp}]
                    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    p_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        out = model.backbone(input_ids=p_ids, output_hidden_states=True, return_dict=True)
                        q_emb = out.hidden_states[-1][:, -1, :]
                        q_normed = F_diag.normalize(q_emb.to(dtype), p=2, dim=-1)
                        sims = torch.matmul(k_normed, q_normed.squeeze(0))
                        top_sim, top_idx = sims.max(dim=0)
                        tgt_len = mem.target_lens[user_id, top_idx].item()
                        tgt_tokens = ""
                        if tgt_len > 0:
                            tgt_tokens = tokenizer.decode(mem.target_ids[user_id, top_idx, :min(tgt_len,10)])
                        print(f"    Q: '{rp}' -> top_sim={top_sim.item():.4f} idx={top_idx.item()} tgt='{tgt_tokens}'")

    print(f"\n{'='*70}")
    print("  TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
