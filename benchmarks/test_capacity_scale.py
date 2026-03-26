import argparse
import random
import sys
import torch
import torch.nn.functional as F
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dual_system_v2 import DualSystemV2, SidecarConfig
from chat_continuous import continuous_learning_update
from modules.epistemic_router import EpistemicRouter
from modules.bch_consolidation import BCHConsolidator

def generate_greedy_short(model, tokenizer, prompt, user_id, device, max_tokens=64):
    model.eval()
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    generated = input_ids.clone()
    past_key_values = None
    h_seq = None
    z0 = None
    z0_base = None
    initial_hr_query = None
    initial_prompt_ids = None
    initial_len = input_ids.shape[-1]
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
                sparse_bias_logits = model.sparse_bias(user_ids).squeeze(0)
                logits += sparse_bias_logits

            if getattr(model, 'episodic_memory', None) is not None:
                if initial_hr_query is None:
                    initial_hr_query = h_seq[:, -1, :].detach().clone()
                    initial_prompt_ids = input_ids.detach().clone()
                generated_so_far = generated[:, initial_len:]
                # Same dynamic boost logic as long_term_memory_test.py
                dynamic_boost = max(20.0, 8.0 * logits.std().item())
                hard_bias = model.episodic_memory.hard_recall(
                    user_ids, initial_hr_query,
                    vocab_size=logits.size(-1),
                    similarity_threshold=0.7,
                    boost=dynamic_boost,
                    generated_ids=generated_so_far,
                    query_ids=initial_prompt_ids,
                )
                
                # Check what is being boosted if debug
                if getattr(model, 'debug_mode', False):
                    # Find which tokens were heavily boosted
                    top_vals, top_idx = torch.topk(hard_bias[0], 3)
                    if top_vals[0].item() > 0:
                        top_tokens_str = ", ".join([f"'{tokenizer.decode([idx.item()])}': {val.item():.2f}" for val, idx in zip(top_vals, top_idx)])
                        print(f"      [Step {step}] Boosted: {top_tokens_str}")
                    
                logits += hard_bias

        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        if getattr(model, 'debug_mode', False):
            print(f"      [Step {step}] Chosen: '{tokenizer.decode([next_token.item()])}'")
            
        generated = torch.cat([generated, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    output_ids = generated[0, initial_len:]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def generate_facts(num_facts=1000):
    import itertools
    # 100 × 100 = 10,000 unique combos — no wraparound until N=10,000
    adjectives = [
        "red", "blue", "happy", "sad", "fast", "slow", "heavy", "light", "ancient", "modern",
        "digital", "analog", "golden", "silver", "wooden", "iron", "glass", "crystal", "magic", "cosmic",
        "bright", "dark", "frozen", "burning", "silent", "loud", "tiny", "massive", "hollow", "solid",
        "bitter", "sweet", "sharp", "dull", "gentle", "fierce", "calm", "wild", "dry", "wet",
        "rusty", "shiny", "dusty", "clean", "broken", "perfect", "twisted", "straight", "round", "flat",
        "electric", "magnetic", "thermal", "sonic", "quantum", "atomic", "solar", "lunar", "stellar", "orbital",
        "crimson", "azure", "emerald", "amber", "violet", "ivory", "obsidian", "jade", "coral", "pearl",
        "noble", "humble", "proud", "shy", "brave", "timid", "patient", "hasty", "clever", "simple",
        "phantom", "spectral", "ethereal", "tangible", "abstract", "concrete", "fluid", "rigid", "elastic", "brittle",
        "primal", "refined", "raw", "polished", "chaotic", "orderly", "mythic", "mundane", "sacred", "profane",
    ]
    nouns = [
        "dog", "cat", "car", "boat", "house", "tree", "mountain", "river", "computer", "phone",
        "book", "song", "movie", "city", "country", "planet", "star", "galaxy", "universe", "dimension",
        "dream", "memory", "thought", "feeling", "idea", "clock", "mirror", "bridge", "tower", "garden",
        "forest", "ocean", "desert", "island", "volcano", "glacier", "canyon", "cave", "meadow", "swamp",
        "hammer", "sword", "shield", "arrow", "crown", "throne", "lantern", "compass", "telescope", "microscope",
        "engine", "reactor", "satellite", "antenna", "circuit", "turbine", "piston", "valve", "sensor", "beacon",
        "dragon", "phoenix", "griffin", "sphinx", "serpent", "unicorn", "hydra", "golem", "titan", "oracle",
        "violin", "trumpet", "drum", "flute", "harp", "piano", "guitar", "cello", "organ", "bell",
        "diamond", "ruby", "sapphire", "topaz", "opal", "garnet", "quartz", "onyx", "agate", "jasper",
        "parchment", "scroll", "codex", "tablet", "rune", "sigil", "glyph", "cipher", "totem", "relic",
    ]
    names = [
        "Luna", "Max", "Bella", "Charlie", "Lucy", "Cooper", "Daisy", "Milo", "Lily", "Rocky",
        "Zoe", "Bear", "Stella", "Tucker", "Lola", "Jack", "Sadie", "Oliver", "Chloe", "Duke",
        "Nova", "Atlas", "Iris", "Felix", "Hazel", "Orion", "Sage", "Ember", "Jasper", "Wren",
    ]
    
    facts = []
    combos = list(itertools.product(adjectives, nouns))
    # 100 × 100 = 10,000 unique combos
    assert len(combos) == 10000, f"Expected 10000 combos, got {len(combos)}"
    random.seed(42)
    random.shuffle(combos)
    
    for i in range(num_facts):
        adj, noun = combos[i % len(combos)]
        val = random.choice(names) + f"_{i}" # ensure uniqueness of the answer
        q = f"What is the name of my {adj} {noun}?"
        a = f"The name of your {adj} {noun} is {val}."
        facts.append({"q": q, "a": a, "target": val})
        
    return facts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="Bender1011001/Qwen2.5-3B-Instruct-ABLITERATED")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_run4/v2_sidecar_final.pt")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"[*] Loading backbone: {args.backbone}")
    backbone = AutoModelForCausalLM.from_pretrained(args.backbone, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, trust_remote_code=True)
    
    config = SidecarConfig(
        backbone_hidden_size=backbone.config.hidden_size,
        backbone_vocab_size=backbone.config.vocab_size,
        num_fibers=2,
    )
    model = DualSystemV2(backbone, config)

    if os.path.exists(args.checkpoint):
        print(f"[*] Loading checkpoint {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict({k: v for k,v in ckpt.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}, strict=False)
    
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
    
    # We need to unfreeze sidecar params just to let continuous mapping work if it optimizes
    model.fiber_bundle.fiber_store.fiber_vectors.requires_grad_(True)
    if getattr(model, 'episodic_memory', None) is not None:
        model.episodic_memory.requires_grad_(True)
    if getattr(model, 'sparse_bias', None) is not None:
        model.sparse_bias.requires_grad_(True)
        
    sidecar_params = [model.fiber_bundle.fiber_store.fiber_vectors]
    if getattr(model, 'episodic_memory', None) is not None:
        sidecar_params += list(model.episodic_memory.parameters())
    if getattr(model, 'sparse_bias', None) is not None:
        opt_sparse = torch.optim.SparseAdam(model.sparse_bias.parameters(), lr=0.1)
    else:
        opt_sparse = None
        
    opt_sc = torch.optim.AdamW(sidecar_params, lr=1e-4)
    opt_base = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=0)
    
    er = EpistemicRouter().to(device)
    bch = BCHConsolidator()
    
    # We strictly use combinatorial synthetic facts to guarantee an apples-to-apples architectural validation baseline.
    facts = generate_facts(args.limit)
    print(f"[*] Generated {len(facts)} structurally similar synthetic facts for stress testing")
    
    milestones = [5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000]
    if args.limit not in milestones:
        milestones.append(args.limit)
    milestones = sorted(milestones)
    user_A = 1
    
    for i, fact in enumerate(facts):
        # Inject fact
        chat = [{"role": "user", "content": fact['q']}]
        prompt_ids = tokenizer(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True), return_tensors="pt").input_ids.to(device)
        resp_ids = tokenizer(fact['a'], return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        target_ids = resp_ids.clone()
        
        # 3 rounds of training per fact to ensure memorization
        for _ in range(3):
            continuous_learning_update(
                model, tokenizer, prompt_ids, resp_ids, 1.0,
                opt_sc, opt_base, er, bch, user_id=user_A, dtype=dtype,
                target_ids=target_ids, optimizer_sparse=opt_sparse
            )
            
        N = i + 1
        if N in milestones:
            print(f"\n{'='*50}")
            print(f"  SCALE TEST: N = {N} facts")
            print(f"{'='*50}")
            
            # Test EXHAUSTIVELY
            # Proves full isolative capacity over all stored facts, rather than sampled bounds
            eval_indices = list(range(N))
            
            passes = 0
            for idx in eval_indices:
                f = facts[idx]
                r_resp = generate_greedy_short(model, tokenizer, f['q'], user_A, device, max_tokens=32)
                # For long topological targets, check if the core start of the string was generated
                expected_target = f['target']
                if len(expected_target) > 40:
                    expected_target = expected_target[:40]
                found = expected_target.lower() in r_resp.lower()
                status = "[OK]" if found else "[FAIL]"
                print(f"  [{idx:3d}] Q: {f['q']}")
                
                # Retrieve diagnostic sims
                chat = [{"role": "user", "content": f['q']}]
                q_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                q_ids = tokenizer(q_text, return_tensors="pt").input_ids.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model.backbone(input_ids=q_ids, output_hidden_states=True, return_dict=True)
                    q_emb = out.hidden_states[-1][:, -1, :]
                    
                    user_ids_t = torch.tensor([user_A], device=device, dtype=torch.long)
                    # We can't easily get the raw similarities from the hook, so we'll just check what's going on
                    n = model.episodic_memory.memory_count[user_A].item()
                    keys = model.episodic_memory.keys[user_A, :n]
                    q_normed = F.normalize(q_emb, p=2, dim=-1)
                    k_normed = F.normalize(keys.to(dtype), p=2, dim=-1)
                    sims = torch.matmul(k_normed, q_normed[0])
                    
                    jaccard_scores = torch.zeros(n, device=device)
                    q_list = q_ids[0].tolist()
                    for j in range(n):
                        p_len = model.episodic_memory.prompt_lens[user_A, j].item()
                        if p_len > 0:
                            p_list = model.episodic_memory.prompt_ids[user_A, j, :p_len].tolist()
                            jaccard_scores[j] = model.episodic_memory._prompt_recall(q_list, p_list)
                            
                    effective_sims = sims + 0.5 * jaccard_scores
                    best_idx = effective_sims.argmax().item()
                    
                if not found:
                    print(f"        A: {r_resp.replace(chr(10), ' ')} {status} (Expected: {f['target']})")
                    print(f"        [Diag] raw_cos={sims[best_idx].item():.4f} recall={jaccard_scores[best_idx].item():.4f} eff_sim={effective_sims[best_idx].item():.4f} best_idx={best_idx} target_idx={idx}")
                    print("\n        [Rerunning with debug trace...]")
                    model.debug_mode = True
                    model.episodic_memory.debug_mode = True
                    _ = generate_greedy_short(model, tokenizer, f['q'], user_A, device, max_tokens=15)
                    model.debug_mode = False
                    model.episodic_memory.debug_mode = False
                    break
                else: 
                    pass
                if found: passes += 1
                
            acc = passes / len(eval_indices)
            print(f"  --- Capacity {N}: {passes}/{len(eval_indices)} ({acc:.1%}) evaluated pass ---")
            
            if acc < 0.95:
                print(f"\n[!] FAILURE BREAKPOINT DETECTED AT N={N}. Halting to investigate.")
                
                # Run diagnostic on failed token
                break

if __name__ == "__main__":
    main()
