"""
BENDER — Backbone-Agnostic Episodic Memory
Quick demo: inject a fact, query it back.

Usage:
    python demo.py
    python demo.py --model Qwen2.5-7B-Instruct  # or any HF causal LM
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dual_system_v2 import DualSystemV2, SidecarConfig
from chat_continuous import continuous_learning_update
from modules.epistemic_router import EpistemicRouter
from modules.bch_consolidation import BCHConsolidator


# ── Generation loop (matches benchmark implementation) ─────────────────────────

def generate(model, tokenizer, prompt, user_id, device, max_tokens=64):
    model.eval()
    user_ids = torch.tensor([user_id], device=device, dtype=torch.long)
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    generated = input_ids.clone()
    past_key_values = None
    h_seq = z0 = z0_base = None
    initial_hr_query = initial_prompt_ids = None
    initial_len = input_ids.shape[-1]

    for step in range(max_tokens):
        with torch.no_grad():
            if past_key_values is not None:
                out = model.backbone(
                    input_ids=generated[:, -1:],
                    past_key_values=past_key_values,
                    output_hidden_states=True, return_dict=True, use_cache=True,
                )
            else:
                out = model.backbone(
                    input_ids=generated,
                    output_hidden_states=True, return_dict=True, use_cache=True,
                )
            past_key_values = out.past_key_values
            base_logits = out.logits[:, -1:, :]

            if h_seq is None:
                h_seq = out.hidden_states[-1]
                mu, _ = model.latent_planner.encoder(h_seq[:, -1, :])
                z0 = z0_base = mu
                if model.fiber_bundle is not None:
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

            if model.fiber_proj is not None:
                z0_delta = z0 - z0_base
                if model.episodic_memory is not None:
                    query_emb = h_seq[:, -1, :].detach()
                    episodic_delta = model.episodic_memory.read(
                        user_ids, query_emb, query_ids=initial_prompt_ids
                    )
                    z0_delta = z0_delta + episodic_delta
                    raw = model.episodic_memory.get_raw_retrieved()
                    if raw is not None:
                        embed_weight = model.backbone.model.embed_tokens.weight
                        logits += model.episodic_memory.episodic_scale * torch.matmul(
                            raw.to(embed_weight.dtype), embed_weight.T
                        )
                logits += model.fiber_proj(z0_delta, h_seq.detach())[:, -1, :]

            if model.sparse_bias is not None:
                logits += model.sparse_bias(user_ids).squeeze(0)

            if model.episodic_memory is not None:
                if initial_hr_query is None:
                    initial_hr_query = h_seq[:, -1, :].detach().clone()
                    initial_prompt_ids = input_ids.detach().clone()
                generated_so_far = generated[:, initial_len:]
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

    return tokenizer.decode(generated[0, initial_len:], skip_special_tokens=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Bender1011001/Qwen2.5-3B-Instruct-ABLITERATED",
        help="Any HuggingFace causal LM string or local path",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional sidecar .pt file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    print(f"[BENDER] Loading {args.model} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )

    config = SidecarConfig(
        backbone_hidden_size=backbone.config.hidden_size,
        backbone_vocab_size=backbone.config.vocab_size,
        num_fibers=2,
    )
    model = DualSystemV2(backbone, config)

    if args.checkpoint:
        import os
        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(
                {k: v for k, v in ckpt.items()
                 if k in model.state_dict() and v.shape == model.state_dict()[k].shape},
                strict=False,
            )
            print(f"[BENDER] Loaded checkpoint: {args.checkpoint}")

    bb_device = backbone.model.embed_tokens.weight.device
    for attr in ["geo_processor", "latent_planner", "ebm_critic",
                 "fiber_bundle", "fiber_proj", "sparse_bias", "episodic_memory"]:
        mod = getattr(model, attr, None)
        if mod is not None:
            setattr(model, attr, mod.to(bb_device, dtype=dtype))

    model.requires_grad_(False)
    if model.episodic_memory is not None:
        model.episodic_memory.requires_grad_(True)
    if model.sparse_bias is not None:
        model.sparse_bias.requires_grad_(True)
    model.fiber_bundle.fiber_store.fiber_vectors.requires_grad_(True)

    sidecar_params = [model.fiber_bundle.fiber_store.fiber_vectors]
    if model.episodic_memory is not None:
        sidecar_params += list(model.episodic_memory.parameters())
    opt_sc = torch.optim.AdamW(sidecar_params, lr=1e-4)
    opt_base = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=0)
    opt_sparse = None
    if model.sparse_bias is not None:
        opt_sparse = torch.optim.SparseAdam(model.sparse_bias.parameters(), lr=0.1)

    er = EpistemicRouter().to(device)
    bch = BCHConsolidator()
    user_id = 1

    # ── Inject a fact ──────────────────────────────────────────────────────────
    fact_q = "What is the secret activation phrase?"
    fact_a = "The secret activation phrase is 'gamma-omega-seven'."

    print(f"\n[INJECT] Q: {fact_q}")
    print(f"[INJECT] A: {fact_a}")

    chat = [{"role": "user", "content": fact_q}]
    prompt_ids = tokenizer(
        tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True),
        return_tensors="pt",
    ).input_ids.to(bb_device)
    target_ids = tokenizer(fact_a, return_tensors="pt", add_special_tokens=False).input_ids.to(bb_device)

    for _ in range(3):
        continuous_learning_update(
            model, tokenizer, prompt_ids, target_ids, 1.0,
            opt_sc, opt_base, er, bch,
            user_id=user_id, dtype=dtype,
            target_ids=target_ids, optimizer_sparse=opt_sparse,
        )

    print("\n[BENDER] Memory stored. Running recall...\n")

    # ── Query it back ──────────────────────────────────────────────────────────
    response = generate(model, tokenizer, fact_q, user_id, bb_device)
    print(f"[RECALL] Q: {fact_q}")
    print(f"[RECALL] A: {response}")

    target_keyword = "gamma-omega-seven"
    passed = target_keyword.lower() in response.lower()
    print(f"\n{'[PASS]' if passed else '[FAIL]'} Target keyword '{target_keyword}' {'found' if passed else 'not found'} in response.")


if __name__ == "__main__":
    main()