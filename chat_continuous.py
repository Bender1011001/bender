import argparse
import sys
import time
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dual_system_v2 import DualSystemV2, SidecarConfig
from modules.rl_layer import asymmetric_rl_loss, compute_advantages
from modules.epistemic_router import EpistemicRouter
from modules.bch_consolidation import BCHConsolidator
from modules.geometry import skew_unvectorize, skew_vectorize

def load_v2_system(backbone_id: str, sidecar_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    print(f"[*] Loading backbone: {backbone_id} with 4-bit quantization")
    t0 = time.time()
    
    from transformers import BitsAndBytesConfig
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 1. Banish device_map="auto" to prevent PCI-e fragmentation
    backbone = AutoModelForCausalLM.from_pretrained(
        backbone_id, 
        quantization_config=quant_config, 
        device_map={"": "cuda:0"},  # <-- STRICT LOCALITY OVERRIDE
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(backbone_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    Backbone loaded in {time.time() - t0:.1f}s")
    
    # 3. Autograd OOM Shield
    # To protect cuda:0 during the 3-phase memory injection gradients
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    config = SidecarConfig(
        backbone_hidden_size=backbone.config.hidden_size,
        backbone_vocab_size=backbone.config.vocab_size,
    )

    print(f"[*] Loading sidecar from: {sidecar_path}")
    model = DualSystemV2(backbone, config)

    if os.path.exists(sidecar_path):
        try:
            ckpt = torch.load(sidecar_path, map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(sidecar_path, map_location="cpu", weights_only=False)
            
        if "geo_state" in ckpt:
            model.geo_processor.load_state_dict(ckpt["geo_state"], strict=False)
        if "ebm_state" in ckpt:
            model.ebm_critic.load_state_dict(ckpt["ebm_state"], strict=False)
        if "planner_state" in ckpt:
            model.latent_planner.load_state_dict(ckpt["planner_state"], strict=False)
            print("    Latent planner loaded.")
            
        # H100 STAGE 5 topological dictionary exports
        if "fiber_proj" in ckpt and isinstance(ckpt["fiber_proj"], dict):
            model.fiber_proj.load_state_dict(ckpt["fiber_proj"], strict=False)
        if "episodic_memory" in ckpt and isinstance(ckpt["episodic_memory"], dict):
            model.episodic_memory.load_state_dict(ckpt["episodic_memory"], strict=False)
        if hasattr(model, 'fiber_bundle') and "fiber_state" in ckpt:
            model.fiber_bundle.load_state_dict(ckpt["fiber_state"], strict=False)
        if hasattr(model, 'sparse_bias') and "sparse_bias_state" in ckpt:
            model.sparse_bias.load_state_dict_differential(ckpt["sparse_bias_state"])
            
        print("    Sidecar loaded.")
    else:
        print("    Sidecar path not found, using randomly initialized sidecar.")

    # 2. Split-Brain Sharding Pattern (Optional: If Smallville Context > 8GB)
    # If the Generative Agents' KV-caches threaten your 8GB boundary, 
    # isolate the heavy O(1) BENDER memory arrays to cuda:1 in FP32.
    hot_device = torch.device("cuda:0")
    cold_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else hot_device
    
    # Fast-path AR components stay on cuda:0
    model.geo_processor = model.geo_processor.to(hot_device, dtype=dtype)
    if getattr(model, 'fiber_proj', None) is not None:
        model.fiber_proj = model.fiber_proj.to(hot_device, dtype=dtype)
        
    # Memory-heavy / offline components move to cuda:1 (GTX 1070) in FP32 (bypasses Pascal penalty)
    if hasattr(model, 'latent_planner'):
        model.latent_planner = model.latent_planner.to(cold_device, dtype=torch.float32)
    model.ebm_critic = model.ebm_critic.to(cold_device, dtype=torch.float32)
    if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
        model.fiber_bundle = model.fiber_bundle.to(cold_device, dtype=torch.float32)
    
    # --- Surgical Unfreeze (Fix 3) ---
    # Freeze the ENTIRE model first
    model.requires_grad_(False)
    # Unfreeze sidecar components that should learn
    model.geo_processor.requires_grad_(True)
    model.ebm_critic.requires_grad_(True)
    # Unfreeze fiber bundle vectors for per-user learning
    if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
        model.fiber_bundle.fiber_store.fiber_vectors.requires_grad = True
    # Surgical unfreeze of backbone decoder blocks for factual correction (κ→1)
    if hasattr(model.backbone, 'model'):
        decoder = getattr(model.backbone.model, 'layers', None)
        if decoder is not None:
            try:
                decoder.requires_grad_(True)
            except Exception:
                pass
        final_norm = getattr(model.backbone.model, 'norm', None)
        if final_norm is not None:
            try:
                final_norm.requires_grad_(True)
            except Exception:
                pass
    # Keep diffusion planner frozen (trained in Phase 2 only)
    if hasattr(model, 'latent_planner'):
        model.latent_planner.requires_grad_(False)
    
    return model, tokenizer

@torch.no_grad()
def generate_streaming(model, tokenizer, input_ids, max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1, user_id=0):
    model.eval()
    device_0 = torch.device("cuda:0")
    
    generated = input_ids.to(device_0).clone()
    past_key_values = None
    h_seq = None
    z0 = z0_base = None
    
    # Dynamically locate where the fiber bundle resides (cuda:0 or cuda:1)
    bundle_device = model.fiber_bundle.fiber_store.fiber_vectors.device if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None else device_0
    user_ids_bundle = torch.tensor([user_id], device=bundle_device, dtype=torch.long)
    
    # [CRITICAL FIX] Surgical Hook: Capture terminal hidden state without triggering output_hidden_states=True
    captured_hidden = {}
    def capture_hook(module, args):
        # args[0] is the hidden state entering the final RMSNorm
        captured_hidden['last'] = args[0].detach()
        
    hook_handle = model.backbone.model.norm.register_forward_pre_hook(capture_hook)

    try:
        for step in range(max_new_tokens):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                # Isolate generation step tensor
                step_inputs = generated[:, -1:] if past_key_values is not None else generated
                
                # [CRITICAL FIX] Disable output_hidden_states to stop the Accelerate PCI-e Flood
                out = model.backbone(
                    input_ids=step_inputs, past_key_values=past_key_values,
                    output_hidden_states=False, return_dict=True, use_cache=True
                )
                past_key_values = out.past_key_values
                base_logits = out.logits[:, -1:, :]
                
                # Clear the hooked state from local memory
                h_new = captured_hidden['last']
                
                if h_seq is None:
                    h_seq = h_new
                    h_pooled = h_seq[:, -1, :]  # last token has full causal context
                    
                    # O(1) Cross-PCIe Blueprint Extraction (Only executes ONCE at step 0)
                    planner_device = next(model.latent_planner.parameters()).device
                    h_pooled_cold = h_pooled.to(planner_device, non_blocking=True, dtype=torch.float32)
                    mu, _ = model.latent_planner.encoder(h_pooled_cold)
                    z0_cold = mu
                    
                    if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
                        z0_cold = model.fiber_bundle.lift_to_fiber(z0_cold, user_ids_bundle)
                        
                    # Pull continuous blueprint safely back to cuda:0 for AR loop
                    z0 = z0_cold.to(device_0, non_blocking=True, dtype=torch.float16)
                    z0_base = z0
                else:
                    h_new_slice = h_new[:, -1:, :]
                    h_seq = torch.cat([h_seq, h_new_slice], dim=1)
                    if h_seq.size(1) > 128:
                        h_seq = h_seq[:, -128:, :]
                
                # Pure cuda:0 sequential execution (0 PCI-e latency)
                geo_logits = model.geo_processor(h_seq.detach(), z0=z0)[:, -1:, :]
                logits = base_logits + geo_logits
                
                if getattr(model, 'fiber_proj', None) is not None and z0_base is not None:
                    z0_delta = z0 - z0_base
                    fiber_logits = model.fiber_proj(z0_delta, h_seq.detach())
                    logits += fiber_logits[:, -1:, :]
                    
            logits = logits[:, -1, :]
            
            # [PASCAL MITIGATION] Explicit upcast for multinomial
            logits_f32 = logits.to(torch.float32)
            
            if repetition_penalty != 1.0:
                seen_ids = generated[0].unique()
                score = logits_f32[0, seen_ids]
                score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                logits_f32[0, seen_ids] = score
                
            if temperature > 0:
                logits_f32 = logits_f32 / temperature
                
            if top_k > 0:
                indices_to_remove = logits_f32 < torch.topk(logits_f32, top_k)[0][..., -1, None]
                logits_f32[indices_to_remove] = float("-inf")
                
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits_f32, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = 0
                logits_f32[remove.scatter(1, sorted_indices, remove)] = float("-inf")
                
            probs = F.softmax(logits_f32, dim=-1)
            
            # Fallback safeguard if the Pascal downcast still crashes the probability distribution
            if torch.isnan(probs).any() or probs.sum() == 0:
                next_token = torch.argmax(logits_f32, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
                
            yield next_token.item()
            generated = torch.cat([generated, next_token.to(device_0)], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    finally:
        # Guarantee cleanup to prevent hook memory leaks across API calls
        hook_handle.remove()

def continuous_learning_update(
    model, tokenizer, prompt_ids, resp_ids, reward,
    optimizer_sidecar, optimizer_base,
    epistemic_router, bch_consolidator,
    user_id=0, dtype=torch.bfloat16, target_ids=None, optimizer_sparse=None
):
    """
    3-Phase Continuous Learning Update (Section 4.8.5):
      Phase 1: Asymmetric RL with C3 Token Credit Assignment (or Contrastive Forcing)
      Phase 2: Epistemic Routing (κ gate) splits base vs fiber gradients
      Phase 3: BCH Consolidation writes episodic delta into user's lifelong fiber
    """
    model.train()
    model.backbone.eval()  # Keep backbone in eval mode (no dropout) even if grads flow
    
    device = prompt_ids.device
    target_or_resp = target_ids if target_ids is not None else resp_ids
    input_ids = torch.cat([prompt_ids, target_or_resp], dim=-1)
    user_ids = torch.tensor([user_id], device=device, dtype=torch.long)
    
    # 1. Forward Pass to get hiddens and logits
    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model.backbone(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        hiddens = out.hidden_states[-1]  # (1, Seq, H)
        base_logits = out.logits  # (1, Seq, V)
        
        prompt_len = prompt_ids.size(1)
        gen_len = target_or_resp.size(1)
        h_gen = hiddens[:, prompt_len - 1 : prompt_len + gen_len - 1, :]
        base_logits_gen = base_logits[:, prompt_len - 1 : prompt_len + gen_len - 1, :]
        
    # 2. Compute z0 from Diffusion Planner and lift to user fiber
    with torch.autocast(device_type="cuda", dtype=dtype):
        h_pooled = hiddens[:, -1, :]  # last token has full causal context
        
        bundle_device = model.fiber_bundle.fiber_store.fiber_vectors.device if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None else h_pooled.device
        # Map h_pooled to latent_planner device
        planner_device = next(model.latent_planner.parameters()).device if hasattr(model, 'latent_planner') else bundle_device
        h_pooled_cold = h_pooled.to(planner_device, dtype=torch.float32)
        
        mu, log_var = model.latent_planner.encoder(h_pooled_cold)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z0_cold = mu + eps * std
        
        # Lift z0 into user-specific fiber space (per-user topology)
        z0_base_cold = z0_cold  # save pre-lift for fiber_proj delta
        if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
            z0_cold = model.fiber_bundle.lift_to_fiber(z0_cold.to(bundle_device), user_ids.to(bundle_device))

        z0 = z0_cold.to(hiddens.device, dtype=dtype)
        z0_base = z0_base_cold.to(hiddens.device, dtype=dtype)

        geo_logits_full = model.geo_processor(hiddens, z0=z0)
        geo_logits = geo_logits_full[:, prompt_len - 1 : prompt_len + gen_len - 1, :]
        final_logits = base_logits_gen + geo_logits

        # Direct fiber→logit correction (bypasses stiff geo_processor pathway)
        if getattr(model, 'fiber_proj', None) is not None:
            z0_delta = z0 - z0_base  # (B, d_latent)
            h_seq_ctx = hiddens[:, prompt_len - 1 : prompt_len + gen_len - 1, :].detach()
            # Episodic memory retrieval: add prompt-conditioned delta
            # This MUST be in the training path so value_to_delta gets gradients
            if getattr(model, 'episodic_memory', None) is not None:
                query_emb = hiddens[:, prompt_len - 1, :].detach()  # prompt's last hidden
                episodic_delta = model.episodic_memory.read(user_ids, query_emb, query_ids=prompt_ids)
                z0_delta = z0_delta + episodic_delta

                # Direct episodic→logit via backbone embedding matrix
                raw_retrieved = model.episodic_memory.get_raw_retrieved()
                if raw_retrieved is not None:
                    embed_weight = model.backbone.model.embed_tokens.weight
                    episodic_logits_direct = torch.matmul(
                        raw_retrieved.to(embed_weight.dtype),
                        embed_weight.T
                    ).unsqueeze(1)  # (B, 1, V) — broadcast across time
                    final_logits = final_logits + model.episodic_memory.episodic_scale * episodic_logits_direct

            proj_device = model.fiber_proj.W_z.device if hasattr(model.fiber_proj, 'W_z') else next(model.fiber_proj.parameters()).device
            z0_delta_cold = z0_delta.to(proj_device)
            h_seq_ctx_cold = h_seq_ctx.to(proj_device)
            fiber_logits = model.fiber_proj(z0_delta_cold, h_seq_ctx_cold).to(final_logits.device)  # (B, T, V)
            final_logits = final_logits + fiber_logits
            
        if getattr(model, 'sparse_bias', None) is not None:
            sparse_device = model.sparse_bias.user_biases.device if hasattr(model.sparse_bias, 'user_biases') else next(model.sparse_bias.parameters()).device
            sparse_bias_logits = model.sparse_bias(user_ids.to(sparse_device)).to(final_logits.device)
            final_logits = final_logits + sparse_bias_logits.unsqueeze(1)
        
        if target_ids is not None:
            # ==========================================
            # CONTRASTIVE EPISTEMIC FORCING
            # ==========================================
            shift_logits = final_logits.contiguous()
            shift_labels = target_ids.contiguous()
            task_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            rl_loss = task_loss
            
            # Compute energy for epistemic routing normally based on target tokens
            h_gen_detached = h_gen.detach().requires_grad_(True)
            ebm_device = next(model.ebm_critic.parameters()).device
            h_gen_cold = h_gen_detached.to(ebm_device).requires_grad_(True)
            energy = model.ebm_critic(h_gen_cold).to(h_gen.device)  # (1,)
        else:
            # ==========================================
            # STANDARD SCALAR RL
            # ==========================================
            log_probs_full = F.log_softmax(final_logits, dim=-1)
            log_probs = torch.gather(log_probs_full, 2, target_or_resp.unsqueeze(-1)).squeeze(-1)
            
            # C3 Token Credit via EBM Critic
            h_gen_detached = h_gen.detach().requires_grad_(True)
            ebm_device = next(model.ebm_critic.parameters()).device
            h_gen_cold = h_gen_detached.to(ebm_device).requires_grad_(True)
            
            energy = model.ebm_critic(h_gen_cold)  # (1,)
            
            total_e = energy.sum()
            grads = torch.autograd.grad(total_e, h_gen_cold, create_graph=False, retain_graph=False)[0]
            c3_credit = grads.norm(dim=-1).detach().to(hiddens.device)  # (1, T) — credit weights are not trained
            
            advantages = torch.ones_like(c3_credit)
            
            # Asymmetric RL Loss (per-sample)
            reward_tensor = torch.tensor([reward], device=device, dtype=dtype)
            c3_weights = c3_credit.abs().detach()
            c3_weights = c3_weights / c3_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            
            rl_loss = asymmetric_rl_loss(
                c3_weights=c3_weights,
                advantages=advantages,
                log_probs=log_probs,
                reaction_reward=reward_tensor
            )
    
    # ===== PHASE 2: Epistemic Routing =====
    # κ splits the gradient: factual errors → base manifold, style errors → user fiber
    try:
        router_device = epistemic_router.fc1.weight.device if hasattr(epistemic_router, 'fc1') else (epistemic_router.mapper[0].weight.device if hasattr(epistemic_router, 'mapper') else next(epistemic_router.parameters()).device)
    except StopIteration:
        router_device = 'cuda:1'
        
    rl_loss_cold = rl_loss.unsqueeze(0).to(router_device)
    energy_cold = energy.unsqueeze(0).to(router_device)
    
    routing = epistemic_router(rl_loss_cold, energy_cold)
    kappa = routing["kappa"].item()
    
    # Zero ALL optimizers at the top of every update
    optimizer_base.zero_grad()
    optimizer_sidecar.zero_grad()
    if optimizer_sparse is not None:
        optimizer_sparse.zero_grad()
    
    # Use a unified backward pass to prevent PyTorch inplace modification graph crashes.
    # The mathematical guarantees of the epistemic router are enforced by explicitly 
    # Scale parameter gradients post-backward to match the routing equations.
    rl_loss.mean().backward()
    
    # Purge any exploding fp16 overflow gradients to zero
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            if p.grad.is_sparse:
                torch.nan_to_num_(p.grad._values(), nan=0.0, posinf=0.0, neginf=0.0)
            else:
                torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
    
    # --- Base manifold update (κ: factual correction) ---
    base_params = [p for p in model.backbone.parameters() if p.requires_grad]
    
    # Clip base gradients FIRST before applying kappa scaling
    if base_params:
        torch.nn.utils.clip_grad_norm_(base_params, 0.5)
        
    original_base_lrs = []
    for group in optimizer_base.param_groups:
        original_base_lrs.append(group['lr'])
        group['lr'] = group['lr'] * kappa
            
    if kappa > 0.01 and base_params:
        optimizer_base.step()
        
    for group, orig_lr in zip(optimizer_base.param_groups, original_base_lrs):
        group['lr'] = orig_lr
    
    # --- Sidecar + Fiber update (1-κ: style correction) ---
    sidecar_params = list(model.geo_processor.parameters()) + list(model.ebm_critic.parameters())
    if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
        sidecar_params += [model.fiber_bundle.fiber_store.fiber_vectors]
    if getattr(model, 'fiber_proj', None) is not None:
        sidecar_params += list(model.fiber_proj.parameters())
    # Sparse bias has explicitly its own optimizer!
        
    # Clip sidecar gradients FIRST before applying (1-kappa) routing penalty
    torch.nn.utils.clip_grad_norm_(sidecar_params, 1.0)
    # Tighter explicit clip on geo_processor to prevent retraining (0.1)
    torch.nn.utils.clip_grad_norm_(model.geo_processor.parameters(), 0.1)
    
    # Clip sparse gradients to prevent F.log_softmax Float16 overflow
    if getattr(model, 'sparse_bias', None) is not None:
        torch.nn.utils.clip_grad_norm_(model.sparse_bias.parameters(), 1.0)
    
    # Scale Sidecar learning rate by (1-kappa), bypassed during contrastive forcing
    # When target_ids is provided, the user is explicitly teaching a fact.
    # The epistemic router (kappa→1 for factual error) must NOT suppress the
    # sidecar/fiber optimizer — otherwise, in fiber-only mode (base lr=0),
    # NOTHING learns. This is the same bypass applied to sparse_bias below.
    original_sidecar_lrs = []
    for group in optimizer_sidecar.param_groups:
        original_sidecar_lrs.append(group['lr'])
        if target_ids is None:
            group['lr'] = group['lr'] * (1.0 - kappa)
        # else: keep full LR for factual injection
        
    # Scale Sparse learning rate by (1-kappa), bypassed during contrastive forcing
    if optimizer_sparse is not None:
        original_sparse_lrs = []
        for group in optimizer_sparse.param_groups:
            original_sparse_lrs.append(group['lr'])
            # Contrastive forcing (target_ids) bypasses epistemic suppression.
            # The user is explicitly teaching a fact — LR must not be zeroed.
            if target_ids is None:
                group['lr'] = group['lr'] * (1.0 - kappa)
            # else: keep full LR for factual injection
            
    # Apply scaled step — bypass the kappa gate during contrastive forcing
    if kappa < 0.99 or target_ids is not None:
        optimizer_sidecar.step()
        if optimizer_sparse is not None:
            optimizer_sparse.step()
            model.sparse_bias.bias.weight.data.clamp_(-10.0, 10.0)

        # DEBUG NaNs
        for name, p in model.named_parameters():
            if p.requires_grad and torch.isnan(p).any():
                print(f"NAN DETECTED AFTER STEP IN {name}")

    
    # Restore sidecar LRs
    for group, orig_lr in zip(optimizer_sidecar.param_groups, original_sidecar_lrs):
        group['lr'] = orig_lr

    # Restore sparse LRs
    if optimizer_sparse is not None:
        for group, orig_lr in zip(optimizer_sparse.param_groups, original_sparse_lrs):
            group['lr'] = orig_lr

    # ZERO OUT MOMENTUM FOR INACTIVE USERS (Adam Momentum Leak Fix)
    # Prevents optimizer state from accumulating and updating isolated users
    
    def zero_inactive_momentum(param_tensor):
        if param_tensor in optimizer_sidecar.state:
            param_state = optimizer_sidecar.state[param_tensor]
            if 'exp_avg' in param_state:
                num_u = param_tensor.size(0)
                mask = torch.ones(num_u, 1, dtype=torch.bool, device=param_tensor.device)
                mask[user_id] = False
                param_state['exp_avg'].masked_fill_(mask, 0.0)
                if 'exp_avg_sq' in param_state:
                    param_state['exp_avg_sq'].masked_fill_(mask, 0.0)

    if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
        zero_inactive_momentum(model.fiber_bundle.fiber_store.fiber_vectors)
        
    if getattr(model, 'sparse_bias', None) is not None:
        # NOTE: This is currently a NO-OP because sparse_bias params live in
        # optimizer_sparse (SparseAdam), not optimizer_sidecar (AdamW).
        # SparseAdam only updates accessed embedding rows, so momentum leak
        # is naturally mitigated. However, if optimizer_sparse is ever changed
        # to a dense optimizer, this must be updated to reference the correct
        # optimizer instance. See code review R-1 (2026-03-25).
        zero_inactive_momentum(model.sparse_bias.bias.weight)
    
    # ===== PHASE 3: BCH Consolidation =====
    # Write episodic perturbation into user's lifelong Lie algebra fiber
    if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
        with torch.no_grad():
            # Compute the episodic delta from the fiber gradient
            fiber_vec = model.fiber_bundle.fiber_store.fiber_vectors
            if fiber_vec.grad is not None:
                delta_vec = fiber_vec.grad[user_id]  # (vec_dim,)
                d_fiber = model.fiber_bundle.d_fiber
                delta_A = skew_unvectorize(delta_vec.unsqueeze(0), d_fiber).squeeze(0)  # (d, d)
                A_old = model.fiber_bundle.fiber_store.get_gauge_connection(
                    user_ids
                ).squeeze(0)  # (d, d)
                
                # BCH consolidation (2nd-order non-commutative integration)
                A_new = bch_consolidator.consolidate(A_old, delta_A)
                
                # Write back
                new_vec = skew_vectorize(A_new.unsqueeze(0)).squeeze(0)
                fiber_vec.data[user_id] = new_vec
    
    # ===== PHASE 4: Episodic Memory Write =====
    # When contrastive forcing was used (target_ids is not None), store the
    # prompt embedding (key) and TARGET response hidden state (value).
    # The target hidden state is PROMPT-SPECIFIC: it encodes what the model
    # should say for THIS particular fact, unlike z0_delta which is the same
    # for all prompts from the same user.
    if target_ids is not None and getattr(model, 'episodic_memory', None) is not None:
        with torch.no_grad():
            # Key: prompt embedding (last-token hidden state of the prompt)
            # 
            # TESTED AND FAILED:
            # - Mean-pool ALL: sims 0.93+ (template dominates) → 2/5
            # - Content-only pool: sims 0.91-0.94 (content still similar) → 1/5
            # - Dual key_proj: sims unstable → 1/5
            # Last-token gives 0.70-0.73 band which, while narrow, is the
            # best discrimination we can get without a learned encoder.
            backbone_out = model.backbone(
                input_ids=prompt_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            prompt_emb = backbone_out.hidden_states[-1][:, -1, :].squeeze(0)  # (d_key,)

            # Value: target response hidden state
            # Run the target through the backbone to get its semantic embedding
            target_out = model.backbone(
                input_ids=target_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            target_emb = target_out.hidden_states[-1][:, -1, :].squeeze(0)  # (d_key,)

            model.episodic_memory.write(
                user_id, prompt_emb, target_emb,
                token_ids=target_ids, prompt_ids=prompt_ids,
            )

    # ===== PHASE 5: Sparse Bias Decay =====
    # Prevent unbounded accumulation of sparse bias over many training rounds.
    # Without decay, bias values accumulate and cause grammatical artifacts
    # (e.g., "You was" at Round 10+). Decay of 0.95 per step preserves recent
    # signal while preventing historical accumulation from distorting grammar.
    if getattr(model, 'sparse_bias', None) is not None:
        model.sparse_bias.decay_bias(user_id, factor=0.95)
    
    return rl_loss.item(), energy.item(), kappa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="Bender1011001/Qwen2.5-3B-Instruct-ABLITERATED")
    parser.add_argument("--sidecar", type=str, default="sidecar_step500.pt")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for sidecar continuous learning")
    parser.add_argument("--lr_base", type=float, default=1e-6, help="Learning rate for base manifold (factual corrections)")
    parser.add_argument("--user_id", type=int, default=0, help="User ID for fiber bundle isolation")
    args = parser.parse_args()

    model, tokenizer = load_v2_system(args.backbone, args.sidecar)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    # --- Two separate optimizers for epistemic routing ---
    sidecar_params = list(model.geo_processor.parameters()) + list(model.ebm_critic.parameters())
    if getattr(model, 'fiber_proj', None) is not None:
        sidecar_params += list(model.fiber_proj.parameters())
        
    optimizer_groups = []
    if sidecar_params:
        optimizer_groups.append({'params': sidecar_params, 'weight_decay': 0.01})
        
    per_user_params = []
    if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
        per_user_params.append(model.fiber_bundle.fiber_store.fiber_vectors)
    # NOTE: sparse_bias uses nn.Embedding(sparse=True) which produces sparse gradients.
    # AdamW does NOT support sparse gradients — it crashes on .step().
    # sparse_bias MUST use torch.optim.SparseAdam instead.
        
    if per_user_params:
        optimizer_groups.append({'params': per_user_params, 'weight_decay': 0.0})
        
    optimizer_sidecar = torch.optim.AdamW(optimizer_groups, lr=args.lr)
    
    # Separate SparseAdam optimizer for sparse_bias (sparse gradients)
    optimizer_sparse = None
    if getattr(model, 'sparse_bias', None) is not None:
        optimizer_sparse = torch.optim.SparseAdam(
            list(model.sparse_bias.parameters()), lr=0.1
        )
        print(f"    [*] SparseAdam optimizer created for sparse_bias (lr=0.1)")
    
    # Base manifold optimizer: decoder blocks + final norm (Fix 3 surgical unfreeze)
    base_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer_base = torch.optim.AdamW(base_params, lr=args.lr_base, weight_decay=0.0)
    
    # Epistemic Router (κ gate) and BCH Consolidator
    epistemic_router = EpistemicRouter(
        energy_threshold_init=5.0, tau_e=0.5,
        eta_base=0.01, eta_user=0.05
    ).to(device)
    epistemic_router.train()  # Enable EMA calibration
    
    bch_consolidator = BCHConsolidator(eta=0.05, order=2)
    
    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_total = sum(p.numel() for p in base_params)
    sidecar_total = sum(p.numel() for p in sidecar_params)
    print(f"\n[*] Trainable params: {trainable_total:,} (base: {base_total:,}, sidecar: {sidecar_total:,})")
    
    print("\n" + "="*60)
    print(" 🧠 DUAL-SYSTEM V2 — CONTINUOUS LEARNING TERMINAL")
    print("    ├─ Epistemic Routing (κ gate): ACTIVE")
    print("    ├─ Fiber Bundle User ID: %d" % args.user_id)
    print("    ├─ BCH Consolidation: ACTIVE (order=2, ln(2) clamped)")
    print("    └─ Base Manifold: SURGICALLY UNFROZEN (Fix 3)")
    print("="*60)
    print("  Type your prompt and press Enter.")
    print("  After generation, reward [-1.0 to 1.0].")
    print("  κ→1: factual fix → base manifold | κ→0: style fix → user fiber")
    print("  Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")

    history = []

    while True:
        try:
            user_input = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if user_input.strip().lower() in ["quit", "exit"]:
            print("\n[*] Saving updated manifold...")
            save_dict = {
                "geo_state": model.geo_processor.state_dict(),
                "ebm_state": model.ebm_critic.state_dict(),
                "planner_state": model.latent_planner.state_dict(),
            }
            if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
                save_dict["fiber_state"] = model.fiber_bundle.state_dict()
            if hasattr(model, 'sparse_bias') and model.sparse_bias is not None:
                save_dict["sparse_bias_state"] = model.sparse_bias.state_dict_differential()

            torch.save(save_dict, args.sidecar)
            print(f"[*] Saved to {args.sidecar}. Exiting.")
            break
            
        if not user_input.strip():
            continue

        history.append({"role": "user", "content": user_input})
        
        prompt_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        
        print("AI: ", end="", flush=True)
        resp_tokens = []
        
        try:
            for token_id in generate_streaming(model, tokenizer, prompt_ids, user_id=args.user_id):
                resp_tokens.append(token_id)
                word = tokenizer.decode([token_id], skip_special_tokens=True)
                sys.stdout.write(word)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print(" [Interrupted]")
            
        response_text = tokenizer.decode(resp_tokens, skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": response_text})
        print()
        
        # Online Learning Step
        resp_ids = torch.tensor([resp_tokens], device=device)
        
        try:
            reward_str = input("\n[System] Reward this response [-1.0 to 1.0, or Enter to skip]: ").strip()
            if reward_str:
                reward = float(reward_str)
                reward = max(-1.0, min(1.0, reward))
                
                print(f"[*] 3-Phase Continuous Learning (Reward: {reward:.2f})...")
                loss, energy, kappa = continuous_learning_update(
                    model, tokenizer, prompt_ids, resp_ids, reward,
                    optimizer_sidecar, optimizer_base,
                    epistemic_router, bch_consolidator,
                    user_id=args.user_id, dtype=dtype,
                    optimizer_sparse=optimizer_sparse
                )
                route_label = "BASE (factual)" if kappa > 0.5 else "FIBER (style)"
                print(f"    [+] Loss: {loss:.4f} | Energy: {energy:.4f} | κ={kappa:.3f} → {route_label}")
                
                # Periodically save
                save_dict = {
                    "geo_state": model.geo_processor.state_dict(),
                    "ebm_state": model.ebm_critic.state_dict(),
                    "planner_state": model.latent_planner.state_dict(),
                }
                if hasattr(model, 'fiber_bundle') and model.fiber_bundle is not None:
                    save_dict["fiber_state"] = model.fiber_bundle.state_dict()
                if hasattr(model, 'sparse_bias') and model.sparse_bias is not None:
                    save_dict["sparse_bias_state"] = model.sparse_bias.state_dict_differential()
                    
                torch.save(save_dict, args.sidecar + ".latest")
        except ValueError:
            print("[System] Invalid reward, skipping update.")
        except KeyboardInterrupt:
            print("\n[System] Skipped reward.")

if __name__ == "__main__":
    main()
