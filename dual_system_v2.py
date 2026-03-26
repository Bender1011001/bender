"""
Dual-System V2 — Sidecar Architecture
======================================

The correct architecture: frozen HuggingFace backbone + trainable geometric sidecars.

Forward pass:
  1. Frozen backbone produces hidden states h and base logits
  2. Geometric sidecar processes h through DiffusionPlanner, TCRS, EBM Critic
  3. Geometric output projected to vocab → geo_logits
  4. final_logits = base_logits + alpha * geo_logits
  5. Loss = CE(final_logits, target) + lambda_ebm * free_energy + lambda_elbo * elbo

This preserves 100% of the pretrained language modeling capability while adding
our geometric reasoning modules as learnable corrections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional
from modules.diffusion_planner import DiffusionPlanner


class ClampedContextualFiberProj(nn.Module):
    """
    Bilinear contextual gate: maps (z_delta, h_seq) to per-user logit corrections.
    z_delta is zero at init (Cayley(0)=I), so output is zero until fibers learn.
    h_seq provides sequence context to prevent context-blind global hijacking.
    Output clamped to ±50 to prevent bilinear explosion.
    """
    def __init__(self, d_latent: int, d_model: int, vocab: int, rank: int = 64):
        super().__init__()
        self.W_z = nn.Linear(d_latent, rank, bias=False)
        self.W_h = nn.Linear(d_model, rank, bias=False)
        self.W_out = nn.Linear(rank, vocab, bias=False)
        self.act = nn.GELU()

    def forward(self, z_delta: Tensor, h_seq: Tensor) -> Tensor:
        """
        Args:
            z_delta: (B, d_latent) fiber delta (zero at init)
            h_seq: (B, T, d_model) or (B, d_model) sequence context
        Returns:
            (B, T, vocab) or (B, vocab) logit correction
        """
        z_proj = self.W_z(z_delta)        # (B, rank)
        h_proj = self.W_h(h_seq)          # (B, T, rank) or (B, rank)
        if h_proj.dim() == 3 and z_proj.dim() == 2:
            z_proj = z_proj.unsqueeze(1)  # (B, 1, rank) for broadcast
        gated = z_proj * h_proj
        gated = F.normalize(gated, p=2, dim=-1)
        out = self.W_out(self.act(gated))
        return torch.clamp(out, min=-50.0, max=50.0)


@dataclass
class SidecarConfig:
    """Configuration for the geometric sidecar modules."""
    # Backbone info (set from loaded model)
    backbone_hidden_size: int = 2048   # will be detected from backbone
    backbone_vocab_size: int = 151936  # will be detected from backbone

    # Geometric latent dimensions
    d_geo: int = 512           # geometric processing dim
    d_latent: int = 256        # diffusion planner latent dim
    n_geo_layers: int = 4      # number of geometric processing layers
    n_geo_heads: int = 8       # attention heads in geo processor

    # Diffusion planner
    n_diffusion_steps: int = 8
    diffusion_beta_start: float = 1e-4
    diffusion_beta_end: float = 0.02

    # EBM Critic
    ebm_hidden_dim: int = 512

    # Fiber Bundle (Phase 4 multi-user topology)
    num_fibers: int = 0  # 0 disables the bundle, >0 enables user fibers
    max_memories: int = 4096  # maximum capacity for episodic memory bank
    hard_recall_boost: float = 20.0  # logit boost for hard recall (must exceed backbone top-1 gap ~16)

    # Loss weights
    lambda_geo: float = 0.1    # weight for geometric logit correction    
    lambda_ebm: float = 0.1   # weight for free energy regularization
    lambda_elbo: float = 0.01  # weight for diffusion planner ELBO

    # Training
    geo_dropout: float = 0.1


class GeometricProcessor(nn.Module):
    """
    Processes backbone hidden states through geometric transformer layers.
    This is the core "System 2" — slower, deliberate processing that adds
    structured reasoning on top of the fast backbone predictions.
    """

    def __init__(self, config: SidecarConfig):
        super().__init__()
        self.config = config

        # Project from backbone hidden size to geometric processing dimension
        self.input_proj = nn.Linear(config.backbone_hidden_size, config.d_geo, bias=False)
        self.z0_proj = nn.Linear(config.d_latent, config.d_geo, bias=False)

        # Geometric transformer layers (lightweight)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_geo,
            nhead=config.n_geo_heads,
            dim_feedforward=config.d_geo * 4,
            dropout=config.geo_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.geo_layers = nn.TransformerEncoder(layer, num_layers=config.n_geo_layers)

        # Output projection back to vocab space for logit correction
        self.output_norm = nn.RMSNorm(config.d_geo)
        self.output_proj = nn.Linear(config.d_geo, config.backbone_vocab_size, bias=False)
        nn.init.zeros_(self.output_proj.weight)

        # Learnable mixing coefficient — controls sidecar contribution to output logits.
        # Init to 0.0 → sigmoid(0)=0.5 → 50/50 base/sidecar mix from step 1.
        # Previous init of -5.0 caused gradient vanishing: sigmoid(-5)≈0.007 meant
        # ∂CE/∂geo_logits ≈ 0, so the sidecar trained behind a closed gate.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_backbone: Tensor, z0: Tensor = None, causal_mask: Tensor = None) -> Tensor:
        """
        Args:
            h_backbone: (B, T, backbone_hidden_size) from frozen backbone
            z0: (B, d_latent) continuous blueprint from Diffusion Planner
            causal_mask: (T, T) causal attention mask

        Returns:
            geo_logits: (B, T, vocab_size) geometric correction to base logits
        """
        # Project to geometric space
        h_geo = self.input_proj(h_backbone)
        
        # Inject continuous blueprint plan into the geometric representation
        if z0 is not None:
            z0_proj = self.z0_proj(z0).unsqueeze(1)  # (B, 1, d_geo)
            h_geo = h_geo + z0_proj

        # Generate causal mask if needed
        T = h_geo.size(1)
        if causal_mask is None and T > 1:
            causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                T, device=h_geo.device, dtype=h_geo.dtype
            )

        # Process through geometric transformer
        h_geo = self.geo_layers(h_geo, mask=causal_mask)

        # Project to vocab and scale by learned alpha
        h_geo = self.output_norm(h_geo)
        geo_logits = self.output_proj(h_geo)

        # Alpha controls how much geometric correction to apply
        # sigmoid bounds to [0, 1]; optimizer learns the natural mixing level
        mixing = torch.sigmoid(self.alpha)
        return geo_logits * mixing


class EBMCriticHead(nn.Module):
    """
    Energy-Based Model critic that evaluates sequence-level quality.
    Low energy = good sequence, high energy = hallucinated/incoherent.
    """

    def __init__(self, config: SidecarConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.backbone_hidden_size, config.ebm_hidden_dim),
            nn.GELU(),
            nn.Linear(config.ebm_hidden_dim, config.ebm_hidden_dim),
            nn.GELU(),
            nn.Linear(config.ebm_hidden_dim, 1),
        )

    def forward(self, h_backbone: Tensor) -> Tensor:
        """
        Returns free energy scalar per sequence.
        """
        # [FIX] Option 2 (Adjusted): Target the Final Token Slice
        # Bypasses permutation-invariance of .mean() while retaining causal sequence context
        h_pooled = h_backbone[:, -1, :]  # (B, hidden_size)
        
        energy = self.net(h_pooled).squeeze(-1)  # (B,)
        return energy


class DualSystemV2(nn.Module):
    """
    The complete Dual-System Architecture v2.

    Frozen HuggingFace backbone (System 1) + trainable geometric sidecars (System 2).
    The backbone handles language; the sidecars add structured reasoning.
    """

    def __init__(self, backbone, config: SidecarConfig):
        super().__init__()
        self.backbone = backbone
        self.config = config

        # Freeze backbone completely
        for param in self.backbone.parameters():
            param.requires_grad = False

        # System 2: Geometric sidecars (all trainable)
        self.geo_processor = GeometricProcessor(config)
        self.latent_planner = DiffusionPlanner(
            d_input=config.backbone_hidden_size,
            d_latent=config.d_latent,
            n_steps=config.n_diffusion_steps,
            beta_start=config.diffusion_beta_start,
            beta_end=config.diffusion_beta_end,
        )
        self.ebm_critic = EBMCriticHead(config)

        if config.num_fibers > 0:
            from modules.fiber_bundle import PrincipalFiberBundle
            self.fiber_bundle = PrincipalFiberBundle(
                num_users=config.num_fibers,
                d_fiber=config.d_latent,
                fiber_weight=0.1
            )
            # Direct fiber→logit projection: bypasses the stiff geo_processor
            # pathway that attenuates Cayley-near-identity perturbations.
            # Input: z0_delta = z0_lifted - z0_base (zero at init).
            # Output: per-user logit correction added to final_logits.
            # Low-rank bottleneck (256→64→V) keeps params at ~9.7M.
            # Zero-init output ensures no behavioral change at init.
            # Gradient still flows at init via ∂cayley/∂A evaluated at A=0.
            self.fiber_proj = ClampedContextualFiberProj(
                config.d_latent, config.backbone_hidden_size, config.backbone_vocab_size
            )
            from modules.sparse_bias import HighGainSparseBias
            self.sparse_bias = HighGainSparseBias(config.num_fibers, config.backbone_vocab_size, gain=5.0, sparse=True)
            # Episodic memory bank for context-dependent factual recall
            from modules.episodic_memory import EpisodicMemoryBank
            self.episodic_memory = EpisodicMemoryBank(
                num_users=config.num_fibers,
                d_key=config.backbone_hidden_size,
                d_value=config.d_latent,
                max_memories=config.max_memories,
                temperature=0.1,
                top_k=4,
            )
        else:
            self.fiber_bundle = None
            self.fiber_proj = None
            self.sparse_bias = None
            self.episodic_memory = None

    def unfreeze_lora(self):
        """Re-enable gradients for LoRA parameters after global freeze."""
        for name, param in self.backbone.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

    def get_trainable_params(self):
        """Returns sidecar + any unfrozen backbone params (e.g., LoRA)."""
        params = []
        modules = [self.geo_processor, self.latent_planner, self.ebm_critic]
        if self.fiber_bundle is not None:
            modules.append(self.fiber_bundle)
        if getattr(self, 'fiber_proj', None) is not None:
            modules.append(self.fiber_proj)
        if getattr(self, 'sparse_bias', None) is not None:
            modules.append(self.sparse_bias)
        if getattr(self, 'episodic_memory', None) is not None:
            modules.append(self.episodic_memory)

        for module in modules:
            params.extend(module.parameters())
        # Include any backbone params that have requires_grad (e.g., LoRA)
        for p in self.backbone.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def count_params(self):
        """Count total, frozen, and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.get_trainable_params())
        return {"total": total, "frozen": frozen, "trainable": trainable}

    @torch.no_grad()
    def svd_init_projections(self):
        """
        SVD-initialize the GeometricProcessor's input_proj from the backbone's
        token embedding matrix.

        Instead of random initialization, compute the top-d_geo right singular
        vectors of embed_tokens (shape: vocab_size × hidden_size). These are the
        principal directions of maximum variance in the backbone's learned
        representation space. Using them as the projection basis gives the
        sidecar a semantically meaningful subspace from step 0.

        Also initializes the DiffusionPlanner's encoder input layer with the
        same SVD basis for consistency.
        """
        embed = self.backbone.model.embed_tokens.weight.float()  # (V, H)
        d_geo = self.config.d_geo
        d_latent = self.config.d_latent

        # Compute top-k right singular vectors of the embedding matrix
        # V_t[:k] gives the top-k directions of maximum variance in hidden_size space
        # pca_lowrank is faster than full SVD for large vocab matrices
        # Determine max components needed across all target layers
        max_components = d_geo
        if hasattr(self.latent_planner, 'encoder') and hasattr(self.latent_planner.encoder, 'net'):
            first_layer = self.latent_planner.encoder.net[0]
            if isinstance(first_layer, nn.Linear):
                max_components = max(max_components, first_layer.out_features)
        # Cap at hidden_size (can't extract more components than dimensions)
        q = min(max_components + 16, embed.shape[1])
        U, S, V_t = torch.pca_lowrank(embed, q=q)
        # V_t columns are the principal components in hidden_size space
        # We want: input_proj.weight @ h = (V_t[:, :d_geo])^T @ h
        # input_proj is Linear(H → d_geo), weight shape is (d_geo, H)

        # Initialize geo_processor.input_proj with top-d_geo components
        svd_proj = V_t[:, :d_geo].T  # (d_geo, H)
        target = self.geo_processor.input_proj.weight
        self.geo_processor.input_proj.weight.copy_(svd_proj.to(device=target.device, dtype=target.dtype))
        print(f"  [SVD Init] geo_processor.input_proj: top-{d_geo} PCA components "
              f"(captured {S[:d_geo].sum() / S.sum():.1%} of total variance)")

        # Initialize DiffusionPlanner encoder's first projection
        if hasattr(self.latent_planner, 'encoder') and hasattr(self.latent_planner.encoder, 'net'):
            first_layer = self.latent_planner.encoder.net[0]  # Linear(d_input, d_hidden)
            if isinstance(first_layer, nn.Linear) and first_layer.in_features == embed.shape[1]:
                d_out = first_layer.out_features
                if d_out <= V_t.shape[1]:
                    svd_planner = V_t[:, :d_out].T  # (d_out, H)
                    target_p = first_layer.weight
                    first_layer.weight.copy_(svd_planner.to(device=target_p.device, dtype=target_p.dtype))
                    print(f"  [SVD Init] latent_planner.encoder: top-{d_out} PCA components")
                else:
                    print(f"  [SVD Init] Skipping planner encoder: needs {d_out} components but only {V_t.shape[1]} available")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        user_ids: Tensor = None,
    ):
        """
        Full forward pass:
          1. Backbone → base_logits + hidden_states
          2. GeometricProcessor → geo_logits (correction)
          3. LatentPlanner → ELBO loss (plan quality)
          4. EBMCritic → free energy (sequence quality)
          5. Combined loss
        """
        # 1. Backbone forward pass
        # If LoRA is active, backbone needs gradients; otherwise use no_grad
        # NOTE: Do NOT cache this — if unfreeze_lora() is called after first
        # forward, a stale False would silently block LoRA gradient flow.
        has_trainable_backbone = any(p.requires_grad for p in self.backbone.parameters())

        if has_trainable_backbone:
            backbone_out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            with torch.no_grad():
                backbone_out = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

        base_logits = backbone_out.logits
        h_last = backbone_out.hidden_states[-1]  # (B, T, hidden_size)
        # Detach hidden states for sidecar processing so geo losses don't affect backbone
        h_last_detached = h_last.detach()
        # Cast to sidecar dtype for mixed-precision compatibility (e.g. backbone=bf16, sidecar=fp32)
        sidecar_dtype = self.geo_processor.input_proj.weight.dtype
        h_last_detached = h_last_detached.to(sidecar_dtype)
        h_pooled = h_last_detached[:, -1, :]  # last token has full causal context
        
        # Extract continuous blueprint from latent planner
        if self.latent_planner is not None:
            mu, log_var = self.latent_planner.encoder(h_pooled)
            if self.training:
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z0 = mu + eps * std
            else:
                z0 = mu
        else:
            z0 = None

        # If user personalization bundle is active, lift to fiber
        fiber_logits = None
        if user_ids is not None and getattr(self, "fiber_bundle", None) is not None:
            z0_base = z0  # save pre-lift for delta computation
            z0 = self.fiber_bundle.lift_to_fiber(z0, user_ids)
            # Direct fiber→logit path (bypasses geo_processor attenuation)
            if getattr(self, "fiber_proj", None) is not None:
                z0_delta = z0 - z0_base  # (B, d_latent) — zero at init
                fiber_logits = self.fiber_proj(z0_delta, h_last_detached)  # (B, T, V)

        # 2. Geometric processor → logit correction (conditioned on continuous blueprint z0)
        geo_logits = self.geo_processor(h_last_detached, z0=z0)

        # 3. Final logits = base + geometric correction + fiber correction
        final_logits = base_logits + geo_logits
        if fiber_logits is not None:
            final_logits = final_logits + fiber_logits  # (B, T, V)

        if user_ids is not None and getattr(self, "sparse_bias", None) is not None:
            sparse_logits = self.sparse_bias(user_ids)  # gain applied internally
            final_logits = final_logits + sparse_logits.unsqueeze(1)

        # 4. Compute losses
        result = {"logits": final_logits}

        if labels is not None:
            # Primary: cross-entropy on corrected logits
            shift_logits = final_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # Secondary: latent planner ELBO
            # NOTE (R-4, 2026-03-25): h_pooled and h_target are both h_last[:, -1, :].
            # This means x=y — the ELBO is autoencoding the last-token hidden state.
            # This is INTENTIONAL: the planner ELBO serves as a VAE regularizer
            # (KL term prevents posterior collapse, reconstruction loss keeps z0
            # informationally grounded), NOT as a cross-prediction objective.
            # The "real" learning signal comes from CE loss + fiber routing.
            # If a non-trivial reconstruction target is desired in the future,
            # use h_last_detached.mean(dim=1) as y while keeping [:, -1, :] as x.
            h_target = h_last_detached[:, -1, :]  # (B, hidden_size) — same as h_pooled
            planner_out = self.latent_planner.compute_elbo_loss(
                h_pooled, y=h_target
            )
            elbo_loss = planner_out["elbo_loss"]

            # Tertiary: EBM with NCE contrastive loss (prevents partition collapse)
            energy = self.ebm_critic(h_last_detached)
            ebm_pos = energy.mean()
            
            B = h_last_detached.size(0)
            T = h_last_detached.size(1)
            negatives = []
            
            # ---------------------------------------------------------
            # [FIX] Option 1: Corruption Negatives
            # Replace ~20% of tokens sequentially with disjoint embeddings
            # fetched from an alternate sequence in the batch dimension.
            # ---------------------------------------------------------
            if B > 1:
                corrupted = h_last_detached.clone()
                # 20% corruption probability mask
                mask = torch.rand(B, T, 1, device=h_last_detached.device) < 0.20
                
                # Fetch alternate sequence in the batch (shifted by 1)
                alt_h = torch.roll(h_last_detached, shifts=1, dims=0)
                corrupted = torch.where(mask, alt_h, corrupted)
                negatives.append(corrupted)
            else:
                # Fallback to shuffle if batch size is 1
                perm = torch.randperm(T, device=h_last_detached.device)
                negatives.append(h_last_detached[:, perm, :])

            # 2. Gaussian noise: add noise proportional to hidden state magnitude
            noise_scale = 0.3 * h_last_detached.std()
            negatives.append(h_last_detached + noise_scale * torch.randn_like(h_last_detached))

            # 3. Cross-batch roll: shift sequences between batch elements
            if B > 1:
                negatives.append(torch.roll(h_last_detached, shifts=1, dims=0))

            # Compute energy on negatives
            neg_energies = []
            for neg in negatives:
                neg_energies.append(self.ebm_critic(neg))
            ebm_neg = torch.stack(neg_energies).mean(dim=0) # Aggregate negatives per sequence before batch mean
            
            # BCE Loss on Energies: Positives pulled towards 0 (low energy), Negatives pulled towards 1 (high energy)
            # This strictly bounds the scaler explosion mathematically.
            ebm_loss = F.binary_cross_entropy_with_logits(
                energy, torch.zeros_like(energy)
            ) + F.binary_cross_entropy_with_logits(
                ebm_neg, torch.ones_like(ebm_neg)
            )

            # Combined Dual-System loss
            total_loss = (
                ce_loss
                + self.config.lambda_elbo * elbo_loss
                + self.config.lambda_ebm * ebm_loss
            )

            result.update({
                "loss": total_loss,
                "ce_loss": ce_loss,
                "elbo_loss": elbo_loss,
                "ebm_loss": ebm_loss,
                "energy": energy.detach().mean(),
                "energy_neg": ebm_neg.detach().mean(),
                "alpha": torch.sigmoid(self.geo_processor.alpha).item(),
                "planner_kl": planner_out["kl_loss"].detach(),
                "planner_recon": planner_out["recon_loss"].detach(),
            })

        return result

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=256,
                 temperature=0.7, top_p=0.9, top_k=50, user_ids=None, **kwargs):
        """
        Generate text using backbone + geometric correction with KV caching.
        
        First pass: full prompt through backbone + sidecar → cache KV states.
        Subsequent passes: single new token through backbone (using cache) + sidecar correction.
        """
        past_key_values = None
        h_seq = None
        z0 = None  # [FIX 4] Initialize continuous blueprint state
        initial_len = input_ids.shape[-1]  # Track prompt length for position-aware recall
        initial_hr_query = None  # Cache prompt embedding for stable hard_recall matching
        initial_prompt_ids = None  # Cache prompt token IDs for Jaccard overlap

        for step in range(max_new_tokens):
            if past_key_values is None:
                # First pass: process full prompt
                backbone_out = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                )
            else:
                # Subsequent passes: only process the new token
                backbone_out = self.backbone(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                )

            past_key_values = backbone_out.past_key_values
            base_logits = backbone_out.logits[:, -1:, :]  # (B, 1, V)
            
            if h_seq is None:
                h_seq = backbone_out.hidden_states[-1]
                
                # Cache true prompt geometry exactly once preventing autoregressive token drift
                if initial_hr_query is None:
                    initial_hr_query = h_seq[:, -1, :].detach().clone()
                    initial_prompt_ids = input_ids[:, :initial_len].detach().clone()
                
                # [FIX 4] Extract the z0 blueprint from the planner on the first step
                # Use FULL prompt context for z0 pooling before capping h_seq
                if self.latent_planner is not None:
                    h_pooled = h_seq[:, -1, :]  # last token has full causal context
                    mu, _ = self.latent_planner.encoder(h_pooled)
                    z0 = mu
                    z0_base = z0  # save pre-lift for delta
                    
                    # Lift to fiber if user personalization bundle is active
                    if user_ids is not None and getattr(self, "fiber_bundle", None) is not None:
                        z0 = self.fiber_bundle.lift_to_fiber(z0, user_ids)
                
                # Cap initial prompt to sliding window AFTER z0 extraction
                if h_seq.size(1) > 128:
                    h_seq = h_seq[:, -128:, :]
            else:
                h_new = backbone_out.hidden_states[-1][:, -1:, :]
                h_seq = torch.cat([h_seq, h_new], dim=1)
                # Cap to sliding window — geo_processor has no KV cache,
                # so full h_seq causes O(n²) recomputation per step.
                # 128 tokens of local context is sufficient for additive corrections.
                if h_seq.size(1) > 128:
                    h_seq = h_seq[:, -128:, :]

            # [FIX 4] Route the z0 blueprint explicitly into the geometric processor
            geo_logits = self.geo_processor(h_seq, z0=z0)[:, -1:, :]  # (B, 1, V)
            logits = (base_logits + geo_logits)[:, -1, :]  # (B, V)

            # Add direct fiber→logit correction
            if (user_ids is not None and getattr(self, "fiber_proj", None) is not None
                    and z0 is not None):
                z0_delta = z0 - z0_base  # (B, d_latent)
                # Add episodic memory retrieval for context-dependent recall
                if getattr(self, 'episodic_memory', None) is not None:
                    uid_tensor = user_ids if user_ids.dim() > 0 else user_ids.unsqueeze(0)
                    episodic_delta = self.episodic_memory.read(
                        uid_tensor, initial_hr_query,
                        query_ids=initial_prompt_ids,
                    )  # (B, d_latent)
                    z0_delta = z0_delta + episodic_delta

                    # DIRECT episodic→logit pathway: project retrieved target
                    # embedding through backbone's own embedding matrix.
                    # This is the same operation as the LM head: h @ W_embed^T → logits
                    # No new params needed — the backbone already knows which tokens
                    # correspond to which hidden states.
                    raw_retrieved = self.episodic_memory.get_raw_retrieved()  # (B, d_key)
                    if raw_retrieved is not None:
                        embed_weight = self.backbone.model.embed_tokens.weight  # (V, d_key)
                        # Project retrieved target embedding to vocab space
                        episodic_logits = torch.matmul(
                            raw_retrieved.to(embed_weight.dtype),
                            embed_weight.T
                        )  # (B, V)
                        # Scale by learned factor to control injection strength
                        logits = logits + self.episodic_memory.episodic_scale * episodic_logits

                fiber_logits = self.fiber_proj(z0_delta, h_seq)  # (B, T, V)
                logits = logits + fiber_logits[:, -1, :]

            if user_ids is not None and getattr(self, "sparse_bias", None) is not None:
                sparse_logits = self.sparse_bias(user_ids)  # gain applied internally
                logits = logits + sparse_logits

            # Hard recall: position-aware boost of stored target tokens
            # Use INITIAL prompt embedding for similarity matching (not current
            # step's embedding which drifts below threshold after step 0)
            if user_ids is not None and getattr(self, 'episodic_memory', None) is not None:
                uid_tensor = user_ids if user_ids.dim() > 0 else user_ids.unsqueeze(0)
                generated_so_far = input_ids[:, initial_len:]  # Only the generated portion
                
                # Adaptive boost via logit std deviation (Proxy for 7B gap scaling)
                dynamic_boost = max(self.config.hard_recall_boost, 8.0 * logits.std().item())
                
                hard_bias = self.episodic_memory.hard_recall(
                    uid_tensor, initial_hr_query,
                    vocab_size=logits.size(-1),
                    similarity_threshold=0.7,
                    boost=dynamic_boost,
                    generated_ids=generated_so_far,
                    query_ids=initial_prompt_ids,
                )
                logits = logits + hard_bias
            logits = logits / temperature  # (B, V)

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones(attention_mask.size(0), 1, device=attention_mask.device, dtype=attention_mask.dtype)
                ], dim=-1)

            # Stop on EOS (batch-safe)
            if hasattr(self.backbone.config, 'eos_token_id'):
                eos_id = self.backbone.config.eos_token_id
                if isinstance(eos_id, list):
                    eos_hit = torch.tensor([t.squeeze().item() in eos_id for t in next_token], device=next_token.device)
                else:
                    eos_hit = (next_token.squeeze(-1) == eos_id)
                if eos_hit.all():
                    break

        return input_ids

