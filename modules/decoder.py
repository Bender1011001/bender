"""
AR Decoder with Geodesic-Traced Hidden States
==============================================

The decoder generates tokens autoregressively, but its hidden states
are *constrained to lie on geodesics* of the learned Riemannian manifold.

At each step t:
  1. TCRS marches the hidden state along the geodesic: h_t -> h_{t+1}
  2. LiSMCA bridge converts h_t to token logits (geometry-aware)
  3. Soft-mask embedding m_t provides prior for next step
  4. Decoder block refines h_{t+1} using attention + FFN

The decoder backbone is a standard transformer-like block (pre-norm)
but operates on hidden states that are geometrically constrained.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from .geometry import make_spd, spd_batchnorm
from .tcrs import TCRSSolver, WoodburyRetractionSolver
from .lismca import LiSMCA_Bridge


class DecoderBlock(nn.Module):
    """
    Single decoder block: self-attention + cross-attention to soft-mask + FFN.

    Pre-norm architecture with residual connections and QK normalization
    (HybridNorm, arXiv 2025) for attention stability in bf16.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int = None,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        if n_kv_heads is None:
            n_kv_heads = n_heads

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.RMSNorm(d_model)
        self.q_norm = nn.RMSNorm(d_model)
        self.k_norm = nn.RMSNorm(d_model)

        # GQA sizing: Q is full dim, K/V are reduced
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * n_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * n_kv_heads, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.RMSNorm(d_model)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.ff_dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(
        self,
        h: Tensor,
        mask: Tensor = None,
        past_kv: tuple = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple]:
        """
        Args:
            h: (B, T, d_model) hidden states
            mask: (T, T) causal attention mask
            past_kv: Tuple of (K, V) from previous step for autoregressive cache
            use_cache: If True, return current step's K, V
        """
        B, T, _ = h.shape

        h_norm = self.norm1(h)
        q_inp = self.q_norm(h_norm)
        k_inp = self.k_norm(h_norm)

        q = self.q_proj(q_inp).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(k_inp)
            .view(B, T, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(h_norm)
            .view(B, T, self.n_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_kv = (k.detach(), v.detach()) if use_cache else None

        # Repeat KV heads to match Q heads for GQA
        if self.n_kv_heads != self.n_heads:
            num_groups = self.n_heads // self.n_kv_heads
            k = torch.repeat_interleave(k, num_groups, dim=1)
            v = torch.repeat_interleave(v, num_groups, dim=1)

        # F.scaled_dot_product_attention handles Flash Attention automatically
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True if mask is None and past_kv is None else False,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        h = h + self.out_proj(attn_out)

        ff_in = self.norm2(h)
        h = h + self.ff_dropout(
            self.w_down(F.silu(self.w_gate(ff_in)) * self.w_up(ff_in))
        )

        return h, current_kv


class ARDecoder(nn.Module):
    """
    Autoregressive Decoder with geodesic-constrained hidden states.

    The full generation pipeline:
      1. Diffusion planner produces z_0
      2. Encoder produces initial h_0 from input x and z_0
      3. For each step t:
         a. TCRS marches h along the geodesic
         b. LiSMCA bridge produces logits and soft-mask
         c. Decoder block refines h using self-attention
      4. EBM critic scores the full sequence

    This module encapsulates steps 2-3. The diffusion planner and EBM
    critic are external.

    Args:
        config: DualSystemConfig with all hyperparameters
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        d = config.d_model
        p = config.d_lie
        V = config.vocab_size

        # Input encoder: project x + z_0 to initial hidden state
        self.input_proj = nn.Linear(d, d)
        self.z0_proj = nn.Linear(config.d_latent, d)
        self.input_norm = nn.RMSNorm(d)
        self.gradient_checkpointing = getattr(config, "gradient_checkpointing", False)

        # Token embedding (shared with LiSMCA output)
        self.token_embedding = nn.Embedding(V, p)
        self.embed_dropout = nn.Dropout(0.1)  # Embedding dropout for regularization

        # Embedding projection to full model dim
        self.embed_up = nn.Linear(p, d)

        # Learned metric tensor factory: z_0 -> G (SPD)
        self.metric_factory = nn.Sequential(
            nn.Linear(config.d_latent, d * config.d_lowrank),
        )

        # Patch embedding (for LiSMCA context)
        self.patch_proj = nn.Linear(d, p)

        # TCRS solver (primary)
        self.tcrs = TCRSSolver(d, config.d_latent, dt=config.tcrs_dt)

        # Woodbury fallback
        self.woodbury = WoodburyRetractionSolver(
            d, config.d_latent, rank=config.d_lowrank, dt=config.tcrs_dt
        )

        # LiSMCA bridge
        self.bridge = LiSMCA_Bridge(
            d_h=d,
            p=p,
            vocab_size=V,
            k=config.lismca_topk,
            d_latent=config.d_latent,
        )

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d,
                    config.n_heads,
                    n_kv_heads=config.n_kv_heads,
                    d_ff=getattr(config, "d_ff", None),
                )
                for _ in range(config.n_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(d)

        # Lie algebra metric for LiSMCA (p x p, learned SPD)
        self.lie_metric_factory = nn.Linear(config.d_latent, p * p)

        # -- LieRE-style positional encoding (arXiv 2406.10322) --
        # Learn a skew-symmetric generator S ∈ so(d) and compute
        # position-dependent rotations R(t) = cayley(t * S)
        # This gives geometrically-principled relative positional awareness
        lie_pe_dim = d
        n_generators = (
            lie_pe_dim * (lie_pe_dim - 1) // 2
        )  # Upper triangle of skew matrix
        self.lie_pe_params = nn.Parameter(torch.randn(n_generators) * 0.01)
        self.lie_pe_scale = nn.Parameter(torch.tensor(0.1))  # Learnable timescale

        # Weight initialization (GPT-2/3 pattern)
        self._init_weights()

    def _init_weights(self):
        """
        GPT-2/3 style initialization:
        - Linear/Embedding: N(0, 0.02)
        - Residual projections: scaled by 1/sqrt(2*n_layers)
        """
        n_layers = self.config.n_layers
        residual_scale = 1.0 / math.sqrt(2 * n_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # Scale residual path outputs
                if "w_down" in name or "out_proj" in name:
                    module.weight.data.mul_(residual_scale)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_causal_mask(self, T: int, device: torch.device) -> Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def _compute_metric(self, z0: Tensor) -> Tensor:
        """
        Compute the learned SPD metric tensor G from z_0.

        Uses G = M @ M^T + eps*I to guarantee SPD.

        Args:
            z0: (B, d_latent)

        Returns:
            G: (B, d, d) SPD metric tensor
        """
        d = self.config.d_model
        r = self.config.d_lowrank
        M = self.metric_factory(z0).view(-1, d, r)  # (B, d, r)
        return make_spd(M)

    def _compute_lie_metric(self, z0: Tensor) -> Tensor:
        """
        Compute the Lie-algebra-space metric (p x p SPD).

        Args:
            z0: (B, d_latent)

        Returns:
            G_lie: (B, p, p) SPD metric
        """
        p = self.config.d_lie
        raw = self.lie_metric_factory(z0).view(-1, p, p)  # (B, p, p)
        return make_spd(raw)

    def _precompute_lie_pe(
        self, T: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        """
        Precompute all T rotation matrices for LieRE positional encoding.

        Batches the linalg.solve across all T positions in a single call
        to avoid per-step overhead.

        Args:
            T: number of sequence positions
            device: target device
            dtype: target dtype

        Returns:
            R_all: (T, d, d) rotation matrices for positions 0..T-1
        """
        d = self.lie_pe_params.shape[0]
        # Recover d from n_generators = d*(d-1)/2
        # d^2 - d = 2*n_gen → d = (1 + sqrt(1 + 8*n_gen)) / 2
        n_gen = self.lie_pe_params.shape[0]
        d = int((1 + (1 + 8 * n_gen) ** 0.5) / 2)

        # Build skew-symmetric generator S
        S = torch.zeros(d, d, device=device, dtype=torch.float32)
        idx = torch.triu_indices(d, d, offset=1, device=device)
        S[idx[0], idx[1]] = self.lie_pe_params.float()
        S = S - S.T

        # Build batched (I + t*scale*S) and (I - t*scale*S) for all t
        I = torch.eye(d, device=device, dtype=torch.float32)
        t_vals = torch.arange(T, device=device, dtype=torch.float32)
        scale = self.lie_pe_scale.float()

        # (T, d, d)
        tS = (scale * t_vals[:, None, None]) * S[None, :, :]
        lhs = I[None, :, :] - tS  # (I - tS)
        rhs = I[None, :, :] + tS  # (I + tS)

        # Batched solve: R[i] = solve(rhs[i], lhs[i])
        R_all = torch.linalg.solve(rhs, lhs)  # (T, d, d)
        return R_all.to(dtype=dtype)

    @staticmethod
    def _lie_pe_cached(h: Tensor, t: int, pe_cache: Tensor) -> Tensor:
        """Apply precomputed LieRE rotation at step t."""
        return h @ pe_cache[t].T

    def forward(
        self,
        x: Tensor,
        z0: Tensor,
        target_ids: Tensor = None,
        max_steps: int = None,
    ) -> dict[str, Tensor]:
        """
        Geodesic forward pass.
        1. Pre-trained Transformer Body processes tokens to deep contextual representations.
        2. Geometric Math Head walks the manifold using those representations and z0.
        """

        B = x.shape[0]
        device = x.device
        p = self.config.d_lie
        T = target_ids.shape[1] if target_ids is not None else self.config.max_seq_len

        # Compute metrics
        G_model = self._compute_metric(z0)  # (B, d, d)
        G_lie = self._compute_lie_metric(z0)  # (B, p, p)

        # We need the spd_batchnorm function from geometry module, but since it's used globally
        # we assume it's imported at the top of the file as `from .geometry import spd_batchnorm`
        if self.training and B > 1:
            G_model = spd_batchnorm(G_model)
            G_lie = spd_batchnorm(G_lie)

        # Token embeddings matrix
        E = self.token_embedding.weight  # (V, p)

        # ---------------------------------------------------------
        # PHASE 1: TRANSFORMER BODY (Pre-trained Intelligence)
        # ---------------------------------------------------------
        x_emb = self.token_embedding(x)  # (B, T, p)
        h_tff = self.embed_up(x_emb)  # (B, T, d)
        h_tff = self.embed_dropout(h_tff)

        causal_mask = self._build_causal_mask(T, device)
        n_blocks = len(self.decoder_blocks)

        for i, block in enumerate(self.decoder_blocks):
            if self.training:
                drop_rate = 0.1 * (i / max(1, n_blocks - 1))
                if random.random() < drop_rate:
                    continue  # Stochastic depth

            if self.gradient_checkpointing and self.training:
                h_tff, _ = cp.checkpoint(block, h_tff, causal_mask, use_reentrant=False)
            else:
                h_tff, _ = block(h_tff, mask=causal_mask, use_cache=False)

        h_tff = self.final_norm(h_tff)  # (B, T, d)

        # ---------------------------------------------------------
        # PHASE 2: GEOMETRIC MATH HEAD & LISMCA
        # ---------------------------------------------------------
        # Initial concealed state seeded from z0 and first token context
        h0 = self.input_norm(self.input_proj(h_tff[:, 0]) + self.z0_proj(z0))  # (B, d)
        A_t = torch.zeros(B, p, p, device=device, dtype=h0.dtype)
        e_mask = E.mean(dim=0, keepdim=True).expand(B, -1)  # (B, p)

        all_logits = []
        all_hidden = [h0]
        all_e_masks = []
        h_t = h0

        B_patches = self.patch_proj(h0).unsqueeze(1).expand(-1, 4, -1)  # (B, 4, p)

        curvature = self.tcrs.compute_curvature(z0)
        use_woodbury = curvature > self.config.curvature_threshold

        tcrs_cache = self.tcrs.precompute(z0)
        wood_cache = self.woodbury.precompute(z0)
        bridge_cache = self.bridge.precompute(z0, E)
        pe_cache = self._precompute_lie_pe(T, device, h0.dtype)

        for t in range(T):
            # Structural geodesic march
            h_tcrs = self.tcrs.step(tcrs_cache, h_t)
            h_wood = self.woodbury.step(wood_cache, h_t)
            h_geo = torch.where(use_woodbury.unsqueeze(-1), h_wood, h_tcrs)

            h_geo = self._lie_pe_cached(h_geo, t, pe_cache)

            # Map coordinates to logits
            logits_t, e_mask, A_t = self.bridge(
                h_geo, A_t, B_patches, E, e_mask, G_lie, z0, cache=bridge_cache
            )
            all_logits.append(logits_t)
            all_e_masks.append(e_mask)

            # Advance trajectory using the LLM's deep contextual intelligence for step t
            mask_info = self.embed_up(self.embed_dropout(e_mask))
            context_injection = self.input_proj(
                h_tff[:, min(t + 1, T - 1)]
            )  # anchor towards next target
            h_t = h_geo + context_injection + 0.1 * mask_info

            all_hidden.append(h_t)

        logits = torch.stack(all_logits, dim=1)  # (B, T, V)
        hidden_states = torch.stack(all_hidden, dim=1)  # (B, T+1, d)
        e_masks = torch.stack(all_e_masks, dim=1)  # (B, T, p)
        pi = F.softmax(logits, dim=-1)  # (B, T, V)

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "pi": pi,
            "metric": G_model,
            "lie_metric": G_lie,
            "e_mask": e_masks,
        }

    def generate(
        self,
        x: torch.Tensor,
        z0: torch.Tensor,
        max_steps: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Fast autoregressive generation (inference mode) with KV caching.
        Delegates to generate_stream() and collects all tokens.
        """
        token_ids = []
        for next_token in self.generate_stream(
            x=x,
            z0=z0,
            max_steps=max_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
        ):
            token_ids.append(next_token)

        if token_ids:
            return torch.cat([x] + token_ids, dim=1)
        return x

    def generate_stream(
        self,
        x: torch.Tensor,
        z0: torch.Tensor,
        max_steps: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ):
        """
        Fast autoregressive generation generator (inference streaming) yielding KV tokens incrementally.

        Args:
            top_p: Nucleus sampling threshold (0.0=disabled). Keep smallest set of tokens
                   with cumulative probability >= top_p. Applied after top_k.
            repetition_penalty: Penalty for repeated tokens (1.0=disabled). Values > 1.0
                   discourage repetition. Applied before temperature scaling.
        """

        B = x.shape[0]
        device = x.device
        p = self.config.d_lie

        G_lie = self._compute_lie_metric(z0)
        E = self.token_embedding.weight

        past_kv_cache = [None for _ in range(len(self.decoder_blocks))]
        x_t_input = x  # Start with the full prompt context

        # Pre-compute
        curvature = self.tcrs.compute_curvature(z0)
        use_woodbury = curvature > self.config.curvature_threshold

        tcrs_cache = self.tcrs.precompute(z0)
        wood_cache = self.woodbury.precompute(z0)
        bridge_cache = self.bridge.precompute(z0, E)
        pe_cache = self._precompute_lie_pe(max_steps, device, z0.dtype)

        # T=0 Math state setup
        h_t = None
        A_t = None
        e_mask = E.mean(dim=0, keepdim=True).expand(B, -1)
        B_patches = None

        # Tracking unfinished sequences for EOS stopping
        unfinished = torch.ones(B, dtype=torch.bool, device=device)
        generated_tokens = []  # Track generated tokens for repetition penalty

        for t in range(max_steps):
            # 1. KV-Cached Transformer Pass
            T_batch = x_t_input.shape[1]
            x_emb = self.token_embedding(x_t_input)
            h_tff = self.embed_up(x_emb)

            causal_mask = None
            if T_batch > 1:
                causal_mask = self._build_causal_mask(T_batch, device)

            for i, block in enumerate(self.decoder_blocks):
                h_tff, new_kv = block(
                    h_tff, mask=causal_mask, past_kv=past_kv_cache[i], use_cache=True
                )
                past_kv_cache[i] = new_kv

            h_tff = self.final_norm(h_tff)

            # The latest deep context from the transformer
            ctx_out = h_tff[:, -1]

            # Initialize math trajectory if t==0
            if h_t is None:
                h_t = self.input_norm(self.input_proj(ctx_out) + self.z0_proj(z0))
                A_t = torch.zeros(B, p, p, device=device, dtype=h_t.dtype)
                B_patches = self.patch_proj(h_t).unsqueeze(1).expand(-1, 4, -1)

            # 2. Geometric Math Step
            h_tcrs = self.tcrs.step(tcrs_cache, h_t)
            h_wood = self.woodbury.step(wood_cache, h_t)
            h_geo = torch.where(use_woodbury.unsqueeze(-1), h_wood, h_tcrs)

            h_geo = self._lie_pe_cached(h_geo, t, pe_cache)

            logits_t, e_mask, A_t = self.bridge(
                h_geo, A_t, B_patches, E, e_mask, G_lie, z0, cache=bridge_cache
            )

            # 3. Sampling
            if temperature == 0.0:
                # Apply repetition penalty even in greedy mode
                if repetition_penalty != 1.0:
                    logits_penalized = logits_t.clone()
                    prev_tokens = torch.cat([x] + [tok for tok in generated_tokens], dim=1) if generated_tokens else x
                    for b in range(B):
                        unique_prev = prev_tokens[b].unique()
                        penalized = logits_penalized[b, unique_prev]
                        logits_penalized[b, unique_prev] = torch.where(
                            penalized > 0, penalized / repetition_penalty, penalized * repetition_penalty
                        )
                    next_token = torch.argmax(logits_penalized, dim=-1, keepdim=True)
                else:
                    next_token = torch.argmax(logits_t, dim=-1, keepdim=True)
            else:
                # Apply repetition penalty before temperature
                logits_scaled = logits_t.clone()
                if repetition_penalty != 1.0:
                    prev_tokens = torch.cat([x] + [tok for tok in generated_tokens], dim=1) if generated_tokens else x
                    for b in range(B):
                        unique_prev = prev_tokens[b].unique()
                        penalized = logits_scaled[b, unique_prev]
                        logits_scaled[b, unique_prev] = torch.where(
                            penalized > 0, penalized / repetition_penalty, penalized * repetition_penalty
                        )
                logits_scaled = logits_scaled / temperature
                if top_k > 0:
                    topk_vals, _ = torch.topk(logits_scaled, top_k, dim=-1)
                    threshold = topk_vals[:, -1].unsqueeze(-1)
                    logits_scaled = logits_scaled.masked_fill(
                        logits_scaled < threshold, float("-inf")
                    )

                # Top-p (nucleus) sampling: keep smallest set with cumulative prob >= p
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(logits_scaled, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Shift right so the first token above threshold is always kept
                    sorted_mask = (cumulative_probs - sorted_probs) >= top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    # Scatter back to original order
                    logits_scaled = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                next_token = torch.where(
                    unfinished.unsqueeze(1),
                    next_token,
                    torch.tensor(eos_token_id, device=device),
                )
                unfinished = unfinished & (next_token.squeeze(1) != eos_token_id)

            yield next_token
            generated_tokens.append(next_token)

            if not unfinished.any():
                break

            # 4. Math Trajectory Update for Next Step
            mask_info = self.embed_up(self.embed_dropout(e_mask))
            h_t = h_geo + self.input_proj(ctx_out) + 0.1 * mask_info

            x_t_input = next_token
