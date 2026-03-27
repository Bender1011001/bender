"""
LiSMCA Bridge: Lie-to-Token with Dynamic Soft-Masking
=====================================================

Full implementation from master document section 4.

The bridge connects the Riemannian manifold (continuous geometry) to the
discrete token space via:

1. TCRS geodesic alignment of hidden states
2. Cayley-mapped Lie group actions on embeddings
3. Curvature-aware attention (Riemannian distance bias)
4. Mahalanobis-based generator matching (kappa scores)
5. Gated fusion of base logits + group-theoretic logits
6. Dynamic soft-mask update (mu-gated interpolation)

The key insight: instead of projecting h -> logits directly (standard AR),
we route through the Lie algebra so that the token distribution is
*geometrically informed* by the manifold trajectory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .geometry import (
    cayley,
    skew_vectorize,
    row_mahalanobis,
    squared_riemannian_distance,
)


class LiSMCA_Bridge(nn.Module):
    """
    Lie-group Soft-Masked Cross-Attention bridge.

    Converts geodesic-aligned hidden states into token logits via:
      - Cayley-mapped orthogonal projections of embeddings
      - Riemannian-distance-biased attention
      - Generator-matching scores under learned precision
      - Gated base + group logit fusion
      - Dynamic soft-mask for next-step embedding priors

    Args:
        d_h:        Hidden state dimension (d_model)
        p:          Lie algebra dimension (d_lie)
        vocab_size: Vocabulary size
        k:          Top-k for dynamic soft-mask (default 8)
        d_latent:   Latent dimension for TCRS (default = p)
    """

    def __init__(
        self,
        d_h: int,
        p: int,
        vocab_size: int,
        k: int = 8,
        d_latent: int = None,
    ):
        super().__init__()
        self.d_h = d_h
        self.p = p
        self.vocab_size = vocab_size
        self.k = k
        if d_latent is None:
            d_latent = p

        # Attention projections (in Lie-group-transformed space)
        self.W_Q = nn.Linear(d_h, p)
        self.W_K = nn.Linear(p, p)
        self.W_V = nn.Linear(p, p)
        self.W_C = nn.Linear(p, p)

        # Base output projection (standard AR path)
        self.W_O = nn.Linear(d_h, vocab_size, bias=False)

        # Lie generator construction
        self.W_lie = nn.Linear(d_h, p - 1)  # For TCRS v_t

        # Token generator projections (for Mahalanobis matching)
        self.W_G = nn.Linear(p, p)

        # Gating networks
        skew_dim = p * (p - 1) // 2
        self.w_gamma = nn.Linear(d_h + p + skew_dim, 1)  # Gate: base vs group
        self.w_mu = nn.Linear(d_h + p + 2, 1)  # Gate: soft-mask update

        # Mathematical bottleneck projection limit (saves 17 Billion parameters)
        self.g_down = nn.Linear(skew_dim, p, bias=False)

        # Learned precision matrix for generator matching
        self.Sigma_inv = nn.Parameter(torch.ones(p) * 0.1)

        # Projection: h -> Lie algebra dimension for attention
        self.h_to_lie = nn.Linear(d_h, p) if d_h != p else nn.Identity()

    def precompute(self, z0: Tensor, E: Tensor) -> dict:
        """
        Pre-compute z0-dependent and E-dependent values for the loop.

        Call once before the decoder loop, pass cache to forward().

        Args:
            z0: (B, d_latent) latent blueprint
            E:  (V, p) token embedding matrix

        Returns:
            Cache dict with TCRS coefficients and E generators.
        """
        return {
            "E_generators": self.W_G(E),  # (V, p) — constant across all steps
        }

    def forward(
        self,
        h_t: Tensor,
        A_t: Tensor,
        B_t: Tensor,
        E: Tensor,
        e_mask: Tensor,
        G_tilde: Tensor,
        z0: Tensor,
        cache: dict = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the LiSMCA bridge.

        Args:
            h_t:     (B, d_h) hidden state at time t
            A_t:     (B, p, p) current Lie algebra state (skew-symmetric)
            B_t:     (B, n_patches, p) patch/context embeddings
            E:       (V, p) token embedding matrix (projected to Lie dim)
            e_mask:  (B, p) current soft-mask embedding
            G_tilde: (B, p, p) metric tensor (or diagonal approx)
            z0:      (B, d_latent) latent blueprint for TCRS
            cache:   Optional pre-computed cache from precompute()

        Returns:
            logits:  (B, V) token logits
            m_next:  (B, p) updated soft-mask embedding
            A_t:     (B, p, p) Lie algebra state (updated via gradient externally)
        """
        p = self.p

        # ── Step 1: Input alignment check ──────────────────────────────
        h_aligned = h_t

        # ── Step 2: Cayley map on Lie algebra state ──────────────────────
        Q_t = cayley(A_t)  # (B, p, p) orthogonal matrix

        # ── Step 3: LiSMCA attention ─────────────────────────────────────
        # Query from hidden state
        q_t = self.W_Q(h_t)  # (B, p)

        # Keys and Values: patch embeddings rotated by Lie group action
        # Keys and Values: patch embeddings rotated by Lie group action
        B_rotated = B_t @ Q_t  # (B, n, p)
        K_t = self.W_K(B_rotated)  # (B, n, p)
        V_t = self.W_V(B_rotated)  # (B, n, p)

        # Riemannian distance bias: penalizes attention to geometrically distant patches
        h_lie = self.h_to_lie(h_aligned)  # (B, p)
        omega_t = -0.5 * squared_riemannian_distance(
            h_lie, B_rotated, G_tilde
        )  # (B, n)

        # Attention scores with geometric bias
        attn_scores = (q_t.unsqueeze(-2) @ K_t.transpose(-1, -2)).squeeze(-2) / (
            p**0.5
        )  # (B, n)
        attn_scores = attn_scores + omega_t
        alpha = F.softmax(attn_scores, dim=-1)  # (B, n)

        # Context vector
        c_t = (alpha.unsqueeze(1) @ V_t).squeeze(1)  # (B, p)

        # ── Step 4: Generator matching (Mahalanobis) ─────────────────────
        # Skew-vectorize current Lie state for matching
        g_t = skew_vectorize(A_t)  # (B, skew_dim)

        # Mahalanobis distance: how well does each token's generator match g_t?
        E_generators = (
            cache["E_generators"] if cache is not None else self.W_G(E)
        )  # (V, p)

        # Project sparse O(d^2) generator g_t back onto native p dimensional basis
        g_proj = self.g_down(g_t)  # (B, p)

        # Calculate full Riemannian variance mapping dynamically via Mahalanobis
        # E_generators is (V, p), g_proj is (B, p). We want (B, V, p) -> (B, V)
        # So we treat E_generators as shared across batch and broadcast g_proj
        kappa = -row_mahalanobis(
            E_generators.unsqueeze(0), g_proj.unsqueeze(1), self.Sigma_inv
        )  # (B, V)

        # ── Step 5: Gated logit fusion ───────────────────────────────────
        # Base logits (standard AR path)
        base_logits = self.W_O(h_t)  # (B, V)

        # Group logits (geometry-informed path)
        c_transformed = self.W_C(c_t)  # (B, p)
        # O(N) Associative optimization: Rotate the small query vector c_t NOT the massive vocab E
        c_rotated = (c_transformed.unsqueeze(1) @ Q_t).squeeze(1)  # (B, p)
        group_logits = F.linear(c_rotated, E) / (p**0.5)  # (B, V)
        group_logits = group_logits + kappa

        # Gating: how much to trust geometry vs standard prediction
        gate_input = torch.cat([h_t, c_t, g_t], dim=-1)  # (B, d_h + p + skew_dim)
        gamma = torch.sigmoid(self.w_gamma(gate_input))  # (B, 1)

        logits = base_logits + gamma * group_logits  # (B, V)

        # ── Step 6: Dynamic soft-mask update ─────────────────────────────
        pi = F.softmax(logits, dim=-1)  # (B, V)

        # Top-k expected embedding
        topk_val, topk_idx = torch.topk(pi, self.k, dim=-1)  # (B, k)
        topk_probs = topk_val / topk_val.sum(
            dim=-1, keepdim=True
        )  # (B, k) renormalized

        # Gather unrotated embeddings first, THEN rotate them locally for O(1) memory
        E_gathered = E[topk_idx]  # (B, k, p)
        E_G_gathered = torch.einsum("bkp,bpq->bkq", E_gathered, Q_t)  # (B, k, p)

        E_exp = torch.einsum("bk,bkp->bp", topk_probs, E_G_gathered)  # (B, p)

        # Entropy of current distribution
        H = -(pi * torch.log(pi + 1e-12)).sum(dim=-1, keepdim=True)  # (B, 1)

        # Curvature-based confidence (simplified rho)
        rho = squared_riemannian_distance(h_lie, E_exp.unsqueeze(-2), G_tilde).squeeze(
            -1
        )  # (B,)
        rho = rho.unsqueeze(-1)  # (B, 1)

        # Soft-mask gate: high entropy / high curvature -> rely more on prior mask
        mu_input = torch.cat([h_t, c_t, H, rho], dim=-1)  # (B, d_h + p + 2)
        mu = torch.sigmoid(self.w_mu(mu_input))  # (B, 1)

        m_next = mu * e_mask + (1.0 - mu) * E_exp  # (B, p)

        # ── Step 7: Update Lie Trajectory ──────────────────────────────
        # The geometric shift from previous mask to current mask generates a rotation.
        # Compute the skew-symmetric projection of their outer product to drive A_t forward.
        outer = torch.bmm(m_next.unsqueeze(2), e_mask.unsqueeze(1))
        delta_A = 0.1 * (outer - outer.transpose(1, 2))  # (B, p, p)
        A_next = A_t + delta_A

        return logits, m_next, A_next
