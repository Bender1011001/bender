"""
Energy-Based Model Critic with C3 Credit Assignment
====================================================

Sequence-level free energy scorer from master document section 2.

The EBM critic scores the quality of a generated sequence by computing
a scalar energy E_eta(e_bar_{1:T}, x, z_0) over the relaxed (soft)
embeddings. Lower energy = better sequence.

The free-energy loss term is:
    L_free = E_eta(e_bar, x, z_0) + tau * sum_t H(pi_t)

where H(pi_t) is the per-step entropy and tau controls exploration.

The C3 (Contextual Credit Conditioning) mechanism provides per-token
credit signals by decomposing the sequence-level energy into per-step
contributions via gradient-based attribution.

Architecture:
  - Bidirectional transformer over relaxed embeddings
  - Conditioned on z_0 (latent blueprint) and x (input)
  - Outputs scalar energy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EBMTransformerLayer(nn.Module):
    """
    Single transformer layer for the energy model.

    Uses pre-norm architecture with bidirectional self-attention
    (no causal mask — the critic sees the full sequence).
    """

    def __init__(
        self, d_model: int, n_heads: int = 8, d_ff: int = None, dropout: float = 0.1
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.norm1 = nn.RMSNorm(d_model)
        self.q_norm = nn.RMSNorm(d_model)
        self.k_norm = nn.RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.RMSNorm(d_model)
        # SwiGLU FFN
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm self-attention with QK normalization (bidirectional)
        h = self.norm1(x)
        q = self.q_norm(h)
        k = self.k_norm(h)
        h = x + self.attn(q, k, h, need_weights=False)[0]
        # SwiGLU FFN
        ff_in = self.norm2(h)
        h = h + self.ff_dropout(
            self.w_down(F.silu(self.w_gate(ff_in)) * self.w_up(ff_in))
        )
        return h


class EBMCritic(nn.Module):
    """
    Energy-Based Model Critic for sequence-level scoring.

    Takes relaxed embeddings e_bar_{1:T} (soft token representations),
    the latent blueprint z_0, and input x, and outputs a scalar energy.

    Lower energy indicates a better sequence under the learned energy landscape.

    The C3 credit assignment is computed by taking the gradient of E w.r.t.
    each token's relaxed embedding, giving per-token importance signals.

    Args:
        d_embed:  Dimension of relaxed embeddings (Lie algebra dim p)
        d_latent: Dimension of latent blueprint z_0
        d_input:  Dimension of input features x
        d_hidden: Hidden dimension of energy network
        n_layers: Number of transformer layers
        n_heads:  Number of attention heads
    """

    def __init__(
        self,
        d_embed: int,
        d_latent: int,
        d_input: int,
        d_hidden: int = 1024,
        n_layers: int = 3,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_hidden = d_hidden

        # Project inputs to hidden dimension (spectrally normalized for Lipschitz)
        self.embed_proj = nn.utils.parametrizations.spectral_norm(
            nn.Linear(d_embed, d_hidden)
        )
        self.latent_proj = nn.utils.parametrizations.spectral_norm(
            nn.Linear(d_latent, d_hidden)
        )
        self.input_proj = nn.utils.parametrizations.spectral_norm(
            nn.Linear(d_input, d_hidden)
        )

        # Learnable [CLS] token for energy readout
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_hidden) * 0.02)

        # Positional encoding for sequence positions
        self.pos_encoding = nn.Parameter(
            torch.randn(1, 2050, d_hidden) * 0.02
        )  # max_len + 2

        # Bidirectional transformer
        self.layers = nn.ModuleList(
            [
                EBMTransformerLayer(d_hidden, n_heads, d_hidden * 4)
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(d_hidden)

        # Residual energy: E = E_base + alpha * E_residual (EDLM, arXiv 2410.17819)
        # Spectral norm on energy head for bounded energy landscape
        self.energy_head = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(d_hidden, d_hidden // 2)),
            nn.GELU(),
            nn.utils.parametrizations.spectral_norm(nn.Linear(d_hidden // 2, 1)),
        )
        # Base energy: simple cosine similarity (provides gradient signal from step 0)
        self.base_energy_proj = nn.Linear(d_embed, d_latent)
        self.residual_alpha = nn.Parameter(
            torch.tensor(0.1)
        )  # Learnable residual weight

    def forward(
        self,
        e_bar: Tensor,
        z0: Tensor,
        x: Tensor,
    ) -> Tensor:
        """
        Compute sequence-level energy using residual formulation.

        E(e_bar, z0, x) = E_base(e_bar, z0) + α * E_residual(e_bar, z0, x)

        Args:
            e_bar: (B, T, d_embed) relaxed embeddings (soft token representations)
            z0:    (B, d_latent) latent blueprint
            x:     (B, d_input) conditioning input

        Returns:
            energy: (B,) scalar energy values (lower = better)
        """
        B, T, _ = e_bar.shape

        # Base energy: negative cosine similarity between mean embedding and z0
        e_mean = e_bar.mean(dim=1)  # (B, d_embed)
        e_proj = self.base_energy_proj(e_mean)  # (B, d_latent)
        base_energy = -F.cosine_similarity(e_proj, z0, dim=-1)  # (B,)

        # Residual energy from transformer
        h_seq = self.embed_proj(e_bar)  # (B, T, d_hidden)
        h_z0 = self.latent_proj(z0).unsqueeze(1)  # (B, 1, d_hidden)
        h_x = self.input_proj(x).unsqueeze(1)  # (B, 1, d_hidden)

        # Prepend: [CLS] [z0] [x] [e_1 ... e_T]
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_hidden)
        sequence = torch.cat([cls, h_z0, h_x, h_seq], dim=1)  # (B, T+3, d_hidden)

        # Add positional encoding
        seq_len = sequence.shape[1]
        sequence = sequence + self.pos_encoding[:, :seq_len, :]

        # Transformer forward (bidirectional)
        for layer in self.layers:
            sequence = layer(sequence)

        sequence = self.final_norm(sequence)

        # Extract CLS token and compute residual energy
        cls_out = sequence[:, 0, :]  # (B, d_hidden)
        residual_energy = self.energy_head(cls_out).squeeze(-1)  # (B,)

        # Combined: E = E_base + alpha * E_residual
        energy = base_energy + self.residual_alpha * residual_energy

        return energy

    def compute_c3_credit(
        self,
        e_bar: Tensor,
        z0: Tensor,
        x: Tensor,
    ) -> Tensor:
        """
        Compute per-token C3 credit via gradient-based attribution.

        The credit for token t is proportional to ||dE/de_t||, which
        measures how much changing that token's embedding would affect
        the total sequence energy.

        Args:
            e_bar: (B, T, d_embed) relaxed embeddings (must require grad)
            z0:    (B, d_latent) latent blueprint
            x:     (B, d_input) conditioning input

        Returns:
            credit: (B, T) per-token credit magnitudes
        """
        # Ensure e_bar requires gradients for attribution
        if not e_bar.requires_grad:
            e_bar = e_bar.detach().requires_grad_(True)

        energy = self.forward(e_bar, z0, x)  # (B,)
        total_energy = energy.sum()

        # Gradient of energy w.r.t. each token embedding
        grads = torch.autograd.grad(
            total_energy,
            e_bar,
            create_graph=True,  # Allow second-order gradients for training
            retain_graph=True,
        )[
            0
        ]  # (B, T, d_embed)

        # Credit = L2 norm of gradient per token
        credit = grads.norm(dim=-1)  # (B, T)

        return credit

    def contrastive_loss(
        self,
        e_bar: Tensor,
        z0: Tensor,
        x: Tensor,
        margin: float = 1.0,
        noise_scale: float = 0.1,
    ) -> Tensor:
        """
        Noise Contrastive Estimation loss for EBM training (EDLM, arXiv 2410.17819).

        Generates negative samples via:
          1. Token-level shuffling across batch (structural corruption)
          2. Gaussian noise perturbation (continuous corruption)

        Loss = E_pos + max(0, margin - E_neg) per sample, averaged.

        This creates a proper energy landscape where:
          - Real sequences → low energy
          - Corrupted sequences → high energy (at least `margin` above real)

        Args:
            e_bar:       (B, T, d_embed) relaxed embeddings (real sequences)
            z0:          (B, d_latent) latent blueprint
            x:           (B, d_input) conditioning input
            margin:      Energy margin between positive and negative samples
            noise_scale: Scale of Gaussian noise for continuous corruption

        Returns:
            loss: scalar NCE loss
        """
        B, T, d = e_bar.shape

        # Strategy 1: Token shuffle
        perm = torch.rand(B, T, device=e_bar.device).argsort(dim=-1)
        e_neg_shuffle = torch.gather(e_bar, 1, perm.unsqueeze(-1).expand(-1, -1, d))

        # Strategy 2: Gaussian noise
        e_neg_noise = e_bar + noise_scale * torch.randn_like(e_bar)

        # Batch all forward passes concurrently for 4x latency reduction
        inputs = [e_bar, e_neg_shuffle, e_neg_noise]

        if B > 1:
            e_neg_cross = torch.roll(e_bar, shifts=1, dims=0)
            inputs.append(e_neg_cross)

        e_all = torch.cat(inputs, dim=0)
        z0_all = z0.repeat(len(inputs), 1)
        x_all = x.repeat(len(inputs), 1)

        E_all = self.forward(e_all, z0_all, x_all)

        E_pos = E_all[:B]
        E_neg_shuffle = E_all[B : 2 * B]
        E_neg_noise = E_all[2 * B : 3 * B]

        if B > 1:
            E_neg_cross = E_all[3 * B : 4 * B]
            E_neg = torch.min(torch.min(E_neg_shuffle, E_neg_noise), E_neg_cross)
        else:
            E_neg = torch.min(E_neg_shuffle, E_neg_noise)

        # Hinge loss: push positive down, push negative up past margin
        # L = E_pos + max(0, margin - E_neg)
        loss = E_pos.mean() + F.relu(margin - E_neg).mean()

        return loss


class FreeEnergyLoss(nn.Module):
    """
    Free-energy loss term: E_eta + tau * entropy.

    L_free = E_eta(e_bar, x, z_0) + tau * sum_t H(pi_t)

    The entropy term encourages exploration early in training and is
    annealed down as the model becomes more confident.
    """

    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        energy: Tensor,
        pi: Tensor,
    ) -> Tensor:
        """
        Compute free-energy loss.

        Args:
            energy: (B,) sequence-level energy from EBM critic
            pi:     (B, T, V) token probability distributions

        Returns:
            loss: scalar free-energy loss
        """
        # Entropy per token: H(pi_t) = -sum_v pi_tv * log(pi_tv)
        entropy = -(pi * torch.log(pi + 1e-12)).sum(dim=-1)  # (B, T)
        entropy_sum = entropy.sum(dim=-1)  # (B,)

        # Free energy = energy + tau * entropy_bonus (entropy is subtracted for exploration)
        loss = energy.mean() + self.tau * entropy_sum.mean()

        return loss
