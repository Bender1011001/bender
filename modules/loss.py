"""
Unified Dual-System Loss: L_Dual
=================================

From master document section 2:

L_Dual = lambda_ELBO * L_ELBO
       + lambda_geo  * L_geo
       + lambda_free * L_free

Where:
  L_ELBO = reconstruction + KL chain (from diffusion planner)
  L_geo  = 0.5 * sum_t xi_t^T G(a_t) xi_t + lambda_acc/2 * sum_t ||v_t+ - v_t-||^2
  L_free = E_eta(e_bar, x, z_0) + tau * sum_t H(pi_t)

Gradient flow: EBM -> relaxed embeddings -> logits -> Lie algebra A_t
(via Cayley Frechet) -> metric tensor G(a_t) -> diffusion prior z_0
(via Jacobi field).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GeodesicLoss(nn.Module):
    """
    Riemannian geodesic regularization loss.

    L_geo = 0.5 * sum_t xi_t^T G(a_t) xi_t  +  lambda_acc/2 * sum_t ||v_t+ - v_t-||^2

    First term: energy of tangent vectors (encourages geodesic paths).
    Second term: acceleration penalty (encourages smooth, non-jerky paths).

    The tangent vector xi_t = h_{t+1} - h_t is the discrete velocity,
    and the acceleration is v_t+ - v_t- = (h_{t+1} - h_t) - (h_t - h_{t-1}).

    Args:
        lambda_acc: Weight on acceleration penalty (default 0.1)
    """

    def __init__(self, lambda_acc: float = 0.1):
        super().__init__()
        self.lambda_acc = lambda_acc

    def forward(
        self,
        hidden_states: Tensor,
        metric_tensors: Tensor,
    ) -> Tensor:
        """
        Compute geodesic regularization loss.

        Args:
            hidden_states:  (B, T, d) sequence of hidden states along the geodesic
            metric_tensors: (B, T, d, d) or (B, d, d) metric tensor(s)
                           If (B, d, d), the same metric is used for all timesteps.

        Returns:
            loss: scalar geodesic loss
        """
        B, T, d = hidden_states.shape

        # Discrete velocities: xi_t = h_{t+1} - h_t
        xi = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]  # (B, T-1, d)

        # Kinetic energy: 0.5 * xi_t^T G xi_t
        if metric_tensors.dim() == 3:
            # Single metric per sample: (B, d, d) -> broadcast
            G_xi = torch.einsum("bde,bte->btd", metric_tensors, xi)  # (B, T-1, d)
        else:
            # Per-timestep metrics: (B, T, d, d). Use metrics[1:T] to match xi[0:T-1]
            G_xi = torch.einsum("btde,bte->btd", metric_tensors[:, 1:, :, :], xi)

        kinetic = 0.5 * (xi * G_xi).sum(dim=-1)  # (B, T-1)
        kinetic_loss = kinetic.mean()

        # Acceleration penalty: ||v_t+ - v_t-||^2 where v_t± are forward/backward velocities
        if T >= 3:
            v_forward = xi[:, 1:, :]  # (B, T-2, d) = h[2:T] - h[1:T-1]
            v_backward = xi[:, :-1, :]  # (B, T-2, d) = h[1:T-1] - h[0:T-2]
            acc = v_forward - v_backward  # (B, T-2, d)

            # Measure acceleration in the metric
            if metric_tensors.dim() == 3:
                G_acc = torch.einsum("bde,bte->btd", metric_tensors, acc)
            else:
                G_acc = torch.einsum("btde,bte->btd", metric_tensors[:, 2:, :, :], acc)

            acc_loss = 0.5 * self.lambda_acc * (acc * G_acc).sum(dim=-1).mean()
        else:
            acc_loss = torch.tensor(0.0, device=hidden_states.device)

        return kinetic_loss + acc_loss


class DualSystemLoss(nn.Module):
    """
    Unified Dual-System Loss combining all objectives (v1.2).

    L_Dual = lambda_ELBO * L_ELBO
           + lambda_geo  * L_geo
           + lambda_free * L_free
           + lambda_RL   * L_RL^asym    (v1.2 addendum)

    The RL term adds human-reaction reinforcement:
      - Positive reactions: linear reinforcement of good token paths.
      - Negative reactions: exponentially amplified punishment.
      - Both flow through C3 credit for per-token attribution.

    Supports warm-up scheduling: during the first warmup_fraction of training,
    only L_ELBO is active to let the diffusion planner converge.

    Args:
        lambda_elbo: Weight on ELBO loss (default 1.0)
        lambda_geo:  Weight on geodesic loss (default 0.5)
        lambda_free: Weight on free-energy loss (default 1.0)
        lambda_rl:   Weight on asymmetric RL loss (default 0.3)
        tau:         Entropy temperature (default 0.1)
        tau_s:       Severity temperature for RL (default 0.3)
        lambda_acc:  Acceleration penalty weight (default 0.1)
    """

    def __init__(
        self,
        lambda_elbo: float = 1.0,
        lambda_geo: float = 0.5,
        lambda_free: float = 1.0,
        lambda_rl: float = 0.3,
        tau: float = 0.1,
        tau_s: float = 0.3,
        lambda_acc: float = 0.1,
        auto_tau: bool = True,
        target_entropy_ratio: float = 0.5,
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.tau_s = tau_s
        self.auto_tau = auto_tau
        self.learnable_weights = learnable_weights

        if learnable_weights:
            # Uncertainty-based multi-task weighting (Kendall & Gal 2017)
            # Learn log(sigma^2) per task. Weight = 1/(2*sigma^2), regularizer = log(sigma)
            # Initialize so exp(log_sigma_sq) ≈ 1/(2*lambda) → lambda = 1/(2*exp(x))

            self.log_sigma_sq_elbo = nn.Parameter(
                torch.tensor(math.log(1.0 / (2 * lambda_elbo)))
            )
            self.log_sigma_sq_geo = nn.Parameter(
                torch.tensor(math.log(1.0 / (2 * lambda_geo)))
            )
            self.log_sigma_sq_free = nn.Parameter(
                torch.tensor(math.log(1.0 / (2 * lambda_free)))
            )
            self.log_sigma_sq_rl = nn.Parameter(
                torch.tensor(math.log(1.0 / (2 * lambda_rl)))
            )
        else:
            self.lambda_elbo = lambda_elbo
            self.lambda_geo = lambda_geo
            self.lambda_free = lambda_free
            self.lambda_rl = lambda_rl

        # Meta-SAC style auto-tuning entropy temperature
        # tau = exp(log_tau) — learned via gradient descent
        # Target entropy = target_entropy_ratio * log(V)
        if auto_tau:
            self.log_tau = nn.Parameter(torch.tensor(float(tau)).log())
            self.target_entropy_ratio = target_entropy_ratio
        else:
            self.tau = tau

        self.geo_loss = GeodesicLoss(lambda_acc=lambda_acc)

    def forward(
        self,
        elbo_loss: Tensor,
        hidden_states: Tensor,
        metric_tensors: Tensor,
        energy: Tensor,
        logits: Tensor,
        warmup: float | bool = False,
        # v1.2 RL inputs (optional — omit during pre-training without reactions)
        c3_weights: Tensor = None,
        advantages: Tensor = None,
        log_probs: Tensor = None,
        reaction_reward: Tensor = None,
    ) -> dict[str, Tensor]:
        """
        Compute the unified loss (v1.2).

        Args:
            elbo_loss:       Scalar ELBO loss from diffusion planner
            hidden_states:   (B, T, d) hidden state trajectory
            metric_tensors:  (B, [T,] d, d) learned SPD metric(s)
            energy:          (B,) sequence energy from EBM critic
            logits:          (B, T, V) raw token logits from decoder
            warmup:          If True, only use ELBO (everything else zeroed)

            # v1.2 RL inputs (all required together, or all None)
            c3_weights:      (B, T) per-token C3 credit weights
            advantages:      (B, T) per-token advantage estimates
            log_probs:       (B, T) log probabilities of generated tokens
            reaction_reward: (B,) human reaction in [-1, +1]

        Returns:
            Dictionary with:
              - "total": scalar total loss
              - "elbo": scalar ELBO component
              - "geo": scalar geodesic component
              - "free": scalar free-energy component
              - "rl": scalar asymmetric RL component
              - "entropy": scalar mean entropy
        """

        device = elbo_loss.device
        zero = torch.tensor(0.0, device=device)

        # ELBO term (always active)
        if self.learnable_weights:
            # Kendall & Gal: w_i = 1/(2*sigma_i^2), regularizer = 0.5*log(sigma_i^2)
            w_elbo = 0.5 * torch.exp(-self.log_sigma_sq_elbo)
            l_elbo = w_elbo * elbo_loss + 0.5 * self.log_sigma_sq_elbo
        else:
            l_elbo = self.lambda_elbo * elbo_loss

        # Determine warmup scaling factor
        # warmup=True or warmup=0.0 → scale=0 (ELBO only)
        # warmup=False or warmup=1.0 → scale=1 (full loss)
        # warmup=float(0..1) → smooth cosine ramp
        if isinstance(warmup, bool):
            ramp_scale = 0.0 if warmup else 1.0
        else:
            progress = max(0.0, min(1.0, float(warmup)))
            ramp_scale = 0.5 * (1.0 - math.cos(math.pi * progress))

        # z-loss: penalize large logit magnitudes for numerical stability (PaLM/ST-MoE)
        if logits is not None:
            log_partition = torch.logsumexp(logits, dim=-1)  # (B, T)
            z_loss = 1e-4 * (log_partition**2).mean()
            l_elbo = l_elbo + z_loss
        else:
            z_loss = zero

        if ramp_scale < 1e-6:
            return {
                "total": l_elbo,
                "elbo": elbo_loss.detach(),
                "geo": zero,
                "free": zero,
                "rl": zero,
                "entropy": zero,
                "tau": zero,
                "z_loss": z_loss.detach() if isinstance(z_loss, Tensor) else zero,
            }

        # Geodesic term
        geo_loss = self.geo_loss(hidden_states, metric_tensors)
        if self.learnable_weights:
            w_geo = 0.5 * torch.exp(-self.log_sigma_sq_geo)
            l_geo = ramp_scale * (w_geo * geo_loss + 0.5 * self.log_sigma_sq_geo)
        else:
            l_geo = ramp_scale * self.lambda_geo * geo_loss

        # Free-energy term with auto-tuned temperature (Meta-SAC, arXiv 2007.01932)
        if logits is not None:
            pi = F.softmax(logits, dim=-1)
            entropy = -(pi * torch.log(pi + 1e-12)).sum(dim=-1)  # (B, T)
            mean_entropy = entropy.mean()
            V = logits.shape[-1]
        else:
            pi = None
            mean_entropy = torch.tensor(0.0, device=device)
            V = 2

        energy_mean = energy.mean()

        if self.auto_tau:
            tau = self.log_tau.exp()
            # Target entropy: half of maximum possible entropy (log V)
            target_H = self.target_entropy_ratio * math.log(max(V, 2))
            # Temperature auto-tuning loss: tau * (H_actual - H_target)
            # If entropy < target, (H_actual - target_H) is negative, so minimizing tau*(negative)
            # INCREASES tau, which properly pushes the model to increase entropy.
            tau_loss = tau * (mean_entropy.detach() - target_H)
        else:
            tau = self.tau
            tau_loss = torch.tensor(0.0, device=device)

        if self.learnable_weights:
            w_free = 0.5 * torch.exp(-self.log_sigma_sq_free)
            l_free = ramp_scale * (
                w_free * (energy_mean + tau * mean_entropy)
                + 0.5 * self.log_sigma_sq_free
            )
        else:
            l_free = ramp_scale * self.lambda_free * (energy_mean + tau * mean_entropy)

        # Asymmetric RL term (v1.2 — only when reaction data is provided)
        has_rl = (
            c3_weights is not None
            and advantages is not None
            and log_probs is not None
            and reaction_reward is not None
        )

        if has_rl:
            from .rl_layer import asymmetric_rl_loss

            rl_loss = asymmetric_rl_loss(
                c3_weights=c3_weights,
                advantages=advantages,
                log_probs=log_probs,
                reaction_reward=reaction_reward,
                tau_s=self.tau_s,
            )
            if self.learnable_weights:
                w_rl = 0.5 * torch.exp(-self.log_sigma_sq_rl)
                l_rl = ramp_scale * (w_rl * rl_loss + 0.5 * self.log_sigma_sq_rl)
            else:
                l_rl = ramp_scale * self.lambda_rl * rl_loss
        else:
            rl_loss = zero
            l_rl = zero

        total = l_elbo + l_geo + l_free + l_rl + tau_loss

        return {
            "total": total,
            "elbo": elbo_loss.detach(),
            "geo": geo_loss.detach(),
            "free": (energy_mean + tau * mean_entropy).detach(),
            "rl": rl_loss.detach() if has_rl else zero,
            "entropy": mean_entropy.detach(),
            "tau": (tau.detach() if isinstance(tau, Tensor) else torch.tensor(tau)),
            "z_loss": z_loss.detach() if isinstance(z_loss, Tensor) else zero,
        }
