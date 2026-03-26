"""
Epistemic Routing (v2.0)
=========================

When a user triggers a severe asymmetric penalty (r << 0), the system must
solve: "Did the model hallucinate (factual error) or just violate the user's
style preference?"

The Epistemic Variance Gate κ ∈ [0,1] routes the RL gradient:
  - κ → 1 (high EBM energy): structural error → update BASE manifold
  - κ → 0 (low EBM energy): style mismatch → update USER FIBER only

This prevents subjective anger from corrupting objective reasoning.
A user who hates verbosity can't accidentally teach the model that
"2+2=4 is wrong" — the base manifold stays protected.

κ = σ((E_ψ - E_threshold) / τ_e)

Route split:
  Δg_B     = κ · η_base · ∇_G L_RL^asym      (factual correction)
  ΔA^(u)   = (1-κ) · η_user · ∇_A L_RL^asym   (style correction)
"""

import torch
import torch.nn as nn
from torch import Tensor


class EpistemicRouter(nn.Module):
    """
    Routes RL gradient updates between base manifold and user fiber
    based on the EBM Critic's energy estimate.

    High-energy sequences indicate structural/factual problems that
    should update the global model. Low-energy sequences where the
    user is still unhappy indicate subjective style mismatches that
    should only affect the user's personal fiber.

    Args:
        energy_threshold: E_threshold — baseline energy for κ sigmoid
        tau_e: Temperature for the sigmoid (lower = sharper routing)
        eta_base: Learning rate multiplier for base manifold updates
        eta_user: Learning rate multiplier for user fiber updates
    """

    def __init__(
        self,
        energy_threshold_init: float = 5.0,
        tau_e: float = 0.5,
        eta_base: float = 0.01,
        eta_user: float = 0.05,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        # Use an Exponential Moving Average (EMA) to robustly track baseline energy,
        # preventing the silent failure of absolute bounds on new models/domains.
        self.register_buffer(
            "energy_threshold_ema", torch.tensor(energy_threshold_init)
        )
        self.ema_decay = ema_decay
        self.tau_e = tau_e
        # NOTE: eta_base and eta_user are stored for documentation purposes but are NOT
        # applied by this module. The actual learning rates come from the external
        # optimizer_base and optimizer_sidecar in chat_continuous.py.
        self.eta_base = eta_base
        self.eta_user = eta_user

    def compute_kappa(self, ebm_energy: Tensor) -> Tensor:
        """
        Compute the epistemic routing gate κ using EMA-calibrated thresholds.

        κ → 1: high relative energy → factual error → base update
        κ → 0: low relative energy → style issue → fiber update

        Args:
            ebm_energy: (B,) per-sample energy from EBM Critic

        Returns:
            kappa: (B,) routing coefficients in [0, 1]
        """
        # Auto-calibrate the underlying baseline energy distribution if training
        if self.training:
            with torch.no_grad():
                batch_mean = ebm_energy.mean()
                # Exponential moving average update
                new_ema = (
                    self.ema_decay * self.energy_threshold_ema
                    + (1.0 - self.ema_decay) * batch_mean
                )
                self.energy_threshold_ema.copy_(new_ema)

        return torch.sigmoid((ebm_energy - self.energy_threshold_ema) / self.tau_e)

    def route_gradients(
        self,
        rl_loss: Tensor,
        ebm_energy: Tensor,
    ) -> dict[str, Tensor]:
        """
        Split the RL loss into base and fiber components via κ routing.

        The total RL gradient is decomposed:
          L_base = κ · η_base · L_RL
          L_fiber = (1-κ) · η_user · L_RL

        The caller applies L_base.backward() to base model params
        and L_fiber.backward() to user fiber params.

        Args:
            rl_loss:    (B,) per-sample RL loss (before mean reduction)
            ebm_energy: (B,) per-sample EBM energy

        Returns:
            Dictionary with:
              - "kappa": (B,) routing coefficients
              - "base_loss": scalar loss for base manifold update
              - "fiber_loss": scalar loss for user fiber update
              - "base_fraction": mean κ (diagnostic)
        """
        kappa = self.compute_kappa(ebm_energy).detach()  # (B,)

        # Route: factual component goes to base, style component goes to fiber
        # Downstream mathematical optimization layers (AdamW / BCH) multiply by unique η values explicitly
        base_loss = (kappa * rl_loss).mean()
        fiber_loss = ((1.0 - kappa) * rl_loss).mean()

        return {
            "kappa": kappa.detach(),
            "base_loss": base_loss,
            "fiber_loss": fiber_loss,
            "base_fraction": kappa.mean().detach(),
        }

    def forward(
        self,
        rl_loss: Tensor,
        ebm_energy: Tensor,
    ) -> dict[str, Tensor]:
        """Alias for route_gradients."""
        return self.route_gradients(rl_loss, ebm_energy)
