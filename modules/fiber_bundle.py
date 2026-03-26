"""
Principal Fiber Bundle (v2.0)
==============================

Elevates the Dual-System Architecture from a monolithic Riemannian manifold
into a Principal SO(d)-Bundle P(B, G) that mathematically decouples:

  BASE SPACE (B):   Objective truth — factual logic, reasoning structure,
                    the unperturbed diffusion prior z₀. Immutable to emotion.
                    "The AI knows WHAT a Python script is."

  USER FIBERS (F_u): Subjective personalization — style, tone, verbosity,
                    relational preferences. Localized per user.
                    "The AI knows HOW User u likes their Python scripts."

Without this separation, User A's exponential anger at verbosity overwrites
User B's approval of the same verbosity → "Topological Schizophrenia."
The fiber bundle prevents this by isolating per-user emotional geometry.

Core component: The Kaluza-Klein Bundle Metric
  ds²_P = g^B_μν dx^μ dx^ν  +  k_ab (θ^a + A^(u),a_μ dx^μ)(θ^b + A^(u),b_ν dx^ν)

Where A^(u) ∈ so(d) is the user-specific gauge connection (Lie algebra element)
and k_ab is the Cartan-Killing form preserving orthogonality.
"""

import math
import torch
import torch.nn as nn
from torch import Tensor

from .geometry import cayley, skew_unvectorize


class UserFiberStore(nn.Module):
    """
    Stores per-user gauge connections A^(u) ∈ so(d).

    Each user has a skew-symmetric matrix representing their personalization
    in the Lie algebra. These are lifelong parameters that accumulate
    learning from the user's reaction history via BCH consolidation.

    Storage is in vectorized form (upper-triangle of skew-symmetric)
    to save memory: d*(d-1)/2 per user instead of d*d.
    """

    def __init__(self, num_users: int, d_fiber: int):
        super().__init__()
        self.num_users = num_users
        self.d_fiber = d_fiber
        self.vec_dim = d_fiber * (d_fiber - 1) // 2

        # Vectorized skew-symmetric gauge connections, initialized to zero
        # (new users start with no personalization — pure base model)
        self.fiber_vectors = nn.Parameter(torch.zeros(num_users, self.vec_dim))

    def get_gauge_connection(self, user_ids: Tensor) -> Tensor:
        """
        Retrieve gauge connections for a batch of users.

        Args:
            user_ids: (B,) integer user IDs

        Returns:
            A_u: (B, d, d) skew-symmetric gauge connections in so(d)
        """
        vecs = self.fiber_vectors[user_ids]  # (B, vec_dim)
        return skew_unvectorize(vecs, self.d_fiber)  # (B, d, d)

    def get_rotation(self, user_ids: Tensor) -> Tensor:
        """
        Get the SO(d) rotation matrix for each user via Cayley map.

        Args:
            user_ids: (B,) integer user IDs

        Returns:
            Q_u: (B, d, d) orthogonal rotation matrices in SO(d)
        """
        A_u = self.get_gauge_connection(user_ids)
        return cayley(A_u)  # (B, d, d)


class KaluzaKleinMetric(nn.Module):
    """
    Kaluza-Klein bundle metric combining base and fiber components.

    ds²_P = g^B(h) + k(θ + A^(u) · h)(θ + A^(u) · h)

    The base metric g^B captures objective geometric structure.
    The fiber contribution rotates the metric by the user's gauge connection,
    adding personalization without altering the base factual geometry.

    The combined metric is used by the TCRS solver and geodesic loss
    to trace user-personalized trajectories on the manifold.
    """

    def __init__(self, d_fiber: int, fiber_weight: float = 0.1):
        super().__init__()
        self.d_fiber = d_fiber
        self.fiber_weight = fiber_weight

        # Learnable base-to-fiber projection (maps hidden states to fiber space)
        self.base_to_fiber = nn.Linear(d_fiber, d_fiber, bias=False)

    def forward(
        self,
        G_base: Tensor,
        A_u: Tensor,
        h: Tensor = None,
    ) -> Tensor:
        """
        Compute the Kaluza-Klein bundle metric for a batch.

        The user's gauge connection A_u rotates the metric via conjugation:
        G_bundle = G_base + w * Q_u^T G_base Q_u

        This ensures the base component is always present (factual reasoning)
        while the fiber adds a user-specific geometric layer.

        Args:
            G_base: (B, d, d) SPD base metric from the decoder's metric factory
            A_u:    (B, d, d) skew-symmetric gauge connection for each user
            h:      (B, d) hidden states (optional, for state-dependent mixing)

        Returns:
            G_bundle: (B, d, d) combined SPD bundle metric
        """
        # User rotation via Cayley map: A_u → Q_u ∈ SO(d)
        Q_u = cayley(A_u)  # (B, d, d)

        # Fiber metric contribution: rotate the base metric by user's rotation
        # G_fiber = Q_u^T @ G_base @ Q_u (conjugation preserves SPD)
        G_fiber = torch.bmm(torch.bmm(Q_u.transpose(-1, -2), G_base), Q_u)

        # Combined bundle metric (base + weighted fiber)
        G_bundle = G_base + self.fiber_weight * G_fiber

        return G_bundle


class PrincipalFiberBundle(nn.Module):
    """
    Principal SO(d)-Bundle for multi-user topology.

    This is the top-level v2.0 module that:
    1. Stores per-user gauge connections (fibers)
    2. Computes bundle metrics for user-personalized inference
    3. Routes RL updates between base and fiber via epistemic κ

    The bundle preserves the base manifold (objective truth) while allowing
    each user to have their own stylistic/relational geometry that evolves
    independently over their interaction history.
    """

    def __init__(
        self,
        num_users: int,
        d_fiber: int,
        fiber_weight: float = 0.1,
    ):
        super().__init__()
        self.d_fiber = d_fiber

        self.fiber_store = UserFiberStore(num_users, d_fiber)
        self.kk_metric = KaluzaKleinMetric(d_fiber, fiber_weight)

    def compute_bundle_metric(
        self,
        G_base: Tensor,
        user_ids: Tensor,
    ) -> Tensor:
        """
        Compute the full Kaluza-Klein bundle metric for a batch of users.

        Args:
            G_base:   (B, d, d) SPD base metric from decoder
            user_ids: (B,) integer user IDs

        Returns:
            G_bundle: (B, d, d) personalized SPD bundle metric
        """
        A_u = self.fiber_store.get_gauge_connection(user_ids)
        return self.kk_metric(G_base, A_u)

    def lift_to_fiber(
        self,
        z0: Tensor,
        user_ids: Tensor,
    ) -> Tensor:
        """
        Lift the base manifold blueprint z₀ into user-specific fiber.

        Applies the user's SO(d) rotation to z₀, giving a personalized
        starting point for the AR decoder while preserving the factual
        content of the latent blueprint.

        Args:
            z0:       (B, d_latent) latent blueprint from diffusion planner
            user_ids: (B,) integer user IDs

        Returns:
            z0_lifted: (B, d_latent) user-personalized latent blueprint
        """
        Q_u = self.fiber_store.get_rotation(user_ids)  # (B, d, d)

        # To prevent the Jacobian vanishing gradient problem (where a consistent
        # style rotation applied to a randomly pointing prompt-dependent z0 averages
        # to 0 over a batch), we translate z0 using a fixed reference vector rotated
        # by the user's fiber. This maps rotational style into a stable, additive
        # latent shift invariant to the user's input prompt!
        v_ref = torch.ones(Q_u.shape[-1], 1, device=z0.device, dtype=z0.dtype)
        v_ref = v_ref / math.sqrt(Q_u.shape[-1])
        
        # Consistent translation shift: (Q - I) * v_ref
        I = torch.eye(Q_u.shape[-1], device=z0.device, dtype=z0.dtype).unsqueeze(0)
        v_style = torch.bmm(Q_u - I, v_ref.expand(Q_u.shape[0], -1, -1)).squeeze(-1)
        
        d_latent = z0.shape[-1]
        z0_lifted = z0.clone()
        
        if d_latent <= self.d_fiber:
            z0_lifted = z0_lifted + v_style[:, :d_latent]
        else:
            z0_lifted[:, :self.d_fiber] = z0_lifted[:, :self.d_fiber] + v_style
            
        return z0_lifted

    def get_user_gauge(self, user_ids: Tensor) -> Tensor:
        """Get gauge connections for a batch. Convenience wrapper."""
        return self.fiber_store.get_gauge_connection(user_ids)
