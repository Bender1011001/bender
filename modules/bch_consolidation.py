"""
Baker-Campbell-Hausdorff Lifelong Consolidation (v2.0)
======================================================

At the end of an interaction episode, volatile perturbations from the
asymmetric RL layer must be permanently written into the user's lifelong
fiber A^(u). Because we operate in the non-Abelian Lie algebra so(d),
simple Euclidean addition causes structural decay over thousands of
interactions — the Lie bracket terms matter.

The 2nd-order BCH expansion:
  A_new = A_old + η·ΔA + (η/2)·[A_old, ΔA] + O(η²)

The Lie bracket [X,Y] = XY - YX captures BEHAVIORAL SHOCK:
  - If a historically kind user suddenly inflicts severe exponential penalty,
    the non-commutativity produces a distinct, amplified correction compared
    to a user who is perpetually angry.
  - The model structurally remembers the CONTEXT of the emotion, not just
    its magnitude.

All operations enforce skew-symmetry to stay in so(d).
"""

import torch
import torch.nn as nn
from torch import Tensor

from .geometry import skew_vectorize


class BCHConsolidator(nn.Module):
    """
    Lifelong memory consolidation via Baker-Campbell-Hausdorff flow.

    Accumulates episodic RL perturbations into a user's lifelong gauge
    connection A^(u) using the non-Abelian structure of so(d).

    The BCH expansion preserves the Lie algebra structure that would
    be lost by naive Euclidean addition:
      A_{T+1} = A_T + η·ΔA_ep + (η/2)·[A_T, ΔA_ep]

    Args:
        eta: Learning rate for lifelong consolidation (default 0.05)
        order: BCH expansion order (1 = linear, 2 = + Lie bracket)
    """

    def __init__(self, eta: float = 0.05, order: int = 2):
        super().__init__()
        self.eta = eta
        self.order = order

    @staticmethod
    def lie_bracket(X: Tensor, Y: Tensor) -> Tensor:
        """
        Compute the Lie bracket [X, Y] = XY - YX.

        For X, Y ∈ so(d) (skew-symmetric), the bracket is also skew-symmetric,
        keeping the result in the Lie algebra.

        Args:
            X: (..., d, d) Lie algebra element
            Y: (..., d, d) Lie algebra element

        Returns:
            [X,Y]: (..., d, d) Lie bracket (skew-symmetric)
        """
        return torch.matmul(X, Y) - torch.matmul(Y, X)

    @staticmethod
    def enforce_skew_symmetry(A: Tensor) -> Tensor:
        """
        Project matrix to skew-symmetric: A_skew = (A - A^T) / 2.

        Numerical errors can break exact skew-symmetry over many updates.
        This projection is O(d²) and keeps everything in so(d).

        Args:
            A: (..., d, d) matrix

        Returns:
            A_skew: (..., d, d) skew-symmetric projection
        """
        return 0.5 * (A - A.transpose(-1, -2))

    def consolidate(
        self,
        A_old: Tensor,
        delta_A_episode: Tensor,
    ) -> Tensor:
        """
        Apply BCH consolidation to update a lifelong gauge connection.
        """
        # Enforce skew-symmetry on the input
        delta_A = self.enforce_skew_symmetry(delta_A_episode)

        # Assess spectral norm budget (ord=2) for strict convergence bounds
        norm_old = torch.linalg.matrix_norm(A_old.float(), ord=2, dim=(-2, -1), keepdim=True).to(A_old.dtype)
        norm_delta = torch.linalg.matrix_norm((self.eta * delta_A).float(), ord=2, dim=(-2, -1), keepdim=True).to(delta_A.dtype)

        # Scale delta_A BEFORE the bracket to ensure BCH series convergence.
        # This prevents the bracket expansion from silently diverging.
        available_radius = torch.clamp(0.693 - norm_old, min=1e-8)
        scale = torch.clamp(available_radius / (norm_delta + 1e-8), max=1.0)
        delta_A_scaled = delta_A * scale

        # 1st order: simple addition
        A_new = A_old + self.eta * delta_A_scaled

        # 2nd order: add Lie bracket (captures behavioral shock)
        if self.order >= 2:
            bracket = self.lie_bracket(A_old, delta_A_scaled)
            A_new = A_new + 0.5 * self.eta * bracket

        # Re-project to ensure strict skew-symmetry after accumulation
        A_new_skew = self.enforce_skew_symmetry(A_new)
        
        # Explicit clamp to prevent convergence lockout
        final_norm = torch.linalg.matrix_norm(A_new_skew.float(), ord=2, dim=(-2, -1), keepdim=True).to(A_new_skew.dtype)
        scale_limit = torch.clamp(0.693 / (final_norm + 1e-8), max=1.0)
        A_new_skew = A_new_skew * scale_limit
        
        return A_new_skew

    def consolidate_batch(
        self,
        fiber_store,
        user_ids: Tensor,
        episodic_deltas: dict[int, Tensor],
    ):
        """
        Batch-consolidate episodic learning into lifelong fibers.

        Called at the end of an episode (conversation turn boundary).
        Each user who interacted gets their fiber updated via BCH.

        Args:
            fiber_store:      UserFiberStore instance
            user_ids:         (B,) user IDs that participated in this episode
            episodic_deltas:  dict mapping user_id → accumulated ΔA for the episode
        """
        unique_users = user_ids.unique()

        for uid in unique_users:
            uid_int = uid.item()
            if uid_int not in episodic_deltas:
                continue

            delta_A = episodic_deltas[uid_int]
            A_old = fiber_store.get_gauge_connection(uid.unsqueeze(0)).squeeze(0)

            A_new = self.consolidate(A_old, delta_A)

            # Write back the consolidated gauge connection
            # We need to update the vectorized storage
            new_vec = skew_vectorize(A_new.unsqueeze(0)).squeeze(0)
            fiber_store.fiber_vectors.data[uid_int] = new_vec

    def compute_behavioral_shock(
        self,
        A_old: Tensor,
        delta_A: Tensor,
    ) -> Tensor:
        """
        Measure behavioral shock: how different the episodic perturbation
        is from the user's accumulated history.

        High shock = the Lie bracket term dominates = the user did
        something very out-of-character.

        Args:
            A_old:   (d, d) current lifelong gauge connection
            delta_A: (d, d) episodic perturbation

        Returns:
            shock: scalar — Frobenius norm of the Lie bracket relative
                   to the norms of the inputs
        """
        bracket = self.lie_bracket(A_old, delta_A)
        bracket_norm = bracket.norm()
        input_norm = (A_old.norm() * delta_A.norm()).clamp(min=1e-8)
        return bracket_norm / input_norm
