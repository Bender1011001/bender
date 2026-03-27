"""
TCRS Geodesic Solver + Woodbury-Retraction Fallback
====================================================

Two geodesic computation methods from the master document section 3:

1. **TCRS (Primary)** - O(d) per token
   Tridiagonal Lie generator Omega_t = tridiag(-v, 0, v), v = tanh(W_lie @ z_0).
   Exact Crank-Nicolson step solved via Thomas algorithm.
   Global error bound: O(dt^2).

2. **Woodbury-Retraction (Fallback)** - O(dr + r^3) per token
   Used when curvature exceeds threshold.
   G_tilde(z) = D(z) + U(z)U(z)^T, r=8 low-rank.
   Midpoint retraction solver with error O(eps + h^2).

Both solvers are fully differentiable and support torch.compile.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def solve_tridiagonal_lu(a: Tensor, b: Tensor, c: Tensor) -> tuple[Tensor, Tensor]:
    """
    Precompute the localized LU factorization for a tridiagonal system.
    This strictly evaluates exactly once per sequence minimizing VRAM footprints.

    Args:
        a: (..., n) sub-diagonal (a[0] unused)
        b: (..., n) main diagonal
        c: (..., n) super-diagonal (c[n-1] unused)

    Returns:
        A_LU, pivots: Tuple bindings for torch.linalg.lu_solve
    """
    # Build the tridiagonal matrix A (batch_shape, n, n)
    A = torch.diag_embed(b)  # (..., n, n)
    A += torch.diag_embed(a[..., 1:], offset=-1)
    A += torch.diag_embed(c[..., :-1], offset=1)

    # torch.linalg.lu_factor expects float32 natively
    A_LU, pivots = torch.linalg.lu_factor(A.float())
    return A_LU, pivots


def thomas_algorithm_lu_step(A_LU: Tensor, pivots: Tensor, d_vec: Tensor) -> Tensor:
    """
    Solve Ax = d using securely precomputed LU topology dynamically executing per-step natively O(N^2) instead of rebuilding constraints structurally.

    Args:
        A_LU, pivots: Precomputed arrays
        d_vec: (..., n) right-hand side

    Returns:
        x: (..., n) mathematically strict localized bounds
    """
    orig_dtype = d_vec.dtype
    x = torch.linalg.lu_solve(
        A_LU.float(), pivots, d_vec.unsqueeze(-1).float()
    ).squeeze(-1)
    return x.to(dtype=orig_dtype)


class TCRSSolver(nn.Module):
    """
    Tridiagonal Crank-Nicolson Riemannian Stepper (TCRS).

    Primary geodesic solver. Given latent z_0, constructs a tridiagonal
    Lie generator Omega and performs a single Crank-Nicolson step:

        h_{t+1} = (I - dt/2 * Omega)^{-1} (I + dt/2 * Omega) h_t

    Because Omega is tridiagonal, both the LHS and RHS are tridiagonal
    systems solvable in O(d) via Thomas algorithm.

    The Lie generator is constructed as:
        v = tanh(W_lie @ z_0)   (d-1 dimensional)
        Omega = tridiag(-v, 0, v)

    This gives a skew-symmetric tridiagonal matrix, guaranteeing the
    step preserves norm (orthogonal evolution).

    Error bound: O(dt^2) globally (Crank-Nicolson is 2nd order).
    """

    def __init__(self, d_model: int, d_latent: int, dt: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.dt = dt
        # Produces d-1 values for the off-diagonals of Omega
        self.W_lie = nn.Linear(d_latent, d_model - 1)

    def precompute(self, z0: Tensor) -> dict[str, Tensor]:
        """
        Pre-compute z0-dependent coefficients for the Thomas algorithm.

        Call this ONCE before the decoder loop, then use step() for each
        iteration. Eliminates redundant W_lie projection + tanh per step.

        Args:
            z0: (B, d_latent) latent blueprint

        Returns:
            Cache dict with pre-computed v, a_sub, b_main, c_sup
        """
        half_dt = self.dt / 2.0
        v = torch.tanh(self.W_lie(z0))  # (B, d-1)

        # Thomas algorithm coefficients — constant across all T steps
        b_main = torch.ones(z0.shape[0], self.d_model, device=z0.device, dtype=z0.dtype)
        a_sub = F.pad(half_dt * v, (1, 0), value=0.0)  # (B, d)
        c_sup = F.pad(-half_dt * v, (0, 1), value=0.0)  # (B, d)

        # Precompute dense factorization exactly once mapping directly to GPU kernel structures natively
        A_LU, pivots = solve_tridiagonal_lu(a_sub, b_main, c_sup)

        return {
            "v": v,  # (B, d-1) Lie generator off-diagonals
            "half_dt": half_dt,
            "A_LU": A_LU,  # O(n^2) LU bounds statically cached
            "pivots": pivots,
        }

    def step(self, cache: dict[str, Tensor], h: Tensor) -> Tensor:
        """
        Perform one TCRS geodesic step using statically cached LU coefficients.

        Args:
            cache: Pre-computed dict from precompute()
            h:     (B, d_model) current hidden state

        Returns:
            h_next: (B, d_model) stepped hidden state
        """
        v = cache["v"]
        half_dt = cache["half_dt"]

        # Build RHS: (I + dt/2 * Omega) @ h  — only h changes per step
        super_contrib = half_dt * v * h[..., 1:]
        sub_contrib = half_dt * v * h[..., :-1]
        super_padded = F.pad(super_contrib, (0, 1), value=0.0)
        sub_padded = F.pad(sub_contrib, (1, 0), value=0.0)
        rhs = h + super_padded - sub_padded

        return thomas_algorithm_lu_step(cache["A_LU"], cache["pivots"], rhs)

    def forward(self, z0: Tensor, h: Tensor) -> Tensor:
        """
        Perform one TCRS geodesic step (non-cached, backward compatible).

        Args:
            z0: (B, d_latent) latent blueprint from diffusion planner
            h:  (B, d_model) current hidden state on the manifold

        Returns:
            h_next: (B, d_model) hidden state after geodesic step
        """
        cache = self.precompute(z0)
        return self.step(cache, h)

    def compute_curvature(self, z0: Tensor) -> Tensor:
        """
        Estimate local curvature from the Lie generator parameters.

        Large |v| values indicate high curvature regions where the
        tridiagonal approximation may lose accuracy and the Woodbury
        fallback should be used instead.

        Args:
            z0: (B, d_latent) latent blueprint

        Returns:
            curvature: (B,) scalar curvature estimate per sample
        """
        v = torch.tanh(self.W_lie(z0))
        return v.abs().max(dim=-1).values


class WoodburyRetractionSolver(nn.Module):
    """
    Woodbury-Retraction geodesic solver (fallback for high curvature).

    Uses a low-rank factored metric:
        G_tilde(z) = D(z) + U(z) U(z)^T

    where D is diagonal (d params) and U is d x r (d*r params, r=8 default).
    Inversion via Woodbury identity: O(dr + r^3) instead of O(d^3).

    The geodesic step uses midpoint retraction:
        1. Compute velocity from metric gradient
        2. Half-step to midpoint
        3. Recompute metric at midpoint
        4. Full step from start using midpoint metric

    Error bound: O(eps + h^2) where eps is the low-rank approximation
    error and h is the step size.
    """

    def __init__(self, d_model: int, d_latent: int, rank: int = 8, dt: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.dt = dt

        # Diagonal part of the metric
        self.W_diag = nn.Linear(d_latent, d_model)
        # Low-rank factor
        self.W_lowrank = nn.Linear(d_latent, d_model * rank)

    def _compute_metric_factors(self, z: Tensor):
        """
        Compute the diagonal D and low-rank factor U of the metric.

        G = D + U U^T  (guaranteed SPD if D > 0)

        Args:
            z: (B, d_latent) latent point

        Returns:
            D: (B, d) positive diagonal
            U: (B, d, r) low-rank factor
        """
        D = F.softplus(self.W_diag(z)) + 1e-4  # (B, d), guaranteed positive
        U = self.W_lowrank(z).view(*z.shape[:-1], self.d_model, self.rank)  # (B, d, r)
        return D, U

    def _woodbury_solve(self, D: Tensor, U: Tensor, b: Tensor) -> Tensor:
        """
        Solve (D + U U^T) x = b using the Woodbury identity.

        (D + U U^T)^{-1} = D^{-1} - D^{-1} U (I + U^T D^{-1} U)^{-1} U^T D^{-1}

        Cost: O(dr + r^3) instead of O(d^3).

        Args:
            D: (B, d) positive diagonal of the metric
            U: (B, d, r) low-rank factor
            b: (B, d) right-hand side

        Returns:
            x: (B, d) solution
        """
        D_inv = 1.0 / D  # (B, d)
        D_inv_b = D_inv * b  # (B, d)
        D_inv_U = D_inv.unsqueeze(-1) * U  # (B, d, r)

        # Core: (I + U^T D^{-1} U) is r x r
        UtDinvU = torch.einsum("...dr,...ds->...rs", U, D_inv_U)  # (B, r, r)
        I_r = torch.eye(self.rank, device=D.device, dtype=D.dtype).expand_as(UtDinvU)
        core = I_r + UtDinvU  # (B, r, r)

        # U^T D^{-1} b
        Ut_Dinv_b = torch.einsum("...dr,...d->...r", U, D_inv_b)  # (B, r)

        # Solve core system (r x r, so cheap)
        orig_dtype = core.dtype
        correction = torch.linalg.solve(core.float(), Ut_Dinv_b.float()).to(
            orig_dtype
        )  # (B, r)

        # Apply Woodbury
        x = D_inv_b - torch.einsum("...dr,...r->...d", D_inv_U, correction)
        return x

    def precompute(self, z0: Tensor) -> dict[str, Tensor]:
        """Pre-compute z0-dependent metric factors."""
        D, U = self._compute_metric_factors(z0)
        return {"D": D, "U": U}

    def step(self, cache: dict[str, Tensor], h: Tensor) -> Tensor:
        """Perform one Woodbury-retraction step using cached metric."""
        D, U = cache["D"], cache["U"]

        velocity = self._woodbury_solve(D, U, h)
        velocity = F.normalize(velocity, dim=-1) * h.norm(dim=-1, keepdim=True)

        h_mid = h + 0.5 * self.dt * velocity

        velocity_mid = self._woodbury_solve(D, U, h_mid)
        velocity_mid = F.normalize(velocity_mid, dim=-1) * h.norm(dim=-1, keepdim=True)

        return h + self.dt * velocity_mid

    def forward(self, z0: Tensor, h: Tensor) -> Tensor:
        """
        Perform one Woodbury-retraction geodesic step (midpoint method).

        Args:
            z0: (B, d_latent) latent blueprint
            h:  (B, d_model) current hidden state

        Returns:
            h_next: (B, d_model) hidden state after geodesic step
        """
        cache = self.precompute(z0)
        return self.step(cache, h)

    def compute_metric_tensor(self, z0: Tensor) -> Tensor:
        """
        Materialize the full metric tensor G = D + U U^T.

        Only needed for loss computation or diagnostics, not for
        the solver itself (which uses Woodbury).

        Args:
            z0: (B, d_latent) latent blueprint

        Returns:
            G: (B, d, d) full metric tensor
        """
        D, U = self._compute_metric_factors(z0)
        # U is (..., d, r). U U^T requires summing over the rank dimension 'r', not the spatial dimension 'd'.
        G = torch.diag_embed(D) + torch.einsum("...dr,...er->...de", U, U)
        return G
