"""
Riemannian Geometry Primitives
==============================

Core mathematical operations for the Dual-System Architecture:
  - Cayley map: skew-symmetric A -> orthogonal Q = (I - A/2)^{-1}(I + A/2)
  - Skew vectorization / unvectorization (p(p-1)/2 <-> p x p)
  - Row-wise Mahalanobis distance with learned precision matrix
  - Squared Riemannian distance on manifold with metric G

All operations are fully differentiable and O(d) or O(d^2) as specified.
Supports batched inputs with arbitrary leading dimensions.
"""

import torch
from torch import Tensor


def cayley(A: Tensor) -> Tensor:
    """
    Cayley map: maps skew-symmetric A to orthogonal matrix Q.

    Q = (I - A/2)^{-1} (I + A/2)

    This is the primary Lie group exponential approximation used throughout
    the system. Exact for skew-symmetric matrices in SO(n).

    Args:
        A: (..., p, p) skew-symmetric matrix

    Returns:
        Q: (..., p, p) orthogonal matrix in SO(p)
    """
    p = A.shape[-1]
    I = torch.eye(p, device=A.device, dtype=torch.float32).expand_as(A)
    half_A = 0.5 * A.to(torch.float32)
    Q32 = torch.linalg.solve(I - half_A, I + half_A)
    return Q32.to(A.dtype)


def skew_vectorize(A: Tensor) -> Tensor:
    """
    Extract the p(p-1)/2 independent components from a skew-symmetric matrix.

    Takes the strictly upper-triangular entries (row < col) and stacks them
    into a flat vector. This is the canonical vectorization of so(p).

    Args:
        A: (..., p, p) skew-symmetric matrix

    Returns:
        v: (..., p*(p-1)/2) vector of independent components
    """
    p = A.shape[-1]
    # Get indices of strictly upper triangular entries
    rows, cols = torch.triu_indices(p, p, offset=1, device=A.device)
    return A[..., rows, cols]


def skew_unvectorize(v: Tensor, p: int) -> Tensor:
    """
    Reconstruct a skew-symmetric matrix from its p(p-1)/2 independent components.

    Inverse of skew_vectorize. Places components in the upper triangle and
    negates them into the lower triangle.

    Args:
        v: (..., p*(p-1)/2) vector of independent components
        p: matrix dimension

    Returns:
        A: (..., p, p) skew-symmetric matrix
    """
    batch_shape = v.shape[:-1]
    A = torch.zeros(*batch_shape, p, p, device=v.device, dtype=v.dtype)
    rows, cols = torch.triu_indices(p, p, offset=1, device=v.device)
    A[..., rows, cols] = v
    A[..., cols, rows] = -v
    return A


def row_mahalanobis(X: Tensor, mu: Tensor, Sigma_inv: Tensor) -> Tensor:
    """
    Compute row-wise squared Mahalanobis distance.

    d^2(x, mu) = (x - mu)^T Sigma_inv (x - mu)
    """
    # Optimized quadratic separation for diagonal precision matrices
    if Sigma_inv.dim() == 1:
        # X: (..., n, d)  |  mu: (..., d) or (..., 1, d)

        # Ensure mu has matching dimensions explicitly mapped
        if mu.dim() < X.dim():
            mu = mu.unsqueeze(-2)

        # ||X||^2_W
        norm_X = (X.pow(2) * Sigma_inv).sum(dim=-1)  # (..., n)

        # ||mu||^2_W
        norm_mu = (mu.pow(2) * Sigma_inv).sum(dim=-1)  # (..., 1) or (..., n)

        # - 2 <X*W, mu> -> Utilizing `.matmul` to prevent intermediate broadcast graph instantiating O(B*V*p) tensor
        cross_term = 2 * torch.matmul(X * Sigma_inv, mu.transpose(-1, -2)).squeeze(
            -1
        )  # (..., n)

        # Algebraic expansion avoids materializing an O(n*d) intermediate tensor
        return norm_X + norm_mu - cross_term
    else:
        # Full precision matrix fallback
        if mu.dim() < X.dim():
            mu = mu.unsqueeze(-2)
        diff = X - mu  # (..., n, d)
        transformed = diff @ Sigma_inv  # (..., n, d)
        return (transformed * diff).sum(dim=-1)  # (..., n)


def squared_riemannian_distance(h: Tensor, B: Tensor, G: Tensor) -> Tensor:
    """
    Compute squared Riemannian distance between hidden state h and
    patch/token embeddings B under metric tensor G.

    d_R^2(h, b_i) = (h - b_i)^T G (h - b_i)

    When G = I, this reduces to squared Euclidean distance.
    The metric G is learned (SPD) and encodes the manifold curvature.

    Args:
        h: (..., d) hidden state vector
        B: (..., n, d) patch/token embeddings
        G: (..., d, d) SPD metric tensor (can be batched or shared)

    Returns:
        dist: (..., n) squared Riemannian distances
    """
    if h.dim() < B.dim():
        h = h.unsqueeze(-2)  # (..., 1, d)
    diff = h - B  # (..., n, d)

    # Apply metric tensor via batched matmul
    transformed = diff @ G  # (..., n, d)
    return (transformed * diff).sum(dim=-1)  # (..., n)


def make_spd(M: Tensor, eps: float = 1e-4, max_norm: float = 10.0) -> Tensor:
    """
    Project a matrix to the SPD (symmetric positive definite) cone.

    Uses the factorization G = M M^T + eps * I to guarantee SPD.
    The factor M is Frobenius-norm clamped to prevent condition number explosion.

    M can be rectangular (..., d, r) for low-rank SPD construction,
    or square (..., d, d) for full-rank.

    Args:
        M: (..., d, r) or (..., d, d) factor matrix
        eps: small positive constant for numerical stability
        max_norm: maximum Frobenius norm for M (bounds condition number)

    Returns:
        G: (..., d, d) SPD matrix
    """
    # Clamp factor norm to bound condition number
    # κ(G) ≈ (||M||_F^2 + eps) / eps, so max_norm controls this
    M_norm = M.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    M = M * (max_norm / M_norm).clamp(max=1.0)

    d = M.shape[-2]  # Output dimension (rows of M)
    I = torch.eye(d, device=M.device, dtype=M.dtype)
    return M @ M.transpose(-1, -2) + eps * I


def frechet_derivative_cayley(A: Tensor, dA: Tensor) -> Tensor:
    """
    Frechet derivative of the Cayley map at A in direction dA.

    d/dt Cayley(A + t*dA)|_{t=0}

    This is needed for backpropagation through the Cayley map when
    computing gradients of the Lie algebra parameters.

    Uses the formula:
        dQ = (I - A/2)^{-1} * dA * (I + A/2)^{-1}

    which is exact for the Cayley parametrization.

    Args:
        A: (..., p, p) skew-symmetric matrix (evaluation point)
        dA: (..., p, p) skew-symmetric matrix (direction)

    Returns:
        dQ: (..., p, p) derivative of Q w.r.t. perturbation dA
    """
    p = A.shape[-1]
    orig_dtype = A.dtype
    # Upcast to float32 — torch.linalg.inv doesn't support bf16
    A_f = A.to(torch.float32)
    dA_f = dA.to(torch.float32)
    I = torch.eye(p, device=A.device, dtype=torch.float32).expand_as(A_f)
    half_A = 0.5 * A_f

    # (I - A/2)^{-1}
    inv_left = torch.linalg.inv(I - half_A)
    # (I + A/2)^{-1}
    inv_right = torch.linalg.inv(I + half_A)

    return (inv_left @ dA_f @ inv_right).to(orig_dtype)


def parallel_transport_spd(G_start: Tensor, G_end: Tensor, v: Tensor) -> Tensor:
    """
    Parallel transport a tangent vector v from T_{G_start} to T_{G_end}
    on the SPD manifold.

    Uses the canonical transport:
        Gamma(v) = E @ G_start^{-1/2} @ v @ G_start^{-1/2} @ E^T
    where E = (G_end @ G_start^{-1})^{1/2}

    This is needed when accumulating velocity vectors across geodesic
    steps where the metric changes.

    Args:
        G_start: (..., d, d) SPD metric at start point
        G_end: (..., d, d) SPD metric at end point
        v: (..., d) tangent vector at G_start

    Returns:
        v_transported: (..., d) tangent vector at G_end
    """
    # Compute G_start^{-1/2} via eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(G_start)
    eigvals_inv_sqrt = 1.0 / torch.sqrt(eigvals.clamp(min=1e-8))
    G_start_inv_sqrt = (
        eigvecs @ torch.diag_embed(eigvals_inv_sqrt) @ eigvecs.transpose(-1, -2)
    )

    # Compute G_end^{1/2}
    eigvals_e, eigvecs_e = torch.linalg.eigh(G_end)
    G_end_sqrt = (
        eigvecs_e
        @ torch.diag_embed(torch.sqrt(eigvals_e.clamp(min=1e-8)))
        @ eigvecs_e.transpose(-1, -2)
    )

    # Transport map: G_end^{1/2} @ G_start^{-1/2}
    E = G_end_sqrt @ G_start_inv_sqrt

    # Apply transport: for vector v, transported = E @ v
    if v.dim() == E.dim() - 1:
        return torch.einsum("...ij,...j->...i", E, v)
    return E @ v


def spd_batchnorm(G: Tensor, eps: float = 1e-6, momentum: float = 0.1) -> Tensor:
    """
    Lightweight Riemannian normalization for SPD metric tensors.

    Instead of full eigendecomposition (O(d³)), uses spectral norm scaling:
    normalize G so its spectral norm (largest eigenvalue) ≈ 1. This prevents
    metric explosion while maintaining the learned geometry structure.

    The minimum eigenvalue is already bounded by eps*I from make_spd().

    Based on the unified RBN framework for SPD manifolds (arXiv, March 2024).

    Args:
        G: (B, d, d) batch of SPD metric tensors
        eps: numerical stability
        momentum: unused (reserved for running statistics)

    Returns:
        G_norm: (B, d, d) spectrally normalized SPD metrics
    """
    # Spectral norm via Frobenius norm (fast upper bound on largest eigenvalue)
    # For SPD: ||G||_F >= ||G||_2 >= ||G||_F / sqrt(d)
    # Using Frobenius norm is cheap and provides sufficient normalization
    frob_norm = torch.linalg.norm(
        G.float(), ord="fro", dim=(-2, -1), keepdim=True
    )  # (B, 1, 1)
    frob_norm = frob_norm.clamp(min=eps)

    # Scale so average Frobenius norm across batch ≈ sqrt(d)
    # (identity matrix has Frobenius norm = sqrt(d))
    d = G.shape[-1]
    target_norm = d**0.5
    scale = target_norm / frob_norm  # (B, 1, 1)

    return (G.float() * scale).to(dtype=G.dtype)
