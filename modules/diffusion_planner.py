"""
LaDiR-style Latent Diffusion Planner
=====================================

Produces continuous blueprint z_0 in R^d from input x (and optionally target y
during training) via a variational diffusion process.

Architecture:
  - Encoder: q_phi(z_{0:K} | x, y) — encodes input+target into latent trajectory
  - Prior:   p_theta(z_{k-1} | z_k, x) — learned denoising prior
  - Decoder: p_omega(y | x, z_0) — reconstructs target from blueprint

The ELBO objective decomposes into:
  - Reconstruction: -log p_omega(y | x, z_0)
  - KL chain: sum_k KL(q(z_{k-1}|z_k,z_0,x,y) || p(z_{k-1}|z_k,x))

During inference, z_0 is obtained by iteratively denoising from z_K ~ N(0,I)
through the learned prior p_theta.

This is a simplified but fully functional implementation. The denoising
network uses a small MLP conditioned on the diffusion timestep.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
    """
    Cosine noise schedule from Nichol & Dhariwal (2021).
    Produces smoother beta values than the linear schedule, especially
    at the start and end of the diffusion process.

    Args:
        timesteps: Number of diffusion steps K
        s: Small offset to prevent beta from being too small at t=0

    Returns:
        betas: (K,) noise schedule
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> Tensor:
    """Standard linear noise schedule."""
    return torch.linspace(beta_start, beta_end, timesteps)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) integer timesteps

        Returns:
            emb: (B, dim) sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DenoisingNetwork(nn.Module):
    """
    Denoising network with adaptive layer normalization (adaLN).

    Instead of concatenating time/context, the conditioning signal modulates
    each hidden layer via learned scale (γ) and shift (β) parameters.
    This is the DiT (Scalable Diffusion Models with Transformers) pattern.
    """

    def __init__(self, d_latent: int, d_context: int, d_hidden: int = 1024):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(d_latent)
        self.context_proj = nn.Linear(d_context, d_latent)

        # Conditioning path: time + context → adaLN parameters (scale + shift per layer)
        n_hidden_layers = 3
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_latent * 2, d_hidden),
            nn.SiLU(),
        )
        # Per-layer scale and shift projections
        self.adaLN_projs = nn.ModuleList(
            [
                nn.Linear(d_hidden, d_hidden * 2)  # γ, β for each layer
                for _ in range(n_hidden_layers)
            ]
        )

        # Main network with adaLN modulation
        self.input_proj = nn.Linear(d_latent, d_hidden)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(d_hidden, d_hidden) for _ in range(n_hidden_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.RMSNorm(d_hidden) for _ in range(n_hidden_layers)]
        )
        self.output_proj = nn.Linear(d_hidden, d_latent)

    def forward(self, z_noisy: Tensor, t: Tensor, x_context: Tensor) -> Tensor:
        """
        Predict noise using adaLN conditioning.

        Args:
            z_noisy:   (B, d_latent) noisy latent
            t:         (B,) integer timesteps
            x_context: (B, d_context) conditioning information

        Returns:
            eps_pred: (B, d_latent) predicted noise
        """
        t_emb = self.time_emb(t)  # (B, d_latent)
        x_proj = self.context_proj(x_context)  # (B, d_latent)

        # Conditioning signal
        cond = self.cond_mlp(torch.cat([t_emb, x_proj], dim=-1))  # (B, d_hidden)

        # Main path with adaLN modulation
        h = self.input_proj(z_noisy)  # (B, d_hidden)
        for linear, norm, adaLN_proj in zip(
            self.hidden_layers, self.norms, self.adaLN_projs
        ):
            res = h  # Residual connection standard for DiT blocks

            # adaLN: norm(h) * (1 + γ) + β
            scale_shift = adaLN_proj(cond)  # (B, d_hidden * 2)
            gamma, beta = scale_shift.chunk(2, dim=-1)  # (B, d_hidden) each

            h = norm(h)
            h = h * (1 + gamma) + beta  # Adaptive modulation
            h = F.silu(linear(h))

            h = h + res  # Apply residual

        return self.output_proj(h)


class LatentEncoder(nn.Module):
    """
    Variational encoder: q_phi(z_0 | x, y).

    Maps input x and target y to the parameters of a diagonal Gaussian
    posterior over the latent blueprint z_0.
    """

    def __init__(self, d_input: int, d_latent: int, d_hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.RMSNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(d_hidden, d_latent)
        self.logvar_head = nn.Linear(d_hidden, d_latent)

    def forward(self, x: Tensor, y: Tensor = None) -> tuple[Tensor, Tensor]:
        """
        Encode input (+ optional target) to latent Gaussian parameters.

        Args:
            x: (B, d_input) input features
            y: (B, d_input) target features (used during training)

        Returns:
            mu:     (B, d_latent) posterior mean
            logvar: (B, d_latent) log-variance
        """
        if y is not None:
            inp = x + y  # Simple fusion; could use cross-attention for larger models
        else:
            inp = x
        h = self.net(inp)
        return self.mu_head(h), self.logvar_head(h)


class DiffusionPlanner(nn.Module):
    """
    LaDiR-style Latent Diffusion Planner.

    Training:
      1. Encode (x, y) -> q(z_0) via LatentEncoder
      2. Sample z_0 ~ q(z_0)
      3. Forward diffuse z_0 -> z_k at random timestep k
      4. Predict noise via DenoisingNetwork
      5. ELBO = reconstruction + KL

    Inference:
      1. Sample z_K ~ N(0, I)
      2. Iteratively denoise: z_{k-1} = denoise(z_k, k, x)
      3. Return z_0 as the continuous blueprint

    Args:
        d_input:   Dimension of input features
        d_latent:  Dimension of latent space
        n_steps:   Number of diffusion steps K
        beta_start: Starting noise level
        beta_end:   Ending noise level
        schedule:  "linear" or "cosine"
    """

    def __init__(
        self,
        d_input: int,
        d_latent: int,
        n_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.d_latent = d_latent
        self.n_steps = n_steps

        # Noise schedule
        if schedule == "cosine":
            betas = cosine_beta_schedule(n_steps)
        else:
            betas = linear_beta_schedule(n_steps, beta_start, beta_end)

        # Pre-compute diffusion constants (registered as buffers for device transfer)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # Posterior variance for q(z_{k-1} | z_k, z_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

        # Posterior mean coefficients
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        # Min-SNR-γ weighting (arXiv 2303.09556) — 3.4x faster convergence
        # SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
        snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)
        self.register_buffer("snr", snr)
        # For noise prediction: w_t = min(SNR(t), γ) / SNR(t) = min(γ/SNR(t), 1)
        gamma = 5.0  # Recommended by the paper
        min_snr_weights = torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp(
            min=1e-8
        )
        self.register_buffer("min_snr_weights", min_snr_weights)

        # Networks
        self.encoder = LatentEncoder(d_input, d_latent)
        self.denoiser = DenoisingNetwork(d_latent, d_input)

    def _extract(self, schedule: Tensor, t: Tensor, shape: tuple) -> Tensor:
        """Extract values from a 1D schedule at timesteps t, broadcast to shape."""
        batch_size = t.shape[0]
        out = schedule.gather(0, t)
        return out.view(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, z0: Tensor, t: Tensor, noise: Tensor = None) -> Tensor:
        """
        Forward diffusion: sample z_k from q(z_k | z_0).

        z_k = sqrt(alpha_bar_k) * z_0 + sqrt(1 - alpha_bar_k) * epsilon

        Args:
            z0: (B, d_latent) clean latent
            t:  (B,) integer timesteps
            noise: (B, d_latent) optional pre-sampled noise

        Returns:
            z_noisy: (B, d_latent) noisy latent at timestep t
        """
        if noise is None:
            noise = torch.randn_like(z0)

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, z0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, z0.shape)

        return sqrt_alpha_bar * z0 + sqrt_one_minus * noise

    def compute_elbo_loss(self, x: Tensor, y: Tensor = None) -> tuple[Tensor, Tensor]:
        """
        Compute the ELBO training loss.

        1. Encode to get q(z_0 | x, y)
        2. Sample z_0 via reparameterization
        3. Sample random timestep k
        4. Forward diffuse to z_k
        5. Predict noise
        6. Loss = MSE(eps_pred, eps_true) + KL(q || N(0,I))

        Args:
            x: (B, d_input) input features
            y: (B, d_input) target features (optional, for training)

        Returns:
            z0: (B, d_latent) sampled latent blueprint
            loss_elbo: scalar ELBO loss
        """
        B = x.shape[0]
        device = x.device

        # Encode
        mu, logvar = self.encoder(x, y)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps_z = torch.randn_like(std)
        z0 = mu + std * eps_z

        # KL divergence: KL(q(z_0|x,y) || N(0,I)) with free bits (Kingma 2016)
        # Free bits prevents posterior collapse by clamping per-dim KL to min λ
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, d_latent)
        # NOTE: free_bits=0.5 with d_latent=256 → min KL = 128.0, which is where
        # ELBO stalled (128.6) across 3 H100 training runs. Lowered to 0.05 to give
        # the encoder 10x more dynamic range (new floor = 256*0.05 = 12.8 nats).
        free_bits = 0.05  # Min nats per dimension (was 0.5, caused floor at 128)
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        kl_loss = kl_per_dim.sum(dim=-1).mean()

        # Random timestep
        t = torch.randint(0, self.n_steps, (B,), device=device, dtype=torch.long)

        # Forward diffuse
        noise = torch.randn_like(z0)
        z_noisy = self.q_sample(z0, t, noise)

        # Classifier-free guidance: randomly drop conditioning during training
        # This trains both conditional p(z|x) and unconditional p(z) models
        cfg_drop_rate = 0.1  # 10% of the time, train unconditionally
        if self.training and torch.rand(1).item() < cfg_drop_rate:
            x_cond = torch.zeros_like(x)  # Drop conditioning
        else:
            x_cond = x

        # Predict noise
        noise_pred = self.denoiser(z_noisy, t, x_cond)

        # Reconstruction loss with Min-SNR-γ weighting (arXiv 2303.09556)
        # Per-sample MSE weighted by min(SNR(t), γ) / SNR(t)
        per_sample_mse = (noise_pred - noise).pow(2).mean(dim=-1)  # (B,)
        snr_weights = self.min_snr_weights.gather(0, t)  # (B,)
        recon_loss = (snr_weights * per_sample_mse).mean()

        loss_elbo = recon_loss + kl_loss

        return {
            "z": z0,
            "elbo_loss": loss_elbo,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def sample(self, x: Tensor, n_steps: int = None, cfg_scale: float = 1.0) -> Tensor:
        """
        Generate z_0 by iterative denoising with classifier-free guidance.

        CFG interpolates between conditional and unconditional noise predictions:
          eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        Starting from z_K ~ N(0,I), iteratively apply the learned prior.

        Args:
            x: (B, d_input) conditioning input
            n_steps: number of denoising steps (default: self.n_steps)
            cfg_scale: classifier-free guidance scale (1.0 = no guidance,
                       3.0-7.0 = strong guidance, higher = sharper samples)

        Returns:
            z0: (B, d_latent) denoised latent blueprint
        """
        if n_steps is None:
            n_steps = self.n_steps

        B = x.shape[0]
        device = x.device

        # Start from pure noise
        z = torch.randn(B, self.d_latent, device=device)

        for k in reversed(range(n_steps)):
            t = torch.full((B,), k, device=device, dtype=torch.long)

            if cfg_scale != 1.0:
                # Classifier-free guidance: batched forward pass for parallelism
                z_both = torch.cat([z, z], dim=0)
                t_both = torch.cat([t, t], dim=0)
                x_both = torch.cat([x, torch.zeros_like(x)], dim=0)
                eps_both = self.denoiser(z_both, t_both, x_both)
                eps_cond, eps_uncond = eps_both.chunk(2, dim=0)
                eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = self.denoiser(z, t, x)

            # Predict z_0 from z_k and eps_pred
            sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, z.shape)
            sqrt_one_minus = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, z.shape
            )
            z0_pred = (z - sqrt_one_minus * eps_pred) / sqrt_alpha_bar

            # Compute posterior mean
            coef1 = self._extract(self.posterior_mean_coef1, t, z.shape)
            coef2 = self._extract(self.posterior_mean_coef2, t, z.shape)
            posterior_mean = coef1 * z0_pred + coef2 * z

            if k > 0:
                noise = torch.randn_like(z)
                posterior_var = self._extract(self.posterior_variance, t, z.shape)
                z = posterior_mean + torch.sqrt(posterior_var) * noise
            else:
                z = posterior_mean

        return z

    @torch.no_grad()
    def sample_ddim(
        self,
        x: Tensor,
        n_steps: int = 50,
        cfg_scale: float = 1.0,
        eta: float = 0.0,
    ) -> Tensor:
        """
        DDIM sampling (arXiv 2010.02502) — deterministic, step-skipping.

        Unlike DDPM which requires all K steps, DDIM can skip steps
        by using a subsequence of timesteps. This enables 20-100x
        faster inference with minimal quality loss.

        Args:
            x: (B, d_input) conditioning input
            n_steps: number of denoising steps (can be << self.n_steps)
            cfg_scale: classifier-free guidance scale
            eta: stochasticity (0=deterministic DDIM, 1=DDPM-like)

        Returns:
            z0: (B, d_latent) denoised latent blueprint
        """
        B = x.shape[0]
        device = x.device

        # Create evenly-spaced timestep subsequence (pure Python integers to prevent D2H stalls)
        step_indices = torch.linspace(self.n_steps - 1, 0, n_steps).long().tolist()

        z = torch.randn(B, self.d_latent, device=device)

        for i, k in enumerate(step_indices):
            t = torch.full((B,), k, device=device, dtype=torch.long)

            # Noise prediction (with optional CFG)
            if cfg_scale != 1.0:
                z_both = torch.cat([z, z], dim=0)
                t_both = torch.cat([t, t], dim=0)
                x_both = torch.cat([x, torch.zeros_like(x)], dim=0)
                eps_both = self.denoiser(z_both, t_both, x_both)
                eps_cond, eps_uncond = eps_both.chunk(2, dim=0)
                eps_pred = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = self.denoiser(z, t, x)

            # Current and previous alpha_bar
            alpha_bar_t = self._extract(self.alphas_cumprod, t, z.shape)
            if i + 1 < len(step_indices):
                t_prev = torch.full(
                    (B,), step_indices[i + 1], device=device, dtype=torch.long
                )
                alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, z.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)  # t=0 → alpha_bar=1

            # Predict z_0
            z0_pred = (z - (1 - alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()

            # DDIM step
            sigma = (
                eta
                * (
                    (1 - alpha_bar_prev)
                    / (1 - alpha_bar_t)
                    * (1 - alpha_bar_t / alpha_bar_prev)
                ).sqrt()
            )
            dir_pointing = (1 - alpha_bar_prev - sigma**2).sqrt()

            z = alpha_bar_prev.sqrt() * z0_pred + dir_pointing * eps_pred
            if sigma.sum() > 0 and i + 1 < len(step_indices):
                z = z + sigma * torch.randn_like(z)

        return z
