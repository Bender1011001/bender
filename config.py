"""
Configuration for the Dual-System Architecture.

All hyperparameters from the master document §6, plus loss weights from §2.
Designed for 1.3B-7B scale models.
"""

from dataclasses import dataclass


@dataclass
class DualSystemConfig:
    """Complete configuration for the Dual-System Synthetic Cognition model."""

    # -- Model dimensions ---------------------------------------------------
    d_model: int = 4096  # Hidden dimension (decoder width)
    d_ff: int = None  # MLP hidden dimension (defaults to 4 * d_model)
    d_lie: int = 512  # Lie algebra / manifold rank p
    d_lowrank: int = 8  # Woodbury low-rank r for metric factorization
    vocab_size: int = 32000  # Tokenizer vocabulary size
    n_layers: int = 32  # Number of AR decoder layers
    n_heads: int = 32  # Attention heads in decoder
    n_kv_heads: int = None  # Grouped Query Attention KV heads (defaults to n_heads)
    max_seq_len: int = 2048  # Maximum sequence length T

    # -- Diffusion planner --------------------------------------------------
    diffusion_steps: int = 1000  # K denoising steps
    diffusion_beta_start: float = 1e-4
    diffusion_beta_end: float = 0.02
    d_latent: int = 512  # Latent z_0 dimension (defaults to d_lie)

    # -- TCRS geodesic solver -----------------------------------------------
    tcrs_dt: float = 0.01  # delta-tau integration step
    curvature_threshold: float = 10.0  # Switch to Woodbury when curvature > this

    # -- LiSMCA bridge ------------------------------------------------------
    lismca_topk: int = 8  # k for dynamic soft-mask top-k

    # -- EBM critic ---------------------------------------------------------
    ebm_hidden: int = 1024  # Hidden dim of energy network
    ebm_n_layers: int = 3  # Depth of energy network

    # -- Loss weights (document section 2) ----------------------------------
    lambda_elbo: float = 1.0
    lambda_geo: float = 0.5
    lambda_free: float = 1.0
    tau: float = 0.1  # Entropy temperature in free-energy term
    lambda_acc: float = 0.1  # Acceleration penalty in geodesic loss

    # -- RL Layer v1.2: Asymmetric Severity Scaling (addendum) ---------------
    lambda_rl: float = 0.3  # Weight for asymmetric RL loss
    tau_s: float = 0.3  # Severity temperature (higher = harsher negative penalty)

    # -- v2.0: Principal Fiber Bundle (multi-user topology) -----------------
    num_users: int = 1024  # Max concurrent user fibers
    d_fiber: int = None  # Fiber dimension (defaults to d_lie)
    ebm_energy_threshold: float = 5.0  # E_threshold for epistemic routing κ
    tau_e: float = 0.5  # Temperature for epistemic κ sigmoid
    eta_base: float = 0.01  # Learning rate for base manifold updates (factual)
    eta_user: float = 0.05  # Learning rate for user fiber updates (stylistic)
    bch_order: int = 2  # BCH expansion order (2 = includes Lie bracket)

    # -- Training -----------------------------------------------------------
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 256
    gradient_accumulation_steps: int = (
        1  # Extends effective batch scaling preventing OOM loops natively
    )
    warmup_fraction: float = 0.1  # Fraction of steps for pure-ELBO warm-up
    max_epochs: int = 100
    grad_clip: float = 1.0
    seed: int = 42

    # -- Scheduler ----------------------------------------------------------
    scheduler: str = "cosine"  # "cosine" | "linear"
    min_lr: float = 1e-6

    # -- Infrastructure -----------------------------------------------------
    compile: bool = True  # torch.compile the model
    fsdp: bool = False  # Fully Sharded Data Parallel
    bf16: bool = True  # bfloat16 mixed precision
    checkpoint_dir: str = "checkpoints_dual"
    log_interval: int = 50

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_latent is None:
            self.d_latent = self.d_lie
        if self.d_fiber is None:
            self.d_fiber = self.d_lie

    def resolve_tokenizer_id(self) -> str:
        """Resolve the correct HuggingFace tokenizer ID based on config.

        Returns the tokenizer_id attribute if set, otherwise infers from
        vocab_size to prevent embedding index-out-of-range crashes.
        """
        explicit = getattr(self, "tokenizer_id", None)
        if explicit:
            return explicit
        if self.vocab_size == 49152:
            return "HuggingFaceTB/SmolLM-135M-Instruct"
        return "HuggingFaceTB/SmolLM3-3B"
