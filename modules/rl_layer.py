"""
Asymmetric RL Layer (v1.2 Addendum)
====================================

Human-Reaction Reinforcement Learning with BOTH positive reinforcement
AND severity-proportional negative reinforcement.

Two fundamental learning signals, just like real humans:

  POSITIVE REINFORCEMENT (r > 0):
    "That response was good — do more of this."
    Linear scaling preserves the reward signal faithfully.
    The model strengthens the geometric paths that led to approval.
    Every token that contributed (via C3 credit) gets its probability
    pushed upward proportionally.

  NEGATIVE REINFORCEMENT (r < 0):
    "That response was bad — the worse it was, the harder the lesson."
    Exponential amplification makes harsh criticism sting
    disproportionately harder. The model warps its manifold AWAY
    from the paths that led to rejection.

Design principle: ALL reactions leave PERMANENT manifold warps.
Positive reactions build confidence on good paths.
Negative reactions burn avoidance into bad paths.
Neither is ever erased — apologies after harsh criticism become
new positive data points, not undo operations. Just like real
human memory, both praise and criticism permanently shape behavior.

Severity function s(r):
  Positive (r >= 0): s(r) = r           (linear, faithful)
  Negative (r < 0):  s(r) = -|r| * exp(|r| / tau_s)  (exponential)

With tau_s = 0.3:
  r = +1.0 → s(r) = +1.00   (full approval: 1× reinforcement)
  r = +0.5 → s(r) = +0.50   (mild approval: proportional reward)
  r = -0.2 → s(r) = -0.39   (mild negative: ~2× linear)
  r = -0.5 → s(r) = -2.65   (moderate: ~5× linear)
  r = -0.8 → s(r) = -11.47  (harsh: ~14× linear)
  r = -1.0 → s(r) = -27.80  (total rejection: ~28× linear)

The RL loss is integrated with C3 credit assignment from the EBM Critic,
keeping per-token attribution precise even under asymmetric scaling.
Both positive and negative paths get credited/blamed at the token level.
"""

import torch
from torch import Tensor


def severity_scale(r: Tensor, tau_s: float = 0.3, max_penalty: float = 30.0) -> Tensor:
    """
    Asymmetric severity function s(r) for human-reaction RL.

    Both sides are active and permanent:
      - Positive r: linear reinforcement — every good reaction rewards
        the model, strengthening paths that produce good outputs.
      - Negative r: exponential penalty — bad reactions punish harder
        the worse they are, rapidly burning avoidance into the manifold.

    s(r) = r                          if r >= 0  (praise → proportional reward)
    s(r) = -|r| * exp(|r| / tau_s)    if r < 0   (criticism → exponential pain)

    Args:
        r: (...,) reaction reward in [-1, +1]
        tau_s: severity temperature (default 0.3)
               Lower = even harsher exponential curve for negatives.
               Higher = gentler curve (approaches linear).
        max_penalty: absolute limit on the negative scalar (default 30.0)

    Returns:
        s: (...,) scaled severity.
           Positive for good reactions (reinforces behavior).
           Strongly negative for bad reactions (punishes behavior).
    """
    positive = r.clamp(min=0.0)  # linear for r >= 0
    negative_mag = (-r).clamp(min=0.0)  # |r| for r < 0
    negative = -negative_mag * torch.exp(
        negative_mag / tau_s
    )  # exponential amplification

    # Cap the explosion so one single bad reaction doesn't permanently scorch the Lie manifold
    negative = negative.clamp(min=-max_penalty)

    return torch.where(r >= 0, positive, negative)


def asymmetric_rl_loss(
    c3_weights: Tensor,
    advantages: Tensor,
    log_probs: Tensor,
    reaction_reward: Tensor,
    tau_s: float = 0.3,
    max_penalty: float = 30.0,
    clip_epsilon: float = 0.2,
    old_log_probs: Tensor = None,
) -> Tensor:
    """
    Asymmetric Reinforcement Learning Loss with C3 credit and PPO clipping.

    L_RL^asym = -E_t[ omega_t^C3 * A_t^C3 * s(r) * log pi(y_t | y_<t, z0) ]

    PPO clipping (when old_log_probs provided) prevents catastrophically
    large policy updates by constraining the ratio pi_new/pi_old to
    [1-epsilon, 1+epsilon].

    This is a REINFORCE-style policy gradient objective that:
      - When s(r) > 0 (good reaction): INCREASES probability of tokens that
        contributed positively (C3 credit > 0) → positive reinforcement.
      - When s(r) < 0 (bad reaction): DECREASES probability of tokens that
        contributed to the bad output, with exponentially amplified magnitude.

    Both directions flow through the same C3 credit mechanism, so per-token
    attribution stays precise regardless of reward sign.

    Args:
        c3_weights:      (B, T) per-token C3 credit weights from EBM Critic.
                         Higher weight = this token had more causal influence.
        advantages:      (B, T) per-token advantage estimates (centered + normalized).
        log_probs:       (B, T) log probabilities of generated tokens under current policy.
        reaction_reward: (B,) scalar reaction reward per sample in [-1, +1].
                         Positive = user liked it. Negative = user didn't.
        tau_s:           severity temperature for the exponential curve.
        clip_epsilon:    PPO clip range (default 0.2).
        old_log_probs:   (B, T) log probs from previous policy (enables PPO clipping).

    Returns:
        rl_loss: scalar loss (to be minimized). Gradient pushes policy
                 toward high-reward outputs and away from low-reward ones.
    """
    # Apply severity scaling to the reaction reward (bounded so one episode can't wipe history)
    scaled_reward = severity_scale(reaction_reward, tau_s, max_penalty)  # (B,)

    # Broadcast reward to token level: (B,) -> (B, 1) -> (B, T)
    scaled_reward_t = scaled_reward.unsqueeze(-1).expand_as(log_probs)

    # Weighted advantage with severity (Must be detached for REINFORCE gradient flow)
    weighted_adv = (c3_weights * advantages * scaled_reward_t).detach()

    if old_log_probs is not None:
        # PPO-style clipping: prevents catastrophic policy updates
        ratio = torch.exp(log_probs - old_log_probs)  # pi_new / pi_old
        clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        # Pessimistic bound: take the worse of clipped and unclipped
        surr1 = weighted_adv * ratio
        surr2 = weighted_adv * clipped_ratio
        per_token = torch.min(surr1, surr2)
    else:
        # Standard REINFORCE (no clipping)
        per_token = weighted_adv * log_probs

    # Negative expectation (we minimize, so negative of the REINFORCE objective)
    return -per_token.mean()


def compute_advantages(
    energy: Tensor,
    c3_credit: Tensor,
) -> Tensor:
    """
    Compute per-token advantages from EBM energy and C3 credit attribution.

    The advantage for token t is how much more (or less) it contributed
    to the overall sequence quality compared to the average token.
    High-advantage tokens were disproportionately responsible for
    the sequence outcome — these are the ones that get the strongest
    reinforcement signal (positive or negative).

    Args:
        energy:     (B,) sequence-level energy from EBM Critic
        c3_credit:  (B, T) per-token C3 credit from EBM Critic
                    (gradient of energy w.r.t. each token embedding)

    Returns:
        advantages: (B, T) normalized per-token advantages
    """
    # Baseline: mean credit across tokens
    baseline = c3_credit.mean(dim=-1, keepdim=True)  # (B, 1)
    advantages = c3_credit - baseline

    # Normalize for stable gradients (prevents exploding updates)
    std = advantages.std(dim=-1, keepdim=True).clamp(min=1e-8)
    return advantages / std
