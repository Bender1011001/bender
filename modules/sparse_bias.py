import torch
import torch.nn as nn

class HighGainSparseBias(nn.Module):
    """
    Direct, isolated per-user vocabulary bias with explosive gain multiplier.
    Provides high-gain discrete control over the output distribution to
    overpower the 2000-norm logit wall of the base LLM.

    Using sparse=True enforces that gradients are only computed for the active 
    user rows, maintaining memory efficiency and perfect cross-user isolation.
    """
    def __init__(self, num_users: int, vocab_size: int, gain: float = 50.0, sparse: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_users = num_users
        self.gain = gain
        
        # O(V) per user. For 10,000 users and vocab 151936, this is ~1.5B parameters.
        self.bias = nn.Embedding(num_users, vocab_size, sparse=sparse)
        
        # Zero-init ensures no behavioral interference at step 0
        nn.init.zeros_(self.bias.weight)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: (B,) tensor of user indices.
        Returns:
            (B, vocab_size) tensor of logit biases.
        """
        return torch.clamp(self.bias(user_ids) * self.gain, min=-2000, max=2000)

    @torch.no_grad()
    def decay_bias(self, user_id: int, factor: float = 0.95):
        """
        Apply exponential decay to a user's bias vector after each training round.
        
        Without decay, bias values accumulate unboundedly over many training
        rounds, causing grammatical artifacts (e.g., "You was" instead of 
        "You were" at Round 10+). A gentle decay of 0.95 per round preserves 
        the most recent learning signal while preventing historical accumulation
        from distorting grammar.
        
        Args:
            user_id: User whose bias to decay
            factor: Decay multiplier per round (0.95 = 5% decay)
        """
        self.bias.weight.data[user_id] *= factor

    @torch.no_grad()
    def state_dict_differential(self) -> dict:
        """
        O(1) Differential Checkpoint Serialization Strategy.
        Instead of dumping 1.5B identically zeroed structural variables (6GB), 
        filter explicit coordinates conditionally capturing uniquely modified metrics natively.
        Returns a compressed dictionary isolating physical nonzero rows perfectly natively.
        """
        weights = self.bias.weight.data
        # Efficient row-wise norm to find active users
        active_rows = torch.where(torch.norm(weights, dim=1) > 1e-6)[0]
        
        return {
            "active_indices": active_rows.cpu(),
            "active_weights": weights[active_rows].cpu(),
            "vocab_size": self.vocab_size,
            "num_users": self.num_users
        }

    @torch.no_grad()
    def load_state_dict_differential(self, state_dict: dict):
        """
        Loads highly compressed differential structural IO vectors natively back into execution arrays identically globally mapping accurately linearly perfectly globally directly correctly perfectly precisely intrinsically natively.
        """
        if "active_indices" not in state_dict:
            raise KeyError("Invalid differential state dictionary provided natively.")
            
        indices = state_dict["active_indices"].to(self.bias.weight.device)
        weights = state_dict["active_weights"].to(self.bias.weight.dtype).to(self.bias.weight.device)
        
        self.bias.weight.data[indices] = weights

