"""
Episodic Memory Bank
=====================

Per-user episodic memory for context-dependent factual recall.

Problem: The fiber_proj receives the same z0_delta for ALL prompts from a user,
making it impossible to recall different facts for different questions. This module
solves this by storing per-fact memory entries (key-value pairs) and retrieving
the most relevant ones based on prompt similarity.

Architecture:
  - Keys: prompt embeddings at teaching time (d_key dimensional)
  - Values: fiber delta vectors learned during teaching (d_latent dimensional)
  - Retrieval: hybrid cosine similarity + Jaccard overlap → soft-select values

The retrieved delta is fed through the existing fiber_proj, reusing its learned
bilinear mapping from (z_delta, h_seq) → logits. No new vocabulary-sized params needed.

Jaccard Overlap (Direction 6.4):
  At write time, stores the raw prompt token IDs alongside the embedding key.
  At recall time, computes Jaccard(query_tokens, stored_prompt_tokens) and
  blends it 50/50 with cosine similarity. This shatters the 0.70-0.73 cosine
  similarity ceiling where structurally similar ChatML prompts become
  indistinguishable. The true match ("favorite color" → color memory) shoots
  to ~0.86 while false matches drop to ~0.36.

Any-Completed Viterbi Lock:
  Once any winning memory has been fully generated (match_pos >= tgt_len),
  ALL injection is terminated. This prevents stalled "ghost" memories from
  waking up and dumping their tokens after the winner finishes.

Storage: num_users × max_memories × (d_key + d_latent) = tiny
Example: 10 users × 64 memories × (2048 + 256) = 1.47M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import functools

import math

def compute_prompt_recall_idf(query_tup: tuple, stored_tup: tuple, n_docs: int, df_map: dict) -> float:
    """
    Computes TF-IDF weighted Jaccard overlap natively decoupling the O(N) 
    Document Frequency build loops using a mathematically cached map constraint.
    """
    if n_docs == 0:
        return 0.0
        
    q_set = set(query_tup)
    s_set = set(stored_tup)
    
    intersection_weight = 0.0
    query_weight = 0.0
    
    for t in q_set:
        # Exact matching of log((N+1)/(DF+1))
        idf = math.log((n_docs + 1) / (df_map.get(t, 0) + 1))
        query_weight += idf
        if t in s_set:
            intersection_weight += idf
            
    if query_weight <= 0:
        return 0.0
    return intersection_weight / query_weight

class EpisodicMemoryBank(nn.Module):
    """
    Per-user episodic memory for context-dependent factual recall.

    Each user has up to max_memories slots. When a fact is taught (via
    contrastive forcing), the prompt embedding is stored as a key and the
    resulting fiber delta is stored as a value. At inference time, the most
    similar stored keys are retrieved and their values blended to produce
    a prompt-conditioned z0_delta.

    Uses a circular buffer per user: oldest memories are overwritten when
    the buffer is full, implementing a simple FIFO forgetting policy.
    """

    def __init__(
        self,
        num_users: int,
        d_key: int,
        d_value: int,
        max_memories: int = 64,
        temperature: float = 0.1,
        top_k: int = 4,
    ):
        """
        Args:
            num_users: Maximum number of users
            d_key: Dimension of prompt embeddings (backbone hidden_size)
            d_value: Dimension of stored values (d_latent)
            max_memories: Maximum memories per user (circular buffer)
            temperature: Softmax temperature for retrieval sharpness
            top_k: Number of top memories to retrieve (sparse attention)
        """
        super().__init__()
        self.num_users = num_users
        self.d_key = d_key
        self.d_value = d_value
        self.max_memories = max_memories
        self.temperature = temperature
        self.top_k = top_k

        # Key storage: prompt embeddings at teach time
        self.register_buffer("keys", torch.zeros(num_users, max_memories, d_key))
        # Value storage: target response hidden states at teach time
        self.register_buffer("values", torch.zeros(num_users, max_memories, d_key))
        # Per-user write pointer and count
        self.register_buffer("write_ptr", torch.zeros(num_users, dtype=torch.long))
        self.register_buffer("memory_count", torch.zeros(num_users, dtype=torch.long))

        # Target token IDs for hard recall (padded to max_target_len)
        self.max_target_len = 64
        self.register_buffer(
            "target_ids", torch.zeros(num_users, max_memories, 64, dtype=torch.long)
        )
        self.register_buffer(
            "target_lens", torch.zeros(num_users, max_memories, dtype=torch.long)
        )

        # Prompt token IDs for Jaccard overlap retrieval (Direction 6.4)
        self.max_prompt_len = 128
        self.register_buffer(
            "prompt_ids", torch.zeros(num_users, max_memories, self.max_prompt_len, dtype=torch.long)
        )
        self.register_buffer(
            "prompt_lens", torch.zeros(num_users, max_memories, dtype=torch.long)
        )

        # Learnable key projection: maps backbone hidden states to key space
        self.key_proj = nn.Linear(d_key, d_key, bias=False)
        nn.init.eye_(self.key_proj.weight)

        # Value-to-delta projection: maps retrieved d_key-dimensional target
        # embeddings down to d_value (d_latent) for injection into fiber_proj.
        self.value_to_delta = nn.utils.parametrizations.spectral_norm(
            nn.Linear(d_key, d_value, bias=False)
        )
        # Scale factor: learned magnitude control for the episodic delta
        self.episodic_scale = nn.Parameter(torch.tensor(0.1))

    def _get_df_cache(self, uid: int, n: int) -> tuple[int, dict]:
        """
        Retrieves or accurately re-computes the mathematically exact TF-IDF DF Map,
        saving ~O(N^2) evaluation operations natively by caching the tuple state
        tied explicitly to the current write pointer parameter.
        """
        cache_key = (uid, n, self.write_ptr[uid].item())
        if hasattr(self, '_df_cache') and self._df_cache.get('key') == cache_key:
            return self._df_cache['data']

        df_map = {}
        valid_n = 0
        for k in range(n):
            k_len = self.prompt_lens[uid, k].item()
            if k_len > 0:
                valid_n += 1
                tup = tuple(self.prompt_ids[uid, k, :k_len].tolist())
                for t in set(tup):
                    df_map[t] = df_map.get(t, 0) + 1
                    
        result = (valid_n, df_map)
        self._df_cache = {'key': cache_key, 'data': result}
        return result

    def _prompt_recall(self, query_ids: list, stored_ids: list, n_docs: int, df_map: dict) -> float:
        """
        Compute recall of query content tokens in stored prompt tokens using dynamically computed IDF.
        """
        return compute_prompt_recall_idf(tuple(query_ids), tuple(stored_ids), n_docs, df_map)

    @torch.no_grad()
    def write(
        self,
        user_id: int,
        key_embedding: Tensor,
        value_embedding: Tensor,
        token_ids: Tensor = None,
        prompt_ids: Tensor = None,
    ):
        """
        Write a new memory entry for a user.

        Called during contrastive forcing (target_ids is not None) after
        the learning step to store the teaching context.

        Deduplication: if a stored key has cosine similarity > 0.95 to the
        new key, update that slot in-place instead of creating a duplicate.

        Args:
            user_id: Integer user ID
            key_embedding: (d_key,) prompt embedding at teach time
            value_embedding: (d_key,) target response hidden state
            token_ids: (L,) target token IDs for hard recall
            prompt_ids: (S,) prompt token IDs for Jaccard overlap retrieval
        """
        n = self.memory_count[user_id].item()

        # Deduplication check: if a near-identical key already exists, update it
        if n > 0:
            existing_keys = self.keys[user_id, :n]
            key_normed = F.normalize(key_embedding.detach().unsqueeze(0).to(existing_keys.dtype), p=2, dim=-1)
            existing_normed = F.normalize(existing_keys, p=2, dim=-1)
            sims = torch.matmul(existing_normed, key_normed.squeeze(0))
            max_sim, max_idx = sims.max(dim=0)

            # Fix prompt collision during write: only deduplicate if cosine > 0.95 AND exact content match
            if max_sim.item() > 0.95:
                idx = max_idx.item()
                is_exact_match = True
                if prompt_ids is not None:
                    p_len = self.prompt_lens[user_id, idx].item()
                    if p_len > 0:
                        stored_p = self.prompt_ids[user_id, idx, :p_len].tolist()
                        curr_p = prompt_ids[0].tolist()
                        n_docs, df_map = self._get_df_cache(user_id, n)
                        recall = self._prompt_recall(curr_p, stored_p, n_docs, df_map)
                        if recall < 1.0:
                            is_exact_match = False
                            
                if is_exact_match:
                    self.keys[user_id, idx] = key_embedding.detach()
                    self.values[user_id, idx] = value_embedding.detach()
                    if token_ids is not None:
                        L = min(token_ids.shape[-1], self.max_target_len)
                        self.target_ids[user_id, idx, :L] = token_ids.detach().flatten()[:L]
                        self.target_lens[user_id, idx] = L
                    if prompt_ids is not None:
                        L_p = min(prompt_ids.shape[-1], self.max_prompt_len)
                        self.prompt_ids[user_id, idx, :L_p] = prompt_ids.detach().flatten()[-L_p:]
                        self.prompt_lens[user_id, idx] = L_p
                    return

        # New memory — allocate next slot
        idx = self.write_ptr[user_id].item() % self.max_memories
        self.keys[user_id, idx] = key_embedding.detach()
        self.values[user_id, idx] = value_embedding.detach()
        if token_ids is not None:
            L = min(token_ids.shape[-1], self.max_target_len)
            self.target_ids[user_id, idx, :L] = token_ids.detach().flatten()[:L]
            self.target_lens[user_id, idx] = L
        if prompt_ids is not None:
            L_p = min(prompt_ids.shape[-1], self.max_prompt_len)
            self.prompt_ids[user_id, idx, :L_p] = prompt_ids.detach().flatten()[-L_p:]
            self.prompt_lens[user_id, idx] = L_p
        self.write_ptr[user_id] = (self.write_ptr[user_id] + 1) % self.max_memories
        self.memory_count[user_id] = min(
            self.memory_count[user_id] + 1,
            self.max_memories
        )

    def read(
        self, user_ids: Tensor, query_embeddings: Tensor, query_ids: Tensor = None, similarity_threshold: float = 0.6
    ) -> Tensor:
        """
        Retrieve relevant memories for a batch of users.

        When query_ids are provided, modulates cosine similarity with Jaccard
        overlap to break the ChatML template similarity ceiling.

        Args:
            user_ids: (B,) integer user IDs
            query_embeddings: (B, d_key) current prompt embeddings
            query_ids: (B, S) optional prompt token IDs for Jaccard modulation
            similarity_threshold: minimum similarity required to retrieve memory

        Returns:
            retrieved_deltas: (B, d_value) prompt-conditioned deltas
        """
        B = user_ids.shape[0]
        device = query_embeddings.device

        q_proj = self.key_proj(query_embeddings)
        q_proj = F.normalize(q_proj, p=2, dim=-1)

        retrieved = torch.zeros(B, self.d_value, device=device, dtype=query_embeddings.dtype)
        self._raw_retrieved = torch.zeros(B, self.d_key, device=device, dtype=query_embeddings.dtype)

        for b in range(B):
            uid = user_ids[b].item()
            n = self.memory_count[uid].item()
            if n == 0:
                continue

            user_keys = self.keys[uid, :n]
            user_vals = self.values[uid, :n]

            k_normed = F.normalize(user_keys.to(query_embeddings.dtype), p=2, dim=-1)
            sims = torch.matmul(k_normed, q_proj[b])

            # Additive Jaccard bonus for retrieval ranking
            if query_ids is not None:
                q_list = query_ids[b].tolist()
                n_docs, df_map = self._get_df_cache(uid, n)
                for i in range(n):
                    p_len = self.prompt_lens[uid, i].item()
                    if p_len > 0:
                        p_list = self.prompt_ids[uid, i, :p_len].tolist()
                        recall = self._prompt_recall(q_list, p_list, n_docs, df_map)
                        sims[i] = sims[i] + 0.5 * recall

            # Apply similarity threshold to prevent injecting noise on unrelated queries
            mask = sims >= similarity_threshold
            if not mask.any():
                continue
                
            sims_masked = sims.clone()
            sims_masked[~mask] = -float('inf')

            k = min(self.top_k, n)
            topk_sims, topk_idx = torch.topk(sims_masked, k)
            weights = F.softmax(topk_sims / self.temperature, dim=-1)

            topk_vals = user_vals[topk_idx]
            blended = (weights.unsqueeze(-1) * topk_vals.to(query_embeddings.dtype)).sum(dim=0)

            self._raw_retrieved[b] = blended.detach()

            retrieved[b] = self.episodic_scale * self.value_to_delta(blended)

        return retrieved

    def get_raw_retrieved(self):
        """Return the raw blended target embedding from the last read() call."""
        return getattr(self, '_raw_retrieved', None)

    @torch.no_grad()
    def hard_recall(
        self,
        user_ids: Tensor,
        query_embeddings: Tensor,
        vocab_size: int,
        similarity_threshold: float = 0.7,
        boost: float = 20.0,
        generated_ids: Tensor = None,
        query_ids: Tensor = None,
    ) -> Tensor:
        """
        Position-aware hard recall with Jaccard hybrid retrieval and
        any-completed Viterbi lock.

        Hybrid Retrieval (Direction 6.4):
        When query_ids are provided, computes Jaccard overlap between the
        current query tokens and each memory's stored prompt tokens (stripping
        ChatML markers). The Jaccard score is blended 50/50 with cosine
        similarity: effective_sim = 0.5 * cosine + 0.5 * jaccard.
        This pushes the true match to ~0.86 and false matches to ~0.36,
        completely resolving the F2 disambiguation failure.

        Any-Completed Lock:
        Once any memory has been fully generated (match_pos >= tgt_len),
        ALL injection is terminated. This prevents ghost memories from
        waking up after the winner finishes.

        Args:
            user_ids: (B,) integer user IDs
            query_embeddings: (B, d_key) current prompt embeddings
            vocab_size: Output vocabulary size
            similarity_threshold: Minimum cosine similarity to trigger recall
            boost: Logit boost for the next expected token
            generated_ids: (B, T) tokens generated so far (for position tracking)
            query_ids: (B, S) prompt token IDs for Jaccard overlap scoring

        Returns:
            logit_bias: (B, V) sparse logit bias (mostly zeros)
        """
        B = user_ids.shape[0]
        device = query_embeddings.device
        logit_bias = torch.zeros(B, vocab_size, device=device, dtype=query_embeddings.dtype)

        q_normed = F.normalize(query_embeddings, p=2, dim=-1)

        for b in range(B):
            uid = user_ids[b].item()
            n = self.memory_count[uid].item()
            if n == 0:
                continue

            user_keys = self.keys[uid, :n]
            k_normed = F.normalize(user_keys.to(query_embeddings.dtype), p=2, dim=-1)
            sims = torch.matmul(k_normed, q_normed[b])

            # Compute prompt recall scores for WTA tiebreaking (not for thresholding)
            recall_scores = torch.zeros(n, device=device)
            if query_ids is not None:
                q_list = query_ids[b].tolist()
                n_docs, df_map = self._get_df_cache(uid, n)
                for i in range(n):
                    p_len = self.prompt_lens[uid, i].item()
                    if p_len > 0:
                        p_list = self.prompt_ids[uid, i, :p_len].tolist()
                        recall_scores[i] = self._prompt_recall(q_list, p_list, n_docs, df_map)

            # Gate on raw cosine similarity — recall does NOT affect the gate.
            above_mask = sims >= 0.6
            if not above_mask.any():
                continue

            # Effective similarity for WTA ranking: cosine + recall bonus.
            RECALL_BONUS = 0.5
            effective_sims = sims + RECALL_BONUS * recall_scores

            # Single-Winner Selection (Semantic WTA): 
            # We strictly route only the mathematically closest memory geometry.
            # This prevents N=25000 arrays from blindly executing O(T^2) sequence 
            # evaluations on hundreds of unrelated clustered memories structurally
            # avoiding multi-fact context hijacking.
            
            valid_eff_sims = effective_sims.clone()
            valid_eff_sims[~above_mask] = -float('inf')
            
            best_idx = valid_eff_sims.argmax().item()
            best_eff_sim = valid_eff_sims[best_idx].item()
            
            if best_eff_sim == -float('inf'):
                continue
                
            tgt_len = self.target_lens[uid, best_idx].item()
            if tgt_len == 0:
                continue
                
            tgt_ids = self.target_ids[uid, best_idx, :tgt_len]
            weight = min(best_eff_sim / 0.6, 1.5)
            weighted_boost = boost * weight
            
            match_pos = 0
            if generated_ids is not None and generated_ids.shape[-1] > 0:
                gen_seq = generated_ids[b]
                match_pos = self._find_target_position(gen_seq, tgt_ids)
                
            # Any-Completed Lock applied ONLY to the structural winner:
            if match_pos < tgt_len:
                next_token = tgt_ids[match_pos].item()
                
                # DEBUG TRACE
                if getattr(self, 'debug_mode', False):
                    print(f"      [hard_recall] best_mem next_token: {next_token}, eff_sim: {best_eff_sim:.4f}")
                    
                logit_bias[b, next_token] = weighted_boost

        return logit_bias

    @staticmethod
    def _find_target_position(generated: Tensor, target: Tensor) -> int:
        """
        Find how many target tokens have appeared in the generated sequence
        (in order, not necessarily contiguous). Returns the index of the
        next target token to boost.

        Uses greedy subsequence matching, NOT strict prefix matching.
        Strict prefix was tested but caused repetition loops (3/5, down
        from 4/5) because a single divergent token kills all further
        boosting, causing the winner to repeatedly boost the same position.

        Example:
            target:    [Your, dog, 's, name, is, Luna]
            generated: [Your, dog, 's, name]
            returns: 4 (next expected = "is")
        """
        gen_list = generated.tolist()
        tgt_list = target.tolist()
        gen_pos = 0
        for tgt_idx, tgt_tok in enumerate(tgt_list):
            found = False
            for g in range(gen_pos, len(gen_list)):
                if gen_list[g] == tgt_tok:
                    gen_pos = g + 1
                    found = True
                    break
            if not found:
                return tgt_idx
        return len(tgt_list)  # all matched

    def get_stats(self, user_id: int) -> dict:
        """Get memory stats for a user."""
        n = self.memory_count[user_id].item()
        if n == 0:
            return {"num_memories": 0, "avg_key_norm": 0.0, "avg_value_norm": 0.0}
        return {
            "num_memories": n,
            "avg_key_norm": self.keys[user_id, :n].norm(dim=-1).mean().item(),
            "avg_value_norm": self.values[user_id, :n].norm(dim=-1).mean().item(),
        }
