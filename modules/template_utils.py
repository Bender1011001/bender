"""
Chat Template Utilities
========================
Extracts content token indices from Qwen chat templates.
Used for content-only mean pooling in episodic memory keys.
"""

import torch

# Qwen chat template special token IDs (verified via tokenizer)
_IM_START_ID = 151644  # <|im_start|>
_IM_END_ID = 151645    # <|im_end|>
_USER_ID = 872         # 'user'
_NEWLINE_ID = 198      # '\n'


def extract_content_token_indices(prompt_ids: torch.Tensor) -> list:
    """
    Extract indices of user content tokens from a Qwen chat template.
    
    Template structure:
      <|im_start|>system\n...system...<|im_end|>\n
      <|im_start|>user\n{CONTENT}<|im_end|>\n
      <|im_start|>assistant\n
    
    Content tokens are between 'user\n' and '<|im_end|>' in the last
    user turn. Returns list of token position indices.
    
    If pattern not found, falls back to all token indices (safe default).
    """
    ids = prompt_ids.squeeze().tolist() if prompt_ids.dim() > 1 else prompt_ids.tolist()
    
    # Find LAST occurrence of <|im_start|>user\n pattern
    content_start = None
    content_end = None
    for i in range(len(ids) - 2, -1, -1):
        if (ids[i] == _IM_START_ID and i + 2 < len(ids) 
                and ids[i + 1] == _USER_ID and ids[i + 2] == _NEWLINE_ID):
            content_start = i + 3  # first content token after 'user\n'
            # Find the <|im_end|> after content_start
            for j in range(content_start, len(ids)):
                if ids[j] == _IM_END_ID:
                    content_end = j  # exclusive end
                    break
            break
    
    if content_start is None or content_end is None or content_start >= content_end:
        # Fallback: use all token positions
        return list(range(len(ids)))
    
    return list(range(content_start, content_end))


def content_mean_pool(h_seq: torch.Tensor, prompt_ids: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool hidden states over content-only tokens.
    
    Args:
        h_seq: (B, Seq, H) hidden states from backbone
        prompt_ids: (B, Seq) or (1, Seq) token IDs
    
    Returns:
        (B, H) content-only mean-pooled embedding
    """
    content_indices = extract_content_token_indices(prompt_ids)
    if len(content_indices) > 0 and max(content_indices) < h_seq.size(1):
        content_h = h_seq[:, content_indices, :]  # (B, C, H)
        return content_h.mean(dim=1)  # (B, H)
    else:
        return h_seq[:, -1, :]  # fallback to last token
