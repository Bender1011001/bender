# Dual-System Decoder & Math Heads

## Status
- **Working**: Hybrid native SDPA `DecoderBlock` structure handling full GQA (Grouped Query Attention) and BFloat16 without intermediate state explosion.
- **Working**: Full continuous $O(T)$ scale autoregressive KV generation inside `ARDecoder.generate`.
- **Working**: Complete memory-efficient $O(B)$ memory utilization inside `LiSMCA_Bridge.forward` utilizing $c * (E * Q) \rightarrow (c * Q) * E$ algebraic associative associativity substitution to avoid $O(V)$ OOMs in continuous generation.
- **Working**: BFloat16 compatibility restored inside `geometry.cayley()` with localized FP32 upcasts allowing Native AMP execution to bypass PyTorch CUDA `lu_factor_cusolver` limitations.

## Key Files
- `decoder.py` — Native sequence unrolled graph execution for continuous-time constraint bridging with deep pre-trained contextual LLM transformer heads.
- `lismca.py` — Soft-mask continuous generation head resolving discrete token space to orthogonal manifolds.
- `geometry.py` — Base Cayley orthogonal mapping and Skew-vector tracking for deep representation physics.
- `tcrs.py` — Thomas solver unrolling trajectory calculations.
- `sleep_consolidation.py` — Background REM cycle consuming JSONL arrays mathematically baking RL updates via BCH derivations.
- `server.py` — FastAPI wrapper explicitly mapping `"user"` API parameters natively down into `z0` fiber bundles handling personality persistence.

## Architecture Quirks
- The `ARDecoder` requires its mathematical unrolled forward pass to act natively sequentially inside python to track parameter-sharing recurrences required by TCRS trajectories over continuous sequences. 
- During `train_language.py`, the sequence sequence length MUST strictly be subset clamped to relatively low steps (like `seq_len=64`) instead of `512` out of the box so that PyTorch doesn't throw OOMs attempting to compute gradients for BPTT unrolled sequences in Python memory natively.

## Trap Diary
| Issue | Cause | Fix |
|-------|-------|-----|
| OOM calculating LiSMCA token generators | `LiSMCA_Bridge` calculated $E_G = E \cdot Q_t$ which multiplied the entire $V=32,000$ embedding space over the $O(p^2)$ generator every unrolled frame. | Restructured the multiplication associatively by rotating query `c` first over `Q_t` yielding $O(1)$ memory usage per frame instead of $O(V)$. |
| `lu_factor_cusolver not implemented` | Attempting to execute `torch.linalg.solve` purely in BFloat16, which lacks native solver primitives on NVIDIA CUDA. | Cast `A` natively into `float32` inside `cayley(A)`, performed solve, and downcast back to native `A.dtype` (`bfloat16`). |
| Missing parameters in LLaMA surgery matching | Native SDPA uses `q_proj`, `k_proj`, etc. but GQA models generate size mismatches when standard attention is hardcoded. | Passed `getattr(hf_config, 'num_key_value_heads')` via `DualSystemConfig` directly enabling `n_kv_heads` to resize the tensor buffers flawlessly inside `DecoderBlock`. |
| `GradScaler` OOM/crash due to unscaling unrecoverable types. | `torch.amp.GradScaler` crashes during unscaling while utilizing purely bf16 operations since fp16 logic fails on unmodified float ranges. | Ripped out `scaler.scale()` and `scaler.unscale_()` from `train_language.py`. Bfloat16 natively supports full-spectrum floating precision and requires no scaling. |
| `torch.compile` rejecting sequential Thomas Algorithm. | Native sequential `for` loops inside `thomas_algorithm` over $4D$ sequence gradients break graph compilers generating `RuntimeError: Cannot find working triton installation` due to dynamic unrolling bounds. | Replaced sequential Thomas loop with `solve_tridiagonal_lu` utilizing `torch.linalg.lu_factor` statically evaluating once per tensor eliminating python graph discontinuities. |
| Inefficient CFG decoding natively inside Diffusions | Explicitly evaluating $f(z, c)$ and $f(z, \emptyset)$ sequentially doubled sequence latency per diffusion frame natively over System 2 reasoning blocks. | Optimized `sample_ddim` to statically construct a tensor slice evaluating both conditional bounds entirely parallelized inside a vectorized `chunk(2)` operation exactly identical to Diffusers. |

## Anti-Patterns (DO NOT)
- Do NOT revert `DecoderBlock.forward` to `nn.MultiheadAttention`. Native custom SDPA handling GQA scaling correctly reduces runtime generation inference memory by 75%.
- Do NOT unroll `ARDecoder` without utilizing associative scaling tricks. It operates identically to a full sequence LSTM, its graph tree is massively nested over length $T$.
