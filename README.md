# BENDER — Bolt Permanent Memory onto Any Frozen LLM

BENDER gives any local LLM persistent per-user memory — no fine-tuning, no RAG pipeline, no context stuffing, no cloud dependency.

The backbone is frozen. You train the sidecar once. After that, memory writes happen in real time from user conversations and cost nothing at inference.

---

## How it works

Most "memory" in AI products works by prepending stored facts to the prompt as text. That costs tokens on every query, bloats context, and gets expensive at scale.

BENDER works differently. It injects memory directly into the token probability distribution during generation — the model doesn't "read" a memory, the memory steers which token gets produced. The result is deterministic recall with zero prompt overhead.

Three mechanisms work together:
- **Cosine retrieval** — embedding similarity between the query and stored memories
- **Prompt recall scoring** — token overlap between query and stored prompt (handles cases where embedding similarity fails)
- **Dynamic logit boost** — scales automatically to the backbone's logit distribution, so the same sidecar works on 3B and 7B models without retuning

---

## Benchmark results

Exhaustive recall — every stored fact queried individually, not sampled.

| Backbone | Type | Hidden Dim | N=1000 |
|---|---|---|---|
| Qwen2.5 7B | Dense standard | 3584 | 1000/1000 ✅ |
| Gemma2 9B | Dense sliding window | 3584 | 1000/1000 ✅ |
| Qwen3-30B-A3B | Sparse 128-expert MoE | 2048 | 1000/1000 ✅ |

Multi-user isolation: 6/6 — users cannot access each other's memories.  
Grammar stability: clean through 20 rounds of continuous training.

Logs: `/logs/scale_1000_exhaustive_pass.log`

## BENDER vs. Context-Window Stuffing

To test scaling behavior, both frameworks were queried identically to recall distinct facts across increasing context sizes. The RAG baseline prepends all facts as text into the prompt (context stuffing). This is the simplest RAG approach — a production RAG system with vector search and top-k retrieval would perform better than this baseline.

| Paradigm | Architecture | N=50 | N=250 | N=500 | N=1000 | Latency Scaling |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Context Stuffing** | Gemma-2-9B (Base) | 100.0% (0.39s) | 70.0% (0.77s) | 20.0% (1.35s) | 0.0% (OOM) | O(N²) attention |
| **BENDER Memory** | Gemma-2-9B (+ Sidecar) | **100.0%** (0.83s) | **100.0%** (0.85s) | **100.0%** (0.80s) | **100.0%** (0.86s) | **Constant** |

Context stuffing collapses due to attention dilution ("Lost in the Middle") and eventually OOMs. BENDER's episodic memory retrieves via `embed_tokens.weight.T` projection, maintaining constant latency regardless of memory count.

**Note:** The memory buffer is a fixed-size circular buffer (`max_memories` slots per user). Latency is constant because the buffer size is constant — not because the architecture handles unlimited memories. For most use cases (chatbot, agent, persona), the default capacity is more than sufficient.

---

## 25-Agent Town Simulation (Smallville, N=25)

We migrated the Smallville multi-agent generative town simulation onto this framework. 25 agents operate independently with persistent memory, zero cross-character contamination, and constant latency — replacing the original external vector database entirely.

`bender_api.py` provides drop-in `/generate` and `/inject_memory` FastAPI endpoints.

---

## Backbone-agnostic

Qwen uses standard dense attention. Gemma 2 uses alternating local/global sliding window attention. Qwen3-30B-A3B uses sparse 128-expert MoE with a different hidden dimension. BENDER reads from `hidden_states[-1]` in all cases and gets clean retrieval geometry either way.

If your model exposes `hidden_states` through HuggingFace's standard interface, it should work.

---

## Quick start

```bash
git clone https://github.com/Bender1011001/bender
cd bender
pip install -r requirements.txt

python demo.py
# Uses Qwen2.5-3B by default — runs on ~6GB VRAM

python demo.py --model /path/to/your/model
# Point at any local HuggingFace causal LM
```

The demo injects one fact and queries it back. You should see the target keyword in the response.

---

## What you need to run it

- Any CUDA GPU with enough VRAM to run your chosen backbone
- A quantized 7B fits in 6–8GB; the sidecar itself adds ~200MB
- The backbone is never modified — load it read-only if you want

---

## What's in the repo

```
bender/
  dual_system_v2.py       — main model class, wraps any HF backbone
  chat_continuous.py      — memory write path (continuous_learning_update)
  demo.py                 — one command, see it work
  bender_api.py           — FastAPI wrapper for multi-agent deployments

  modules/
    episodic_memory.py    — retrieval, hybrid scoring, Viterbi lock
    sparse_bias.py        — per-user vocab bias with decay
    fiber_bundle.py       — per-user style personalization (SO(d) fibers)
    geometry.py           — Cayley map, skew-symmetric utilities
    diffusion_planner.py  — latent blueprint encoder (VAE)
    epistemic_router.py   — routes learning: factual vs style errors
    bch_consolidation.py  — lifelong fiber consolidation via BCH expansion

benchmarks/
  test_capacity_scale.py      — exhaustive N-fact recall test
  test_multi_user_isolation.py — cross-user isolation
  test_repetition_recall.py   — multi-round regression
```

---

## Known limitations

- Fact capacity depends on prompt diversity. Synthetic benchmarks use maximally similar prompts (stress test). Real user queries are more semantically distinct and easier to retrieve.
- The sidecar needs a one-time training run to initialize. A pre-trained checkpoint is available on HuggingFace: [`Bender1011001/Qwen2.5-3B-DualSystem-V2`](https://huggingface.co/Bender1011001/Qwen2.5-3B-DualSystem-V2).
- Memory is stored in a fixed-size circular buffer per user. Once full, oldest memories are overwritten.

---

## The name

The backbone this was developed on is `Bender1011001/Qwen2.5-7B-ABLITERATED`. The system literally bends logit distributions to inject memory. And Bender remembers everything about himself and doesn't answer to anyone. Seemed right.