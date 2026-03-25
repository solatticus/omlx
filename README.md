<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/icon-rounded-dark.svg" width="140">
    <source media="(prefers-color-scheme: light)" srcset="docs/images/icon-rounded-light.svg" width="140">
    <img alt="oMLX" src="docs/images/icon-rounded-light.svg" width="140">
  </picture>
</p>

<h1 align="center">oMLX</h1>
<p align="center"><b>LLM inference with persistent sessions, optimized for your Mac</b><br>Stateful KV caching, TurboQuant compression, continuous batching — for people who own their hardware.</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple" alt="Apple Silicon">
</p>

<p align="center">
  <a href="#whats-different">What's Different</a> ·
  <a href="#install">Install</a> ·
  <a href="#sessions">Sessions</a> ·
  <a href="#turboquant">TurboQuant</a> ·
  <a href="#benchmark-results">Benchmarks</a> ·
  <a href="#features">All Features</a> ·
  <a href="#cli-configuration">CLI Configuration</a>
</p>

---

> ### Recent: dtype fix turns a 2x slowdown into a 16% speedup
>
> The benchmark suite caught a critical bug: TurboQuant was decompressing
> KV tensors to float32 instead of the model's native bfloat16. Every
> decode step ran at half speed with double memory. One-line fix — store
> the original dtype during compression, cast back on decompression.
>
> **Before fix** (float32 KV):
> ```
> 8-turn session total: 50.6s (87% SLOWER than stateless)
> ```
> **After fix** (bfloat16 KV):
> ```
> 8-turn session total: 32.3s (16% FASTER than stateless)
> Turn 8: 87% cache hit, 2.74s vs 3.16s stateless
> Memory: 45MB compressed (was 155MB uncompressed)
> Park: 0.12s to SSD | Resume: instant
> ```
>
> This is what benchmarks are for. Functional tests passed — every cache
> hit was correct, every response was valid. But the model was doing
> attention in float32 instead of bfloat16 and nobody would have noticed
> without wall-clock timing.

---

> **This is a fork of [jundot/omlx](https://github.com/jundot/omlx)** — an excellent MLX inference server with continuous batching, tiered KV caching, multi-model serving, and a polished admin dashboard. All credit for the foundation goes to [@jundot](https://github.com/jundot) and the oMLX contributors. We built on top of their work.

---

## What's Different

Every cloud LLM provider is stateless because they have to be. They serve 200 million users — they can't hold your KV cache between turns. Every request re-sends the entire conversation. The server recomputes everything from scratch.

**You're not at scale.** You're one household. One small team. One mesh of machines. The constraint doesn't apply.

This fork adds two things:

### 1. Persistent Sessions

After a chat completion, the KV cache stays in memory, tagged with a session ID. Next turn, only the new tokens are computed — everything prior is already there.

```
Turn 1:   0% cache hit   (cold start)
Turn 2:  56% cache hit   (prior context reused)
Turn 3:  68% cache hit   (compounding)
Turn 4:  74% cache hit   (3/4 of prefill is free)
Turn 20: nearly everything cached
```

**Park** a session to SSD when you walk away. **Resume** it hours later — KV state loads from a safetensors file in seconds. Pick up exactly where you left off.

### 2. TurboQuant KV Compression

Session KV cache is compressed 4x in memory using the algorithm from [Zandieh et al. (ICLR 2026)](https://arxiv.org/abs/2504.19874). Random rotation + Lloyd-Max scalar quantization at 3 bits per coordinate. MSE 0.034 — near-zero quality loss.

| | Without TurboQuant | With TurboQuant |
|---|---|---|
| Session memory (short conv) | ~155 MB | ~40 MB |
| Concurrent sessions (18GB headroom) | ~1 large | ~4 large |
| Park file size | ~155 MB | ~155 MB (raw on SSD) |

Compression is transparent — enabled by default, applied on store, reversed on retrieval.

### Everything Else Still Works

The existing `/v1/chat/completions` endpoint, admin dashboard, multi-model serving, tiered KV cache, VLM support, embeddings, reranking — all unchanged. Sessions are additive.

---

## Install

### From This Fork

```bash
git clone https://github.com/solatticus/omlx.git
cd omlx
pip install -e .
```

### From the Original (no sessions)

If you don't need sessions, use the original: [github.com/jundot/omlx](https://github.com/jundot/omlx). It has a macOS app, Homebrew tap, and auto-updates.

### Requirements

- macOS 15.0+ (Sequoia)
- Python 3.10+
- Apple Silicon (M1/M2/M3/M4)

## Quickstart

```bash
omlx serve --model-dir ~/models --paged-ssd-cache-dir ~/.omlx/cache
```

The server discovers models from subdirectories automatically. Any OpenAI-compatible client can connect to `http://localhost:8000/v1`.

---

## Sessions

Sessions make the server remember. Create one, chat in it, and KV state persists between turns.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/sessions` | Create a session |
| `POST` | `/v1/sessions/{id}/chat` | Chat (KV retained) |
| `GET` | `/v1/sessions` | List all sessions |
| `GET` | `/v1/sessions/{id}` | Session details |
| `POST` | `/v1/sessions/{id}/park` | Serialize KV to SSD |
| `POST` | `/v1/sessions/{id}/resume` | Load KV from SSD |
| `DELETE` | `/v1/sessions/{id}` | Destroy session |

### Create and Chat

```bash
# Create a session
SESSION=$(curl -s http://localhost:8000/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-32B-4bit"}' | jq -r .session_id)

# Turn 1 — cold start, full prefill
curl http://localhost:8000/v1/sessions/$SESSION/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Explain KV caching in transformers."}], "max_tokens": 256}'

# Turn 2 — prior context cached, only new tokens computed
curl http://localhost:8000/v1/sessions/$SESSION/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [
    {"role": "user", "content": "Explain KV caching in transformers."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "How does paged attention improve on this?"}
  ], "max_tokens": 256}'
```

The client sends full message history each turn (same as stateless). The server detects the shared prefix and skips recomputing it.

### Park and Resume

```bash
# Park — KV serialized to SSD, memory freed
curl -X POST http://localhost:8000/v1/sessions/$SESSION/park

# Hours later...
curl -X POST http://localhost:8000/v1/sessions/$SESSION/resume

# Chat continues with cached KV intact
```

### Streaming

```bash
curl -N http://localhost:8000/v1/sessions/$SESSION/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [...], "stream": true}'
```

SSE chunks include `session_id`, `content`, `finished`. The final chunk has `usage`, `turn`, and `cached_tokens`.

### How It Works

1. **On generation complete**: Scheduler extracts KV tensors from the batch, stores them in a `SessionKVStore` keyed by session ID (instead of freeing them)
2. **On next turn**: Scheduler detects the session, finds the longest matching token prefix, reconstructs cache objects from stored tensors, injects them into the batch — skipping prefill for cached tokens
3. **TurboQuant**: KV tensors are compressed 4x on store and decompressed on retrieval, transparent to the scheduler
4. **Park**: Decompresses, serializes raw tensors to safetensors on SSD, frees memory
5. **Resume**: Loads from SSD, recompresses, session continues

The existing `/v1/chat/completions` stateless endpoint is completely unchanged.

---

## TurboQuant

KV cache compression based on [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh, Daliri, Hadian, Mirrokni — Google Research / NYU, ICLR 2026).

### Algorithm

1. **Rotate** each KV vector by a random orthogonal matrix (QR decomposition, computed once per head dimension)
2. After rotation, coordinates are approximately i.i.d. from a known distribution (Beta converging to Gaussian)
3. **Quantize** each coordinate independently using precomputed Lloyd-Max centroids (3 bits = 8 levels)
4. Store the quantized indices (uint8) + vector norms (float16)
5. **Dequantize**: look up centroids, rotate back, rescale

### Results

Tested on Qwen3.5-VL-122B-A10B (MoE, head_dim 128/256):

- **4x compression** (3-bit indices stored as uint8)
- **MSE 0.034** (theoretical bound: 0.043)
- **Zero quality degradation** on multi-turn conversations
- Enabled by default, configurable via `--kv-compress-bits 0/2/3/4`

## Benchmark Results

8-turn coding conversation on Qwen3.5-VL-122B-A10B-4bit-CRACK (M3 Ultra, 96GB):

```
Turn  Stateless          Session + TurboQuant
      time   prompt hit  time   ttft   prompt cached  hit
  1  18.13s     56   0%  13.89s 0.216s     56      0   0%
  2   2.65s    126   0%   2.64s 0.352s    126     54  42%
  3   2.74s    181   0%   2.58s 0.288s    181    124  68%
  4   2.83s    253   0%   2.63s 0.329s    253    179  70%
  5   2.91s    318   0%   2.62s 0.319s    318    251  78%
  6   3.01s    396   0%   2.64s 0.337s    396    316  79%
  7   3.08s    452   0%   2.60s 0.292s    452    394  87%
  8   3.16s    515   0%   2.74s 0.423s    515    450  87%
                -----                       -----
Total: 38.5s              32.3s (+16% faster)

Session memory: 45.4 MB (TurboQuant 3-bit compressed)
Park to SSD: 0.12s | Resume: instant
```

Cache hits compound every turn. By turn 8, 87% of the prompt is served from
session KV cache. Sessions are faster than stateless at ~500 tokens — the gap
widens with longer conversations.

Run the benchmark yourself:
```bash
python scripts/benchmark_sessions.py \
  --url http://localhost:8080 \
  --model YOUR_MODEL \
  --turns 10 --max-tokens 128
```

---

## Features

Everything from upstream oMLX, plus sessions and TurboQuant.

### From Upstream

- **Tiered KV Cache** — hot (RAM) + cold (SSD) with prefix sharing and copy-on-write
- **Continuous Batching** — concurrent request handling via mlx-lm BatchGenerator
- **Multi-Model Serving** — LLMs, VLMs, embeddings, rerankers with LRU eviction
- **Vision-Language Models** — multi-image chat, OCR auto-detection
- **Admin Dashboard** — real-time monitoring, model management, chat UI, benchmarks
- **macOS Menubar App** — native, not Electron
- **Per-Model Settings** — sampling params, TTL, aliases, chat template kwargs
- **Tool Calling** — function calling, JSON schema, MCP integration
- **API Compatibility** — OpenAI and Anthropic drop-in replacement

### Added in This Fork

- **Persistent Sessions** — KV cache retained across turns, park/resume to SSD
- **TurboQuant Compression** — 4x KV memory reduction with near-zero quality loss
- **Session-Aware Memory Enforcement** — auto-parks LRU sessions under memory pressure
- **Thinking Model Support** — handles generation prompt tokens correctly for reasoning models

## CLI Configuration

```bash
# Basic: serve models with SSD cache
omlx serve --model-dir ~/models --paged-ssd-cache-dir ~/.omlx/cache

# Memory limits
omlx serve --model-dir ~/models --max-model-memory 32GB --max-process-memory 80%

# Hot cache + large initial block pool
omlx serve --model-dir ~/models --hot-cache-max-size 4GB --initial-cache-blocks 512

# With MCP tools
omlx serve --model-dir ~/models --mcp-config mcp.json

# API key authentication
omlx serve --model-dir ~/models --api-key your-secret-key
```

Sessions and TurboQuant are enabled automatically. No additional flags needed.

<details>
<summary>Architecture</summary>

```
FastAPI Server (OpenAI / Anthropic / Sessions API)
    │
    ├── SessionManager (session lifecycle, KV store, park/resume)
    │   └── SessionKVStore (in-memory, TurboQuant compressed)
    │       └── TurboQuant (rotate → quantize → store, 4x compression)
    │
    ├── EnginePool (multi-model, LRU eviction, TTL)
    │   ├── BatchedEngine (LLMs, continuous batching)
    │   ├── VLMEngine (vision-language models)
    │   ├── EmbeddingEngine
    │   └── RerankerEngine
    │
    ├── ProcessMemoryEnforcer (memory limit + session auto-park)
    │
    ├── Scheduler (FCFS, session cache injection)
    │   └── mlx-lm BatchGenerator
    │
    └── Cache Stack
        ├── PagedCacheManager (block-based, CoW, prefix sharing)
        ├── Hot Cache (in-memory tier, write-back)
        └── PagedSSDCacheManager (SSD cold tier, safetensors)
```

</details>

## Models

Point `--model-dir` at a directory containing MLX-format model subdirectories.

| Type | Models |
|------|--------|
| LLM | Any model supported by [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| VLM | Qwen3.5 Series, GLM-4V, Pixtral, and other [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) models |
| Embedding | BERT, BGE-M3, ModernBERT |
| Reranker | ModernBERT, XLM-RoBERTa |

## License

[Apache 2.0](LICENSE)

## Acknowledgments

- **[jundot/omlx](https://github.com/jundot/omlx)** — the foundation. Continuous batching, tiered KV caching, multi-model serving, admin dashboard, macOS app. This fork builds directly on their work.
- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) by Apple
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — Vision-language model inference on Apple Silicon
- [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026) — KV cache quantization algorithm
- [PolarQuant](https://arxiv.org/abs/2502.02617) (Han et al., AISTATS 2026) — random preconditioning for KV compression
- [vllm-mlx](https://github.com/waybarrios/vllm-mlx) — where oMLX started
- [venvstacks](https://venvstacks.lmstudio.ai) — portable Python environment layering
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) — embedding model support
