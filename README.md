# Anvit More — ML Systems · GPU Kernels · Applied RL

ML engineer focused on the systems layer: how models run fast, how decisions get made under uncertainty, how inference holds up in production.

---

## What I build

**GPU kernel engineering** — custom Triton kernels for LLM primitives, benchmarked against PyTorch baselines on real hardware.

| Kernel | Speedup | Peak throughput |
|---|---|---|
| Fused Bias + GELU | **14.65×** | 172 GB/s |
| FlashAttention (T=2048) | **2.52×** | 11.4 GB/s |
| Fused AdamW (50M params) | **3.45×** | 177 GB/s |
| Inference attention (B=2) | **3.94×** | 95 GB/s |

→ [`llm-sys`](https://github.com/dunkinflicka/llm-sys) — RMSNorm, LayerNorm, FlashAttention, fused AdamW, inference attention. Every kernel validated against fp32 reference, benchmarked with `triton.testing.do_bench`.

---

**Production LLM inference** — async serving stack on a 6 GB GPU, built from first principles.

| Metric | Value |
|---|---|
| TTFT P50 | 28 ms |
| Decode speed | 39.4 tok/s (~85% of memory bandwidth) |
| Cache hit latency P50 | 2 ms |
| Cache hit rate | 81% |
| Success rate @ concurrency=10 | 100% |

→ [`llm-inference-stack`](https://github.com/dunkinflicka/llm-inference-stack) — FastAPI gateway → Redis cache → FP16 PyTorch → RTX 4050L. Fused Triton attention kernel, asyncio-locked GPU access, fire-and-forget cache writes.

---

**Reinforcement learning for real-time decisions** — physics-informed simulation + PPO agent for F1 race strategy. The same architecture applies to ADAS planning, EV energy management, and hybrid powertrain arbitration.

| Agent | E[Position] | E[Points] |
|---|---|---|
| Rule-based baseline (1-stop M→H) | 3.09 | 15.8 |
| **PPO agent** | **1.00** | **25.0** |

+58% points vs baseline. Monte Carlo planner runs at 870 rollouts/second on a single CPU core.

→ [`autonomous-strategy-engine`](https://github.com/dunkinflicka/autonomous-strategy-engine) — physics-informed tyre/fuel/weather models, 10k–100k MC rollouts, PPO on 8-dim sensor observation, 27 passing tests.

---

## Stack

`Python` · `Triton` · `PyTorch` · `CUDA` · `FastAPI` · `Redis` · `Docker` · `Stable-Baselines3` · `NumPy` · `scikit-learn`

Production experience: `LoRA/QLoRA` fine-tuning · `Whisper` ASR · `RAG` (FAISS, Pinecone) · `Gemini Vision` · medical NLP

---

## Background

MSc Data Science — University of Edinburgh (2024)  
Currently: ML & AI Engineer @ Plus91 Technology, Pune  
Target: ML Systems / LLM Inference / Automotive AI — open to relocate to Germany, Switzerland, Poland, Norway, Finland

📧 anviit22@gmail.com · [LinkedIn](https://linkedin.com/in/anvit22)
