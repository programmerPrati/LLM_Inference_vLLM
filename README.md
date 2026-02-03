# LLM Inference Optimization using vLLM

## Summary
This research report analyzes Large Language Model (LLM) performance, spanning from **compute-bound prefill tasks** to **memory-bound decoding phases**. 

Using the **vLLM engine** on NVIDIA L4 hardware, I implemented and benchmarked several state-of-the-art optimization techniques—including Prefix Caching, Chunked Prefill, Speculative Decoding, and Continuous Batching. These implementations achieved a cumulative throughput improvement of up to **1.8x** in high-concurrency environments.

---

## Technical Overview
LLM inference is fundamentally bifurcated into two distinct operational stages:

1.  **Prefill Phase (Compute-Bound):** The model processes input tokens in parallel to generate KV caches. Performance here is limited by the GPU’s **TFLOPS**.
2.  **Decode Phase (Memory-Bound):** Individual tokens are generated autoregressively. The primary bottleneck is **memory bandwidth** (moving weights and KV cache data) rather than raw computation.

---

## Key Experiments & Achievements

* **Speculative Decoding:** Achieved a **1.06x (5.8%) speedup** in throughput (Tokens/Sec) by implementing N-Gram Speculative Decoding for predictable technical contexts.
* **Prefix Caching:** Realized a **1.10x speedup** using automatic prefix caching to eliminate redundant prefill computation for 80 concurrent requests sharing a 3,000-token context.
* **Continuous Batching:** Achieved a **1.09x speedup** over standard batching methods.
* **Chunked Prefill & Latency:** Mitigated the "compute-bound hijacking" effect by interleaving a 3,500-token "blocker" request with smaller queries, achieving a **1.8x speedup** for concurrent users.
* **Memory Engineering:** Optimized VRAM overhead by tuning GPU utilization to `0.85` and implementing a Draft-and-Verify pipeline with **FP8 KV cache quantization**.
* **Systems Implementation:** Developed custom background server management and cleanup routines in Google Colab to prevent OOM (Out of Memory) errors during stress tests.

---

## Performance Results

| Optimization Technique | Baseline Performance | Optimized Performance | Speedup / Improvement |
| :--- | :--- | :--- | :--- |
| **Automatic Prefix Caching** | 1.18 Tokens/Sec (126.85s) | 1.30 Tokens/Sec (115.36s) | **1.10x** |
| **Concurrency Stress Test** | Standard Batching | Continuous Batching | **1.09x** |
| **N-Gram Speculative Decoding** | 21.21 Tokens/Sec | 22.45 Tokens/Sec | **1.06x** |
| **Chunked Prefill** | Blocker: 3.69s / Victim: 1.72s | Blocker: 2.89s / Victim: 0.93s | **1.8x (Victim)** |

---

## Abstract
This research demonstrates that through hardware-aware software orchestration—specifically the integration of Prefix Caching, Continuous Batching, Chunked Prefill, and Speculative Decoding—LLM inference can be shifted from memory-bound bottlenecks to high-throughput, memory-efficient streams. 

By implementing manual GPU utilization caps and FP8 KV-cache quantization on NVIDIA L4 hardware, I achieved significant throughput speedups and reduced inter-token latency. These results provide a technical proof-of-concept for maximizing **utilization density** in high-concurrency environments under strict VRAM constraints.

> **Note:** Benchmarks were conducted on an **NVIDIA L4 GPU** within a **Google Colab** environment. Performance results reflect real-world overhead and system-level constraints of a hosted notebook instance.

For a more detailed explanation of the theory and implementation, see my **[LLM Inference Overview Document](./LLM_Inference_Overview.pdf)**.
