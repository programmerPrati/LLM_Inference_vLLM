# LLM Inference Overview using vLLM

This repository explores high-performance LLM inference techniques designed to overcome the memory-bound bottlenecks of transformer architectures. Using the vLLM library in Python on NVIDIA L4 GPU hardware, I explored and benchmarked state-of-the-art optimization techniques. I focused on LLMs at inference time, detailing the two stages of **Prefill** and **Decode**, as well as other essential concepts regarding Transformer deployment.

To evaluate these optimizations, I implemented these techniques and benchmarked them against vanilla baselines to quantify their performance gains.

* **PagedAttention:** Efficient KV cache management using non-contiguous memory allocation to reduce fragmentation.
* **Continuous Batching:** Dynamic request scheduling to eliminate GPU "bubble" time.
* **Automatic Prefix Caching:** Hashing shared contexts (e.g., System Prompts/RAG documents) to bypass redundant prefill computations.
* **Speculative Decoding:** Leveraging N-Gram lookup and small draft models to "guess" tokens, utilizing excess memory bandwidth to accelerate generation.
* **Chunked Prefill:** Interleaving large prompt processing with active decoding sequences to maintain low Inter-Token Latency (ITL).

---

### Performance Results
| Optimization Technique | Speedup Factor |
| :--- | :--- |
| **Automatic Prefix Caching** | **1.10x** |
| **Continuous Batching** | **1.09x** |
| **N-Gram Speculative Decoding**| **1.06x** |

> **Note:** Benchmarks were conducted on an **NVIDIA L4 GPU** within a **Google Colab** environment. Performance results reflect real-world overhead and system-level constraints of a hosted notebook instance.

For a more detailed explanation of the theory and implementation, see my **[LLM Inference Overview Document](./LLM_Inference_Overview.pdf)**.
