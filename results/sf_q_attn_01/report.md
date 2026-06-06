I now have a complete picture across all five research tracks and ~35 verified papers. Writing the final synthesis.

---

## Long-Context Attention for LLMs: Evolution and 2025–2026 State of the Art

---

### 1. The Canonical Approach and Its Key Prior Art

**FlashAttention** (Dao et al., 2022; [2205.14135] *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*) is the canonical baseline this whole field is defined against. It solved the *IO bottleneck* of exact softmax attention: by tiling key-query-value blocks and keeping them in SRAM, it avoided repeated HBM reads/writes, achieving ~3× wall-clock speedup with identical outputs and O(n) peak memory — without touching the O(n²) FLOP count. With 4,411+ citations, it became the universal implementation baseline for Transformers on long sequences.

Two direct successors hardened it into production infrastructure:
- **FlashAttention-2** (Dao, 2023; [2307.08691]) redesigned the work partitioning and parallelism for modern multi-GPU setups (2,779 citations).
- **FlashAttention-3** (Shah et al., 2024; [2407.08608]) exploited H100-specific hardware — asynchrony and FP8 low-precision — to reach up to 1.2 PFLOPs/s and 2× additional speedup over FA-2.

Supporting infrastructure that the field treats as standard:
- **Multi-Query Attention / GQA** ([1911.02150] Shazeer, 2019; [2305.13245] Ainslie et al., 2023): reducing KV head count cuts the KV cache proportionally, the first architectural lever on memory.
- **Ring Attention** ([2310.01889] Liu et al., 2023): distributes FlashAttention across device rings for near-infinite context on clusters.
- **Self-attention does not need O(n²) memory** ([2112.05682] Rabe & Staats, 2021): the theoretical basis for memory-efficient recomputation that FA builds on.

**Why this era is being superseded:** FlashAttention is IO-efficient but remains fundamentally O(n²) in FLOPs. At context lengths of 64K–1M tokens, even IO-efficient exact attention becomes compute-prohibitive during prefill, and the KV cache becomes the dominant memory cost at decode time. No kernel trick changes that.

---

### 2. How the Field Evolved: Three Parallel Tracks

#### Track A — Sub-quadratic / Linear Attention (the "remove softmax" track)

The theoretical goal: replace the softmax outer product with a kernel decomposition that allows the attention matrix to be computed as a product of low-dimensional features, reducing complexity to O(n).

**Foundations (2019–2021):**
- **Sparse Transformers** ([1904.10509] Child et al., 2019): the first systematic sparse-pattern attention, showing O(n√n) is achievable with strided + local blocks.
- **Longformer** ([2004.05150] Beltagy et al., 2020) and **Big Bird** ([2007.14062] Zaheer et al., 2020): established sliding-window + global-token hybrid patterns as standard baselines for document-scale tasks.
- **Reformer** ([2001.04451] Kitaev et al., 2020): locality-sensitive hashing for approximate attention.
- **"Transformers are RNNs"** ([2006.16236] Katharopoulos et al., 2020): the first practical kernel linearization, recasting attention as an RNN with O(1) per-step cost at inference.
- **Performers** ([2009.14794] Choromanski et al., 2020): FAVOR+ random feature maps as unbiased softmax approximations.
- **Random Feature Attention** ([2103.02143] Peng et al., 2021).

**The recurrent/SSM wave (2022–2023):** Linear kernel attention was faster but consistently underperformed softmax in quality — particularly on tasks requiring precise in-context retrieval. State-space models (SSMs) addressed this with structured recurrences.
- **RWKV** ([2305.13048] Peng et al., 2023): reconciled Transformer-parallel training with RNN O(1) inference by recasting attention as a time-mix and channel-mix, achieving parity with same-scale Transformers (1,066 citations).
- **RetNet** ([2307.08621] Sun et al., 2023): added exponential decay ("retention") to linear attention, enabling parallel training, recurrent inference, and chunk-wise modes simultaneously (671 citations).
- **Mamba** (Gu & Dao, 2023; arXiv 2312.00752, not indexed in this corpus): introduced *selective* SSMs where the state-transition matrix is input-dependent, achieving near-Transformer quality for the first time in an SSM while staying O(n) in flops and O(1) in state.

**The hardware-efficient linear attention wave (2023–2024):**
- **Gated Linear Attention (GLA)** ([2312.06635] Yang et al., 2023; ICML): made gated linear attention hardware-efficient via chunked parallel algorithms, achieving 2× training speedup over naïve linear attention baselines and strong length generalization to 20K (435 citations).
- **DeltaNet / Parallelizing the Delta Rule** ([2406.06484] Yang et al., 2024): enabled hardware-efficient parallel training of the associative-memory-style delta update, previously sequential.
- **Gated DeltaNet** ([2412.06464] Yang et al., 2024; ICLR 2025): combined gating for selective forgetting with the delta write rule — the clearest current improvement on GLA and Mamba-2, consistently outperforming both on language modeling and long-context benchmarks (318 citations).
- **RWKV-7 "Goose"** ([2503.14456] Peng et al., 2025): expressive dynamic state evolution achieving new SOTA for 3B non-attention models on multilingual and English benchmarks (116 citations).

**The key limitation of pure linear attention:** the fixed-size hidden state compresses all history. For tasks requiring exact token retrieval (needle-in-haystack, multi-hop QA), linear models still lag softmax attention noticeably. This motivated hybrid architectures (see Track C) rather than pure linear replacement.

#### Track B — Sparse Attention at Scale (the "keep softmax, reduce coverage" track)

Rather than approximating softmax, this track keeps exact softmax attention but restricts which token pairs communicate.

**KV cache eviction / selection (inference-time sparsity):**
- **H2O: Heavy-Hitter Oracle** ([2306.14048] Zhang et al., 2023): identified that a small number of "heavy hitter" tokens dominate KV cache usage and proposed evicting the rest (741 citations).
- **Scissorhands** ([2305.17118] Liu et al., 2023): *persistence-of-importance* hypothesis — important tokens stay important across decoding steps.
- **Attention Sinks** ([2309.17453] Xiao et al., 2023): revealed that initial tokens act as "sink" tokens absorbing large attention mass; using a sliding window with these sinks enables streaming inference to millions of tokens with no quality degradation on perplexity (1,912 citations).
- **SnapKV** ([2404.14469] Li et al., 2024): observation-window KV compression without retraining.

**Dynamic sparse prefill:**
- **MInference** ([2407.02490] Jiang et al., 2024): discovered that LLM attention has typed patterns (A-shape, vertical-slash, block-diagonal); exploiting these dynamically achieves 10× prefill speedup with negligible quality loss on 1M-token contexts (361 citations).
- **Quest** ([2406.10774] Tang et al., 2024): query-aware KV page selection for hardware-efficient sparse decoding (368 citations).
- **SeerAttention** ([2410.13276] Gao et al., 2024): trains a lightweight gating head to predict block-level sparsity from within the model itself, enabling learned sparse attention (105 citations).

**Dilated / hierarchical sparse attention:**
- **LongNet** ([2307.02486] Ding et al., 2023): dilated attention with exponentially increasing segment sizes achieves O(n log n) and demonstrated Transformer operation at 1 billion tokens.

---

### 3. 2025–2026 Work That Supersedes the Established Answer

This is where the landscape most clearly breaks from "FlashAttention + KV compression heuristics" to something fundamentally different.

#### 3a. Natively Trainable Hardware-Aligned Sparse Attention

**Native Sparse Attention (NSA)** ([2502.11089] Yuan et al., DeepSeek, 2025; ACL 2025): the single most-cited 2025 paper in this space (370 citations within 4 months). NSA makes three advances over prior sparse attention: (1) it is *natively trained* end-to-end with sparse attention rather than applied post-hoc, (2) it uses a hierarchical combination of compressed-block attention (coarse global context), selected-block attention (dynamically routed important blocks), and sliding window attention (local tokens), and (3) the kernel is *hardware-aligned* — designed around Tensor Core tiling for Ampere/Hopper GPUs so the sparse pattern maps to dense tensor operations. Evaluated on a 27B-total/3B-active MoE backbone with 64K training contexts, NSA matches or exceeds full-attention quality on general benchmarks, reasoning, and long-context tasks while being substantially faster on 64K sequences.

**MoBA: Mixture of Block Attention** ([2502.13189] Lu et al., Moonshot AI/Kimi, 2025): applies MoE-style routing to attention blocks — each query token routes to a top-K subset of the context's blocks. This eliminates the need for hand-crafted sparse patterns; the model learns which blocks are relevant per query. Deployed in production in Kimi's 1M-context system (158 citations). **FlashMoBA** ([2511.11571] Xiao et al., 2025) added a statistical model for MoBA block selection and a CUDA kernel achieving 14.7× speedup over FlashAttention-2 while matching dense quality.

**XAttention** ([2503.16428] Xu et al., MIT Han Lab, 2025; ICML 2025): uses *antidiagonal scoring* to identify sparse blocks — the insight is that the antidiagonal of a query-key block captures both vertical (constant-column) and slash (diagonal) attention patterns simultaneously with a single metric. Achieved 13.5× prefill acceleration on Llama-3.1-8B at 128K context with near full-attention quality (108 citations, hardware-level acceleration for video generation, NLP, VLMs).

**IndexCache** ([2603.12201] Bai et al., 2026): observes that sparse attention indices are highly stable across adjacent transformer layers; reusing them across layers cuts indexer computation by 75%, adding another 1.82× prefill speedup on top of existing sparse methods.

#### 3b. Production Hybrid Linear Architectures

The most important architectural shift of 2025 is the move from "pure Transformer" or "pure SSM" to **hybrid models** that interleave full/sparse softmax attention layers with linear recurrent layers. The key finding: a small number of softmax-attention layers handle exact retrieval tasks that pure-linear models fail on, while linear layers handle recurrent in-context computation cheaply.

**Kimi Linear** ([2510.26692] Kimi Team, 2025): Moonshot AI's production hybrid — interleaves gated linear attention with standard softmax heads. Achieves 6× decoding throughput vs. full attention, 75% KV cache reduction at 1M context, while *outperforming* full attention on standard benchmarks. A production deployment at scale, not just a research prototype (71 citations).

**Samba** ([2406.07522] Ren et al., Microsoft, 2024): Mamba layers + sliding window attention in alternating blocks; 3.73× throughput over Transformer at long sequences, unlimited context extrapolation (153 citations). The archetype of the hybrid pattern.

**Falcon-H1** ([2507.22448] Zuo et al., TII, 2025): hybrid-head language model family (0.5B–34B) combining Transformer attention with SSM. The 34B model matches or outperforms 70B-scale competitors; the 1.5B-Deep rivals 7B–10B models, demonstrating the efficiency of hybrid scaling (39 citations).

**L2A: Learning When to Attend** ([2603.17484] Choudhary et al., 2026): a layer-level gating mechanism that conditionally invokes global softmax attention only when needed — ~80% of tokens skip global attention entirely. Extends effective context from 32K to 128K, ~2× training throughput improvement.

**HyLo** ([2604.24715] Ashrafi Fashi et al., 2026): converts existing pretrained Transformer checkpoints into hybrid Transformer+linear via *upcycling*, extending context by 32× and reducing KV cache by >90% with minimal additional training.

#### 3c. Latest Linear Attention Advances

**RWKV-7 "Goose"** ([2503.14456] Peng et al., 2025): adds *expressive dynamic state evolution* — a token-mixing formulation where the state matrix evolves expressively through a learned update rule, achieving new SOTA for 3B non-attention models on multilingual tasks with O(1) inference memory.

**Softmax Linear Attention (SLA)** ([2602.01744] Xu et al., 2026): restores global competition to linear attention via head-level softmax, improving perplexity and retrieval robustness on LongBench without sacrificing linear complexity.

---

### 4. Synthesis: What Is State of the Art in 2025–2026?

The honest answer is that FlashAttention (dense softmax) has **not been replaced** as infrastructure — FlashAttention-3 and FlashInfer ([2501.01005]) remain the kernels inside every major framework. What has changed is which *attention pattern* those kernels compute:

| Context length | Dominant 2025–2026 approach | Key papers |
|---|---|---|
| ≤32K tokens | Dense softmax + FlashAttention-3 | [2407.08608] |
| 32K–256K | Trainable sparse (NSA) or hybrid (Kimi Linear) | [2502.11089], [2510.26692] |
| 256K–1M+ | Hybrid linear+sparse (Kimi Linear, MoBA, L2A) | [2510.26692], [2502.13189], [2603.17484] |
| Pure efficiency / streaming | Linear/SSM (Gated DeltaNet, RWKV-7) | [2412.06464], [2503.14456] |

**The field consensus as of mid-2025:** pure dense attention above ~64K is unacceptable in new models. NSA ([2502.11089]) demonstrated that *natively training with sparse attention is better than training dense and then compressing* — a decisive break with the "train dense, apply heuristic sparsity at inference" paradigm that dominated 2023–2024. Simultaneously, hybrid linear architectures (Kimi Linear, Samba) showed that interleaving recurrent layers with sparse attention can *exceed* full-attention quality at a fraction of the KV cache cost.

**What remains open:** linear attention still underperforms on tasks requiring hard in-context retrieval without softmax layers. Gated DeltaNet ([2412.06464]) is the best pure-linear result today but still needs hybrid scaffolding in production systems (Kimi Linear, Samba, Falcon-H1) to match dense Transformer quality across all tasks.

---

### Full Citation Index

| arXiv ID | Title |
|---|---|
| [1706.03762] | Attention Is All You Need |
| [1904.10509] | Generating Long Sequences with Sparse Transformers |
| [1911.02150] | Fast Transformer Decoding: One Write-Head is All You Need |
| [2001.04451] | Reformer: The Efficient Transformer |
| [2004.05150] | Longformer: The Long-Document Transformer |
| [2006.16236] | Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention |
| [2007.14062] | Big Bird: Transformers for Longer Sequences |
| [2009.14794] | Rethinking Attention with Performers |
| [2103.02143] | Random Feature Attention |
| [2112.05682] | Self-attention Does Not Need O(n²) Memory |
| [2205.14135] | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| [2305.13048] | RWKV: Reinventing RNNs for the Transformer Era |
| [2305.13245] | GQA: Training Generalized Multi-Query Transformer Models |
| [2306.14048] | H₂O: Heavy-Hitter Oracle for Efficient Generative Inference |
| [2307.02486] | LongNet: Scaling Transformers to 1,000,000,000 Tokens |
| [2307.08621] | Retentive Network: A Successor to Transformer for Large Language Models |
| [2307.08691] | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning |
| [2309.17453] | Efficient Streaming Language Models with Attention Sinks |
| [2310.01889] | Ring Attention with Blockwise Transformers for Near-Infinite Context |
| [2312.06635] | Gated Linear Attention Transformers with Hardware-Efficient Training |
| [2404.07143] | Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention |
| [2404.14469] | SnapKV: LLM Knows What You are Looking for Before Generation |
| [2406.06484] | Parallelizing Linear Transformers with the Delta Rule over Sequence Length |
| [2406.07522] | Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling |
| [2406.10774] | Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference |
| [2407.02490] | MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention |
| [2407.08608] | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision |
| [2410.13276] | SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs |
| [2412.06464] | Gated Delta Networks: Improving Mamba2 with Delta Rule |
| [2501.01005] | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving |
| [2502.11089] | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention |
| [2502.13189] | MoBA: Mixture of Block Attention for Long-Context LLMs |
| [2503.14456] | RWKV-7 "Goose" with Expressive Dynamic State Evolution |
| [2503.16428] | XAttention: Block Sparse Attention with Antidiagonal Scoring |
| [2507.22448] | Falcon-H1: A Family of Hybrid-Head Language Models |
| [2510.26692] | Kimi Linear: An Expressive, Efficient Attention Architecture |
| [2511.11571] | Optimizing Mixture of Block Attention (FlashMoBA) |
| [2602.01744] | Softmax Linear Attention: Reclaiming Global Competition |
| [2603.12201] | IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse |
| [2603.17484] | Learning When to Attend: Conditional Memory Access for Long-Context LLMs (L2A) |
| [2604.24715] | Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling (HyLo) |