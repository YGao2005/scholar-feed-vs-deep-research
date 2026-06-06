I now have a comprehensive picture across all phases. Let me synthesize the full lineage report.

---

# KV-Cache Compression and Eviction for Efficient LLM Inference: Lineage Report

---

## 1. Canonical Anchor and Key Prior Art

### Architecture-Level Foundations (the "reduce before evict" axis)

The very first reduction in KV footprint happened at the model architecture level, before eviction existed as a concept. **Multi-Query Attention (MQA)** — introduced in *"Fast Transformer Decoding: One Write-Head is All You Need"* [arXiv:1911.02150, Shazeer 2019] — collapsed all attention heads to share a single KV head, slashing KV memory linearly with head count. **Grouped Query Attention (GQA)** [arXiv:2305.13245, Ainslie et al. 2023, 1,480 citations] generalised this to groups of heads sharing KV pairs, making the trade-off tunable; GQA is now the default in every major open-weight family (LLaMA-2/3, Mistral, Qwen). These are not eviction methods but they define the baseline KV budget that later eviction work operates on.

The system context was set by *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU* [arXiv:2303.06865, Sheng et al. 2023, 730 citations], which treated the KV cache as a first-class memory-management object (offloading, recompute vs. keep trade-offs) and implicitly framed the question every subsequent eviction paper answers: *which KV pairs are worth keeping in fast memory?*

### The Three Classical Eviction/Compression Anchors (2023)

Three papers, all from mid-2023, constitute the canonical eviction baseline every downstream paper compares against:

**H2O — Heavy-Hitter Oracle** [arXiv:2306.14048, Zhang et al. 2023, 741 citations] is the most-cited pure-eviction method. Its core insight is the *heavy-hitter* phenomenon: a small subset of tokens (~20%) accumulate disproportionate cumulative attention mass across all layers and queries. H2O keeps a fixed-size cache of these heavy-hitters plus the most recent tokens, evicting everything else greedily. It demonstrated up to 29× throughput gain over DeepSpeed with 5–10× KV memory reduction at comparable accuracy on OPT and LLaMA.

**Scissorhands** [arXiv:2305.17118, Liu et al. 2023, 427 citations] formalised the *Persistence of Importance Hypothesis*: tokens that are important now will remain important in future decoding steps. This justified evicting tokens with low past-attention score, achieving up to 5× KV compression with minimal quality loss.

**StreamingLLM — Efficient Streaming Language Models with Attention Sinks** [arXiv:2309.17453, Xiao et al. 2023, 1,912 citations] identified a different phenomenon: the initial tokens of any sequence act as *attention sinks* — they accumulate anomalously high attention regardless of content because the softmax must sum to 1. Any eviction policy that drops these sinks causes catastrophic perplexity spikes. StreamingLLM's solution: always keep the first 4 tokens plus a rolling window, enabling infinite-context streaming without fine-tuning. This "sink + window" insight is now a hard constraint embedded in almost every subsequent eviction method.

Concurrent with these, **FASTGEN / "Model Tells You What to Discard"** [arXiv:2310.01801, Ge et al. 2023, 465 citations] proposed that different attention heads exhibit different structural patterns (heavy local attention, heavy punctuation attention, etc.) and that a profile-guided, head-adaptive policy outperforms the one-size-fits-all approach of H2O.

---

## 2. Evolution of the Method

### Phase 1 — Quantization as a Parallel Track (early 2024)

While eviction-based methods kept or dropped whole tokens, the quantization track compressed the *precision* of every token's stored vectors:

- **KVQuant** [arXiv:2401.18079, Hooper et al. 2024, NeurIPS 2024, 536 citations]: Non-uniform quantization (per-channel, per-vector) of keys and values to 3–4 bits with <0.1 perplexity increase; claimed to support 10M-context inference on 8 GPUs.
- **KIVI** [arXiv:2402.02750, Liu et al. 2024, ICML 2024, 497 citations]: Asymmetric 2-bit quantization — keys quantised per-channel (stationary statistics), values per-token (volatile statistics) — achieving 2.6× memory reduction and 2.35–3.47× throughput gain with no fine-tuning.
- **No Token Left Behind** [arXiv:2402.18096, Yang et al. 2024, 89 citations]: Mixed-precision hybrid — "evicted" tokens survive at very low bit-width rather than being fully dropped, recovering accuracy at aggressive compression ratios.

### Phase 2 — Prompt-Aware Eviction; the SnapKV Paradigm (mid 2024)

H2O and Scissorhands score tokens *online during decoding*, meaning they discard tokens before the model has seen the full query. **SnapKV** [arXiv:2404.14469, Li et al. 2024, NeurIPS 2024, 650 citations] broke this constraint by using an *observation window* at the end of the prompt (the instructions/question itself) to vote on which context tokens matter *before generation begins*. Tokens with high attention from this window are kept; the rest are dropped. SnapKV achieved 3.6× speed and 8.2× memory improvement on 16K inputs and became the de-facto baseline for all subsequent work.

**PyramidKV** [arXiv:2406.02069, Cai et al. 2024, 305 citations] and **PyramidInfer** [arXiv:2405.12532, Yang et al. 2024, 146 citations] added a layer-wise insight: information funnels through layers, so lower layers need larger KV budgets while upper layers can operate with far fewer tokens. Both use a pyramidal allocation scheme.

**Quest** [arXiv:2406.10774, Tang et al. 2024, 368 citations] introduced fully query-aware *per-decoding-step* selection: rather than deciding at prefill time, Quest retrieves a different sparse KV set for each new query token using min-max key statistics per page, avoiding the stale-importance problem inherent in offline eviction.

**Ada-KV** [arXiv:2407.11550, Feng et al. 2024, 167 citations] showed that uniform per-head KV budgets are suboptimal — some heads are more important than others — and formulated budget allocation as an optimisation problem, yielding substantial quality gains across 29 datasets.

**ThinK** [arXiv:2407.21018, Xu et al. 2024, 54 citations] added a *channel* (feature-dimension) pruning axis to the K cache rather than only token-level pruning, reducing KV memory >20% beyond token-level methods.

### Phase 3 — Head Differentiation and System Integration (late 2024)

**DuoAttention** [arXiv:2410.10819, Xiao et al. 2024, ICLR 2025, 243 citations] formalised FASTGEN's head-profile observation into a principled two-class taxonomy: *retrieval heads* need full KV cache (they do long-range lookup); *streaming heads* only need attention sinks + a recent window. DuoAttention identifies this split at a one-time calibration cost and then applies KV compression only to streaming heads, delivering 2.55× memory reduction and 2.18× decode speedup.

**ShadowKV** [arXiv:2410.21465, Sun et al. 2024, ICML 2025, 95 citations] tackled the GPU-RAM bottleneck differently: compress keys with SVD for GPU storage, offload full values to CPU, and prefetch the dynamically selected sparse KV pairs needed per step. Achieved 3.04× throughput on A100 for long-context inference.

**MInference** [arXiv:2407.02490, Jiang et al. 2024, 361 citations] targeted the *prefill* phase (which prior work mostly ignored) with three dynamic sparse attention patterns (A-shape, vertical-slash, block) achieving up to 10× prefill speedup.

**InfiniGen** [arXiv:2406.19707, Lee et al. 2024, 254 citations] combined speculative computation with KV sparsity: a lightweight draft pass predicts which KV pairs the full model will attend to, pre-fetching only those from CPU memory.

**Palu** [arXiv:2407.21118, Chang et al. 2024, 30 citations] applied low-rank projection to the KV cache — decomposing head dimensions to halve memory with up to 1.89× attention speedup.

**Dynamic Memory Compression (DMC)** [arXiv:2403.09636, Nawrot et al. 2024, 114 citations] took a training-time approach: fine-tuning LLMs to learn to *merge* consecutive KV pairs, producing a compressed sequence rather than evicting.

**L₂-Norm Importance Score** [arXiv:2406.11430, Devoto et al. 2024, 73 citations] showed that the L₂ norm of key vectors is a simple, hardware-cheap, and surprisingly competitive importance proxy — it requires no attention computation and has become a reference baseline.

**Cross-Layer Attention (CLA)** [arXiv:2405.12981, Brandon et al. 2024, 115 citations] and **MiniCache** [arXiv:2405.14366, Liu et al. 2024, 122 citations] explored sharing or merging KV caches across adjacent layers, exploiting high cross-layer similarity in deep layers.

### Phase 4 — Query-Agnostic Compression; Bridging Eviction and Quantization (2025)

**KVzip** [arXiv:2505.23416, Kim et al. 2025, 34 citations, novelty 0.85] is the highest-novelty 2025 eviction paper in the corpus. Its key insight: SnapKV and Quest score KV importance relative to *specific queries*, which is brittle in multi-turn and reuse scenarios. KVzip instead uses a *context reconstruction* objective — it treats the full context as a set of self-queries and compresses based on which tokens are reconstructable with low error, making the eviction *query-agnostic* and reusable across sessions. It achieves 3–4× size reduction and ~2× decode latency improvement with minimal loss.

**LAVa** [arXiv:2509.09754, Shen et al. 2025, EMNLP 2025, 9 citations, novelty 0.70] unifies token selection, head selection, and cross-layer budget allocation into one framework with dynamic allocation during inference, outperforming SnapKV and Ada-KV on LongBench and Needle-in-a-Haystack.

**SpeCache** [arXiv:2503.16163, Jie et al. 2025, ICML 2025, 5 citations] introduced *speculative KV caching*: offload the full KV cache to CPU, then predict which pairs will be needed in the next step using a lightweight draft model, prefetching them before they are requested — achieving up to 10× VRAM reduction.

**EVICPRESS** [arXiv:2512.14946, Feng et al. 2025, 3 citations] tackled the under-studied joint eviction+compression scheduling problem: when memory is at risk of overflow in a serving system, it jointly decides *which* KV pairs to evict vs. *how much* to quantise remaining ones, resulting in 2.19× TTFT speedup.

---

## 3. Most Recent Work (2025–2026) That Supersedes or Substantially Improves the Established Answer

The current frontier moves along four orthogonal axes simultaneously:

### 3a. Rate-Distortion Joint Optimisation

**RDKV: Rate-Distortion Bit Allocation for Joint Eviction and Quantization** [arXiv:2605.08317, Zhang et al. 2026, novelty 0.85] is the most principled recent unification. Rather than treating eviction and quantisation as sequential decisions, RDKV frames the entire compression problem as a rate-distortion optimisation: given a total bit budget, jointly decide *which tokens to drop* and *at what precision to store the survivors*, solving for the Pareto-optimal allocation per head. On LongBench it achieves 9.1% average improvement over SnapKV/Ada-KV baselines and **recovers 97.81% of full-attention accuracy retaining only 2.48% of the cache**, with 4.5× decode speedup and 1.9× memory reduction. This directly supersedes the common practice of running eviction and quantisation as independent post-hoc steps.

### 3b. Closing the Directional Gap

**MomentKV: Closing the Directional Gap in KV Cache Eviction for Long-Context Inference** [arXiv:2606.01563, Li et al. 2026, novelty 0.75] identifies a structural flaw in all attention-score–based eviction methods: when tokens are dropped, the surviving KV set no longer represents the original attention distribution — there is a *directional mismatch* (the mean of retained vectors shifts). MomentKV computes first-order moment statistics for evicted tokens and uses them to apply a closed-form correction to attention outputs, compensating for this bias without storing the evicted pairs. It outperforms all baselines on LongBench and RULER at every budget, with the largest gains under aggressive compression.

### 3c. Gated, Efficient Query-Agnostic Eviction

**Fast KVzip** [arXiv:2601.17668, Kim et al. 2026, novelty 0.70] follows KVzip (2505.23416) with a computationally lighter gating-based eviction mechanism, reaching up to 70% KV reduction with near-lossless performance and eliminating the context reconstruction overhead of the original KVzip.

### 3d. Million-Token-Scale Retrieval

**ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs** [arXiv:2602.07721, Qi et al. 2026, Harvard/Microsoft Research, novelty 0.85] targets the regime where the KV cache is too large even for CPU offload — million-token contexts. It builds a retrieval index over the KV cache that is robust to the *distribution drift* problem (query embeddings change over decoding, so ANN indices built at prefill become stale). ParisKV matches or exceeds full-attention throughput at batch size 1, achieving 2.8× higher throughput and reducing decode latency by 17× vs. MagicPIG and 44× vs. PQCache at million-token scale.

### 3e. Reasoning-Model-Specific Eviction

**VaSE: Value-Aware Stochastic KV Cache Eviction for Reasoning Models** [arXiv:2606.03928, Chang et al. 2026] targets the qualitatively different workload of chain-of-thought reasoning models (DeepSeek-R1, o3-style), where standard selection-based eviction degrades accuracy badly. VaSE replaces deterministic top-k selection with stochastic sampling weighted by attention scores, achieving **4× compression with higher accuracy than selection-based methods** and outperforming the strongest deterministic eviction by >4% across six reasoning benchmarks.

### 3f. Beyond Per-Vector Quantisation

**FibQuant** [arXiv:2605.11478, Lee & Kim 2026, novelty 0.75] and **VQKV** [arXiv:2603.16435, Wang et al. 2026] apply vector quantisation (codebook-based) rather than scalar bit-reduction. FibQuant achieves 5× compression at 0.99 cosine similarity (to full-precision attention), and is within 0.10 perplexity of FP16 at 4× compression while keeping full random-access capability — important for systems that need to retrieve arbitrary KV pairs at decode time.

---

## 4. Consolidated Lineage Summary

| Era | Representative Papers | Key Advance |
|-----|----------------------|-------------|
| Pre-eviction (2019–2022) | MQA [1911.02150], GQA [2305.13245], FlexGen [2303.06865] | Reduce KV count at model design time; treat KV as memory object |
| Classical eviction (mid-2023) | H2O [2306.14048], Scissorhands [2305.17118], StreamingLLM [2309.17453], FASTGEN [2310.01801] | Attention-score eviction; attention-sink invariant; head profiling |
| Quantisation track (early 2024) | KVQuant [2401.18079], KIVI [2402.02750] | Sub-4-bit lossless-ish quantisation of KV vectors |
| Prompt-aware eviction (mid 2024) | SnapKV [2404.14469], Quest [2406.10774], PyramidKV [2406.02069] | Prefill-time importance scoring; query-driven selection; layer budget pyramids |
| Head & layer differentiation (late 2024) | DuoAttention [2410.10819], Ada-KV [2407.11550], ShadowKV [2410.21465] | Head taxonomy (retrieval vs. streaming); adaptive cross-head budgeting; CPU offload |
| Joint & query-agnostic (2025) | KVzip [2505.23416], LAVa [2509.09754], EVICPRESS [2512.14946] | Query-agnostic scoring via reconstruction; unified dynamic allocation; joint eviction+compression scheduling |
| Frontier (2026) | RDKV [2605.08317], MomentKV [2606.01563], ParisKV [2602.07721], Fast KVzip [2601.17668], VaSE [2606.03928], FibQuant [2605.11478] | Rate-distortion joint optimisation; directional-gap correction; drift-robust million-token retrieval; reasoning-model stochastic eviction; vector quantisation |

---

**The settled consensus** is that any production KV eviction system must (a) always preserve attention sinks [2309.17453], (b) use head-adaptive rather than uniform budgets [2407.11550, 2410.10819], and (c) combine token-level eviction with quantisation of survivors rather than running them independently [2402.18096, 2605.08317]. **The actively contested frontier** is whether to score importance *at prefill* (SnapKV paradigm, fast but query-specific) vs. *per decoding step* (Quest paradigm, accurate but costly) vs. *query-agnostic via reconstruction* (KVzip paradigm, reusable across queries but requires a calibration forward pass). **The most significant 2026 shift** is RDKV's rate-distortion unification, which provides a theoretically grounded answer to the "how do we allocate bits across tokens and layers simultaneously" question that prior work answered with heuristics.