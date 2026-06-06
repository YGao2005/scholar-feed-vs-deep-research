The workflow completed: 107 agents, 25 sources, 124 claims extracted, 25 adversarially verified (13 confirmed, 12 killed). Here is the full report:

---

# KV-Cache Compression & Eviction for Efficient LLM Inference: Lineage to Frontier

*107-agent deep research · 25 sources · 25 claims adversarially verified (13 confirmed, 12 killed)*

---

## 1. Canonical Anchor Methods

The field crystallized around two orthogonal axes: **token eviction** (drop KV states of unimportant tokens) and **quantization** (reduce bit-width of retained states). A third axis — **low-rank decomposition** — emerged in 2024–2025.

### Eviction: H2O (NeurIPS 2023)

**arXiv:2306.14048 — "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"**

H2O is the canonical citation anchor for score-based eviction. It observes that attention scores are highly non-uniform: a small subset of tokens accumulates disproportionately high cumulative attention mass (the "heavy hitters"). H2O retains a fixed-size KV budget holding the top-scoring tokens, evicting the rest greedily. On OPT-6.7B/30B it demonstrated **up to 29x throughput improvement** over DeepSpeed/HuggingFace Accelerate by enabling batch sizes ~100× larger. *(Vote: 2-1 on the throughput figure; core mechanism is field consensus.)*

> **Caveat:** The claim that retaining only 20% of the KV cache is sufficient for competitive quality was **refuted** (0-3 vote) — the threshold is task- and model-dependent.

### Attention Sinks: StreamingLLM (ICLR 2024)

**arXiv:2309.17453 — "Efficient Streaming Language Models with Attention Sinks"**

StreamingLLM coined "attention sinks": LLMs trained with softmax attention assign disproportionately large scores to initial tokens regardless of semantic relevance. A pure sliding window without sink tokens causes catastrophic perplexity collapse (~5158 on PG19/Llama-2-13B); retaining 4 initial sink tokens alongside a recent-token window recovers perplexity to ~5.40. *(Vote: 3-0 on both the phenomenon and the recovery result.)*

This is the complementary prior art to H2O — eviction methods that ignore initial tokens fail even when retaining high-attention tokens.

> **Caveat:** StreamingLLM does **not** extend effective context length; it only enables unbounded streaming at fixed memory. The claim of supporting 4M-token inference was **refuted** (0-3 vote).

---

## 2. Evolution of the Methods

### 2a. Eviction Track

| Year | Paper | Key Advance |
|------|-------|-------------|
| 2023 | **H2O** arXiv:2306.14048 | Score-based greedy eviction; heavy-hitter oracle |
| 2023 | **StreamingLLM** arXiv:2309.17453 | Attention sinks; initial-token retention |
| 2024 | **PyramidKV** arXiv:2406.02069 | Non-uniform layer-wise budget allocation |
| 2025 | **RocketKV** arXiv:2502.14051 | Two-stage permanent eviction + dynamic top-k |

**PyramidKV** (arXiv:2406.02069) introduced the insight that attention patterns differ systematically by layer depth. Lower layers show diffuse attention (many tokens compete), while higher layers concentrate on fewer. PyramidKV allocates larger KV budgets to lower layers and smaller budgets to higher layers via a linear schedule. *(Vote: 3-0 on the non-uniform allocation mechanism; the quantitative "12% retention matches full performance" claim was refuted 0-3.)*

### 2b. Quantization Track

| Year | Paper | Key Advance |
|------|-------|-------------|
| 2024 | **KVQuant** arXiv:2401.18079 | Extreme-context sub-4-bit; 1M-token LLaMA-7B on single A100 |
| 2024 | **KIVI** arXiv:2402.02750 | Asymmetric per-channel (keys) / per-token (values) quantization |
| 2024 | **GEAR** arXiv:2403.05527 | Three-component hybrid: quant + low-rank residual + sparse outliers |

**KIVI** (arXiv:2402.02750, ICML 2024) identified the structural asymmetry between key and value caches: the key cache has channel-wise outliers (fixed channels with large magnitudes), making per-channel grouping effective; the value cache lacks this pattern and benefits from per-token quantization. KIVI achieves 2-bit compression with **2.6x peak memory reduction** while maintaining nearly equivalent output quality on Llama, Falcon, and Mistral. *(Vote: 3-0 on the asymmetry insight; 2-1 on the 2.6x figure.)*

> 2025 work (PolarQuant arXiv:2502.00527; RotateKV arXiv:2501.16383) explores alternative rotation-based key quantization strategies but does not refute the underlying distribution asymmetry.

**KVQuant** (arXiv:2401.18079, NeurIPS 2024) pushed to extreme context lengths via sub-4-bit quantization, claiming LLaMA-7B serving at **1M-token context on a single A100-80GB** GPU. *(Vote: 2-1; requires batch size 1, nuq2 precision, and carries 18-30% dequantization overhead. The 10M-context title claim was refuted 0-3.)*

**GEAR** (arXiv:2403.05527, NeurIPS 2024) introduced a three-component decomposition: *(a)* uniformly quantized low-precision matrix for typical entries, *(b)* a low-rank matrix approximating quantization residuals, *(c)* a sparse correction matrix for outliers. The three-component design targets near-lossless compression at sub-4-bit precision where pure quantization or pure low-rank individually fail. *(Vote: 3-0 on the design; throughput and memory multiplier claims were refuted.)*

### 2c. Low-Rank Decomposition Track (Emerging 2024–2025)

**KQ-SVD** (arXiv:2512.05916, late 2024) identified a fundamental flaw in prior low-rank KV compression: applying SVD to keys alone (K-SVD) or embedding queries and keys jointly (EigenAttention) is suboptimal because attention depends on the inner product QK^T, not on K or Q individually. KQ-SVD provides:
- **Theorem 1**: K-SVD error strictly exceeds KQ-SVD error unless K's top singular subspace coincidentally aligns with the top subspace of KQ^T.
- **Theorem 2**: Optimal solution A* = K⁺ Û, B* = K^T Û where Û = top-R left singular vectors of KQ^T, with O(Td²) complexity.

*(Vote: 3-0 on the suboptimality critique and theorem statements; the complexity figure was 1-2 contested due to amortization questions.)*

**ReCalKV** (arXiv:2505.24357, 2025) builds on this, attempting to handle keys and values with distinct compression strategies. *(Vote: 1-2 — performance claims against the Palu baseline were not sufficiently corroborated to confirm.)*

---

## 3. Current Frontier (2025–2026)

### RocketKV — Two-Stage Eviction (ICML 2025)

**arXiv:2502.14051 — "RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression"**

RocketKV is the current high-water mark for token eviction. It identified empirically that single-stage eviction methods (SnapKV, H2O, Quest, SparQ) all fail to accurately predict top-k KV tokens at budgets below 1024 tokens. Its two-stage pipeline:
1. **Stage 1 (permanent eviction):** Coarse-budget eviction using attention statistics at prefill
2. **Stage 2 (dynamic top-k selection):** Fine-grained per-query selection on the Stage 1 survivors at decode time

Results on Llama3.1-8B-Instruct:
- **100% accuracy on NIAH** at a 256-token budget (≈426x compression ratio vs. 109K context)
- **Up to 3.7x decode speedup** (batch size 1, FP16, A100)
- **Up to 32.6% peak memory savings**
- Negligible accuracy loss on LongBench

*(Vote: 3-0 on NIAH result and two-stage design motivation; 2-1 on "prior methods fail" claim.)*

> **Caveat:** The 3.7x speedup is an "up to" peak at the lowest budgets; a 1.1% average LongBench drop is present at 256 tokens; multi-turn tasks required a separate RocketKV-MT variant.

### Additional 2025–2026 Frontier Work

- **RotateKV** (arXiv:2501.16383): Rotation-based key transformation to reduce outliers before quantization, complementing KIVI's asymmetric strategy.
- **PolarQuant** (arXiv:2502.00527): Polar decomposition of key matrices as an alternative to per-channel quantization.
- **"Don't be so Stief"** (arXiv:2601.21686): Cites KQ-SVD; addresses amortization of the O(Td²) cost for practical deployment.
- Attention-sink analysis follow-up (arXiv:2410.10781, ICLR 2025): Confirmed sinks emerge universally in softmax LLMs trained with standard procedures.

---

## 4. Full Citation Register

| Paper | arXiv ID | Venue | Role |
|-------|----------|-------|------|
| H2O: Heavy-Hitter Oracle for Efficient Generative Inference | 2306.14048 | NeurIPS 2023 | Canonical eviction anchor |
| Efficient Streaming Language Models with Attention Sinks | 2309.17453 | ICLR 2024 | Attention sink; streaming complement |
| KVQuant: Towards 10M Context LLM Inference with KV Cache Quantization | 2401.18079 | NeurIPS 2024 | Extreme-context quantization |
| KIVI: A Plug-and-Play 2bit KV Cache Quantization by Identifying Key Influential Tokens | 2402.02750 | ICML 2024 | Asymmetric K/V quantization |
| GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference | 2403.05527 | NeurIPS 2024 | Hybrid quant+low-rank+sparse |
| PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling | 2406.02069 | 2024 | Non-uniform layer allocation |
| KQ-SVD: Optimal Low-Rank Approximation of KV Cache via KQ^T Decomposition | 2512.05916 | Late 2024 | Low-rank track anchor |
| RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression | 2502.14051 | ICML 2025 | Current eviction frontier |
| RotateKV | 2501.16383 | 2025 | Rotation-based key quantization |
| PolarQuant | 2502.00527 | 2025 | Polar-decomposition key quantization |
| ReCalKV | 2505.24357 | 2025 | Low-rank K/V asymmetric compression |

---

## Open Questions

1. **Eviction vs. quantization at matched memory:** How do RocketKV-style methods compare against KIVI/KVQuant/GEAR at equal memory budgets on long-context reasoning tasks, and is there a dominant strategy or are they complementary?
2. **KQ-SVD practical cost:** KQ-SVD's O(Td²) complexity per forward pass raises inference latency questions; do ReCalKV (arXiv:2505.24357) and "Don't be so Stief" (arXiv:2601.21686) successfully amortize this?
3. **Post-softmax architectures:** The attention sink phenomenon is softmax-specific; as sigmoid-attention and linear-attention variants become more prevalent, do score-based eviction heuristics remain valid?
4. **Layer/head heterogeneity:** PyramidKV's layer-wise allocation was validated mechanistically but its quantitative claims were refuted — are there principled budget-setting methods that provably beat uniform allocation while accounting for head-level heterogeneity?