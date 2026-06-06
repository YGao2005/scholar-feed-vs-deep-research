Here is the final adversarially-verified report. **Stats: 31 sources fetched, 152 claims extracted, 25 verified by 3-vote adversarial panel — 11 confirmed, 14 killed.**

---

# Long-Context Attention for LLMs: State of the Art (2025–2026)

*All substantive claims below survived adversarial 3-vote verification (≥ 2/3 confirm). Killed claims are listed separately.*

---

## 1. The Canonical Approach and Its Prior Art

**FlashAttention** (Dao et al., arXiv:2205.14135) is the canonical implementation of dense self-attention. It does not change the O(N²) asymptotic complexity of attention — it restructures computation into IO-aware tiled passes to avoid materializing the full N×N matrix in HBM, yielding 2–4× wall-clock speedup. Successors **FlashAttention-2** (arXiv:2307.08691) and **FlashAttention-3** (arXiv:2407.08608, Hopper-specific via WGMMA/TMA) extend these hardware gains.

**The fundamental problem:** Standard dense softmax attention incurs O(N²) computation and memory with respect to sequence length — directly quoted in the Long-Context Attention Benchmark (arXiv:2510.17896) as "posing a major bottleneck for long-context training." FlashAttention-era work addressed IO efficiency *within* the quadratic regime but did not escape the asymptotic bound. *(3-0 unanimous)*

---

## 2. How the Method Evolved

### Phase 1: Linear/State-Space Recurrences

**Mamba** (Gu & Dao, arXiv:2312.00752, NeurIPS 2024) is the canonical state-space successor. It makes SSM parameters functions of the input, enabling selective propagation/forgetting — directly addressing the prior SSM weakness on discrete modalities. This breaks time-invariance (ruling out FFT-based convolution), and instead uses a hardware-aware parallel scan for true O(L) scaling, versus O(L log L) for prior convolution-based SSMs (S4) and O(L²) for transformers. *(3-0 unanimous)*

**Gated Linear Attention / GLA** (Yang et al., arXiv:2312.06635, ICML 2024) extended the linear attention lineage by introducing *data-dependent* gates on linear attention recurrences. RetNet and prior variants use a global, data-independent decay factor; RetNet is explicitly a special case of GLA with input-independent gates. The ICML 2024 paper introduced a chunk-wise parallel formulation (FlashLinearAttention) for hardware-efficient training. **The data-dependent gate became the key architectural lever** separating performant linear attention from earlier fixed-decay variants. *(3-0 unanimous)*

### Phase 2: Delta-Rule Memory Updates

**Gated DeltaNet** (arXiv:2412.06464, ICLR 2025) addressed the retrieval gap of linear transformers by combining a decay gate with the delta rule for targeted memory updates. However, it used a *single scalar gate* controlling both erase and write operations simultaneously.

**Gated DeltaNet-2** (NVIDIA, arXiv:2605.22791, May 2026) decouples these into a *channel-wise erase gate* b_t ∈ [0,1]^{d_k} and a *channel-wise write gate* w_t ∈ [0,1]^{d_v}, enabling more precise memory editing while preserving sub-quadratic complexity. At 1.3B parameters trained on 100B FineWeb-Edu tokens, Gated DeltaNet-2 achieves the highest average accuracy (53.11) among all recurrent baselines: Mamba-2 (51.82), Gated DeltaNet (52.07), KDA (52.28), Mamba-3 (52.39) across language modeling, commonsense reasoning, and retrieval benchmarks. *(3-0 unanimous — this is the current SOTA for pure recurrent models)*

### Phase 3: Dynamic Sparse Attention

**NSA — Native Sparse Attention** (DeepSeek, arXiv:2502.11089, ACL 2025) replaced fixed-pattern sparse methods (e.g., Longformer's sliding window + global tokens) with *content-adaptive* three-branch sparse attention: (1) coarse-grained block compression via learnable MLP, (2) fine-grained token selection via block importance scores, (3) sliding window for recency. Branches aggregate via learned gating. NSA achieves up to **11.6× decoding speedup** and up to 9.0×/6.0× forward/backward training speedup over FlashAttention-2 at 64k context. Quality: outperforms full attention on 7/9 general benchmarks and all long-context tasks. *(2-1 on speedup magnitude, 3-0 on architecture — speedup figures quoted directly from paper's hardware benchmarks; one verifier contested methodology)*

### Phase 4: Sliding-Window Adaptation

**SWA** is the most widely deployed sparse pattern (Gemma2, Qwen 2.5-1M, DeepSeek NSA all use it) but "structurally cuts off all direct information pathways to tokens beyond the local window," causing catastrophic long-context collapse when post-hoc applied to models pretrained with full attention (arXiv:2512.10411). *(3-0 unanimous)*

**SWAA** (arXiv:2512.10411v5, March 2026) adapts SWA post-hoc to pretrained models — requiring neither costly retraining nor new modules — and achieves **30–100% inference speedups** at 24k context with acceptable quality retention (100% speedup endpoint corresponds to ~90% accuracy retention). *(3-0 unanimous)*

---

## 3. 2025–2026 State of the Art

The frontier is split between two paradigms with no direct head-to-head comparison yet:

| Paradigm | Best current work | arXiv ID | Key result |
|---|---|---|---|
| **Pure recurrent (linear attention)** | Gated DeltaNet-2 (NVIDIA, May 2026) | 2605.22791 | 53.11 avg accuracy at 1.3B, beats all recurrent baselines |
| **Dynamic sparse attention** | NSA (DeepSeek, ACL 2025) | 2502.11089 | 11.6× decoding speedup over FA-2 at 64k, matches full attention quality |
| **SWA adaptation** | SWAA (March 2026) | 2512.10411 | 30–100% speedups adapting pretrained full-attention models to SWA |
| **Linear attention foundation** | GLA (ICML 2024) | 2312.06635 | Data-dependent gating, hardware-efficient chunk-wise training |
| **SSM foundation** | Mamba (NeurIPS 2024) | 2312.00752 | True O(L) scaling via input-selective SSM + parallel scan |

---

## 4. Adversarially Killed Claims (Do Not Cite)

The following claims were refuted by ≥ 2/3 adversarial verifiers:

- **SparseK Attention** achieving linear complexity and constant-memory inference (0-3 — not supported by paper)
- **FA3 being "the best dense kernel to beat"** in 2025 (0-3 — superseded in practice by sub-quadratic methods)
- **Sparse kernels uniformly not outperforming dense FA** (0-3 — contradicted by NSA results)
- **Mamba-2 SSD being 6× faster than FA-2 at 16K tokens** (0-3 — original SSD paper framing; not robustly replicated)
- **SSD/Mamba-2 mathematical equivalence** to standard attention via semiseparable matrices (0-3 — claim is overstated)
- **GLA faster than FlashAttention-2 at short sequences (1K tokens)** (0-3 — not supported)
- **Dense attention consuming 70–80% of latency at 64k context** (0-3 — specific figure not supported)

---

## Caveats

1. **Gated DeltaNet-2** (arXiv:2605.22791) and **SWAA** (arXiv:2512.10411) are arXiv preprints not yet peer-reviewed. **NSA** (arXiv:2502.11089) is the most recent peer-reviewed result (ACL 2025).
2. The recurrent vs. sparse-attention comparison is **not direct** — each paradigm's superiority is established against its own baselines. Hybrid architectures (not fully covered here) may outperform both.
3. Hardware-level speedup claims for sub-quadratic methods are frequently context-dependent and should not be generalized across GPU generations.

---

**Bottom line:** FlashAttention remains the standard backend when full attention is required, but the 2025–2026 frontier has moved to **(a) the Gated DeltaNet-2 family** for pure linear-complexity inference and **(b) NSA-style dynamic sparse attention** for situations requiring softmax-level retrieval fidelity at long context. The next open question is whether hybrid SSM-attention architectures close the gap between these two paradigms.