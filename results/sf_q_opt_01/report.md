I now have the full picture across the citation graph. Let me write the synthesis report.

---

## Optimizer Lineage for Small LM Pretraining Under Fixed Compute

### (1) The Canonical/Established Approach and Its Key Prior Art

**The established answer — "just use AdamW" — was correct from roughly 2019 through 2024.** Its foundations are:

- **Adam** — Kingma & Ba (2015, arXiv `1412.6980`). Introduced coordinate-wise adaptive learning rates via first- and second-moment estimates, stabilizing training over SGD.
- **AdamW** — Loshchilov & Hutter, *"Decoupled Weight Decay Regularization"* (2017, arXiv `1711.05101`, ~5,200 citations). Decoupled L2 regularization from the gradient update, which is the standard LLM training optimizer today.

Why transformers *specifically* need Adam-style adaptivity is well-characterized: Zhang et al., *"Why Transformers Need Adam: A Hessian Perspective"* (2024, arXiv `2402.16788`, 121 citations) shows that transformer weight matrices exhibit **block Hessian heterogeneity** — different parameter blocks have very different curvature scales — which single-learning-rate SGD cannot handle but Adam's per-coordinate rates manage naturally.

Two 2024 ablation papers confirmed the "just use AdamW" heuristic was reasonable but not unassailable: Zhao et al., *"Deconstructing What Makes a Good Optimizer for Language Models"* (arXiv `2407.07972`, 43 citations) found that Adam, Lion, and Sophia all perform similarly when tuned optimally, with no clear winner — though SGD remained significantly weaker.

---

### (2) How the Method Evolved (The Lineage)

**Pre-cursor preconditioned methods (2018–2023):**

- **Shampoo** — Gupta, Koren & Singer (2018, arXiv `1802.09568`, 439 citations). The original Kronecker-factored full-matrix preconditioner. Showed faster convergence than Adam but costly to run. Used in Gemini-1.5 Flash at scale.
- **Sophia** — Liu et al., *"Sophia: A Scalable Stochastic Second-Order Optimizer for Language Model Pre-training"* (2023, arXiv `2305.14342`, 277 citations). Diagonal Hessian estimate + clipping; claimed 2× compute speedup over Adam for GPT up to 1.5B — a significant early challenge to AdamW's dominance.
- **Lion** — Chen et al., *"Symbolic Discovery of Optimization Algorithms"* (2023, arXiv `2302.06675`, 642 citations). Evolutionary search discovered a sign-based momentum update (cheaper memory than Adam); useful for vision but later benchmarks showed it didn't reliably beat AdamW for language models.

**The SOAP breakthrough (2024):**

Vyas, Morwani et al., *"SOAP: Improving and Stabilizing Shampoo using Adam"* (2024, arXiv `2409.11321`, 170 citations) is the key transitional paper. SOAP establishes that **Shampoo with the ½-power is equivalent to running Adafactor in Shampoo's preconditioner eigenbasis**, which leads to the design of running AdamW directly in that rotated space. Results at 360M–660M parameter scale: **>40% fewer iterations and >35% less wall-clock time vs AdamW**, with only one extra hyperparameter (preconditioning frequency). This was the first reproducible, well-ablated evidence that non-diagonal preconditioners consistently beat AdamW at LM pretraining scale.

**Muon enters (late 2024):**

Keller Jordan's **Muon optimizer** (blog post/GitHub, November 2024; no formal arXiv preprint, cited throughout literature as `jordan2024muon`) applies Nesterov momentum followed by Newton-Schulz iteration to approximate the matrix polar factor (the orthogonal component of the gradient's SVD). The update rule is:

```
M_t = μ·M_{t-1} + ∇L(W_{t-1})
O_t = NewtonSchulz(M_t)   # approximate (M·Mᵀ)^{-1/2}·M  
W_t = W_{t-1} - η·O_t
```

This is **steepest descent under the spectral norm** (equivalently, the nuclear norm), constraining updates to have orthogonal structure. On small NanoGPT-scale experiments this outperformed AdamW clearly.

---

### (3) The 2025–2026 Papers That Supersede "Just Use AdamW"

**The landmark scaling paper:**

Liu et al. (Moonshot AI), *"Muon is Scalable for LLM Training"* (2025, arXiv `2502.16982`, **247 citations**). This is the most important paper in this space. Key contributions:
1. Identified that weight decay and per-parameter update-scale normalization are necessary for Muon to work at billion-parameter scale without re-tuning.
2. Scaling law experiments comparing Muon vs AdamW: **Muon achieves ~2× compute efficiency** in compute-optimal training.
3. Released Moonlight, a 3B/16B MoE model trained on 5.7T tokens with Muon, improving the Pareto frontier versus all comparable prior models.
4. Open-sourced a distributed ZeRO-1-style Muon implementation.

**This directly falsifies "just use AdamW" for compute-optimal small LM pretraining.** The 2× efficiency gain means for a fixed FLOP budget you can either train a better model or train faster.

**Independent benchmarking:**

Semenov, Pagliardini & Jaggi, *"Benchmarking Optimizers for Large Language Model Pretraining"* (2025, arXiv `2509.01440`, 41 citations). A systematic comparison across many optimizers including Muon, D-Muon (the Liu et al. variant), SOAP, AdEMAMix, Lion, Signum. Finds that matrix-based optimizers (Muon/SOAP) are among the leading candidates at smaller scales.

**The important nuance / counterpoint:**

Wen, Hall, Ma & Liang (Stanford), *"Fantastic Pretraining Optimizers and Where to Find Them"* (2025, arXiv `2509.02046`, **61 citations**) delivers the most careful methodological critique: claimed optimizer speedups are **often overestimated due to unfair comparisons** (e.g., under-tuned baselines, mismatched token horizons). Their controlled study finds matrix-based optimizers show only **1.4× speedup for small models and ~1.1× for large models** when comparisons are fair. This paper does not say AdamW wins — it says the gap is smaller than advertised, and that the advantage is scale-dependent (favoring small models more).

**2025–2026 Muon-family successors:**

| Paper | arXiv ID | Year | Key improvement |
|---|---|---|---|
| Preconditioning Benefits of Spectral Orthogonalization in Muon | `2601.13474` | 2026 | Theoretical analysis: linear convergence, condition-number-independent |
| Muon+ (extra normalization step) | `2602.21545` | 2026 | Consistent PPL improvements 60M–1B, up to 200 tokens/param |
| HTMuon (heavy-tailed spectral correction) | `2603.10067` | 2026 | Fixes Muon's suppression of heavy-tailed weight spectra; −0.98 PPL on LLaMA |
| Mousse (curvature-aware preconditioning) | `2603.09697` | 2026 | Adds second-order info to Muon; ~12% fewer training steps (160M–800M) |
| Muon² (adaptive second-moment before orthogonalization) | `2604.09967` | 2026 | 40% fewer Newton-Schulz iterations; beats Muon 60M–1.3B |
| DynMuon (dynamic spectral exponent scheduling) | `2605.17109` | 2026 | Schedules spectral exponent from positive→negative; 10–27% fewer steps to target |
| Why Muon Outperforms Adam: A Curvature Perspective | `2606.04662` | 2026 | Explains advantage via lower Normalized Directional Sharpness |

**SOAP's ongoing development:**

DASH — *"Faster Shampoo via Batched Block Preconditioning and Efficient Inverse-Root Solvers"* (2026, arXiv `2602.02016`) — achieves up to 4.83× faster optimizer steps than standard Shampoo/SOAP, improving both step time and validation perplexity per iteration.

---

### (4) Where "Just Use AdamW" Is Now Wrong

The "just use AdamW" heuristic fails specifically in this regime:

**Small models (≤1B parameters), compute-optimal or compute-constrained pretraining:**

1. **If you can afford Muon's per-step overhead**: Muon (with Liu et al.'s weight-decay + per-parameter scaling fix) achieves ~2× compute efficiency vs AdamW (`2502.16982`), meaning for the same FLOP budget you reach substantially lower loss. The 2× efficiency means you effectively get twice as much "model" for your training budget.

2. **If your matrices are square or nearly square**: Muon's orthogonalization is most effective on weight matrices with balanced dimensions. The Newton-Schulz overhead is O(mn·min(m,n)) per step — RMNP (`2603.20527`) reduces this to O(mn) with row-wise normalization at small cost to quality.

3. **Large-batch or high-parallelism settings**: SOAP (`2409.11321`) is the stronger choice when you have high-frequency preconditioner updates. Its >40% iteration reduction holds in the large-batch regime (2M token batch size) where Muon's advantage is also strong.

**Where AdamW remains competitive or preferred:**

- **Very large models (>3B parameters)**: The Fantastic Pretraining Optimizers paper (`2509.02046`) shows the Muon advantage shrinks to ~1.1× at large scale after fair comparison.
- **1D parameters (embeddings, biases, LayerNorm)**: Muon is typically applied only to matrix parameters; 1D parameters still use AdamW. Hybrid setups (Muon for matrices, AdamW for 1D) are standard.
- **Fine-tuning**: The evidence for Muon over AdamW in fine-tuning is weaker than in pretraining; AdamW with careful hyperparameters remains a strong default here.
- **Simplicity-bias-sensitive tasks**: To Use or Not to Use Muon (`2603.00742`) shows Muon removes a simplicity bias preserved by SGD/Adam, potentially causing more susceptibility to spurious features — a non-obvious cost.

---

### Summary Verdict

For **small LM pretraining under a fixed compute budget in 2025–2026**:

- **Muon (Liu et al. 2025 variant, `2502.16982`)** is the current best-supported choice, with ~2× compute efficiency over AdamW in scaling law experiments at compute-optimal training points, and a strong empirical track record (the Moonlight model).
- **SOAP (`2409.11321`)** is the closest competitor, with >35% wall-clock savings, and is preferred when you have access to a good preconditioner-update schedule.
- **AdamW (`1711.05101`)** remains the default for large models, fine-tuning, and settings where implementation simplicity matters — but for the specific regime of small model pretraining under a FLOP budget, the "just use AdamW" heuristic is demonstrably suboptimal as of 2025.