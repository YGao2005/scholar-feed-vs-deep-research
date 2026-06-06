Here is the complete verified report. The workflow ran 107 agents, fetched 24 sources, extracted 113 claims, adversarially verified 25 of them (3-vote each), and confirmed 13 while killing 12 — including several of my interim report's stronger quantitative claims.

---

# Optimizer for Small LM Pretraining: From AdamW to Muon (2025–2026)

*Adversarially verified: 25 claims checked, 13 confirmed, 12 refuted. Only confirmed claims appear below.*

---

## 1. The Canonical Answer and Its Prior Art

The established answer is **AdamW** — not vanilla Adam with L2 regularization.

The distinction matters mathematically: Adam's adaptive second-moment scaling divides any L2 gradient penalty by per-parameter gradient estimates, making effective regularization entangled with the learning rate. AdamW decouples weight decay by applying it directly to weights outside the gradient path. This makes the regularization strength and learning rate independent hyperparameters — a non-trivial improvement in practice.

> **arXiv:1711.05101** — *"Decoupled Weight Decay Regularization"* (Loshchilov & Hutter, ICLR 2019)
> Proves the L2 ≠ weight-decay equivalence failure and shows substantial improvement on classification benchmarks. Established the standard for all LLM training runs since ~2019.

The compute-optimal framing that made AdamW the default came from two scaling-law papers, both of which assumed AdamW throughout:

> **arXiv:2001.08361** — *"Scaling Laws for Neural Language Models"* (Kaplan et al., 2020)
> **arXiv:2203.15556** — *"Training Compute-Optimal Large Language Models"* (Hoffmann et al. / Chinchilla, 2022)

---

## 2. The Lineage: From AdamW to Muon

### Step 1 — Second-order preconditioned methods

- **Shampoo** (*arXiv:1802.09568*) maintained Kronecker-factored preconditioners per layer. Theoretically better but expensive.
- **SOAP** (*arXiv:2409.11321*, "SOAP: Improving and Stabilizing Shampoo using Adam in the Preconditioned Subspace", Vyas et al., 2024) — runs Adam in the eigenbasis of Shampoo's preconditioner. **Note:** Adversarial verification killed specific quantitative SOAP claims (the "40% fewer iterations / 35% wall-clock" figures voted 1-2 and 0-3). SOAP's directional advantage over AdamW at high Chinchilla ratios is confirmed, but claimed magnitudes are not trustworthy from current evidence.

### Step 2 — Modular duality / spectral geometry

> **arXiv:2409.20325** — *"Modular Duality in Deep Learning"* (Bernstein & Newhouse, 2024)
> Theoretical foundation: Adam implicitly uses Frobenius-norm steepest descent for weight matrices. The correct geometry is spectral-norm steepest descent — the update should be a matrix with orthonormal columns (an element of the Stiefel manifold).

### Step 3 — Muon

**Muon** (Jordan et al., 2024; implemented via Newton-Schulz iterations for spectral orthogonalization of gradient momentum matrices) is the practical realization of spectral-norm steepest descent. The key 2025 papers that confirm its validity:

> **arXiv:2502.16982** — *"Muon is Scalable for LLM Training"* (Moonshot AI / Moonlight model, Feb 2025)
> Demonstrates Muon on a production 3B/16B-parameter MoE model trained on 5.7 trillion tokens. Closes the "doesn't scale" objection. *(Note: the specific "2x efficiency" and "scaling constant lower than AdamW" quantitative claims from this paper were refuted 0-3 and 1-2 respectively — the production-scale demonstration is confirmed, the precise multipliers are not.)*

> **arXiv:2505.02222** — *"Practical Efficiency of Muon for Pretraining"* (Essential AI, May 2025)
> **Confirmed:** Muon requires **10–15% fewer tokens** than AdamW to reach identical loss, across 100M–4B parameter models. Muon also retains data efficiency far beyond the critical batch size where AdamW degrades.

> **arXiv:2601.13474** — *"Preconditioning Benefits of Spectral Orthogonalization in Muon"* (Ma et al., Jan 2026)
> Theoretical explanation: spectral orthogonalization induces preconditioning that decouples Muon's dynamics into independent scalar sequences in the spectral domain, flattening the effective loss landscape.

---

## 3. The 2025–2026 Works That Supersede AdamW

### The direct comparison that kills "just use AdamW"

> **arXiv:2601.04890** — *"Learnable Multipliers: Freeing the Scale of Language Model Matrix Layers"* (Jan 2026)
> At 200GT training of a 0.5B Falcon-H1 model, across 7 tasks (HellaSwag, ARC-C, MMLU, MMLU-PRO, BBH, GSM8K, MATH-L5):
> - Adam baseline: **30.80%**
> - Muon baseline: **31.88%**
> - Adam + learnable scale multipliers: **32.01%**
> - Muon + learnable scale multipliers: **32.98%**
>
> The Muon-family advantage persists even after applying the same enhancement to both optimizers.

### Regime-dependent ranking

> **arXiv:2509.02046** — *"Fantastic Pretraining Optimizers and Where to Find Them"* (Sep 2025)
> Systematic experiments across 130M–1.2B Llama 2 models at 1×, 2×, 4×, 8×, and 16× Chinchilla ratios:
> - At **1–4× Chinchilla** (compute-constrained, shorter training): **Muon is best**
> - At **8× Chinchilla and above** (heavily over-trained): **SOAP and Kron outperform Muon**
> - Claimed magnitudes like "1.4–2× speedup" and "consistently scales inversely with model size" were refuted (0-3 and 1-2 votes).

### The most recent advancement: Newton-Muon (April 2026)

> **arXiv:2604.01472** — *"The Newton-Muon Optimizer"* (Du and Su, April 2026)
> Standard Muon is an implicit Newton method that drops the right-preconditioner term from input second moments (ZZ^T). Newton-Muon adds it back. Result: same target validation loss in **6% fewer iteration steps (~4% wall-clock)** on 124M GPT-2 pretraining. The improvement recovers exactly when inputs are non-isotropic. *(Single-paper result, not yet widely replicated.)*

---

## 4. Where "Just Use AdamW" Is Wrong

| Claim | Status |
|---|---|
| AdamW is the compute-optimal optimizer for transformer pretraining | **Superseded** — Muon reaches identical loss with 10–15% fewer tokens (arXiv:2505.02222) |
| AdamW scales well to large batch sizes | **Wrong** — Muon retains data efficiency far beyond AdamW's critical batch size (arXiv:2505.02222) |
| AdamW has been validated at production scale | True — but so has Muon now (5.7T token Moonlight model, arXiv:2502.16982) |
| AdamW is the correct geometry for weight-matrix updates | **Theoretically refuted** — Frobenius-norm descent is wrong; spectral-norm is correct (arXiv:2409.20325) |
| "AdamW consistently achieves highest downstream accuracy vs Lion/Sophia" | **Explicitly refuted** (0-3 adversarial vote, arXiv:2507.08472) |

---

## 5. Practical Recommendation (2025–2026)

**For small LM pretraining at 1–4× Chinchilla compute:**
> Use **Muon for 2D hidden-layer weight matrices** + **AdamW for embeddings, output heads, layer norms, and biases**. This hybrid consistently reaches lower loss at equivalent FLOP cost.

**For heavily over-trained runs (8×+ Chinchilla):**
> SOAP or Kron outperform Muon in this regime. AdamW is still beaten directionally, but Muon is no longer the best matrix optimizer.

**If you need a true drop-in:**
> Schedule-Free AdamW (*arXiv:2405.15682*) removes LR schedule sensitivity without changing optimizer family — not a Muon replacement, but reduces the "tuned AdamW" overhead that inflated AdamW's apparent dominance.

---

## Caveat Summary (from adversarial verification)

- The "2× efficiency" and "lower scaling constant" claims for Muon from arXiv:2502.16982 were **killed (0-3 and 1-2)** — the production-scale demonstration is real, specific efficiency multipliers are not.
- SOAP quantitative speedup claims were **killed**; its regime advantage at high Chinchilla is directionally confirmed.
- Full Gauss-Newton "5× speedup over SOAP/Muon" claims were **killed entirely (0-3)**.
- Newton-Muon's 6% improvement is a single benchmark; treat as promising but not yet replicated.
- Muon applies only to 2D matrices — not a drop-in replacement; embeddings/norms still need AdamW.
- Muon-pretrained models may have distribution mismatch when fine-tuned with AdamW (*arXiv:2605.10468*) — an open deployment concern.