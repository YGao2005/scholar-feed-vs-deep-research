# Corpus MCP vs. web deep-research: a small head-to-head

A reproducible comparison of two ways an AI agent can do literature research on the
same question:

- **Web deep-research** — Claude Code's built-in `/deep-research`, a ~100-agent web
  fan-out (scope → parallel search → fetch → adversarial 3-vote verify → synthesize).
- **Corpus MCP** — the [Scholar Feed](https://www.scholarfeed.org) MCP server, which
  gives the agent a curated corpus of 600k+ CS/AI/ML arXiv papers plus a citation
  graph, foundational-lineage lookup, and full-text extraction. The agent runs a
  corpus loop: search → trace lineage → walk forward citations → read full text.

Both were given the **same prompt**, explicitly triggered (so this measures output
quality, not which tool an agent picks on its own), on three depth-shaped CS/AI/ML
questions. A **blind judge** then scored the two reports per question with the tool
labels stripped and the order randomized.

> **Honest scope first:** this is a small study — **3 questions, one run each, a
> single model (Claude Sonnet 4.6)**, on questions chosen in fast-moving CS/AI/ML
> areas where a citation graph has the most to offer. It is not a large benchmark.
> The cost and citation-grounding results are mechanical counts and robust; the
> quality result is one blind judge's preference. Read it as a directional signal
> plus a reproducible harness, not a settled benchmark. All raw outputs are in
> [`results/`](results/) so you can check the claims yourself.

## Results (3 questions, blind-judged)

| | Corpus MCP (Scholar Feed) | Web deep-research (built-in) |
|---|---|---|
| Blind-judge preference | **won 3 / 3** (two high-confidence, one medium) | won 0 / 3 |
| Mean cost per question | **$0.63** | $26.60 |
| Mean wall-clock per question | **214 s** | 890 s |
| Distinct real arXiv IDs cited (across 3) | **100** | 38 |
| Fabricated (nonexistent) IDs | 0 | 0 |
| Rubric means 1–5 (rec / depth / cov / grnd / corr) | **5.0 / 5.0 / 5.0 / 5.0 / 4.0** | 3.7 / 3.3 / 3.3 / 4.3 / 3.0 |

Per-question blind winners: optimizer selection (corpus, medium), long-context
attention (corpus, high), KV-cache compression (corpus, high). Full numbers in
[`results/_summary.json`](results/_summary.json); the judge's de-blinded verdicts and
rationales are in [`results/_judge/`](results/_judge/).

## What the numbers mean

**Cost / speed.** The built-in's quality comes from brute parallelism — ~100
sub-agents, dozens of fetches, a verification panel — which is why it runs ~$27 and
~15 min per question. The corpus loop reaches a comparable-or-deeper answer with a
single agent making ~15 citation-graph-aware tool calls.

**Recency.** The corpus reports cited about twice as many genuinely recent (2025–2026)
papers per question. The mechanism is the citation graph: starting from a canonical
paper and walking forward to the newer work that cites it surfaces recent results a
model can't recall from training and a keyword search ranks poorly.

**Grounding.** Neither tool invented nonexistent arXiv IDs. (Scholar Feed shows one
"unresolved" ID in the raw verification, `2112.05682` — that is a transient arXiv-API
false negative; the paper is real and correctly cited, so corrected both arms are at
zero fabricated.) The difference is in *binding*: does the ID match the title and date
attached to it? The corpus copies the ID, title, and year out of a tool result, so it
can't misbind unless it mistypes — zero binding errors across 100+ citations. The web
pipeline reconstructs citations from memory and snippets and slipped a few times
(e.g. citing one ID under the wrong paper's title; dating a December-2025 paper "late
2024"). The blind judge caught one such date error on its own.

**Where the built-in wins.** It runs an adversarial verification pass the corpus does
not — and it correctly flagged overstated speedup magnitudes that the corpus report
repeated without checking. It also works on any topic, while the corpus only knows
CS/AI/ML papers. The honest read is that the two are complementary: use the built-in
for breadth and claim-verification on any subject; use the corpus for cheap, current,
correctly-cited CS/AI/ML literature research.

## Reproduce it

The harness is [`skill_vs_dr.py`](skill_vs_dr.py). It shells out to the `claude` CLI
once per (arm, question), captures each final report + a transcript, then runs a blind
judge and verifies cited IDs.

```bash
# Web arm (built-in deep-research; no MCP):
python3 skill_vs_dr.py run --arm dr --q all --watch

# Corpus arm (Scholar Feed MCP; no web). Needs the scholar-feed MCP wired to the
# claude CLI — see https://www.npmjs.com/package/scholar-feed-mcp (npx scholar-feed-mcp init):
python3 skill_vs_dr.py run --arm sf --q all --watch

python3 skill_vs_dr.py judge --q q_opt,q_attn,q_kv   # blind pairwise + rubric
python3 skill_vs_dr.py verify                          # arXiv ID existence check
python3 skill_vs_dr.py aggregate                       # win-rate + cost/coverage table
```

Requirements: the [`claude` CLI](https://www.npmjs.com/package/@anthropic-ai/claude-code),
Python 3, and (for the corpus arm) the
[`scholar-feed-mcp`](https://www.npmjs.com/package/scholar-feed-mcp) server. Only the
Python standard library is used. The questions are defined at the top of
`skill_vs_dr.py`; swap in your own and rerun.

## What's in `results/`

Per `(arm, question)` directory (`dr_*` = web, `sf_*` = corpus): `report.md` (the
final synthesized report), `run_meta.json` (cost, wall-clock, tool calls), and
`ids.json` (every cited arXiv ID + whether it resolves). `_judge/` holds the blind
verdicts; `_summary.json` is the aggregate; `_id_verify.json` is the full ID-existence
log.

## Context

Write-up: <https://www.scholarfeed.org/compare/deep-research-vs-scholar-feed> ·
Scholar Feed MCP: <https://www.npmjs.com/package/scholar-feed-mcp> ·
source: <https://github.com/YGao2005/scholar-feed-mcp>. MIT licensed.
