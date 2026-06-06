#!/usr/bin/env python3
"""Skill-vs-deep-research driver.

Clean head-to-head: does our CORPUS deep-research (scholar-feed skill + MCP) beat
the Claude Code built-in WEB deep-research, when BOTH are explicitly triggered on
the SAME question? Triggering is removed as a variable; this measures OUTPUT
quality only. See HANDOFF-skill-vs-deepresearch.md.

Arms:
  dr  -> built-in `/deep-research` (web fan-out workflow). No MCP, no corpus.
  sf  -> our `/scholar-feed` skill over the local MCP build. Pure corpus, no web.

Subcommands:
  run       run one (arm, question, rep); capture report + transcript + meta
  judge     blind pairwise + rubric over each sf/dr report pair (per question/rep)
  verify    extract arXiv ids from every report, check they resolve (hallucination)
  aggregate compile win-rate, per-metric deltas, recency/grounding/cost

Usage:
  python3 skill_vs_dr.py run --arm dr --q q_opt --rep 1 --timeout 1500 --watch
  python3 skill_vs_dr.py run --arm sf --q all   --reps 2
  python3 skill_vs_dr.py judge
  python3 skill_vs_dr.py verify
  python3 skill_vs_dr.py aggregate
"""
import argparse
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results2")
SKILL_SRC = os.path.expanduser("~/.claude/skills/scholar-feed/SKILL.md")
SF_MCP = os.path.join(ROOT, "arms", "papers-local.json")

# Identical synthesis contract appended to every question, both arms.
SUFFIX = (
    "\n\nIn your FINAL report: (1) name the canonical/established approach and its "
    "key prior art; (2) trace how the method evolved; (3) identify the most recent "
    "work (2025-2026) that supersedes or improves the established answer; (4) cite "
    "every substantive claim to a specific paper with its arXiv id AND title. Prefer "
    "recent, high-impact work over generic defaults. Do not ask clarifying "
    "questions -- the scope is sufficient; produce the report directly."
)

# Depth-shaped, recency-sensitive CS/AI/ML questions where curation CAN win.
# No paper id is hardcoded as a win condition.
QUESTIONS = {
    "q_opt": (
        "What is the current best optimizer for small language-model pretraining "
        "under a fixed compute budget, and what recent work supersedes the "
        "established answer (Adam/AdamW)? Trace the lineage from the canonical "
        "optimizers to the newest matrix/spectral (Muon-family) and other 2025-2026 "
        "successors, and say where the recalled 'just use AdamW' is now wrong."
    ),
    "q_attn": (
        "Long-context attention for LLMs: what has replaced the FlashAttention-era "
        "dense-attention approach in the most recent work, and why? Trace how the "
        "method evolved (linear/state-space, sparse, and sub-quadratic successors) "
        "and identify the 2025-2026 work that is now state of the art."
    ),
    "q_kv": (
        "KV-cache compression and eviction for efficient LLM inference: what are the "
        "canonical anchor methods, and what 2025-2026 work improves on them? Trace "
        "the lineage from the foundational eviction/compression papers to the current "
        "frontier."
    ),
    "q_tts": (
        "Inference-time / test-time compute scaling for LLM reasoning: what is the "
        "current state of the art and the newest directions that supersede the "
        "original best-of-N / chain-of-thought approaches? Trace the lineage and "
        "name the specific 2025-2026 papers that move the frontier."
    ),
}

ARMS = ("dr", "sf")


def run_dir(arm, q, rep):
    return os.path.join(OUT, f"{arm}_{q}_{rep:02d}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def build_cmd(arm, q, model, workdir):
    question = QUESTIONS[q] + SUFFIX
    if arm == "dr":
        prompt = "/deep-research " + question
        # built-in web workflow: all built-in tools, NO mcp (strict + no config).
        tools = "default"
        mcp = None
    elif arm == "sf":
        prompt = "/scholar-feed " + question
        # pure corpus: no Web tools; MCP tools auto-added via --mcp-config.
        tools = "Read,Glob,Grep,Bash"
        mcp = SF_MCP
    else:
        raise ValueError(arm)

    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--dangerously-skip-permissions",
        "--setting-sources", "project",   # clean room: no user CLAUDE.md/hooks/memory
        "--strict-mcp-config",            # ignore ambient MCP servers
        "--tools", tools,
        "--output-format", "stream-json",
        "--verbose",
        "--no-session-persistence",
    ]
    if mcp:
        cmd += ["--mcp-config", os.path.abspath(mcp)]
    return cmd, prompt


def pretty(line):
    try:
        ev = json.loads(line)
    except Exception:
        return None
    t = ev.get("type")
    if t == "assistant":
        out = []
        for b in ev.get("message", {}).get("content", []):
            if b.get("type") == "text" and b.get("text", "").strip():
                out.append("💬 " + b["text"].strip().replace("\n", " ")[:180])
            elif b.get("type") == "tool_use":
                inp = json.dumps(b.get("input", {}))[:110]
                out.append(f"🔧 {b.get('name')}  {inp}")
        return "\n".join(out) or None
    if t == "result":
        return (f"✅ done  ({ev.get('num_turns','?')} turns, "
                f"{ev.get('duration_ms',0)/1000:.0f}s, ${ev.get('total_cost_usd',0):.3f})")
    if t == "system" and ev.get("subtype") not in (None, "thinking_tokens"):
        return f"⚙️  {ev.get('subtype')}"
    return None


def run_agent(cmd, workdir, transcript_path, timeout, watch):
    deadline = time.time() + timeout
    timed_out = False
    with open(transcript_path, "w") as tf:
        proc = subprocess.Popen(
            cmd, cwd=workdir, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        while True:
            if time.time() > deadline:
                proc.terminate()
                timed_out = True
                break
            rlist, _, _ = select.select([proc.stdout], [], [], 1.0)
            if rlist:
                line = proc.stdout.readline()
                if not line:
                    break
                tf.write(line)
                tf.flush()
                if watch:
                    msg = pretty(line)
                    if msg:
                        print(msg, flush=True)
            elif proc.poll() is not None:
                break
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return proc.returncode, timed_out


def parse_transcript(path):
    """Extract final report, tool histogram, cost/turns/wall from a stream-json log."""
    tools = {}
    result_ev = None
    last_texts = []
    # DR's web fan-out hides inside the Workflow; count its phases from task_progress
    # descriptions (each unique tool_use_id is one workflow sub-step) for fair tooling.
    wf_phases = {}
    seen_subtasks = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("type") == "assistant":
                texts = []
                for b in ev.get("message", {}).get("content", []):
                    if b.get("type") == "tool_use":
                        tools[b["name"]] = tools.get(b["name"], 0) + 1
                    elif b.get("type") == "text" and b.get("text", "").strip():
                        texts.append(b["text"])
                if texts:
                    last_texts = texts  # keep the most recent assistant text turn
            elif ev.get("type") == "system" and ev.get("subtype") == "task_progress":
                desc = (ev.get("description") or "")
                phase = desc.split(":", 1)[0].strip() if desc else "?"
                key = (ev.get("tool_use_id"), desc)
                if key not in seen_subtasks:
                    seen_subtasks.add(key)
                    wf_phases[phase] = wf_phases.get(phase, 0) + 1
            elif ev.get("type") == "result":
                result_ev = ev
    report = ""
    meta = {}
    if result_ev:
        report = result_ev.get("result") or ""
        meta = {
            "num_turns": result_ev.get("num_turns"),
            "duration_ms": result_ev.get("duration_ms"),
            "total_cost_usd": result_ev.get("total_cost_usd"),
            "subtype": result_ev.get("subtype"),
            "usage": result_ev.get("usage"),
        }
    if wf_phases:
        meta["workflow_phases"] = wf_phases  # e.g. {Scope, Search, Fetch, Verify, Synthesize}
    if not report:  # timeout / no result event: fall back to last assistant text
        report = "\n\n".join(last_texts)
        meta["fallback_report"] = True
    return report, tools, meta


def do_run(arm, q, rep, model, timeout, watch):
    rd = run_dir(arm, q, rep)
    workdir = os.path.join(rd, "work")
    if os.path.exists(rd):
        shutil.rmtree(rd)
    os.makedirs(workdir)
    if arm == "sf":  # make /scholar-feed resolvable as a project skill
        skd = os.path.join(workdir, ".claude", "skills", "scholar-feed")
        os.makedirs(skd)
        shutil.copy2(SKILL_SRC, os.path.join(skd, "SKILL.md"))

    cmd, prompt = build_cmd(arm, q, model, workdir)
    print(f"\n=== {arm}_{q}_{rep:02d}  (model {model}, timeout {timeout}s) ===", flush=True)
    json.dump({"arm": arm, "q": q, "rep": rep, "model": model, "prompt": prompt,
               "cmd": cmd}, open(os.path.join(rd, "cmd.json"), "w"), indent=2)

    t0 = time.time()
    rc, timed_out = run_agent(cmd, workdir, os.path.join(rd, "transcript.jsonl"),
                              timeout, watch)
    wall = round(time.time() - t0, 1)

    report, tools, meta = parse_transcript(os.path.join(rd, "transcript.jsonl"))
    with open(os.path.join(rd, "report.md"), "w") as f:
        f.write(report)
    json.dump({
        "arm": arm, "q": q, "rep": rep, "model": model,
        "returncode": rc, "wall_s": wall, "timed_out": timed_out,
        "tool_calls": tools, "report_chars": len(report), **meta,
    }, open(os.path.join(rd, "run_meta.json"), "w"), indent=2)

    shutil.rmtree(workdir, ignore_errors=True)  # teardown; keep transcript/report/meta
    print(f"--- {arm}_{q}_{rep:02d}: {len(report)} chars, {sum(tools.values())} tool calls, "
          f"{wall}s{'  [TIMEOUT]' if timed_out else ''}  tools={tools}", flush=True)
    return len(report) > 200 and not timed_out


def cmd_run(args):
    qs = list(QUESTIONS) if args.q == "all" else [args.q]
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    for q in qs:
        for arm in arms:
            for rep in range(1, args.reps + 1):
                if args.rep and rep != args.rep:
                    continue
                do_run(arm, q, rep, args.model, args.timeout, args.watch)


# ---------------------------------------------------------------------------
# verify ids (hallucination / grounding)
# ---------------------------------------------------------------------------
ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")


def arxiv_exists(aid, cache):
    if aid in cache:
        return cache[aid]
    url = f"http://export.arxiv.org/api/query?id_list={aid}&max_results=1"
    ok, title = False, None
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            body = r.read().decode("utf-8", "replace")
        # a real entry has a <title> inside an <entry>; the feed title is separate
        m = re.search(r"<entry>.*?<title>(.*?)</title>", body, re.S)
        if m and "Error" not in m.group(1):
            ok, title = True, re.sub(r"\s+", " ", m.group(1)).strip()
    except Exception as e:
        title = f"(lookup error: {e})"
    cache[aid] = (ok, title)
    time.sleep(3)  # arxiv API politeness
    return cache[aid]


def cmd_verify(args):
    cache = {}
    cache_path = os.path.join(OUT, "_arxiv_cache.json")
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
        cache = {k: tuple(v) for k, v in cache.items()}
    rows = []
    for d in sorted(os.listdir(OUT)):
        rd = os.path.join(OUT, d)
        rep = os.path.join(rd, "report.md")
        if not os.path.isdir(rd) or not os.path.exists(rep):
            continue
        arm, q, _ = d.split("_", 2)
        text = open(rep).read()
        ids = sorted({m.group(1) for m in ARXIV_RE.finditer(text)})
        results = {}
        for aid in ids:
            ok, title = arxiv_exists(aid, cache)
            results[aid] = {"exists": ok, "arxiv_title": title}
            json.dump({k: list(v) for k, v in cache.items()}, open(cache_path, "w"))
        n = len(ids)
        bad = [a for a in ids if not results[a]["exists"]]
        row = {"run": d, "arm": arm, "q": q, "n_ids": n,
               "n_unresolved": len(bad), "unresolved": bad, "ids": results}
        rows.append(row)
        json.dump(row, open(os.path.join(rd, "ids.json"), "w"), indent=2)
        print(f"{d}: {n} ids, {len(bad)} unresolved {bad if bad else ''}")
    json.dump(rows, open(os.path.join(OUT, "_id_verify.json"), "w"), indent=2)


# ---------------------------------------------------------------------------
# judge (blind pairwise + rubric)
# ---------------------------------------------------------------------------
JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "report_A": {"$ref": "#/$defs/scores"},
        "report_B": {"$ref": "#/$defs/scores"},
        "pairwise_winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "winner_confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "rationale": {"type": "string"},
    },
    "required": ["report_A", "report_B", "pairwise_winner", "winner_confidence", "rationale"],
    "$defs": {
        "scores": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "recency": {"type": "integer", "minimum": 1, "maximum": 5},
                "depth_lineage": {"type": "integer", "minimum": 1, "maximum": 5},
                "coverage": {"type": "integer", "minimum": 1, "maximum": 5},
                "grounding": {"type": "integer", "minimum": 1, "maximum": 5},
                "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
                "n_recent_papers_2025_2026": {"type": "integer", "minimum": 0},
                "notes": {"type": "string"},
            },
            "required": ["recency", "depth_lineage", "coverage", "grounding",
                         "correctness", "n_recent_papers_2025_2026", "notes"],
        }
    },
}

JUDGE_INSTR = """You are a blind, rigorous judge of two research reports (A and B) that
answer the SAME question. You do not know how either was produced. Score each on a
1-5 rubric and pick a pairwise winner. Be skeptical and specific.

Rubric (1=poor, 5=excellent), per report:
- recency: how well it surfaces genuinely recent (2025-2026) work, not just classics.
- depth_lineage: does it TRACE how the method evolved and what supersedes the
  established answer, vs. just listing papers?
- coverage: are BOTH the canonical prior art AND the current frontier present?
- grounding: is every substantive claim tied to a specific paper with an arXiv id +
  title? Penalize vague, uncited, or hand-wavy claims. (Judge groundedness of form;
  you cannot look ids up.)
- correctness: do the technical claims and the who-superseded-whom story look right
  for this subfield, from your own knowledge? Penalize confident-but-wrong claims.
- n_recent_papers_2025_2026: count distinct papers it cites that are dated 2025 or 2026.

Also give a pairwise_winner (A, B, or tie) for which report better answers the
question overall (depth + recency + grounding + correctness), a confidence, and a
2-4 sentence rationale naming concrete differences.

Output ONLY the structured object.

================ QUESTION ================
{question}

================ REPORT A ================
{A}

================ REPORT B ================
{B}
"""


def load_report(arm, q, rep):
    p = os.path.join(run_dir(arm, q, rep), "report.md")
    return open(p).read() if os.path.exists(p) else None


def deterministic_order(q, rep):
    """Stable A/B assignment without Date/random: hash of the pair key parity."""
    h = sum(ord(c) for c in f"{q}_{rep}")
    return ("sf", "dr") if h % 2 == 0 else ("dr", "sf")  # (which arm is A, which is B)


def cmd_judge(args):
    os.makedirs(os.path.join(OUT, "_judge"), exist_ok=True)
    reps = args.reps
    qs = list(QUESTIONS) if args.q == "all" else [s.strip() for s in args.q.split(",")]
    out_rows = []
    for q in qs:
        for rep in range(1, reps + 1):
            sf = load_report("sf", q, rep)
            dr = load_report("dr", q, rep)
            if not sf or not dr:
                print(f"skip {q} rep{rep}: missing report (sf={bool(sf)} dr={bool(dr)})")
                continue
            if len(sf) < 500 or len(dr) < 500:  # guard against limit-killed stubs
                print(f"skip {q} rep{rep}: stub report (sf={len(sf)}c dr={len(dr)}c)")
                continue
            armA, armB = deterministic_order(q, rep)
            A = sf if armA == "sf" else dr
            B = sf if armB == "sf" else dr
            prompt = JUDGE_INSTR.format(question=QUESTIONS[q], A=A, B=B)
            jd = os.path.join(OUT, "_judge", f"{q}_{rep:02d}")
            os.makedirs(jd, exist_ok=True)
            cmd = [
                "claude", "-p", prompt,
                "--model", args.model,
                "--dangerously-skip-permissions",
                "--setting-sources", "project",
                "--strict-mcp-config",
                "--tools", "",
                "--output-format", "json",
                "--json-schema", json.dumps(JUDGE_SCHEMA),
                "--no-session-persistence",
            ]
            print(f"judging {q} rep{rep}  (A={armA}, B={armB}) ...", flush=True)
            try:
                p = subprocess.run(cmd, cwd=jd, capture_output=True, text=True,
                                   timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"  judge timeout {q} rep{rep}")
                continue
            open(os.path.join(jd, "raw.json"), "w").write(p.stdout)
            try:
                outer = json.loads(p.stdout)
                # with --json-schema the validated object is in `structured_output`,
                # not `result` (which holds a human summary string).
                verdict = outer.get("structured_output")
                if verdict is None:
                    r = outer.get("result")
                    verdict = json.loads(r) if isinstance(r, str) else r
            except Exception as e:
                print(f"  parse fail {q} rep{rep}: {e}\n  stdout head: {p.stdout[:300]}")
                continue
            row = {"q": q, "rep": rep, "armA": armA, "armB": armB, "verdict": verdict}
            # de-blind: map A/B back to sf/dr
            winner_arm = armA if verdict["pairwise_winner"] == "A" else (
                armB if verdict["pairwise_winner"] == "B" else "tie")
            row["winner_arm"] = winner_arm
            row["sf_scores"] = verdict["report_A"] if armA == "sf" else verdict["report_B"]
            row["dr_scores"] = verdict["report_A"] if armA == "dr" else verdict["report_B"]
            out_rows.append(row)
            json.dump(row, open(os.path.join(jd, "verdict.json"), "w"), indent=2)
            print(f"  winner={winner_arm}  conf={verdict['winner_confidence']}")
    # merge into all.json by (q,rep) so prior verdicts (e.g. q_opt) survive
    all_path = os.path.join(OUT, "_judge", "all.json")
    merged = {}
    if os.path.exists(all_path):
        for r in json.load(open(all_path)):
            merged[(r["q"], r["rep"])] = r
    for r in out_rows:
        merged[(r["q"], r["rep"])] = r
    final = [merged[k] for k in sorted(merged)]
    json.dump(final, open(all_path, "w"), indent=2)
    print(f"\nwrote {len(out_rows)} new verdicts; all.json now has {len(final)}")


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def cmd_aggregate(args):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    judge = json.load(open(os.path.join(OUT, "_judge", "all.json")))
    ids = {r["run"]: r for r in json.load(open(os.path.join(OUT, "_id_verify.json")))} \
        if os.path.exists(os.path.join(OUT, "_id_verify.json")) else {}

    # pairwise win-rate
    wins = {"sf": 0, "dr": 0, "tie": 0}
    for r in judge:
        wins[r["winner_arm"]] += 1
    n = sum(wins.values())

    dims = ["recency", "depth_lineage", "coverage", "grounding", "correctness",
            "n_recent_papers_2025_2026"]
    agg = {"sf": {d: [] for d in dims}, "dr": {d: [] for d in dims}}
    for r in judge:
        for d in dims:
            agg["sf"][d].append(r["sf_scores"].get(d))
            agg["dr"][d].append(r["dr_scores"].get(d))

    # cost / wall / tool calls / id grounding from run_meta + ids
    runmeta = {"sf": [], "dr": []}
    for d in sorted(os.listdir(OUT)):
        mp = os.path.join(OUT, d, "run_meta.json")
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        if m["arm"] in runmeta:
            idrow = ids.get(d, {})
            m["n_ids"] = idrow.get("n_ids")
            m["n_unresolved"] = idrow.get("n_unresolved")
            runmeta[m["arm"]].append(m)

    def cost_stats(arm):
        ms = runmeta[arm]
        return {
            "runs": len(ms),
            "mean_wall_s": mean([m.get("wall_s") for m in ms]),
            "mean_cost_usd": mean([m.get("total_cost_usd") for m in ms]),
            "mean_tool_calls": mean([sum(m.get("tool_calls", {}).values()) for m in ms]),
            "mean_n_ids": mean([m.get("n_ids") for m in ms]),
            "total_ids": sum((m.get("n_ids") or 0) for m in ms),
            "total_unresolved": sum((m.get("n_unresolved") or 0) for m in ms),
            "n_timeouts": sum(1 for m in ms if m.get("timed_out")),
        }

    out = {
        "n_judged_pairs": n,
        "pairwise": wins,
        "sf_win_rate_excl_ties": round(wins["sf"] / (wins["sf"] + wins["dr"]), 3)
            if (wins["sf"] + wins["dr"]) else None,
        "sf_win_rate_incl_ties": round(wins["sf"] / n, 3) if n else None,
        "rubric_means": {
            "sf": {d: mean(agg["sf"][d]) for d in dims},
            "dr": {d: mean(agg["dr"][d]) for d in dims},
        },
        "cost": {"sf": cost_stats("sf"), "dr": cost_stats("dr")},
        "per_pair": [
            {"q": r["q"], "rep": r["rep"], "winner": r["winner_arm"],
             "conf": r["verdict"]["winner_confidence"],
             "sf_recency": r["sf_scores"]["recency"], "dr_recency": r["dr_scores"]["recency"],
             "sf_recent_n": r["sf_scores"]["n_recent_papers_2025_2026"],
             "dr_recent_n": r["dr_scores"]["n_recent_papers_2025_2026"]}
            for r in judge
        ],
    }
    json.dump(out, open(os.path.join(OUT, "_summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--arm", choices=["dr", "sf", "all"], required=True)
    r.add_argument("--q", default="all", help="question id or 'all'")
    r.add_argument("--reps", type=int, default=1)
    r.add_argument("--rep", type=int, default=0, help="run only this rep index")
    r.add_argument("--model", default="claude-sonnet-4-6")
    r.add_argument("--timeout", type=int, default=1500)
    r.add_argument("--watch", action="store_true")
    r.set_defaults(func=cmd_run)

    j = sub.add_parser("judge")
    j.add_argument("--model", default="claude-opus-4-8")
    j.add_argument("--q", default="all", help="comma-separated question ids or 'all'")
    j.add_argument("--reps", type=int, default=1)
    j.add_argument("--timeout", type=int, default=600)
    j.set_defaults(func=cmd_judge)

    v = sub.add_parser("verify")
    v.set_defaults(func=cmd_verify)

    a = sub.add_parser("aggregate")
    a.set_defaults(func=cmd_aggregate)

    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
