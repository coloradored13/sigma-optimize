#!/usr/bin/env python3
"""Combinatorial Phase — sigma-optimize Experiment 2.

Systematically tests cross-strategy hybrids, token-level ablations,
pairwise winning-token combinations, and modifier tests from both
conservative and aggressive search results.

Uses evaluate_batch for concurrent evaluation at N=10 per task.
Results: results/multi-combinatorial.json
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

from experiment_multi import (
    evaluate_batch, Candidate, ALL_TASKS, score_response,
    SCORE_FUNCTIONS, MAX_WORKERS
)

RESULTS_DIR = Path(__file__).parent / "results"
RUNS_PER_TASK = 10

# ---------------------------------------------------------------------------
# Top-4 from each search (from final_top_10 in result JSONs, N=10)
# ---------------------------------------------------------------------------

# Conservative search — best at gen 10 (full 10 gens, N=10/task)
CONSERVATIVE_TOP4 = [
    "vulnerability find code :: !",                        # 5.15, var=0.083
    ":: just find vulnerability :: identify code",         # 5.125, var=0.169
    "just find | vulnerability :: code",                   # 5.10, var=0.153
    "just find vulnerability :: :: identify # code",       # 5.10, var=0.167
]

# Aggressive search — best at gen 10 (full 10 gens, N=10/task)
AGGRESSIVE_TOP4 = [
    "!identify : bias fix \u2192 evidence fix",                  # 5.225, var=0.162
    "flaw reliability evidence | bias bias \u2192 ? fix",         # 5.225, var=0.449
    "flaw reliability ? | bias bias \u2192 evidence fix",         # 5.175, var=0.303
    "!identify : bias fix \u2192 evidence zx explain \u2192 evidence fix",  # 5.15, var=0.11
]

# Baselines for comparison (best from each search)
BASELINES = [
    "identify the flaws in this reasoning",     # best NL baseline (aggressive: 4.67)
    "what is wrong with this analysis",         # strong NL baseline
    "analyze this data and find the problem",   # generic baseline
]

# ---------------------------------------------------------------------------
# Category 1: Cross-strategy hybrids
# Conservative tokens + aggressive structure, and vice versa
# ---------------------------------------------------------------------------

CROSS_STRATEGY_HYBRIDS = [
    # Conservative tokens in aggressive structure (!X : Y | Z → W)
    "!find : vulnerability code \u2192 identify fix",
    "!vulnerability : code | find \u2192 fix",
    "!find : code | vulnerability \u2192 identify",
    "!vulnerability : find code \u2192 evidence fix",

    # Aggressive tokens in conservative structure (X Y Z :: W)
    "bias fix evidence :: !",
    "flaw reliability evidence :: fix",
    "identify bias fix :: evidence",
    "flaw evidence :: fix !",

    # Mixed token combinations
    "!identify : vulnerability | evidence \u2192 fix",
    "flaw ? vulnerability | bias \u2192 code fix",
    "vulnerability bias fix \u2192 evidence code",
    "!find : bias | evidence \u2192 fix code",
    "flaw reliability | vulnerability :: evidence fix",
    "!identify : flaw | code \u2192 evidence fix",
]

# ---------------------------------------------------------------------------
# Category 2: Token-level ablation of top candidates
# Drop each token one at a time and measure impact
# ---------------------------------------------------------------------------

def ablate(prompt: str) -> list[tuple[str, str]]:
    """Generate all single-token-drop variants with labels."""
    tokens = prompt.split()
    variants = []
    for i, dropped in enumerate(tokens):
        ablated = " ".join(tokens[:i] + tokens[i+1:])
        if ablated.strip():
            variants.append((f"ABLATE:{prompt}:-{dropped}", ablated))
    return variants

ABLATION_CANDIDATES = []
# Ablate all 8 top candidates
for prompt in CONSERVATIVE_TOP4 + AGGRESSIVE_TOP4:
    for label, ablated in ablate(prompt):
        ABLATION_CANDIDATES.append((label, ablated))

# ---------------------------------------------------------------------------
# Category 3: Pairwise winning-token combinations
# Most frequent tokens in top candidates from both searches
# ---------------------------------------------------------------------------

# Extracted from frequency analysis of top-10 from each search:
WINNING_NOUNS = ["vulnerability", "flaw", "bias", "evidence", "fix", "code", "reliability"]
WINNING_VERBS = ["find", "identify"]
WINNING_OPERATORS = ["::", "|", "\u2192", ":", "?", "!"]

# Systematic noun+operator+verb grid — focused: top-3 nouns × top-3 ops × 2 verbs = 18
PAIRWISE_CANDIDATES = []
TOP_NOUNS = ["flaw", "bias", "evidence"]
TOP_OPS = ["|", "::", "\u2192"]
for noun in TOP_NOUNS:
    for op in TOP_OPS:
        for verb in WINNING_VERBS:
            prompt = f"{noun} {op} {verb}"
            PAIRWISE_CANDIDATES.append((f"PAIR:{noun}+{op}+{verb}", prompt))

# Additional single-noun tests with key operators (tokens not in top-3)
for noun in ["vulnerability", "fix", "reliability"]:
    for op in ["|", "::"]:
        PAIRWISE_CANDIDATES.append((f"PAIR:{noun}+{op}+find", f"{noun} {op} find"))

# Noun+noun+operator combos (focused subset)
TOP_NOUN_PAIRS = [
    ("flaw", "evidence"), ("bias", "fix"), ("flaw", "bias"),
    ("evidence", "fix"), ("bias", "evidence"),
]
for n1, n2 in TOP_NOUN_PAIRS:
    for op in ["|", "\u2192"]:
        PAIRWISE_CANDIDATES.append((f"PAIR:{n1}+{n2}+{op}", f"{n1} {op} {n2}"))

# ---------------------------------------------------------------------------
# Category 4: Modifier tests
# Prefix/suffix modifiers on top-2 from each search
# ---------------------------------------------------------------------------

MODIFIERS_PREFIX = ["!", "::", "|", "?"]
MODIFIERS_SUFFIX = ["!", "::", "|", "?", "fix"]
BASE_FOR_MODIFIERS = [
    CONSERVATIVE_TOP4[0],  # "vulnerability find code :: !"
    AGGRESSIVE_TOP4[0],    # "!identify : bias fix → evidence fix"
]

MODIFIER_CANDIDATES = []
for base in BASE_FOR_MODIFIERS:
    for mod in MODIFIERS_PREFIX:
        variant = f"{mod} {base}"
        MODIFIER_CANDIDATES.append((f"MOD-PRE:{mod}+{base}", variant))
    for mod in MODIFIERS_SUFFIX:
        variant = f"{base} {mod}"
        MODIFIER_CANDIDATES.append((f"MOD-SUF:{base}+{mod}", variant))

# ---------------------------------------------------------------------------
# Category 5: Domain-specific vs domain-neutral comparison
# Key question: do domain-specific tokens (vulnerability, code) hurt
# compared to domain-neutral tokens (bias, evidence, fix)?
# ---------------------------------------------------------------------------

DOMAIN_COMPARISON = [
    # Domain-specific (code-oriented) — same structures
    ("DOMAIN:specific-1", "vulnerability code :: find !"),
    ("DOMAIN:specific-2", "!find : vulnerability | code \u2192 fix"),
    ("DOMAIN:specific-3", "vulnerability :: code"),
    ("DOMAIN:specific-4", "code vulnerability find"),

    # Domain-neutral (analytical) — same structures
    ("DOMAIN:neutral-1", "bias evidence :: find !"),
    ("DOMAIN:neutral-2", "!find : bias | evidence \u2192 fix"),
    ("DOMAIN:neutral-3", "bias :: evidence"),
    ("DOMAIN:neutral-4", "evidence bias find"),

    # Hybrid domain tokens
    ("DOMAIN:hybrid-1", "vulnerability evidence :: fix !"),
    ("DOMAIN:hybrid-2", "!find : bias | code \u2192 fix"),
    ("DOMAIN:hybrid-3", "flaw :: code"),
    ("DOMAIN:hybrid-4", "code bias evidence fix"),
]

# ---------------------------------------------------------------------------
# Category 6: Minimal effective prompts
# Test whether very short prompts with high-signal tokens match top-K
# ---------------------------------------------------------------------------

MINIMAL_CANDIDATES = [
    ("MIN:2tok-1", "flaw fix"),
    ("MIN:2tok-2", "bias evidence"),
    ("MIN:2tok-3", "evidence fix"),
    ("MIN:2tok-4", "flaw evidence"),
    ("MIN:2tok-5", "vulnerability !"),
    ("MIN:2tok-6", "! fix"),
    ("MIN:3tok-1", "flaw evidence fix"),
    ("MIN:3tok-2", "bias evidence fix"),
    ("MIN:3tok-3", "flaw | fix"),
    ("MIN:3tok-4", "bias \u2192 fix"),
    ("MIN:3tok-5", "evidence | fix"),
    ("MIN:3tok-6", "flaw ? evidence"),
    ("MIN:3tok-7", "! bias fix"),
    ("MIN:3tok-8", "flaw :: fix"),
]


def build_all_candidates() -> list[tuple[str, Candidate]]:
    """Build the full combination matrix with labels."""
    candidates = []
    seen_prompts = set()

    def add(label: str, prompt: str):
        prompt = prompt.strip()
        # Deduplicate
        if prompt in seen_prompts:
            return
        seen_prompts.add(prompt)
        c = Candidate(tokens=prompt.split())
        candidates.append((label, c))

    # Original top-4 from each search (for re-evaluation at N=10)
    for i, p in enumerate(CONSERVATIVE_TOP4):
        add(f"CON-TOP{i+1}", p)
    for i, p in enumerate(AGGRESSIVE_TOP4):
        add(f"AGG-TOP{i+1}", p)

    # Baselines
    for i, p in enumerate(BASELINES):
        add(f"BASELINE-{i+1}", p)

    # Cross-strategy hybrids
    for i, p in enumerate(CROSS_STRATEGY_HYBRIDS):
        add(f"CROSS-{i+1}", p)

    # Ablation variants
    for label, p in ABLATION_CANDIDATES:
        add(label, p)

    # Pairwise winning-token combos
    for label, p in PAIRWISE_CANDIDATES:
        add(label, p)

    # Modifier tests
    for label, p in MODIFIER_CANDIDATES:
        add(label, p)

    # Domain comparison
    for label, p in DOMAIN_COMPARISON:
        add(label, p)

    # Minimal prompts
    for label, p in MINIMAL_CANDIDATES:
        add(label, p)

    return candidates


def main():
    start_time = time.time()
    print(f"=== Combinatorial Phase — Experiment 2 ===", flush=True)
    print(f"Started: {datetime.now().isoformat()}", flush=True)

    all_candidates = build_all_candidates()
    print(f"Total unique candidates: {len(all_candidates)}", flush=True)
    print(f"API calls estimate: {len(all_candidates)} × {RUNS_PER_TASK} runs × {len(ALL_TASKS)} tasks = {len(all_candidates) * RUNS_PER_TASK * len(ALL_TASKS)}", flush=True)

    # Category counts
    cats = {}
    for label, _ in all_candidates:
        cat = label.split(":")[0].split("-")[0]
        cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count} candidates", flush=True)

    # Batch evaluate — all candidates at once using concurrent API
    candidate_objs = [c for _, c in all_candidates]
    label_map = {c.prompt: label for label, c in all_candidates}

    print(f"\nEvaluating {len(candidate_objs)} candidates with N={RUNS_PER_TASK}/task, {len(ALL_TASKS)} tasks...", flush=True)
    print(f"Using {MAX_WORKERS} concurrent workers", flush=True)

    api_calls = asyncio.run(evaluate_batch(candidate_objs, ALL_TASKS, RUNS_PER_TASK))
    elapsed = time.time() - start_time

    print(f"\nEvaluation complete: {api_calls} API calls in {elapsed:.0f}s", flush=True)

    # Build results
    results = []
    for label, c in all_candidates:
        entry = {
            "label": label,
            "prompt": c.prompt,
            "avg_score": round(c.avg_score, 3),
            "variance": round(c.score_variance, 3),
            "per_task": {},
        }
        for task_id in ["simpsons_paradox", "methodology_critique", "causal_attribution", "divergent_problem"]:
            scores = c.task_scores.get(task_id, [])
            entry["per_task"][task_id] = {
                "avg": round(c.task_avg(task_id), 2),
                "var": round(c.task_var(task_id), 3),
                "scores": scores,
            }
        results.append(entry)

    # Sort by avg_score descending
    results.sort(key=lambda x: x["avg_score"], reverse=True)

    # Save
    output = {
        "experiment": "sigma-optimize-exp2-combinatorial",
        "phase": "combinatorial",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "runs_per_task": RUNS_PER_TASK,
            "tasks": 4,
            "total_candidates": len(results),
            "max_workers": MAX_WORKERS,
            "api_calls": api_calls,
            "elapsed_s": round(elapsed, 1),
        },
        "source_candidates": {
            "conservative_top4": CONSERVATIVE_TOP4,
            "aggressive_top4": AGGRESSIVE_TOP4,
            "baselines": BASELINES,
        },
        "categories": {
            "originals": len([r for r in results if r["label"].startswith(("CON-", "AGG-"))]),
            "baselines": len([r for r in results if r["label"].startswith("BASELINE")]),
            "cross_strategy": len([r for r in results if r["label"].startswith("CROSS")]),
            "ablation": len([r for r in results if r["label"].startswith("ABLATE")]),
            "pairwise": len([r for r in results if r["label"].startswith("PAIR")]),
            "modifier": len([r for r in results if r["label"].startswith("MOD")]),
            "domain_comparison": len([r for r in results if r["label"].startswith("DOMAIN")]),
            "minimal": len([r for r in results if r["label"].startswith("MIN")]),
        },
        "results": results,
    }

    out_path = RESULTS_DIR / "multi-combinatorial.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}", flush=True)

    # Print top-20
    print(f"\n=== TOP-20 CANDIDATES ===", flush=True)
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:2d}. [{r['label']:40s}] avg={r['avg_score']:+.3f} var={r['variance']:.3f} | "
              f"SP={r['per_task']['simpsons_paradox']['avg']:+.1f} "
              f"MC={r['per_task']['methodology_critique']['avg']:+.1f} "
              f"CA={r['per_task']['causal_attribution']['avg']:+.1f} "
              f"DP={r['per_task']['divergent_problem']['avg']:+.1f} "
              f"| {r['prompt']}", flush=True)

    # Print category analysis
    print(f"\n=== CATEGORY AVERAGES ===", flush=True)
    for cat_prefix, cat_name in [
        ("CON-", "Conservative originals"),
        ("AGG-", "Aggressive originals"),
        ("BASELINE", "NL Baselines"),
        ("CROSS", "Cross-strategy hybrids"),
        ("ABLATE", "Ablation variants"),
        ("PAIR", "Pairwise combinations"),
        ("MOD", "Modifier tests"),
        ("DOMAIN:specific", "Domain-specific"),
        ("DOMAIN:neutral", "Domain-neutral"),
        ("DOMAIN:hybrid", "Domain-hybrid"),
        ("MIN", "Minimal prompts"),
    ]:
        cat_results = [r for r in results if r["label"].startswith(cat_prefix)]
        if cat_results:
            avg = sum(r["avg_score"] for r in cat_results) / len(cat_results)
            best = max(r["avg_score"] for r in cat_results)
            worst = min(r["avg_score"] for r in cat_results)
            print(f"  {cat_name:25s}: n={len(cat_results):3d} avg={avg:+.3f} best={best:+.3f} worst={worst:+.3f}", flush=True)

    # Ablation analysis — which tokens are load-bearing?
    print(f"\n=== ABLATION ANALYSIS ===", flush=True)
    for parent_prompt in CONSERVATIVE_TOP4 + AGGRESSIVE_TOP4:
        parent_results = [r for r in results if r["prompt"] == parent_prompt]
        if not parent_results:
            continue
        parent_score = parent_results[0]["avg_score"]
        print(f"\n  Parent: \"{parent_prompt}\" = {parent_score:+.3f}", flush=True)
        ablations = [r for r in results if r["label"].startswith(f"ABLATE:{parent_prompt}:")]
        for ab in sorted(ablations, key=lambda x: x["avg_score"], reverse=True):
            dropped = ab["label"].split(":-")[1] if ":-" in ab["label"] else "?"
            delta = ab["avg_score"] - parent_score
            direction = "+" if delta >= 0 else ""
            print(f"    drop \"{dropped:15s}\": {ab['avg_score']:+.3f} ({direction}{delta:.3f}) | {ab['prompt']}", flush=True)

    print(f"\n=== DONE === elapsed: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
