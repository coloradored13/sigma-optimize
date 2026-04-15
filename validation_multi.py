#!/usr/bin/env python3
"""Experiment 2 Validation: Retest top candidates at N=20 per task.

Selected candidates:
  - Top 4 from combinatorial (actual best scores from JSON)
  - Top 1 from conservative (if not in combinatorial)
  - Top 1 from aggressive (if not in combinatorial)
  - Exp 1 winner (H[3] transfer test)
  - 3 natural-language baselines

Total: 10 candidates × 4 tasks × 20 runs = 800 API calls
Estimated time: ~15 min with 10 workers
"""

import asyncio
import json
import sys
import time
from pathlib import Path

# Import from the harness
sys.path.insert(0, str(Path(__file__).parent))
from experiment_multi import (
    Candidate, ALL_TASKS, evaluate_batch, score_response,
    RESULTS_DIR, MAX_WORKERS,
)

VALIDATION_N = 20  # runs per task per candidate
OUTPUT_FILE = RESULTS_DIR / "multi-validation.json"


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------
CANDIDATES = {
    # --- Combinatorial top 4 (by actual JSON scores, not workspace report) ---
    "COMBO-1: flaw reliability evidence | bias → ? fix":
        "flaw reliability evidence | bias → ? fix",
    "COMBO-2: !find : vulnerability | code → fix":
        "!find : vulnerability | code → fix",
    "COMBO-3: flaw reliability evidence :: fix":
        "flaw reliability evidence :: fix",
    "COMBO-4: !identify : bias fix → evidence fix ::":
        "!identify : bias fix → evidence fix ::",

    # --- Conservative top 1 ---
    "CON-1: flaw | evidence fix":
        "flaw | evidence fix",

    # --- Aggressive top 1 ---
    "AGG-1: !identify : bias fix → evidence | → evidence fix":
        "!identify : bias fix → evidence | → evidence fix",

    # --- Exp 1 winner (H[3] transfer test) ---
    "EXP1-WINNER: just find vulnerability :: code":
        "just find vulnerability :: code",

    # --- Natural-language baselines ---
    "BASELINE-1: identify the flaws in this reasoning":
        "identify the flaws in this reasoning",
    "BASELINE-2: what is wrong with this analysis":
        "what is wrong with this analysis",
    "BASELINE-3: analyze this data and find the problem":
        "analyze this data and find the problem",
}


async def main():
    print(f"=" * 70)
    print(f"Experiment 2 Validation — N={VALIDATION_N} per task per candidate")
    print(f"Candidates: {len(CANDIDATES)}")
    print(f"Tasks: {len(ALL_TASKS)}")
    print(f"Total API calls: {len(CANDIDATES) * len(ALL_TASKS) * VALIDATION_N}")
    print(f"=" * 70)

    # Build Candidate objects
    candidates = []
    labels = []
    for label, prompt_text in CANDIDATES.items():
        tokens = prompt_text.split()
        c = Candidate(tokens=tokens, generation=-1)  # gen=-1 = validation
        candidates.append(c)
        labels.append(label)

    # Run evaluation
    t0 = time.time()
    print(f"\nStarting evaluation at {time.strftime('%H:%M:%S')}...")
    api_calls = await evaluate_batch(candidates, ALL_TASKS, VALIDATION_N)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({api_calls} API calls)")

    # Build results
    results = {
        "experiment": "sigma-optimize-exp2-validation",
        "phase": "statistical-validation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "runs_per_task": VALIDATION_N,
            "tasks": len(ALL_TASKS),
            "task_ids": [t["id"] for t in ALL_TASKS],
            "candidates": len(CANDIDATES),
            "total_api_calls": api_calls,
            "elapsed_s": round(elapsed, 1),
            "max_workers": MAX_WORKERS,
        },
        "candidates": [],
    }

    for label, c in zip(labels, candidates):
        entry = {
            "label": label,
            "prompt": c.prompt,
            "avg_score": round(c.avg_score, 4),
            "variance": round(c.score_variance, 4),
            "per_task": {},
        }
        for task in ALL_TASKS:
            tid = task["id"]
            scores = c.task_scores.get(tid, [])
            avg = sum(scores) / len(scores) if scores else 0
            var = (sum((s - avg)**2 for s in scores) / (len(scores) - 1)) if len(scores) > 1 else 0
            entry["per_task"][tid] = {
                "avg": round(avg, 4),
                "var": round(var, 4),
                "scores": scores,
                "raw_responses": [
                    d.raw_response for d in c.task_details.get(tid, [])
                ],
            }
        results["candidates"].append(entry)

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Quick summary
    print(f"\n{'Label':<55s} {'Avg':>6s} {'SP':>6s} {'MC':>6s} {'CA':>6s} {'DP':>6s}")
    print("-" * 85)
    for label, c in zip(labels, candidates):
        sp = c.task_avg("simpsons_paradox")
        mc = c.task_avg("methodology_critique")
        ca = c.task_avg("causal_attribution")
        dp = c.task_avg("divergent_problem")
        print(f"{label:<55s} {c.avg_score:>+6.2f} {sp:>+6.2f} {mc:>+6.2f} {ca:>+6.2f} {dp:>+6.2f}")


if __name__ == "__main__":
    asyncio.run(main())
