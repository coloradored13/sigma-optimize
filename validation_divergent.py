#!/usr/bin/env python3
"""Experiment 2 Validation Loop-Back: Divergent Problem at N=40.

Targeted retest of top 2 candidates + best baseline on the task
showing largest effect sizes (d≈0.85-0.89) from N=20 validation.

Total: 3 candidates × 40 runs × 1 task = 120 API calls
"""

import asyncio
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from experiment_multi import (
    Candidate, ALL_TASKS, evaluate_batch,
    RESULTS_DIR, MAX_WORKERS,
)

VALIDATION_N = 40
TASK = [ALL_TASKS[3]]  # divergent_problem only
OUTPUT_FILE = RESULTS_DIR / "multi-validation-divergent.json"

CANDIDATES = {
    "COMBO-1: flaw reliability evidence | bias → ? fix":
        "flaw reliability evidence | bias → ? fix",
    "COMBO-2: !find : vulnerability | code → fix":
        "!find : vulnerability | code → fix",
    "BASELINE-1: identify the flaws in this reasoning":
        "identify the flaws in this reasoning",
}


async def main():
    print(f"=" * 70)
    print(f"Divergent Problem Validation — N={VALIDATION_N}")
    print(f"Candidates: {len(CANDIDATES)}")
    print(f"Task: {TASK[0]['id']}")
    print(f"Total API calls: {len(CANDIDATES) * VALIDATION_N}")
    print(f"=" * 70)

    candidates = []
    labels = []
    for label, prompt_text in CANDIDATES.items():
        tokens = prompt_text.split()
        c = Candidate(tokens=tokens, generation=-1)
        candidates.append(c)
        labels.append(label)

    t0 = time.time()
    print(f"\nStarting evaluation at {time.strftime('%H:%M:%S')}...")
    api_calls = await evaluate_batch(candidates, TASK, VALIDATION_N)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s ({api_calls} API calls)")

    results = {
        "experiment": "sigma-optimize-exp2-validation-divergent",
        "phase": "loop-back-divergent-problem",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "runs_per_task": VALIDATION_N,
            "tasks": 1,
            "task_ids": ["divergent_problem"],
            "candidates": len(CANDIDATES),
            "total_api_calls": api_calls,
            "elapsed_s": round(elapsed, 1),
        },
        "candidates": [],
    }

    for label, c in zip(labels, candidates):
        tid = "divergent_problem"
        scores = c.task_scores.get(tid, [])
        avg = sum(scores) / len(scores) if scores else 0
        var = (sum((s - avg)**2 for s in scores) / (len(scores) - 1)) if len(scores) > 1 else 0
        entry = {
            "label": label,
            "prompt": c.prompt,
            "avg_score": round(avg, 4),
            "variance": round(var, 4),
            "scores": scores,
            "raw_responses": [
                d.raw_response for d in c.task_details.get(tid, [])
            ],
        }
        results["candidates"].append(entry)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")

    print(f"\n{'Label':<55s} {'Avg':>6s} {'SD':>6s}")
    print("-" * 70)
    for label, c in zip(labels, candidates):
        tid = "divergent_problem"
        scores = c.task_scores.get(tid, [])
        import numpy as np
        m = np.mean(scores)
        sd = np.std(scores, ddof=1)
        print(f"{label:<55s} {m:>+6.2f} {sd:>6.2f}")


if __name__ == "__main__":
    asyncio.run(main())
