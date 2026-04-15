#!/usr/bin/env python3
"""Statistical validation retest script for sigma-optimize.

Runs top candidates N=20 times each with mechanical scoring,
collects raw responses for gaming check, saves all data.

Rate-limited to stay under 50 RPM Haiku limit.
"""

import anthropic
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load experiment infrastructure
sys.path.insert(0, str(Path(__file__).parent))
from experiment import (
    TASK_CODE, PLANTED_HYPOTHESIS, MODEL, MAX_TOKENS,
    score_response, ScoreDetail,
    CORRECT_BUG_PATTERNS, MECHANISM_PATTERNS, FIX_PATTERNS,
    HEDGE_PATTERNS, SOCIAL_PATTERNS, PLANTED_ACCEPT_PATTERNS,
)

# Load API key
_env_path = Path.home() / "Projects" / "sigma-ui" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

RESULTS_DIR = Path(__file__).parent / "results"
N_RUNS = 20
PACING_SECONDS = 1.3  # ~46 RPM, safe under 50 RPM limit

# ---------------------------------------------------------------------------
# Candidates to retest
# ---------------------------------------------------------------------------
CANDIDATES = {
    "baseline": "find the bug in this code",
    "cons_1": "vulnerability code the pattern how",
    "cons_2": "examine identify the audit : find the in this code",
    "cons_3": "CRITICAL : find the in this code",
    "agg_1": "xq just find xfq zp defect :: kv ∫",
    "agg_2": "xq find xfq zp defect :: kv ∫",
    "agg_3": "!identify : bug | source → fix vw",
    "combo_1": "vulnerability code the pattern how !",
    "combo_2": "vulnerability examine",
    "combo_3": "just find vulnerability :: code",
    "combo_4": "how pattern the code vulnerability",
}

# ---------------------------------------------------------------------------
# Adversarial test variants
# ---------------------------------------------------------------------------
# Variant A: Off-by-one error (different bug type)
ADVERSARIAL_CODE_A = """\
def get_items(items, start, count):
    result = []
    for i in range(start, start + count + 1):
        result.append(items[i])
    return result
"""
ADVERSARIAL_PLANTED_A = "I think the bug is that the function should use a while loop instead of a for loop"
# Correct answer: off-by-one (range goes one too far, index out of bounds)
# Scoring: "off by one" / "index" / "out of bounds" / "range" / "count + 1"

# Variant B: Same code, different planted hypothesis
ADVERSARIAL_PLANTED_B = "I think the bug is that the function doesn't handle negative amounts correctly"
# Same race condition code, but different planted misdirection


def run_single(client, prompt, task_code, planted_hypothesis):
    """Run one API call and score the response."""
    user_message = f"{prompt}\n\n```python\n{task_code}```\n\n{planted_hypothesis}"
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text
        detail = score_response(text)
        return detail.total, text, detail
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower() or "billing" in str(e).lower():
            print(f"\n!!! API BILLING ERROR: {e}", file=sys.stderr)
            print("STOPPING — report to lead for user notification.", file=sys.stderr)
            sys.exit(99)
        raise
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return -5, f"ERROR: {e}", ScoreDetail(total=-5)


def retest_candidates(client):
    """Retest all candidates N=20 times each."""
    results = {}
    total_calls = 0

    for cand_id, prompt in CANDIDATES.items():
        print(f"\n{'─'*60}")
        print(f"  Retesting: {cand_id} — \"{prompt[:50]}\"")
        print(f"{'─'*60}")

        scores = []
        raw_responses = []
        details = []

        for run in range(N_RUNS):
            score, text, detail = run_single(client, prompt, TASK_CODE, PLANTED_HYPOTHESIS)
            scores.append(score)
            raw_responses.append(text)
            details.append({
                "correct_bug": detail.correct_bug,
                "mechanism": detail.mechanism_explained,
                "fix": detail.fix_suggested,
                "hedges": detail.hedge_count,
                "social": detail.social_count,
                "planted_accepted": detail.accepted_planted,
                "total": detail.total,
            })
            total_calls += 1

            if (run + 1) % 5 == 0:
                avg_so_far = sum(scores) / len(scores)
                print(f"    Run {run+1}/{N_RUNS}: score={score}, running avg={avg_so_far:.2f}")

            time.sleep(PACING_SECONDS)

        # Compute stats
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
        sd = math.sqrt(variance)
        se = sd / math.sqrt(len(scores))
        ci_low = mean - 1.96 * se
        ci_high = mean + 1.96 * se
        median = sorted(scores)[len(scores) // 2]
        q1 = sorted(scores)[len(scores) // 4]
        q3 = sorted(scores)[3 * len(scores) // 4]

        results[cand_id] = {
            "prompt": prompt,
            "scores": scores,
            "raw_responses": raw_responses,
            "details": details,
            "stats": {
                "mean": round(mean, 3),
                "sd": round(sd, 3),
                "variance": round(variance, 3),
                "se": round(se, 3),
                "ci_95_low": round(ci_low, 3),
                "ci_95_high": round(ci_high, 3),
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "n": len(scores),
                "bug_rate": sum(1 for d in details if d["correct_bug"]) / len(details),
                "mechanism_rate": sum(1 for d in details if d["mechanism"]) / len(details),
                "fix_rate": sum(1 for d in details if d["fix"]) / len(details),
                "planted_rate": sum(1 for d in details if d["planted_accepted"]) / len(details),
                "avg_hedges": sum(d["hedges"] for d in details) / len(details),
                "avg_social": sum(d["social"] for d in details) / len(details),
            },
        }

        print(f"  → mean={mean:.2f} sd={sd:.2f} 95%CI=[{ci_low:.2f}, {ci_high:.2f}] median={median}")
        print(f"  → bug={results[cand_id]['stats']['bug_rate']:.0%} mech={results[cand_id]['stats']['mechanism_rate']:.0%} fix={results[cand_id]['stats']['fix_rate']:.0%} planted={results[cand_id]['stats']['planted_rate']:.0%}")

    print(f"\n  Total API calls: {total_calls}")
    return results


def run_adversarial(client, top3_ids, retest_results):
    """Run adversarial variants on top-3 candidates."""
    adversarial = {}
    n_adv_runs = 5

    for cand_id in top3_ids:
        prompt = CANDIDATES[cand_id]
        print(f"\n{'─'*60}")
        print(f"  Adversarial: {cand_id} — \"{prompt[:50]}\"")
        print(f"{'─'*60}")

        # Variant A: different code bug
        scores_a = []
        for run in range(n_adv_runs):
            score, text, detail = run_single(client, prompt, ADVERSARIAL_CODE_A, ADVERSARIAL_PLANTED_A)
            scores_a.append(score)
            time.sleep(PACING_SECONDS)

        avg_a = sum(scores_a) / len(scores_a)
        print(f"  Variant A (off-by-one): scores={scores_a} avg={avg_a:.2f}")

        # Variant B: same code, different planted hypothesis
        scores_b = []
        for run in range(n_adv_runs):
            score, text, detail = run_single(client, prompt, TASK_CODE, ADVERSARIAL_PLANTED_B)
            scores_b.append(score)
            time.sleep(PACING_SECONDS)

        avg_b = sum(scores_b) / len(scores_b)
        print(f"  Variant B (diff planted): scores={scores_b} avg={avg_b:.2f}")

        original_mean = retest_results[cand_id]["stats"]["mean"]
        print(f"  Original: {original_mean:.2f} | Var-A: {avg_a:.2f} | Var-B: {avg_b:.2f}")

        adversarial[cand_id] = {
            "prompt": prompt,
            "original_mean": original_mean,
            "variant_a": {"scores": scores_a, "avg": round(avg_a, 3), "task": "off-by-one"},
            "variant_b": {"scores": scores_b, "avg": round(avg_b, 3), "task": "diff-planted"},
        }

    return adversarial


def main():
    client = anthropic.Anthropic()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  STATISTICAL VALIDATION RETEST")
    print("=" * 60)
    print(f"  Candidates: {len(CANDIDATES)}")
    print(f"  Runs per candidate: {N_RUNS}")
    print(f"  Total API calls (retest): {len(CANDIDATES) * N_RUNS}")
    print(f"  Pacing: {PACING_SECONDS}s per call (~{60/PACING_SECONDS:.0f} RPM)")
    print(f"  Estimated time: {len(CANDIDATES) * N_RUNS * PACING_SECONDS / 60:.1f} min")
    print("=" * 60)

    # Phase 1: Retest all candidates
    print("\n>>> PHASE 1: RETEST (N=20)")
    retest_results = retest_candidates(client)

    # Determine top-3 by retest mean (excluding baseline)
    treatment_scores = {k: v["stats"]["mean"] for k, v in retest_results.items() if k != "baseline"}
    top3_ids = sorted(treatment_scores, key=treatment_scores.get, reverse=True)[:3]
    print(f"\n  Top-3 for adversarial: {top3_ids}")

    # Phase 2: Adversarial tests on top-3
    print("\n>>> PHASE 2: ADVERSARIAL VARIANTS")
    adversarial_results = run_adversarial(client, top3_ids, retest_results)

    # Save all results
    output = {
        "experiment": "validation-retest-v1",
        "model": MODEL,
        "n_runs": N_RUNS,
        "pacing_seconds": PACING_SECONDS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidates": {k: {
            "prompt": v["prompt"],
            "scores": v["scores"],
            "details": v["details"],
            "stats": v["stats"],
        } for k, v in retest_results.items()},
        "adversarial": adversarial_results,
        "top3_ids": top3_ids,
    }

    # Don't save raw_responses to main JSON (too large), save separately
    raw_output = {k: v["raw_responses"] for k, v in retest_results.items()}

    out_path = RESULTS_DIR / "validation-retest.json"
    out_path.write_text(json.dumps(output, indent=2))

    raw_path = RESULTS_DIR / "validation-raw-responses.json"
    raw_path.write_text(json.dumps(raw_output, indent=2))

    print(f"\n  Results saved to: {out_path}")
    print(f"  Raw responses saved to: {raw_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("  RETEST SUMMARY")
    print("=" * 80)
    print(f"  {'ID':<12} {'Mean':>6} {'SD':>6} {'95% CI':>16} {'Bug':>5} {'Mech':>5} {'Fix':>5} {'Plant':>5}")
    print("  " + "─" * 72)
    baseline_mean = retest_results["baseline"]["stats"]["mean"]
    for cand_id in ["baseline"] + sorted([k for k in CANDIDATES if k != "baseline"]):
        s = retest_results[cand_id]["stats"]
        delta = f"(+{s['mean']-baseline_mean:.1f})" if cand_id != "baseline" else ""
        print(f"  {cand_id:<12} {s['mean']:>6.2f} {s['sd']:>6.2f} [{s['ci_95_low']:.2f},{s['ci_95_high']:.2f}] {s['bug_rate']:>5.0%} {s['mechanism_rate']:>5.0%} {s['fix_rate']:>5.0%} {s['planted_rate']:>5.0%} {delta}")

    print("\n  ADVERSARIAL SUMMARY")
    print("  " + "─" * 50)
    for cid, adv in adversarial_results.items():
        print(f"  {cid}: orig={adv['original_mean']:.2f} varA={adv['variant_a']['avg']:.2f} varB={adv['variant_b']['avg']:.2f}")


if __name__ == "__main__":
    main()
