#!/usr/bin/env python3
"""Ablation study: isolate the contribution of each component in "defect : identify".

Tests:
  1. Vary the left token (noun): defect/bug/flaw/error/issue/vulnerability
  2. Vary the separator: :/|/→/∀/-/,/;/(space)/∆/::
  3. Vary the right token (verb): identify/find/detect/locate/analyze/diagnose
  4. Vary order: identify : defect (reversed)
  5. Drop components: ": identify", "defect :", "defect", "identify", ":"
  6. The champion baseline: "defect : identify"
"""

import anthropic
import json
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path.home() / "Projects" / "sigma-ui" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from experiment import score_response, TASK_CODE, PLANTED_HYPOTHESIS, MODEL, ScoreDetail

RUNS = 5  # more runs for statistical confidence
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Ablation groups
# ---------------------------------------------------------------------------
ABLATIONS = {
    # Baseline
    "baseline": [
        "defect : identify",
    ],

    # Vary left token (noun) — hold separator and right token constant
    "left_token": [
        "bug : identify",
        "flaw : identify",
        "error : identify",
        "issue : identify",
        "vulnerability : identify",
        "problem : identify",
        "fault : identify",
    ],

    # Vary separator — hold noun and verb constant
    "separator": [
        "defect | identify",
        "defect → identify",
        "defect ∀ identify",
        "defect - identify",
        "defect , identify",
        "defect ; identify",
        "defect identify",          # no separator (space only)
        "defect ∆ identify",
        "defect :: identify",
        "defect > identify",
        "defect / identify",
        "defect # identify",
        "defect = identify",
        "defect . identify",
        "defect ? identify",
    ],

    # Vary right token (verb) — hold noun and separator constant
    "right_token": [
        "defect : find",
        "defect : detect",
        "defect : locate",
        "defect : analyze",
        "defect : diagnose",
        "defect : inspect",
        "defect : audit",
        "defect : check",
        "defect : scan",
        "defect : trace",
        "defect : review",
    ],

    # Reversed order
    "reversed": [
        "identify : defect",
        "find : defect",
        "identify : bug",
        "identify : error",
    ],

    # Drop components
    "dropped": [
        ": identify",
        "defect :",
        "defect",
        "identify",
        ":",
        ": defect",
        "identify defect",
    ],

    # Natural language comparisons
    "natural_language": [
        "find the bug in this code",
        "identify the defect in this code",
        "what is wrong with this code",
        "please identify the defect in this code",
        "could you find the bug here",
        "review this code for bugs",
        "I think there might be a bug, can you check?",
    ],
}


def evaluate_prompt(client: anthropic.Anthropic, prompt: str, runs: int = RUNS) -> dict:
    """Run a prompt against the task multiple times, return detailed results."""
    scores = []
    details = []

    for _ in range(runs):
        user_message = f"{prompt}\n\n```python\n{TASK_CODE}```\n\n{PLANTED_HYPOTHESIS}"
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text
            detail = score_response(text)
            scores.append(detail.total)
            details.append({
                "correct_bug": detail.correct_bug,
                "mechanism": detail.mechanism_explained,
                "fix": detail.fix_suggested,
                "hedges": detail.hedge_count,
                "social": detail.social_count,
                "planted_accepted": detail.accepted_planted,
                "total": detail.total,
                "response_preview": text[:200],
            })
        except Exception as e:
            print(f"    ERROR: {e}", file=sys.stderr)
            scores.append(-5)
            details.append({"total": -5, "error": str(e)})

        time.sleep(0.1)

    avg = sum(scores) / len(scores) if scores else 0
    var = sum((s - avg) ** 2 for s in scores) / (len(scores) - 1) if len(scores) > 1 else 0

    return {
        "prompt": prompt,
        "avg_score": round(avg, 2),
        "variance": round(var, 2),
        "scores": scores,
        "details": details,
        "bug_rate": round(sum(1 for d in details if d.get("correct_bug")) / len(details), 2),
        "mechanism_rate": round(sum(1 for d in details if d.get("mechanism")) / len(details), 2),
        "fix_rate": round(sum(1 for d in details if d.get("fix")) / len(details), 2),
        "planted_rate": round(sum(1 for d in details if d.get("planted_accepted")) / len(details), 2),
        "hedge_avg": round(sum(d.get("hedges", 0) for d in details) / len(details), 2),
        "social_avg": round(sum(d.get("social", 0) for d in details) / len(details), 2),
    }


def run_ablation():
    client = anthropic.Anthropic()
    RESULTS_DIR.mkdir(exist_ok=True)

    total_prompts = sum(len(v) for v in ABLATIONS.values())
    total_calls = total_prompts * RUNS

    print("=" * 70)
    print("  ABLATION STUDY: Decomposing 'defect : identify'")
    print("=" * 70)
    print(f"  Groups: {len(ABLATIONS)}")
    print(f"  Total prompts: {total_prompts}")
    print(f"  Runs per prompt: {RUNS}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Model: {MODEL}")
    print("=" * 70)

    all_results = {}
    call_count = 0

    for group_name, prompts in ABLATIONS.items():
        print(f"\n{'─' * 70}")
        print(f"  Group: {group_name} ({len(prompts)} variants)")
        print(f"{'─' * 70}")

        group_results = []
        for prompt in prompts:
            result = evaluate_prompt(client, prompt)
            group_results.append(result)
            call_count += RUNS

            # Display
            bar = "█" * int(max(0, result["avg_score"] + 5))
            marker = " ★" if result["avg_score"] >= 6.0 else ""
            print(f"    [{result['avg_score']:+.2f} var={result['variance']:.1f}] {prompt:45s} {bar}{marker}")

        # Sort group by score
        group_results.sort(key=lambda r: r["avg_score"], reverse=True)
        all_results[group_name] = group_results

        # Save after each group
        _save_results(all_results, call_count)

    # Final report
    _print_report(all_results, call_count)


def _save_results(all_results, call_count):
    results = {
        "experiment": "ablation-v1",
        "task": "race-condition + planted-hypothesis-resistance",
        "model": MODEL,
        "runs_per_prompt": RUNS,
        "total_api_calls": call_count,
        "groups": {
            name: [
                {k: v for k, v in r.items() if k != "details"}
                for r in group
            ]
            for name, group in all_results.items()
        },
        "full_details": all_results,
    }
    path = RESULTS_DIR / "ablation-latest.json"
    path.write_text(json.dumps(results, indent=2, default=str))


def _print_report(all_results, call_count):
    print("\n" + "=" * 70)
    print("  ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total API calls: {call_count}")

    # Baseline
    baseline = all_results["baseline"][0]
    print(f"\n  Baseline: [{baseline['avg_score']:+.2f} var={baseline['variance']:.1f}] {baseline['prompt']}")

    for group_name, results in all_results.items():
        if group_name == "baseline":
            continue

        print(f"\n  {group_name}:")
        for r in results:
            diff = r["avg_score"] - baseline["avg_score"]
            diff_str = f"{diff:+.2f}" if diff != 0 else " 0.00"
            bar = "█" * int(max(0, r["avg_score"] + 5))
            print(f"    [{r['avg_score']:+.2f} var={r['variance']:.1f}] ({diff_str}) {r['prompt']:40s} {bar}")

    # Key findings
    print(f"\n{'─' * 70}")
    print("  KEY COMPARISONS:")
    print(f"{'─' * 70}")

    # Best per group
    for group_name, results in all_results.items():
        if group_name == "baseline":
            continue
        best = results[0]
        worst = results[-1]
        print(f"  {group_name}:")
        print(f"    Best:  [{best['avg_score']:+.2f}] {best['prompt']}")
        print(f"    Worst: [{worst['avg_score']:+.2f}] {worst['prompt']}")
        print(f"    Range: {best['avg_score'] - worst['avg_score']:.2f}")

    print(f"\n  Results saved to: {RESULTS_DIR / 'ablation-latest.json'}")
    print("=" * 70)


if __name__ == "__main__":
    run_ablation()
