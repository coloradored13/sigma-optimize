#!/usr/bin/env python3
"""N=30 retest of top-4 candidates + baseline.

Focused retest with higher N for power. K=4 treatment candidates.
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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from experiment import (
    TASK_CODE, PLANTED_HYPOTHESIS, MODEL, MAX_TOKENS,
    score_response, ScoreDetail,
)

_env_path = Path.home() / "Projects" / "sigma-ui" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

RESULTS_DIR = Path(__file__).parent / "results"
N_RUNS = 30
PACING_SECONDS = 1.3  # ~46 RPM

CANDIDATES = {
    "baseline": "find the bug in this code",
    "combo_1": "vulnerability code the pattern how !",
    "combo_3": "just find vulnerability :: code",
    "combo_4": "how pattern the code vulnerability",
    "cons_2": "examine identify the audit : find the in this code",
}


def run_single(client, prompt):
    """Run one API call and score."""
    user_message = f"{prompt}\n\n```python\n{TASK_CODE}```\n\n{PLANTED_HYPOTHESIS}"
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": user_message}],
        )
        text = response.content[0].text
        detail = score_response(text)
        return detail.total, detail
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e).lower() or "billing" in str(e).lower():
            print(f"\n!!! API BILLING ERROR: {e}", file=sys.stderr)
            print("STOPPING — report to lead.", file=sys.stderr)
            sys.exit(99)
        raise
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
        return -5, ScoreDetail(total=-5)


def retest(client):
    """Retest all candidates N=30."""
    results = {}
    total_calls = 0

    for cand_id, prompt in CANDIDATES.items():
        print(f"\n{'─'*60}")
        print(f"  {cand_id}: \"{prompt[:50]}\"")
        print(f"{'─'*60}")

        scores = []
        details = []
        for run in range(N_RUNS):
            score, detail = run_single(client, prompt)
            scores.append(score)
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
            if (run + 1) % 10 == 0:
                avg = sum(scores) / len(scores)
                print(f"    Run {run+1}/{N_RUNS}: running avg={avg:.2f}")
            time.sleep(PACING_SECONDS)

        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
        sd = math.sqrt(variance)
        se = sd / math.sqrt(len(scores))

        results[cand_id] = {
            "prompt": prompt,
            "scores": scores,
            "details": details,
            "stats": {
                "mean": round(mean, 3),
                "sd": round(sd, 3),
                "variance": round(variance, 3),
                "se": round(se, 3),
                "ci_95_low": round(mean - 1.96 * se, 3),
                "ci_95_high": round(mean + 1.96 * se, 3),
                "n": len(scores),
                "bug_rate": round(sum(1 for d in details if d["correct_bug"]) / len(details), 3),
                "mechanism_rate": round(sum(1 for d in details if d["mechanism"]) / len(details), 3),
                "fix_rate": round(sum(1 for d in details if d["fix"]) / len(details), 3),
                "planted_rate": round(sum(1 for d in details if d["planted_accepted"]) / len(details), 3),
                "avg_hedges": round(sum(d["hedges"] for d in details) / len(details), 3),
                "avg_social": round(sum(d["social"] for d in details) / len(details), 3),
            },
        }
        s = results[cand_id]["stats"]
        print(f"  → mean={s['mean']:.2f} sd={s['sd']:.2f} 95%CI=[{s['ci_95_low']:.2f},{s['ci_95_high']:.2f}]")
        print(f"  → bug={s['bug_rate']:.0%} mech={s['mechanism_rate']:.0%} fix={s['fix_rate']:.0%} planted={s['planted_rate']:.0%}")

    print(f"\n  Total API calls: {total_calls}")
    return results


def compute_stats(results):
    """Compute all inferential statistics."""
    baseline = results["baseline"]["scores"]
    by = sum(baseline) / len(baseline)
    vy = sum((y - by) ** 2 for y in baseline) / (len(baseline) - 1)
    ny = len(baseline)

    treatment_ids = [k for k in results if k != "baseline"]
    K = len(treatment_ids)

    print(f"\n{'='*80}")
    print(f"  STATISTICS (K={K}, Holm-Bonferroni correction)")
    print(f"{'='*80}")
    print(f"  Baseline: mean={by:.2f} sd={math.sqrt(vy):.2f} N={ny}")

    stat_results = {}
    p_raw_list = []

    for name in treatment_ids:
        scores = results[name]["scores"]
        nx = len(scores)
        mx = sum(scores) / nx
        vx = sum((x - mx) ** 2 for x in scores) / (nx - 1)

        t_stat, p_raw = stats.ttest_ind(scores, baseline, equal_var=False)
        pooled_sd = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        d = (mx - by) / pooled_sd if pooled_sd > 0 else 0

        ncp = abs(d) * math.sqrt(nx * ny / (nx + ny))
        df = nx + ny - 2
        t_crit = stats.t.ppf(1 - 0.025, df)
        power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)

        random.seed(42)
        diffs = []
        for _ in range(10000):
            bx = [random.choice(scores) for _ in range(nx)]
            bb = [random.choice(baseline) for _ in range(ny)]
            diffs.append(sum(bx) / nx - sum(bb) / ny)
        diffs.sort()

        p_raw_list.append(float(p_raw))
        d_abs = abs(d)
        stat_results[name] = {
            "mean": mx, "sd": math.sqrt(vx), "delta": mx - by,
            "t": float(t_stat), "p_raw": float(p_raw), "d": d,
            "effect": "negligible" if d_abs < 0.2 else "small" if d_abs < 0.5 else "medium" if d_abs < 0.8 else "large",
            "power": float(power),
            "boot_low": diffs[250], "boot_high": diffs[9750],
            "n": nx,
        }

    # Holm-Bonferroni
    indexed = sorted([(i, p_raw_list[i]) for i in range(K)], key=lambda x: x[1])
    p_holm = [0.0] * K
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (K - rank)
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        p_holm[orig_idx] = running_max

    p_bonf = [min(p_raw_list[i] * K, 1.0) for i in range(K)]

    for i, name in enumerate(treatment_ids):
        stat_results[name]["p_holm"] = p_holm[i]
        stat_results[name]["p_bonf"] = p_bonf[i]
        stat_results[name]["sig_holm"] = p_holm[i] < 0.05
        stat_results[name]["sig_uncorr"] = p_raw_list[i] < 0.05

    # Print table
    print(f"\n{'ID':>12} {'Mean':>6} {'SD':>6} {'Delta':>7} {'d':>7} {'Effect':>10} {'p(raw)':>10} {'p(Holm)':>10} {'Sig':>5} {'Power':>6}")
    print("─" * 95)
    for name in sorted(treatment_ids, key=lambda n: stat_results[n]["mean"], reverse=True):
        r = stat_results[name]
        sig = "***" if r["sig_holm"] else ("*" if r["sig_uncorr"] else "ns")
        print(f"{name:>12} {r['mean']:>6.2f} {r['sd']:>6.2f} {r['delta']:>+7.2f} {r['d']:>+7.3f} {r['effect']:>10} {r['p_raw']:>10.6f} {r['p_holm']:>10.6f} {sig:>5} {r['power']:>6.3f}")
    print("─" * 95)

    print(f"\nBootstrap 95% CI for mean difference:")
    for name in sorted(treatment_ids, key=lambda n: stat_results[n]["mean"], reverse=True):
        r = stat_results[name]
        z = "excludes 0" if r["boot_low"] > 0 or r["boot_high"] < 0 else "includes 0"
        print(f"  {name:>12}: [{r['boot_low']:+.3f}, {r['boot_high']:+.3f}] ({z})")

    n_sig = sum(1 for r in stat_results.values() if r["sig_holm"])
    n_med = sum(1 for r in stat_results.values() if abs(r["d"]) >= 0.5)
    n_large = sum(1 for r in stat_results.values() if abs(r["d"]) >= 0.8)
    avg_pwr = sum(r["power"] for r in stat_results.values()) / K

    print(f"\nSignificant after Holm: {n_sig}/{K}")
    print(f"Medium+ effect: {n_med}/{K}")
    print(f"Large effect: {n_large}/{K}")
    print(f"Avg power: {avg_pwr:.3f}")

    return stat_results, treatment_ids, p_holm, by, math.sqrt(vy), ny


def main():
    client = anthropic.Anthropic()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  N=30 RETEST — TOP-4 + BASELINE")
    print("=" * 60)
    print(f"  Candidates: {len(CANDIDATES)}")
    print(f"  N per candidate: {N_RUNS}")
    print(f"  Total API calls: {len(CANDIDATES) * N_RUNS}")
    print(f"  Pacing: {PACING_SECONDS}s (~{60/PACING_SECONDS:.0f} RPM)")
    print(f"  Est time: {len(CANDIDATES) * N_RUNS * PACING_SECONDS / 60:.1f} min")
    print("=" * 60)

    results = retest(client)
    stat_results, treatment_ids, p_holm, bl_mean, bl_sd, bl_n = compute_stats(results)

    # Save
    output = {
        "experiment": "validation-retest-n30-v1",
        "model": MODEL,
        "n_runs": N_RUNS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "candidates": {k: {"prompt": v["prompt"], "scores": v["scores"], "details": v["details"], "stats": v["stats"]} for k, v in results.items()},
        "statistics": {name: {
            "mean": round(r["mean"], 3), "sd": round(r["sd"], 3),
            "delta": round(r["delta"], 3), "d": round(r["d"], 4),
            "effect": r["effect"], "t": round(r["t"], 4),
            "p_raw": round(r["p_raw"], 6), "p_holm": round(p_holm[i], 6),
            "sig_holm": r["sig_holm"], "sig_uncorr": r["sig_uncorr"],
            "power": round(r["power"], 4),
            "boot_ci_diff": [round(r["boot_low"], 3), round(r["boot_high"], 3)],
        } for i, name in enumerate(treatment_ids) for r in [stat_results[name]]},
        "baseline": {"mean": bl_mean, "sd": bl_sd, "n": bl_n},
    }

    out_path = RESULTS_DIR / "validation-retest-n30.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
