#!/usr/bin/env python3
"""Statistical analysis for validation retest results.

Computes: Welch's t-test, Cohen's d, bootstrap CI, Bonferroni correction,
power analysis for all candidates vs baseline.
"""

import json
import math
import random
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def welch_t_test(x, y):
    """Welch's t-test (unequal variance). Returns t-statistic, df, p-value."""
    nx, ny = len(x), len(y)
    mx, my = sum(x) / nx, sum(y) / ny
    vx = sum((xi - mx) ** 2 for xi in x) / (nx - 1)
    vy = sum((yi - my) ** 2 for yi in y) / (ny - 1)

    se = math.sqrt(vx / nx + vy / ny)
    if se == 0:
        return float('inf') if mx != my else 0.0, nx + ny - 2, 0.0 if mx != my else 1.0

    t = (mx - my) / se

    # Welch-Satterthwaite degrees of freedom
    num = (vx / nx + vy / ny) ** 2
    den = (vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1)
    df = num / den if den > 0 else nx + ny - 2

    # Two-tailed p-value using t-distribution approximation
    # Using the regularized incomplete beta function approximation
    p = two_tailed_p(abs(t), df)
    return t, df, p


def two_tailed_p(t, df):
    """Approximate two-tailed p-value from t-distribution.
    Uses the approximation from Abramowitz and Stegun.
    """
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    # Regularized incomplete beta function I_x(a, b) where a=df/2, b=0.5
    p = regularized_beta(x, df / 2, 0.5)
    return p


def regularized_beta(x, a, b, n_terms=200):
    """Regularized incomplete beta function using continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the continued fraction representation
    # First compute the log of the beta function
    log_beta = log_gamma(a) + log_gamma(b) - log_gamma(a + b)

    # Front factor
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - log_beta) / a

    # Lentz's continued fraction
    # I_x(a,b) = front * cf where cf is the continued fraction
    # Using the standard continued fraction for incomplete beta

    # If x < (a+1)/(a+b+2), use direct; else use 1 - I_{1-x}(b,a)
    if x > (a + 1) / (a + b + 2):
        return 1.0 - regularized_beta(1 - x, b, a, n_terms)

    # Continued fraction (Lentz's method)
    tiny = 1e-30
    f = tiny
    c = tiny
    d = 0.0

    for m in range(n_terms):
        if m == 0:
            alpha_m = 1.0
        else:
            # Even/odd terms
            k = m
            if k % 2 == 0:
                j = k // 2
                alpha_m = (j * (b - j) * x) / ((a + 2 * j - 1) * (a + 2 * j))
            else:
                j = (k + 1) // 2
                alpha_m = -((a + j) * (a + b + j) * x) / ((a + 2 * j) * (a + 2 * j + 1))

        beta_m = 1.0

        d = beta_m + alpha_m * d
        if abs(d) < tiny:
            d = tiny
        d = 1.0 / d

        c = beta_m + alpha_m / c
        if abs(c) < tiny:
            c = tiny

        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-12:
            break

    return front * (f - 1.0)  # subtract 1 because we started at tiny not 1


def log_gamma(x):
    """Lanczos approximation for log(Gamma(x))."""
    if x <= 0:
        return float('inf')
    coefs = [
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
    ]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for c in coefs:
        y += 1
        ser += c / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def cohens_d(x, y):
    """Cohen's d effect size (pooled SD)."""
    nx, ny = len(x), len(y)
    mx, my = sum(x) / nx, sum(y) / ny
    vx = sum((xi - mx) ** 2 for xi in x) / (nx - 1) if nx > 1 else 0
    vy = sum((yi - my) ** 2 for yi in y) / (ny - 1) if ny > 1 else 0
    pooled_sd = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return float('inf') if mx != my else 0.0
    return (mx - my) / pooled_sd


def bootstrap_ci(x, y, n_boot=10000, ci=0.95):
    """Bootstrap 95% CI for the mean difference (x - y)."""
    random.seed(42)  # reproducibility
    diffs = []
    for _ in range(n_boot):
        bx = [random.choice(x) for _ in range(len(x))]
        by = [random.choice(y) for _ in range(len(y))]
        diffs.append(sum(bx) / len(bx) - sum(by) / len(by))
    diffs.sort()
    alpha = (1 - ci) / 2
    low_idx = int(alpha * n_boot)
    high_idx = int((1 - alpha) * n_boot)
    return round(diffs[low_idx], 3), round(diffs[high_idx], 3)


def statistical_power(d, n1, n2, alpha=0.05):
    """Approximate power for two-sample t-test given effect size d and sample sizes.
    Uses non-central t approximation.
    """
    if d == 0:
        return alpha  # power = alpha under null
    # Non-centrality parameter
    ncp = d * math.sqrt(n1 * n2 / (n1 + n2))
    df = n1 + n2 - 2
    # Critical t-value (two-tailed)
    t_crit = t_critical(alpha / 2, df)
    # Power = P(|T| > t_crit) under non-central t
    # Approximate using normal distribution
    z = ncp - t_crit
    power = normal_cdf(z)
    return min(power, 0.999)


def t_critical(alpha, df):
    """Approximate t critical value using normal approximation for large df."""
    # For df > 30, t approaches z
    from math import sqrt
    # Use Cornish-Fisher expansion
    z = normal_ppf(1 - alpha)
    g1 = (z ** 3 + z) / (4 * df)
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
    return z + g1 + g2


def normal_cdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_ppf(p):
    """Inverse normal CDF (percent point function). Rational approximation."""
    if p <= 0:
        return -float('inf')
    if p >= 1:
        return float('inf')
    if p == 0.5:
        return 0.0

    if p < 0.5:
        return -normal_ppf(1 - p)

    t = math.sqrt(-2 * math.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)


def classify_effect(d):
    """Classify Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def main():
    # Load retest results
    retest_path = RESULTS_DIR / "validation-retest.json"
    if not retest_path.exists():
        print("ERROR: validation-retest.json not found. Run validation_retest.py first.")
        sys.exit(1)

    with open(retest_path) as f:
        data = json.load(f)

    candidates = data["candidates"]
    baseline_scores = candidates["baseline"]["scores"]
    baseline_mean = candidates["baseline"]["stats"]["mean"]
    baseline_sd = candidates["baseline"]["stats"]["sd"]

    print("=" * 80)
    print("  STATISTICAL ANALYSIS — VALIDATION RETEST")
    print("=" * 80)

    print(f"\n  Baseline: \"{candidates['baseline']['prompt']}\"")
    print(f"  Baseline scores: {baseline_scores}")
    print(f"  Baseline mean={baseline_mean:.3f} sd={baseline_sd:.3f}")
    print(f"  Baseline 95% CI: [{candidates['baseline']['stats']['ci_95_low']:.3f}, {candidates['baseline']['stats']['ci_95_high']:.3f}]")

    # Collect all treatment p-values for multiple comparison correction
    treatment_ids = [k for k in candidates if k != "baseline"]
    K = len(treatment_ids)
    bonferroni_alpha = 0.05 / K

    print(f"\n  Treatment candidates: {K}")
    print(f"  Bonferroni-corrected α: {bonferroni_alpha:.4f} (0.05 / {K})")

    # Compute statistics for each candidate
    results = {}
    raw_pvalues = []

    for cand_id in treatment_ids:
        cand = candidates[cand_id]
        scores = cand["scores"]
        mean = cand["stats"]["mean"]
        sd = cand["stats"]["sd"]

        # Welch's t-test
        t, df, p = welch_t_test(scores, baseline_scores)

        # Cohen's d
        d = cohens_d(scores, baseline_scores)

        # Bootstrap CI for mean difference
        boot_low, boot_high = bootstrap_ci(scores, baseline_scores)

        # Power analysis
        power = statistical_power(abs(d), len(scores), len(baseline_scores))

        raw_pvalues.append(p)

        results[cand_id] = {
            "prompt": cand["prompt"],
            "mean": mean,
            "sd": sd,
            "ci_95": [cand["stats"]["ci_95_low"], cand["stats"]["ci_95_high"]],
            "delta_vs_baseline": round(mean - baseline_mean, 3),
            "t_stat": round(t, 4),
            "df": round(df, 2),
            "p_raw": round(p, 6),
            "cohens_d": round(d, 4),
            "effect_class": classify_effect(d),
            "boot_ci_diff": [boot_low, boot_high],
            "power": round(power, 4),
            "n": len(scores),
        }

    # Apply Holm-Bonferroni correction
    holm_adjusted = holm_bonferroni(raw_pvalues)
    bonferroni_adjusted = [min(p * K, 1.0) for p in raw_pvalues]

    for i, cand_id in enumerate(treatment_ids):
        results[cand_id]["p_bonferroni"] = round(bonferroni_adjusted[i], 6)
        results[cand_id]["p_holm"] = round(holm_adjusted[i], 6)
        results[cand_id]["sig_bonferroni"] = bonferroni_adjusted[i] < 0.05
        results[cand_id]["sig_holm"] = holm_adjusted[i] < 0.05
        results[cand_id]["sig_uncorrected"] = raw_pvalues[i] < 0.05

    # Print results table
    print(f"\n{'─'*100}")
    print(f"  {'ID':<12} {'Mean':>6} {'SD':>6} {'Delta':>6} {'d':>7} {'Effect':>10} {'p(raw)':>9} {'p(Holm)':>9} {'Sig':>4} {'Power':>6}")
    print(f"{'─'*100}")

    for cand_id in sorted(results, key=lambda k: results[k]["mean"], reverse=True):
        r = results[cand_id]
        sig = "***" if r["sig_holm"] else ("*" if r["sig_uncorrected"] else "ns")
        print(f"  {cand_id:<12} {r['mean']:>6.2f} {r['sd']:>6.2f} {r['delta_vs_baseline']:>+6.2f} {r['cohens_d']:>+7.3f} {r['effect_class']:>10} {r['p_raw']:>9.5f} {r['p_holm']:>9.5f} {sig:>4} {r['power']:>6.3f}")

    print(f"{'─'*100}")
    print(f"  Baseline: mean={baseline_mean:.2f} sd={baseline_sd:.2f}")
    print(f"  Significance: *** = Holm-corrected p<0.05, * = uncorrected p<0.05, ns = not significant")

    # Count significant results
    n_sig_holm = sum(1 for r in results.values() if r["sig_holm"])
    n_sig_uncorr = sum(1 for r in results.values() if r["sig_uncorrected"])
    n_medium_plus = sum(1 for r in results.values() if abs(r["cohens_d"]) >= 0.5)
    n_large = sum(1 for r in results.values() if abs(r["cohens_d"]) >= 0.8)
    avg_power = sum(r["power"] for r in results.values()) / len(results)
    max_d = max(abs(r["cohens_d"]) for r in results.values())
    max_d_id = max(results, key=lambda k: abs(results[k]["cohens_d"]))

    print(f"\n  SUMMARY:")
    print(f"  Significant after Holm correction: {n_sig_holm}/{K}")
    print(f"  Significant uncorrected: {n_sig_uncorr}/{K}")
    print(f"  Medium+ effect size (|d|≥0.5): {n_medium_plus}/{K}")
    print(f"  Large effect size (|d|≥0.8): {n_large}/{K}")
    print(f"  Average power: {avg_power:.3f}")
    print(f"  Largest effect: {max_d_id} d={max_d:.3f} ({classify_effect(max_d)})")

    # Bootstrap details
    print(f"\n  BOOTSTRAP 95% CI FOR MEAN DIFFERENCE (candidate - baseline):")
    for cand_id in sorted(results, key=lambda k: results[k]["mean"], reverse=True):
        r = results[cand_id]
        print(f"  {cand_id:<12} [{r['boot_ci_diff'][0]:+.3f}, {r['boot_ci_diff'][1]:+.3f}]")

    # Adversarial results
    if "adversarial" in data:
        print(f"\n{'─'*80}")
        print(f"  ADVERSARIAL TEST RESULTS")
        print(f"{'─'*80}")
        for cand_id, adv in data["adversarial"].items():
            print(f"  {cand_id}: original={adv['original_mean']:.2f} | variant-A(off-by-one)={adv['variant_a']['avg']:.2f} | variant-B(diff-planted)={adv['variant_b']['avg']:.2f}")
            # Check for overfitting
            orig = adv["original_mean"]
            va = adv["variant_a"]["avg"]
            vb = adv["variant_b"]["avg"]
            # Overfit if variant scores drop by >50% of the original-baseline delta
            baseline_delta = orig - baseline_mean
            if baseline_delta > 0:
                drop_a = (orig - va) / baseline_delta if baseline_delta > 0 else 0
                drop_b = (orig - vb) / baseline_delta if baseline_delta > 0 else 0
                overfit_a = drop_a > 0.5
                overfit_b = drop_b > 0.5
                print(f"    Drop vs original: A={drop_a:.0%} B={drop_b:.0%} | Overfit: A={'YES' if overfit_a else 'no'} B={'YES' if overfit_b else 'no'}")

    # Save analysis
    analysis = {
        "experiment": "validation-stats-v1",
        "baseline": {
            "prompt": candidates["baseline"]["prompt"],
            "mean": baseline_mean,
            "sd": baseline_sd,
            "n": len(baseline_scores),
            "scores": baseline_scores,
        },
        "correction": {
            "method": "Holm-Bonferroni",
            "K": K,
            "bonferroni_alpha": bonferroni_alpha,
        },
        "results": results,
        "summary": {
            "n_sig_holm": n_sig_holm,
            "n_sig_uncorrected": n_sig_uncorr,
            "n_medium_effect": n_medium_plus,
            "n_large_effect": n_large,
            "avg_power": round(avg_power, 4),
            "max_effect_id": max_d_id,
            "max_effect_d": round(max_d, 4),
        },
    }

    out_path = RESULTS_DIR / "validation-stats.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"\n  Analysis saved to: {out_path}")


if __name__ == "__main__":
    main()
