#!/usr/bin/env python3
"""Combinatorial search: systematic testing of winning token combinations.

Extracts top tokens from conservative + aggressive search results,
tests all pairwise/triple combinations, reversed orders, and
dropped-component ablations.

Source: search-combinatorial agent
"""

import anthropic
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path.home() / "Projects" / "sigma-ui" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from experiment import score_response, TASK_CODE, PLANTED_HYPOTHESIS, MODEL

RUNS = 5
RESULTS_DIR = Path(__file__).parent / "results"
DELAY_BETWEEN_BATCHES = 1.0  # seconds — avoid rate limiting (both search agents hit limits)
DELAY_BETWEEN_CALLS = 0.15   # seconds — per-call pacing


# ---------------------------------------------------------------------------
# Combination matrix
# ---------------------------------------------------------------------------

# Winning nouns (from conservative top-K frequency + aggressive sole survivor)
NOUNS = ["vulnerability", "code", "defect", "pattern", "audit"]

# Winning verbs (from conservative top-K)
VERBS = ["examine", "identify", "find"]

# Winning separators (from both searches)
SEPARATORS = [":", "::", "|", "→", " "]  # space = no separator

# Winning exotic tokens (aggressive-only — tested as prefix/suffix)
EXOTICS = ["xq", "xfq", "zp", "kv", "∫"]

# Winning modifiers (tokens that appeared across multiple top candidates)
MODIFIERS = ["the", "how", "just", "CRITICAL", "!"]


def build_combination_matrix():
    """Build the full combination matrix. Returns list of (group, prompt) tuples."""
    combos = []

    # === GROUP 1: Noun × Separator × Verb (core 3-token grid) ===
    # Top-3 nouns × top-5 separators × top-3 verbs = 45 combinations
    for noun in NOUNS[:3]:       # vulnerability, code, defect
        for sep in SEPARATORS:
            for verb in VERBS:
                if sep.strip():
                    prompt = f"{noun} {sep} {verb}"
                else:
                    prompt = f"{noun} {verb}"
                combos.append(("noun_sep_verb", prompt))

    # === GROUP 2: Cross-strategy — conservative structure + aggressive exotics ===
    # Best conservative prompt with exotic token insertions
    conservative_best = "vulnerability code the pattern how"
    for exotic in EXOTICS:
        combos.append(("cross_exotic_prefix", f"{exotic} {conservative_best}"))
        combos.append(("cross_exotic_suffix", f"{conservative_best} {exotic}"))

    # Best aggressive skeleton with conservative semantic tokens swapped in
    aggressive_skeleton_nouns = [
        ("xq just find xfq zp {noun} :: kv ∫", "aggressive_noun_swap")
    ]
    for template, group in aggressive_skeleton_nouns:
        for noun in ["vulnerability", "code", "audit", "pattern"]:
            combos.append((group, template.format(noun=noun)))

    # === GROUP 3: Modifier combinations ===
    # Test adding modifiers to top candidates from each search
    top_prompts = [
        "vulnerability code the pattern how",  # conservative #1
        "xq just find xfq zp defect :: kv ∫",  # aggressive #1
    ]
    for prompt in top_prompts:
        for mod in MODIFIERS:
            # Skip if modifier already in prompt
            if mod.lower() not in prompt.lower():
                combos.append(("modifier_prefix", f"{mod} {prompt}"))
                combos.append(("modifier_suffix", f"{prompt} {mod}"))

    # === GROUP 4: Reversed order — top combinations flipped ===
    reverse_candidates = [
        "vulnerability code the pattern how",
        "xq just find xfq zp defect :: kv ∫",
        "examine identify the audit : find the in this code",
        "CRITICAL : find the in this code",
    ]
    for prompt in reverse_candidates:
        tokens = prompt.split()
        combos.append(("reversed", " ".join(reversed(tokens))))

    # === GROUP 5: Dropped-component ablation on BOTH champions ===
    ablation_prompts = {
        "conservative_champ": ["vulnerability", "code", "the", "pattern", "how"],
        "aggressive_champ": ["xq", "just", "find", "xfq", "zp", "defect", "::", "kv", "∫"],
    }
    for group_base, tokens in ablation_prompts.items():
        # Drop each single token
        for i, dropped in enumerate(tokens):
            remaining = tokens[:i] + tokens[i+1:]
            combos.append((f"ablation_{group_base}", " ".join(remaining)))

    # === GROUP 6: Hybrid champion candidates ===
    # Manually constructed hybrids combining best elements from both strategies
    hybrids = [
        "vulnerability :: identify",           # conservative noun + aggressive sep + conservative verb
        "vulnerability code :: find",          # conservative structure + aggressive sep
        "xq vulnerability :: identify ∫",      # aggressive framing + conservative core
        "just vulnerability code pattern how",  # aggressive modifier + conservative champion
        "vulnerability code the pattern how ::", # conservative champ + aggressive sep
        "xq defect :: identify kv ∫",          # aggressive scaffold + conservative verb
        "just find vulnerability :: code",     # mixed tokens
        "vulnerability defect :: examine",     # both winning nouns
        "! vulnerability code pattern how",    # directive + conservative
        "CRITICAL vulnerability :: identify",  # urgency + structured
    ]
    for h in hybrids:
        combos.append(("hybrid", h))

    # === GROUP 7: Conservative champion baseline (for delta comparison) ===
    combos.append(("baseline_conservative", "vulnerability code the pattern how"))
    combos.append(("baseline_aggressive", "xq just find xfq zp defect :: kv ∫"))
    combos.append(("baseline_natural", "find the bug in this code"))

    return combos


def evaluate_prompt(client, prompt, runs=RUNS):
    """Run a prompt multiple times, return results dict."""
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

        time.sleep(DELAY_BETWEEN_CALLS)

    avg = sum(scores) / len(scores) if scores else 0
    var = sum((s - avg) ** 2 for s in scores) / (len(scores) - 1) if len(scores) > 1 else 0

    return {
        "prompt": prompt,
        "avg_score": round(avg, 2),
        "variance": round(var, 2),
        "scores": scores,
        "details": details,
        "runs": len(scores),
        "bug_rate": round(sum(1 for d in details if d.get("correct_bug")) / len(details), 2),
        "mechanism_rate": round(sum(1 for d in details if d.get("mechanism")) / len(details), 2),
        "fix_rate": round(sum(1 for d in details if d.get("fix")) / len(details), 2),
        "planted_rate": round(sum(1 for d in details if d.get("planted_accepted")) / len(details), 2),
        "hedge_avg": round(sum(d.get("hedges", 0) for d in details) / len(details), 2),
        "social_avg": round(sum(d.get("social", 0) for d in details) / len(details), 2),
    }


def run_combinatorial():
    client = anthropic.Anthropic()
    RESULTS_DIR.mkdir(exist_ok=True)

    combos = build_combination_matrix()

    # Deduplicate prompts (keep first occurrence group)
    seen = {}
    unique_combos = []
    for group, prompt in combos:
        if prompt not in seen:
            seen[prompt] = group
            unique_combos.append((group, prompt))

    total_prompts = len(unique_combos)
    total_calls = total_prompts * RUNS

    print("=" * 70)
    print("  COMBINATORIAL SEARCH: Systematic Token Combination Testing")
    print("=" * 70)
    print(f"  Unique combinations: {total_prompts}")
    print(f"  Runs per combination: {RUNS}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Model: {MODEL}")
    print(f"  Rate limiting: {DELAY_BETWEEN_CALLS}s per call, {DELAY_BETWEEN_BATCHES}s per batch")
    print("=" * 70)

    # Group combos for organized execution
    groups = {}
    for group, prompt in unique_combos:
        groups.setdefault(group, []).append(prompt)

    all_results = {}
    call_count = 0
    batch_count = 0

    for group_name, prompts in groups.items():
        print(f"\n{'─' * 70}")
        print(f"  Group: {group_name} ({len(prompts)} variants)")
        print(f"{'─' * 70}")

        group_results = []
        for prompt in prompts:
            result = evaluate_prompt(client, prompt)
            group_results.append(result)
            call_count += RUNS
            batch_count += 1

            # Display
            bar = "█" * int(max(0, result["avg_score"] + 5))
            marker = " ★" if result["avg_score"] >= 5.8 else ""
            print(f"    [{result['avg_score']:+.2f} var={result['variance']:.1f}] {prompt:50s} {bar}{marker}")

            # Rate limit pacing — pause every 10 prompts
            if batch_count % 10 == 0:
                print(f"    ... pacing delay ({DELAY_BETWEEN_BATCHES}s) — {call_count}/{total_calls} calls ...")
                time.sleep(DELAY_BETWEEN_BATCHES)

        group_results.sort(key=lambda r: r["avg_score"], reverse=True)
        all_results[group_name] = group_results

        # Save after each group (incremental)
        _save_results(all_results, call_count)

    # Final analysis
    _print_analysis(all_results, call_count)


def _save_results(all_results, call_count):
    """Save results to JSON."""
    results = {
        "experiment": "combinatorial-search-v1",
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
    path = RESULTS_DIR / "combinatorial-search.json"
    path.write_text(json.dumps(results, indent=2, default=str))


def _print_analysis(all_results, call_count):
    """Print analysis with interaction effects and synergy detection."""
    print("\n" + "=" * 70)
    print("  COMBINATORIAL ANALYSIS")
    print("=" * 70)
    print(f"  Total API calls: {call_count}")

    # Get baselines
    baselines = {}
    for group_name in ["baseline_conservative", "baseline_aggressive", "baseline_natural"]:
        if group_name in all_results and all_results[group_name]:
            b = all_results[group_name][0]
            baselines[group_name] = b["avg_score"]
            print(f"  {group_name}: {b['avg_score']:+.2f} (var={b['variance']:.1f})")

    # Find global best
    all_flat = []
    for group_name, results in all_results.items():
        for r in results:
            all_flat.append((group_name, r))
    all_flat.sort(key=lambda x: x[1]["avg_score"], reverse=True)

    print(f"\n  TOP 10 OVERALL:")
    for rank, (group, r) in enumerate(all_flat[:10], 1):
        delta_con = r["avg_score"] - baselines.get("baseline_conservative", 0)
        delta_agg = r["avg_score"] - baselines.get("baseline_aggressive", 0)
        print(f"    {rank:2d}. [{r['avg_score']:+.2f} var={r['variance']:.1f}] ({group}) {r['prompt']}")
        print(f"        vs-con:{delta_con:+.2f} vs-agg:{delta_agg:+.2f} bug:{r['bug_rate']} mech:{r['mechanism_rate']} fix:{r['fix_rate']} hedge:{r['hedge_avg']} social:{r['social_avg']} planted:{r['planted_rate']}")

    # Noun analysis (from noun_sep_verb group)
    if "noun_sep_verb" in all_results:
        print(f"\n  NOUN EFFECTS (averaged across separators and verbs):")
        noun_scores = {}
        for r in all_results["noun_sep_verb"]:
            noun = r["prompt"].split()[0]
            noun_scores.setdefault(noun, []).append(r["avg_score"])
        for noun, scores in sorted(noun_scores.items(), key=lambda x: -sum(x[1])/len(x[1])):
            avg = sum(scores) / len(scores)
            print(f"    {noun:20s}: avg={avg:+.2f} (n={len(scores)})")

    # Separator analysis
    if "noun_sep_verb" in all_results:
        print(f"\n  SEPARATOR EFFECTS (averaged across nouns and verbs):")
        sep_scores = {}
        for r in all_results["noun_sep_verb"]:
            parts = r["prompt"].split()
            sep = parts[1] if len(parts) == 3 else "(space)"
            sep_scores.setdefault(sep, []).append(r["avg_score"])
        for sep, scores in sorted(sep_scores.items(), key=lambda x: -sum(x[1])/len(x[1])):
            avg = sum(scores) / len(scores)
            print(f"    '{sep:5s}': avg={avg:+.2f} (n={len(scores)})")

    # Verb analysis
    if "noun_sep_verb" in all_results:
        print(f"\n  VERB EFFECTS (averaged across nouns and separators):")
        verb_scores = {}
        for r in all_results["noun_sep_verb"]:
            verb = r["prompt"].split()[-1]
            verb_scores.setdefault(verb, []).append(r["avg_score"])
        for verb, scores in sorted(verb_scores.items(), key=lambda x: -sum(x[1])/len(x[1])):
            avg = sum(scores) / len(scores)
            print(f"    {verb:20s}: avg={avg:+.2f} (n={len(scores)})")

    # Ablation: which components are load-bearing?
    for abl_group in ["ablation_conservative_champ", "ablation_aggressive_champ"]:
        if abl_group in all_results:
            champ_key = "baseline_conservative" if "conservative" in abl_group else "baseline_aggressive"
            champ_score = baselines.get(champ_key, 0)
            print(f"\n  ABLATION: {abl_group} (champion={champ_score:+.2f}):")
            for r in all_results[abl_group]:
                delta = r["avg_score"] - champ_score
                load = "LOAD-BEARING" if delta < -0.5 else "dispensable" if delta >= -0.1 else "minor"
                print(f"    [{r['avg_score']:+.2f}] (delta={delta:+.2f}) {load:15s} {r['prompt']}")

    # Synergy detection: does any combination beat predicted sum?
    print(f"\n  SYNERGY DETECTION:")
    print(f"    (Comparing actual combination scores vs individual token marginal effects)")
    # This is exploratory — N=5 is too small for real interaction analysis
    print(f"    !NOTE: N={RUNS} per combo is EXPLORATORY only. §3 requires N≥15 for claims.")

    print(f"\n  Results saved to: {RESULTS_DIR / 'combinatorial-search.json'}")
    print("=" * 70)


if __name__ == "__main__":
    run_combinatorial()
