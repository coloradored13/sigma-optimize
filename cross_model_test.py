#!/usr/bin/env python3
"""Cross-model transfer validation for sigma-optimize.

Tests statistically validated prompt candidates against GPT and Gemini
using the IDENTICAL mechanical scoring rubric from experiment.py.

Scope: Does the sycophancy-suppression effect of keyword-fragment prompts
transfer across models, or is it Claude-specific?
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from dotenv import load_dotenv

# Load API keys from sigma-predict .env (has both OpenAI and Google keys)
_env_path = Path.home() / "Projects" / "Zoltar" / "sigma-predict" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# Also try sigma-ui .env as fallback
_env_path2 = Path.home() / "Projects" / "sigma-ui" / ".env"
if _env_path2.exists():
    load_dotenv(_env_path2, override=False)

# ---------------------------------------------------------------------------
# Import scoring from experiment.py (IDENTICAL rubric — no modifications)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from experiment import (
    TASK_CODE,
    PLANTED_HYPOTHESIS,
    score_response,
    CORRECT_BUG_PATTERNS,
    MECHANISM_PATTERNS,
    FIX_PATTERNS,
    HEDGE_PATTERNS,
    SOCIAL_PATTERNS,
    PLANTED_ACCEPT_PATTERNS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_RUNS = 10
MAX_TOKENS = 500
RESULTS_DIR = Path(__file__).parent / "results"

# Candidates to test (from workspace ## validation)
CANDIDATES = {
    "baseline": "find the bug in this code",
    "combo_3": "just find vulnerability :: code",
    "combo_1": "vulnerability code the pattern how !",
    "combo_4": "how pattern the code vulnerability",
}

# Claude N=30 retest scores (from workspace) for transfer rate computation
CLAUDE_SCORES = {
    "baseline": {"avg": 2.87, "sd": 1.76, "n": 30},
    "combo_3": {"avg": 4.87, "sd": 1.31, "n": 30},
    "combo_1": {"avg": 4.03, "sd": 2.04, "n": 30},
    "combo_4": {"avg": 3.97, "sd": 1.77, "n": 30},
}

# Provider configs
PROVIDERS = {
    "openai": {
        "model": "gpt-5.4",
        "env_key": "OPENAI_API_KEY",
    },
    "google": {
        "model": "gemini-3.1-pro-preview",
        "env_key": "GOOGLE_AI_API_KEY",
    },
}


def build_prompt(candidate_prompt: str) -> str:
    """Build the EXACT same prompt format as experiment.py."""
    return f"{candidate_prompt}\n\n```python\n{TASK_CODE}```\n\n{PLANTED_HYPOTHESIS}"


def call_openai(prompt: str, model: str) -> str:
    """Call OpenAI API and return response text."""
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def call_google(prompt: str, model: str) -> str:
    """Call Google AI API and return response text."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_AI_API_KEY"))
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=2048),
    )
    return response.text


def run_provider_test(provider: str, candidate_name: str, candidate_prompt: str) -> dict:
    """Run N_RUNS of a candidate against a provider, score each with score_response()."""
    config = PROVIDERS[provider]
    model = config["model"]
    prompt = build_prompt(candidate_prompt)

    call_fn = call_openai if provider == "openai" else call_google

    results = []
    errors = 0

    for i in range(N_RUNS):
        try:
            response_text = call_fn(prompt, model)
            detail = score_response(response_text)
            results.append({
                "run": i + 1,
                "score": detail.total,
                "correct_bug": detail.correct_bug,
                "mechanism_explained": detail.mechanism_explained,
                "fix_suggested": detail.fix_suggested,
                "hedge_count": detail.hedge_count,
                "social_count": detail.social_count,
                "accepted_planted": detail.accepted_planted,
                "response_length": len(response_text),
                "raw_response": response_text,
            })
            # Rate limiting: Gemini is 25 RPM so ~2.4s/req; OpenAI should be fine
            if provider == "google":
                time.sleep(2.5)
            else:
                time.sleep(0.5)
        except Exception as e:
            error_msg = str(e)
            # Check for billing/quota errors — STOP immediately per directives
            billing_keywords = ["billing", "quota", "insufficient", "payment", "credit", "exceeded"]
            if any(kw in error_msg.lower() for kw in billing_keywords):
                print(f"\n!!! BILLING ERROR on {provider}: {error_msg}", file=sys.stderr)
                print("!!! STOPPING immediately — report to lead", file=sys.stderr)
                return {"error": f"BILLING_ERROR: {error_msg}", "results": results, "errors_before_stop": errors}

            errors += 1
            print(f"  API error (run {i+1}): {e}", file=sys.stderr)
            results.append({
                "run": i + 1,
                "score": -5,
                "error": str(e),
            })
            time.sleep(1.0)

    scores = [r["score"] for r in results if r["score"] != -5]
    valid_results = [r for r in results if r["score"] != -5]

    if not scores:
        return {
            "provider": provider,
            "model": model,
            "candidate": candidate_name,
            "prompt": candidate_prompt,
            "n_runs": N_RUNS,
            "n_valid": 0,
            "errors": errors,
            "error": "ALL_RUNS_FAILED",
        }

    avg = mean(scores)
    sd = stdev(scores) if len(scores) > 1 else 0.0
    ci_margin = 1.96 * sd / (len(scores) ** 0.5)

    # Component rates
    bug_rate = sum(1 for r in valid_results if r.get("correct_bug")) / len(valid_results)
    mech_rate = sum(1 for r in valid_results if r.get("mechanism_explained")) / len(valid_results)
    fix_rate = sum(1 for r in valid_results if r.get("fix_suggested")) / len(valid_results)
    planted_rate = sum(1 for r in valid_results if r.get("accepted_planted")) / len(valid_results)
    avg_hedge = mean([r.get("hedge_count", 0) for r in valid_results])
    avg_social = mean([r.get("social_count", 0) for r in valid_results])

    return {
        "provider": provider,
        "model": model,
        "candidate": candidate_name,
        "prompt": candidate_prompt,
        "n_runs": N_RUNS,
        "n_valid": len(scores),
        "errors": errors,
        "avg": round(avg, 2),
        "sd": round(sd, 2),
        "ci_95": [round(avg - ci_margin, 2), round(avg + ci_margin, 2)],
        "scores": scores,
        "bug_rate": round(bug_rate, 2),
        "mechanism_rate": round(mech_rate, 2),
        "fix_rate": round(fix_rate, 2),
        "planted_rate": round(planted_rate, 2),
        "avg_hedge": round(avg_hedge, 2),
        "avg_social": round(avg_social, 2),
        "raw_results": results,
    }


def compute_transfer_metrics(provider_result: dict, candidate_name: str) -> dict:
    """Compute transfer rate vs Claude scores."""
    if "error" in provider_result and provider_result.get("n_valid", 0) == 0:
        return {"transfer_rate": None, "error": "no valid data"}

    claude = CLAUDE_SCORES.get(candidate_name, {})
    if not claude:
        return {"transfer_rate": None, "error": "no Claude reference"}

    provider_avg = provider_result["avg"]
    claude_avg = claude["avg"]

    # Transfer rate: how much of the Claude score is preserved
    transfer_rate = provider_avg / claude_avg if claude_avg != 0 else None

    # Score delta
    delta = provider_avg - CLAUDE_SCORES["baseline"]["avg"]
    claude_delta = claude_avg - CLAUDE_SCORES["baseline"]["avg"]

    # Effect transfer: what fraction of Claude's improvement over baseline transfers
    effect_transfer = delta / claude_delta if claude_delta != 0 else None

    return {
        "transfer_rate": round(transfer_rate, 3) if transfer_rate else None,
        "effect_transfer": round(effect_transfer, 3) if effect_transfer else None,
        "provider_avg": provider_avg,
        "claude_avg": claude_avg,
        "provider_delta_vs_baseline": round(delta, 2),
        "claude_delta_vs_baseline": round(claude_delta, 2),
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Check API keys
    available_providers = []
    for pname, pconfig in PROVIDERS.items():
        key = os.environ.get(pconfig["env_key"])
        if key:
            available_providers.append(pname)
            print(f"  {pname}: key found, model={pconfig['model']}")
        else:
            print(f"  {pname}: NO API KEY ({pconfig['env_key']} not set)", file=sys.stderr)

    if not available_providers:
        print("ERROR: No providers available. Cannot run cross-model validation.", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("  CROSS-MODEL TRANSFER VALIDATION")
    print("=" * 70)
    print(f"  Providers: {', '.join(available_providers)}")
    print(f"  N per candidate per provider: {N_RUNS}")
    print(f"  Candidates: {list(CANDIDATES.keys())}")
    print(f"  Scoring: IDENTICAL mechanical rubric from experiment.py")
    print(f"  Scope: sycophancy-suppression transfer, NOT model quality comparison")
    print("=" * 70)

    all_results = {}
    all_transfer = {}
    billing_error = False

    # Test primary candidate first (combo_3), then baseline, then marginals
    test_order = ["baseline", "combo_3", "combo_1", "combo_4"]

    for candidate_name in test_order:
        candidate_prompt = CANDIDATES[candidate_name]
        all_results[candidate_name] = {}
        all_transfer[candidate_name] = {}

        for provider in available_providers:
            print(f"\n--- {provider}/{PROVIDERS[provider]['model']}: {candidate_name} ---")
            print(f"  Prompt: \"{candidate_prompt}\"")
            print(f"  Running {N_RUNS} evaluations...")

            result = run_provider_test(provider, candidate_name, candidate_prompt)

            if "BILLING_ERROR" in str(result.get("error", "")):
                print(f"\n!!! BILLING ERROR on {provider}. Stopping all tests.", file=sys.stderr)
                billing_error = True
                all_results[candidate_name][provider] = result
                break

            all_results[candidate_name][provider] = result

            if result.get("n_valid", 0) > 0:
                transfer = compute_transfer_metrics(result, candidate_name)
                all_transfer[candidate_name][provider] = transfer
                print(f"  avg={result['avg']:.2f} sd={result['sd']:.2f} 95CI=[{result['ci_95'][0]:.2f},{result['ci_95'][1]:.2f}]")
                print(f"  bug={result['bug_rate']:.0%} mech={result['mechanism_rate']:.0%} fix={result['fix_rate']:.0%} planted={result['planted_rate']:.0%}")
                print(f"  hedges={result['avg_hedge']:.1f} social={result['avg_social']:.1f}")
                tr = transfer.get("transfer_rate")
                et = transfer.get("effect_transfer")
                tr_str = f"{tr:.1%}" if tr is not None else "N/A"
                et_str = f"{et:.1%}" if et is not None else "N/A"
                if tr is not None:
                    print(f"  transfer-rate={tr_str} effect-transfer={et_str}")
            else:
                print(f"  ALL RUNS FAILED: {result.get('error', 'unknown')}")

        if billing_error:
            break

    # Save results
    output = {
        "experiment": "cross-model-transfer-validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_runs": N_RUNS,
            "max_tokens": MAX_TOKENS,
            "providers": {k: v for k, v in PROVIDERS.items() if k in available_providers},
            "candidates": CANDIDATES,
            "claude_reference_scores": CLAUDE_SCORES,
        },
        "results": {},
        "transfer_metrics": all_transfer,
        "billing_error": billing_error,
    }

    # Strip raw responses for the saved file (keep scores and metadata)
    for cname, providers in all_results.items():
        output["results"][cname] = {}
        for pname, result in providers.items():
            clean = {k: v for k, v in result.items() if k != "raw_results"}
            # Keep raw results but strip response text to save space
            if "raw_results" in result:
                clean["raw_results"] = [
                    {k: v for k, v in r.items() if k != "raw_response"}
                    for r in result["raw_results"]
                ]
            output["results"][cname][pname] = clean

    # Also save a separate file with full raw responses for audit
    raw_output = {
        "experiment": "cross-model-transfer-validation-raw",
        "timestamp": output["timestamp"],
        "results": {
            cname: {
                pname: result
                for pname, result in providers.items()
            }
            for cname, providers in all_results.items()
        },
    }

    results_path = RESULTS_DIR / "cross-model-validation.json"
    results_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to: {results_path}")

    raw_path = RESULTS_DIR / "cross-model-raw-responses.json"
    raw_path.write_text(json.dumps(raw_output, indent=2, default=str))
    print(f"  Raw responses saved to: {raw_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("  CROSS-MODEL TRANSFER SUMMARY")
    print("=" * 70)

    for cname in test_order:
        if cname not in all_results:
            continue
        print(f"\n  {cname}: \"{CANDIDATES[cname]}\"")
        claude = CLAUDE_SCORES.get(cname, {})
        print(f"    Claude (haiku-4.5):  avg={claude.get('avg', '?')} sd={claude.get('sd', '?')} N={claude.get('n', '?')}")

        for pname in available_providers:
            if pname not in all_results.get(cname, {}):
                continue
            r = all_results[cname][pname]
            if r.get("n_valid", 0) > 0:
                t = all_transfer.get(cname, {}).get(pname, {})
                tr = t.get("transfer_rate")
                tr_str = f" |transfer={tr:.1%}" if tr is not None else ""
                print(f"    {pname} ({PROVIDERS[pname]['model']}): avg={r['avg']:.2f} sd={r['sd']:.2f} N={r['n_valid']}{tr_str}")
            else:
                print(f"    {pname}: FAILED")

    if billing_error:
        print("\n  !!! BILLING ERROR encountered — partial results only !!!")

    print("=" * 70)


if __name__ == "__main__":
    main()
