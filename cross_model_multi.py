#!/usr/bin/env python3
"""Cross-model validation for sigma-optimize Experiment 2.

Tests whether validated findings transfer across model families.
Uses OpenAI (GPT-5.4) and Google (Gemini 3.1 Pro) via their SDKs,
scores with the SAME mechanical rubric from experiment_multi.py.

Candidates:
  COMBO-2: "!find : vulnerability | code → fix" (best on divergent_problem, d=1.047)
  COMBO-1: "flaw reliability evidence | bias → ? fix" (second best, d=0.695)
  BASELINE: "identify the flaws in this reasoning" (best NL baseline)

Tests all 4 tasks at N=10 per candidate per provider.
Total: 3 candidates × 4 tasks × 10 runs × 2 providers = 240 calls.

Results: results/multi-cross-model.json
"""

import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Load API keys from ~/.claude.json mcpServers.sigma-verify.env
_claude_json = Path.home() / ".claude.json"
if _claude_json.exists():
    with open(_claude_json) as f:
        _config = json.load(f)
    _sv_env = _config.get("mcpServers", {}).get("sigma-verify", {}).get("env", {})
    for k, v in _sv_env.items():
        if v and k not in os.environ:
            os.environ[k] = v

# Now import provider SDKs
from openai import OpenAI
from google import genai

# Import scoring from experiment_multi.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from experiment_multi import (
    score_response,
    score_response_agnostic,
    ALL_TASKS,
    TASK_1, TASK_2, TASK_3, TASK_4,
)

# Scoring mode: "regex" (original) or "behavioral" (model-agnostic, B2)
SCORING_MODE = "regex"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RUNS_PER_TASK = 10
MAX_TOKENS = 500
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "multi-cross-model.json"

# Candidates to test
CANDIDATES = {
    "COMBO-2": "!find : vulnerability | code → fix",
    "COMBO-1": "flaw reliability evidence | bias → ? fix",
    "BASELINE": "identify the flaws in this reasoning",
}

# Provider configs
PROVIDERS = {
    "openai": {
        "model": "gpt-5.4",
        "max_tokens_key": "max_completion_tokens",
    },
    "google": {
        "model": "gemini-3.1-pro-preview",
    },
}

# Rate limiting
OPENAI_DELAY = 0.3    # seconds between OpenAI calls
GOOGLE_DELAY = 0.5    # seconds between Google calls
MAX_RETRIES = 5
CALL_TIMEOUT = 90     # seconds per external provider call (A4: replaces signal.alarm)

# ---------------------------------------------------------------------------
# Model tiers (A6) — tier-matched cross-model testing
# ---------------------------------------------------------------------------
MODEL_TIERS = {
    "search": {
        # Cheap, high volume — used during evolutionary search
        "anthropic": "claude-haiku-4-5-20251001",
    },
    "validation": {
        # Mid-tier — used during statistical validation
        "anthropic": "claude-sonnet-4-5-20250514",
    },
    "cross_model": {
        # Matched peers — used for cross-model comparison
        "anthropic": "claude-sonnet-4-5-20250514",
        "openai": "gpt-5.4",
        "google": "gemini-3.1-pro-preview",
    },
}


# ---------------------------------------------------------------------------
# Provider call functions (async with timeout — A4)
# ---------------------------------------------------------------------------
_openai_client = None
_google_client = None


async def call_openai(prompt: str, stimulus: str, planted: str) -> str:
    """Call OpenAI GPT-5.4 with the experiment prompt format."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""), timeout=60.0)
    user_message = f"{prompt}\n\n{stimulus}\n\n{planted}"

    # Run sync SDK call in thread pool to allow asyncio.wait_for timeout
    loop = asyncio.get_event_loop()
    response = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: _openai_client.chat.completions.create(
                model=PROVIDERS["openai"]["model"],
                messages=[{"role": "user", "content": user_message}],
                temperature=1.0,
                max_completion_tokens=MAX_TOKENS,
            ),
        ),
        timeout=CALL_TIMEOUT,
    )
    return response.choices[0].message.content or ""


async def call_google(prompt: str, stimulus: str, planted: str) -> str:
    """Call Google Gemini 3.1 Pro with the experiment prompt format."""
    global _google_client
    if _google_client is None:
        _google_client = genai.Client(api_key=os.environ.get("GOOGLE_AI_API_KEY", ""))
    user_message = f"{prompt}\n\n{stimulus}\n\n{planted}"

    loop = asyncio.get_event_loop()
    response = await asyncio.wait_for(
        loop.run_in_executor(
            None,
            lambda: _google_client.models.generate_content(
                model=PROVIDERS["google"]["model"],
                contents=user_message,
                config=genai.types.GenerateContentConfig(
                    temperature=1.0,
                    max_output_tokens=MAX_TOKENS,
                ),
            ),
        ),
        timeout=CALL_TIMEOUT,
    )
    return response.text or ""


PROVIDER_CALLERS = {
    "openai": call_openai,
    "google": call_google,
}

PROVIDER_DELAYS = {
    "openai": OPENAI_DELAY,
    "google": GOOGLE_DELAY,
}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
async def evaluate_single(
    provider: str,
    candidate_name: str,
    candidate_prompt: str,
    task: dict,
    run_idx: int,
) -> dict:
    """Run a single evaluation: provider x candidate x task x run.
    Uses asyncio.wait_for for timeout instead of signal.alarm (A4).
    """
    task_id = task["id"]
    stimulus = task["stimulus"]
    planted = task["planted_hypothesis"]
    caller = PROVIDER_CALLERS[provider]
    error_str = ""

    for attempt in range(MAX_RETRIES):
        try:
            # A4: asyncio.wait_for replaces signal.alarm — works with async and threads
            response_text = await caller(candidate_prompt, stimulus, planted)

            scorer = score_response if SCORING_MODE == "regex" else score_response_agnostic
            detail = scorer(response_text, task_id)
            return {
                "provider": provider,
                "model": PROVIDERS[provider]["model"],
                "candidate": candidate_name,
                "prompt": candidate_prompt,
                "task_id": task_id,
                "run": run_idx,
                "score": detail.total,
                "correct_identification": detail.correct_identification,
                "mechanism_explained": detail.mechanism_explained,
                "fix_suggested": detail.fix_suggested,
                "hedge_count": detail.hedge_count,
                "social_count": detail.social_count,
                "accepted_planted": detail.accepted_planted,
                "response_text": response_text,
                "error": None,
            }
        except asyncio.TimeoutError:
            error_str = f"TimeoutError: API call timed out after {CALL_TIMEOUT}s"
            wait = min(2 ** attempt * 2, 60)
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {provider}/{candidate_name}/{task_id}/run{run_idx}: {error_str} — waiting {wait}s", flush=True)
            await asyncio.sleep(wait)
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            wait = min(2 ** attempt * 2, 60)
            print(f"  [retry {attempt+1}/{MAX_RETRIES}] {provider}/{candidate_name}/{task_id}/run{run_idx}: {error_str} — waiting {wait}s", flush=True)
            await asyncio.sleep(wait)

    # All retries exhausted
    return {
        "provider": provider,
        "model": PROVIDERS[provider]["model"],
        "candidate": candidate_name,
        "prompt": candidate_prompt,
        "task_id": task_id,
        "run": run_idx,
        "score": -5,
        "correct_identification": False,
        "mechanism_explained": False,
        "fix_suggested": False,
        "hedge_count": 0,
        "social_count": 0,
        "accepted_planted": False,
        "response_text": "",
        "error": error_str,
    }


async def run_provider(provider: str) -> list[dict]:
    """Run all candidates x tasks x runs for a single provider (async)."""
    results = []
    delay = PROVIDER_DELAYS[provider]
    total = len(CANDIDATES) * len(ALL_TASKS) * RUNS_PER_TASK
    done = 0

    for cand_name, cand_prompt in CANDIDATES.items():
        for task in ALL_TASKS:
            for run_idx in range(RUNS_PER_TASK):
                result = await evaluate_single(provider, cand_name, cand_prompt, task, run_idx)
                results.append(result)
                done += 1

                score_str = f"{result['score']:+d}" if result['error'] is None else "ERR"
                print(f"  [{provider}] {done}/{total} {cand_name}/{task['id']}/r{run_idx}: {score_str}", flush=True)

                if done < total:
                    await asyncio.sleep(delay)

    return results


# ---------------------------------------------------------------------------
# Provider quota pre-check (A5)
# ---------------------------------------------------------------------------
def check_provider_quotas() -> dict[str, dict]:
    """Check remaining quota for each provider before starting.
    Returns dict of {provider: {status, message, remaining_estimate}}.
    """
    quotas = {}

    # Gemini: 250 req/day limit on paid tier
    if "google" in PROVIDERS:
        total_google_calls = len(CANDIDATES) * len(ALL_TASKS) * RUNS_PER_TASK
        quotas["google"] = {
            "status": "warning" if total_google_calls > 200 else "ok",
            "planned_calls": total_google_calls,
            "daily_limit": 250,
            "message": (
                f"Planning {total_google_calls} calls against 250/day limit. "
                f"{'WARNING: close to or exceeding daily quota!' if total_google_calls > 200 else 'Within quota.'}"
            ),
        }

    # OpenAI: estimate cost
    if "openai" in PROVIDERS:
        total_openai_calls = len(CANDIDATES) * len(ALL_TASKS) * RUNS_PER_TASK
        # GPT-5.4 pricing estimate: ~$2.50/MTok input, ~$10/MTok output
        est_input = total_openai_calls * 300 / 1_000_000
        est_output = total_openai_calls * 400 / 1_000_000
        est_cost = est_input * 2.50 + est_output * 10.00
        quotas["openai"] = {
            "status": "ok",
            "planned_calls": total_openai_calls,
            "estimated_cost_usd": round(est_cost, 2),
            "message": f"Planning {total_openai_calls} calls, est. cost ~${est_cost:.2f}",
        }

    return quotas


# ---------------------------------------------------------------------------
# Incremental save (A2) — save partial results after each provider
# ---------------------------------------------------------------------------
PARTIAL_RESULTS_FILE = RESULTS_DIR / "multi-cross-model-partial.json"


def save_partial_results(provider: str, results: list[dict]):
    """Save results incrementally after each provider completes."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing partial results if any
    existing = []
    if PARTIAL_RESULTS_FILE.exists():
        try:
            with open(PARTIAL_RESULTS_FILE) as f:
                data = json.load(f)
            existing = data.get("raw_results", [])
        except (json.JSONDecodeError, KeyError):
            pass

    # Append new results
    all_results = existing + [
        {k: v for k, v in r.items() if k != "response_text"}
        for r in results
    ]

    partial = {
        "experiment": "sigma-optimize-exp2-cross-model-partial",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers_completed": list(set(r["provider"] for r in all_results)),
        "total_results": len(all_results),
        "last_provider": provider,
        "raw_results": all_results,
    }

    with open(PARTIAL_RESULTS_FILE, "w") as f:
        json.dump(partial, f, indent=2)
    print(f"  Partial results saved ({len(all_results)} total) to {PARTIAL_RESULTS_FILE}", flush=True)


def compute_statistics(results: list[dict]) -> dict:
    """Compute per-provider, per-candidate, per-task statistics."""
    import statistics

    stats = {}
    for provider in PROVIDERS:
        stats[provider] = {"model": PROVIDERS[provider]["model"], "candidates": {}}
        for cand_name in CANDIDATES:
            cand_stats = {"prompt": CANDIDATES[cand_name], "tasks": {}, "aggregate": {}}
            all_task_avgs = []

            for task in ALL_TASKS:
                task_id = task["id"]
                scores = [
                    r["score"] for r in results
                    if r["provider"] == provider
                    and r["candidate"] == cand_name
                    and r["task_id"] == task_id
                    and r["error"] is None
                ]
                if not scores:
                    cand_stats["tasks"][task_id] = {"n": 0, "error": "all runs failed"}
                    continue

                n = len(scores)
                mean = statistics.mean(scores)
                sd = statistics.stdev(scores) if n > 1 else 0.0
                se = sd / (n ** 0.5) if n > 0 else 0.0
                ci_lo = mean - 1.96 * se
                ci_hi = mean + 1.96 * se

                # Component analysis
                task_results = [
                    r for r in results
                    if r["provider"] == provider
                    and r["candidate"] == cand_name
                    and r["task_id"] == task_id
                    and r["error"] is None
                ]
                correct_rate = sum(1 for r in task_results if r["correct_identification"]) / n
                mechanism_rate = sum(1 for r in task_results if r["mechanism_explained"]) / n
                fix_rate = sum(1 for r in task_results if r["fix_suggested"]) / n
                planted_rate = sum(1 for r in task_results if r["accepted_planted"]) / n
                avg_hedges = statistics.mean([r["hedge_count"] for r in task_results])
                avg_social = statistics.mean([r["social_count"] for r in task_results])

                cand_stats["tasks"][task_id] = {
                    "n": n,
                    "mean": round(mean, 3),
                    "sd": round(sd, 3),
                    "se": round(se, 3),
                    "ci_95": [round(ci_lo, 3), round(ci_hi, 3)],
                    "scores": scores,
                    "components": {
                        "correct_identification_rate": round(correct_rate, 3),
                        "mechanism_explained_rate": round(mechanism_rate, 3),
                        "fix_suggested_rate": round(fix_rate, 3),
                        "accepted_planted_rate": round(planted_rate, 3),
                        "avg_hedge_count": round(avg_hedges, 3),
                        "avg_social_count": round(avg_social, 3),
                    },
                }
                all_task_avgs.append(mean)

            # Aggregate across tasks
            if all_task_avgs:
                agg_mean = statistics.mean(all_task_avgs)
                agg_sd = statistics.stdev(all_task_avgs) if len(all_task_avgs) > 1 else 0.0
                cand_stats["aggregate"] = {
                    "mean_of_task_means": round(agg_mean, 3),
                    "sd_of_task_means": round(agg_sd, 3),
                    "task_means": {tid: round(m, 3) for tid, m in zip(
                        [t["id"] for t in ALL_TASKS], all_task_avgs
                    )},
                }

            stats[provider]["candidates"][cand_name] = cand_stats

    return stats


def compute_transfer_rates(stats: dict, claude_data: dict) -> dict:
    """Compute transfer rates: external_score / claude_score for each candidate × task."""
    transfers = {}
    for provider in PROVIDERS:
        transfers[provider] = {}
        for cand_name in CANDIDATES:
            transfers[provider][cand_name] = {}
            ext_cand = stats[provider]["candidates"].get(cand_name, {})
            claude_cand = claude_data.get(cand_name, {})

            for task in ALL_TASKS:
                task_id = task["id"]
                ext_task = ext_cand.get("tasks", {}).get(task_id, {})
                claude_task = claude_cand.get("tasks", {}).get(task_id, {})

                ext_mean = ext_task.get("mean")
                claude_mean = claude_task.get("mean")

                if ext_mean is not None and claude_mean is not None and claude_mean != 0:
                    transfers[provider][cand_name][task_id] = {
                        "external_mean": round(ext_mean, 3),
                        "claude_mean": round(claude_mean, 3),
                        "transfer_rate": round(ext_mean / claude_mean, 3),
                        "delta": round(ext_mean - claude_mean, 3),
                    }

            # Aggregate transfer
            ext_agg = ext_cand.get("aggregate", {}).get("mean_of_task_means")
            claude_agg = claude_cand.get("aggregate", {}).get("mean")
            if ext_agg is not None and claude_agg is not None and claude_agg != 0:
                transfers[provider][cand_name]["aggregate"] = {
                    "external_mean": round(ext_agg, 3),
                    "claude_mean": round(claude_agg, 3),
                    "transfer_rate": round(ext_agg / claude_agg, 3),
                    "delta": round(ext_agg - claude_agg, 3),
                }

    return transfers


async def async_main():
    """Async main entry point — uses asyncio for provider calls (A4)."""
    print("=" * 70)
    print("Cross-Model Validation — sigma-optimize Experiment 2")
    print(f"Candidates: {len(CANDIDATES)} x Tasks: {len(ALL_TASKS)} x Runs: {RUNS_PER_TASK} x Providers: {len(PROVIDERS)}")
    print(f"Total API calls: {len(CANDIDATES) * len(ALL_TASKS) * RUNS_PER_TASK * len(PROVIDERS)}")
    print(f"Timeout per call: {CALL_TIMEOUT}s (asyncio.wait_for)")
    print(f"Model tiers: {MODEL_TIERS['cross_model']}")
    print("=" * 70)

    # Verify API keys
    for provider in PROVIDERS:
        key_name = "OPENAI_API_KEY" if provider == "openai" else "GOOGLE_AI_API_KEY"
        if not os.environ.get(key_name):
            print(f"ERROR: {key_name} not set. Cannot test {provider}.")
            sys.exit(1)
    print("API keys verified: OpenAI, Google")

    # A5: Provider quota pre-check
    quotas = check_provider_quotas()
    print("\nQuota pre-check:")
    for provider, info in quotas.items():
        status_icon = "!" if info["status"] == "warning" else "+"
        print(f"  [{status_icon}] {provider}: {info['message']}")
    print()

    # Load Claude validation data for transfer rate computation
    validation_file = RESULTS_DIR / "multi-validation.json"
    divergent_file = RESULTS_DIR / "multi-validation-divergent.json"
    claude_data = {}

    if validation_file.exists():
        with open(validation_file) as f:
            val = json.load(f)
        # Extract Claude scores for our candidates
        label_map = {
            "COMBO-2: !find : vulnerability | code → fix": "COMBO-2",
            "COMBO-1: flaw reliability evidence | bias → ? fix": "COMBO-1",
            "BASELINE-1: identify the flaws in this reasoning": "BASELINE",
        }
        for cand in val.get("candidates", []):
            label = cand.get("label", "")
            for full_label, short_name in label_map.items():
                if full_label in label:
                    per_task = cand.get("per_task", {})
                    claude_data[short_name] = {
                        "tasks": {
                            tid: {"mean": tdata.get("avg", 0), "n": 20}
                            for tid, tdata in per_task.items()
                        },
                        "aggregate": {"mean": cand.get("avg_score", 0)},
                    }
                    break

    # Override DP scores with N=40 loop-back data if available
    if divergent_file.exists():
        with open(divergent_file) as f:
            div = json.load(f)
        div_label_map = {
            "COMBO-2": "COMBO-2",
            "COMBO-1": "COMBO-1",
            "BASELINE-1": "BASELINE",
        }
        for cand in div.get("candidates", []):
            label = cand.get("label", "")
            for div_label, short_name in div_label_map.items():
                if div_label in label and short_name in claude_data:
                    dp_data = cand.get("per_task", {}).get("divergent_problem", {})
                    if dp_data:
                        claude_data[short_name]["tasks"]["divergent_problem"] = {
                            "mean": dp_data.get("avg", 0),
                            "n": 40,
                        }
                    break

    print(f"\nClaude reference data loaded: {list(claude_data.keys())}")
    for name, data in claude_data.items():
        agg = data.get("aggregate", {}).get("mean", "?")
        dp = data.get("tasks", {}).get("divergent_problem", {}).get("mean", "?")
        print(f"  {name}: aggregate={agg}, DP={dp}")

    # Run evaluations
    all_results = []
    start_time = time.time()

    # Clean up stale partial results
    if PARTIAL_RESULTS_FILE.exists():
        PARTIAL_RESULTS_FILE.unlink()

    data_gaps = []
    for provider in PROVIDERS:
        print(f"\n{'─' * 50}")
        print(f"Running {provider} ({PROVIDERS[provider]['model']})")
        print(f"{'─' * 50}")
        provider_start = time.time()
        provider_results = await run_provider(provider)
        all_results.extend(provider_results)
        elapsed = time.time() - provider_start
        errors = sum(1 for r in provider_results if r["error"] is not None)
        valid = len(provider_results) - errors
        print(f"  {provider} complete: {len(provider_results)} runs, {errors} errors, {valid} valid, {elapsed:.1f}s")

        # A2: Save incrementally after each provider
        save_partial_results(provider, provider_results)

        if errors > 0:
            # Log which candidates/tasks have gaps
            for cand_name in CANDIDATES:
                for task in ALL_TASKS:
                    task_errors = [r for r in provider_results
                                  if r["candidate"] == cand_name
                                  and r["task_id"] == task["id"]
                                  and r["error"] is not None]
                    if task_errors:
                        gap = f"{provider}/{cand_name}/{task['id']}: {len(task_errors)}/{RUNS_PER_TASK} runs failed"
                        data_gaps.append(gap)
                        print(f"    GAP: {gap}", flush=True)

    total_elapsed = time.time() - start_time

    # Compute statistics
    stats = compute_statistics(all_results)
    transfers = compute_transfer_rates(stats, claude_data)

    # Save results
    output = {
        "experiment": "sigma-optimize-exp2-cross-model",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "candidates": list(CANDIDATES.keys()),
            "candidate_prompts": CANDIDATES,
            "tasks": [t["id"] for t in ALL_TASKS],
            "runs_per_task": RUNS_PER_TASK,
            "providers": {p: PROVIDERS[p]["model"] for p in PROVIDERS},
            "total_calls": len(all_results),
            "errors": sum(1 for r in all_results if r["error"] is not None),
            "elapsed_s": round(total_elapsed, 1),
            "data_gaps": data_gaps,
            "data_gap_note": "Gemini daily quota (250 req/day for gemini-3.1-pro) caused partial data loss. Gaps are in baseline candidate on late tasks. COMBO-2 and COMBO-1 data is complete for both providers. Consider Gemini baseline results as having reduced N for affected tasks." if data_gaps else None,
        },
        "statistics": stats,
        "transfer_rates": transfers,
        "claude_reference": claude_data,
        "raw_results": [
            {k: v for k, v in r.items() if k != "response_text"}
            for r in all_results
        ],
        "raw_responses": [
            {
                "provider": r["provider"],
                "candidate": r["candidate"],
                "task_id": r["task_id"],
                "run": r["run"],
                "score": r["score"],
                "response_text": r["response_text"],
            }
            for r in all_results
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for provider in PROVIDERS:
        model = PROVIDERS[provider]["model"]
        print(f"\n{provider} ({model}):")
        for cand_name in CANDIDATES:
            cand = stats[provider]["candidates"].get(cand_name, {})
            agg = cand.get("aggregate", {})
            agg_mean = agg.get("mean_of_task_means", "?")
            print(f"  {cand_name}: aggregate={agg_mean}")
            for task in ALL_TASKS:
                task_id = task["id"]
                t = cand.get("tasks", {}).get(task_id, {})
                mean = t.get("mean", "?")
                sd = t.get("sd", "?")
                n = t.get("n", 0)
                print(f"    {task_id}: mean={mean} sd={sd} n={n}")

    print(f"\nTransfer rates (external/claude):")
    for provider in PROVIDERS:
        print(f"\n  {provider}:")
        for cand_name in CANDIDATES:
            t = transfers.get(provider, {}).get(cand_name, {})
            agg = t.get("aggregate", {})
            rate = agg.get("transfer_rate", "?")
            delta = agg.get("delta", "?")
            print(f"    {cand_name}: aggregate transfer={rate} delta={delta}")
            dp = t.get("divergent_problem", {})
            if dp:
                print(f"      divergent_problem: transfer={dp.get('transfer_rate', '?')} delta={dp.get('delta', '?')}")

    print(f"\nTotal elapsed: {total_elapsed:.1f}s")
    return output


def main():
    """Sync wrapper for backward compatibility."""
    return asyncio.run(async_main())


if __name__ == "__main__":
    main()
