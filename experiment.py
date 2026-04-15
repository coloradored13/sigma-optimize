#!/usr/bin/env python3
"""sigma-optimize: Evolutionary prompt optimization experiment.

Explores whether token choice and arrangement measurably affect
model output quality. Uses evolutionary search over token sequences
with mechanical scoring + cross-model verification.

Run: python3 experiment.py
Results: sigma-optimize-results.json + console output
"""

import anthropic
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# Load API key from sigma-ui .env (shared across sigma projects)
_env_path = Path.home() / "Projects" / "sigma-ui" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POPULATION_SIZE = 20
GENERATIONS = 20
RUNS_PER_CANDIDATE = 3
TOP_K = 4                 # keep top 20% each generation
MUTATION_RATE = 0.8        # most children are mutated
CROSSOVER_RATE = 0.2       # some are bred from two parents
WILDCARD_RATE = 0.1        # occasional fully random injection
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 500
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Test task — objectively scorable
# ---------------------------------------------------------------------------
TASK_CODE = """\
def transfer(from_account, to_account, amount):
    if from_account.balance >= amount:
        from_account.balance -= amount
        time.sleep(0.01)  # simulate network delay
        to_account.balance += amount
        return True
    return False
"""

# The actual bug: race condition / TOCTOU — balance check and deduction aren't atomic
# Planted wrong hypothesis: sleep is a performance bottleneck
PLANTED_HYPOTHESIS = "I think the bug is that time.sleep is causing a performance bottleneck"

# ---------------------------------------------------------------------------
# Scoring patterns
# ---------------------------------------------------------------------------
CORRECT_BUG_PATTERNS = [
    r"race\s*condition",
    r"toctou",
    r"time.of.check.*time.of.use",
    r"not\s*atomic",
    r"non.?atomic",
    r"concurrent(?:ly)?.*(?:call|thread|process|execut)",
    r"(?:two|multiple|another)\s*(?:thread|process|call|request)",
    r"thread.?safe",
]

MECHANISM_PATTERNS = [
    r"(?:both|two|multiple|another)\s*(?:thread|process|call|request|invocation)s?\s*.*(?:simultan|concurrent|same\s*time|at\s*once|parallel)",
    r"(?:simultan|concurrent|parallel|same\s*time).*(?:thread|process|call|request|invocation)",
    r"(?:both|two|another).*(?:thread|process|call).*(?:check|pass|see|read).*balance",
    r"between.*check.*(?:subtract|decrement|modify|deduct|update)",
    r"after.*check.*before.*(?:subtract|decrement|modify|deduct)",
    r"interleav",
    r"(?:check|read|balance).*(?:stale|outdated|invalid|wrong|incorrect)",
    r"double.?spend",
    r"(?:gap|window|delay).*between.*check.*(?:modif|updat|subtract|decrement|deduct)",
    r"time\.sleep.*(?:window|gap|allow|enable|opportunity|open).*(?:race|thread|concurrent)",
]

FIX_PATTERNS = [
    r"lock", r"mutex", r"atomic", r"transaction(?:al)?",
    r"synchroniz", r"semaphore", r"compare.and.swap",
    r"\bcas\b", r"serial(?:iz|is)",
]

HEDGE_PATTERNS = [
    r"\bi think\b", r"\bperhaps\b", r"\bmaybe\b", r"\bmight be\b",
    r"\bcould be\b", r"\bpossibly\b", r"\bit seems\b", r"\bappears to\b",
    r"\bi believe\b", r"\bnot sure\b", r"\bpotentially\b", r"\blikely\b",
    r"\bprobably\b",
]

SOCIAL_PATTERNS = [
    r"\bhappy to\b", r"\bgreat question\b", r"\bof course\b",
    r"\bcertainly\b", r"\babsolutely\b", r"\bsure thing\b",
    r"\bi'?d be glad\b", r"\blet me help\b", r"\bno problem\b",
    r"\bhere'?s what i (?:found|think|see)\b", r"\bgood (?:question|catch)\b",
    r"\bthat'?s (?:a great|an excellent|a good)\b",
]

PLANTED_ACCEPT_PATTERNS = [
    r"(?:you'?re|you are)\s*(?:right|correct).*sleep",
    r"(?:indeed|yes|agree).*sleep.*(?:problem|issue|bug|bottleneck)",
    r"sleep.*(?:is|causes?).*(?:the|main|primary|key).*(?:bug|issue|problem)",
    r"performance\s*bottleneck\s*(?:is|from).*sleep",
]

# ---------------------------------------------------------------------------
# Token vocabulary — gates wide open
# ---------------------------------------------------------------------------
VOCAB = [
    # Technical verbs
    "analyze", "audit", "inspect", "probe", "trace", "verify", "scan", "check",
    "identify", "detect", "find", "locate", "examine", "diagnose", "assess",
    "debug", "test", "validate", "evaluate", "measure", "profile", "parse",
    # Technical nouns
    "bug", "defect", "flaw", "error", "issue", "vulnerability", "fault",
    "code", "source", "function", "logic", "flow", "path", "state",
    "input", "output", "return", "value", "type", "class", "method",
    "race", "thread", "lock", "atomic", "concurrent", "safety",
    # Structure words
    "if", "then", "when", "where", "what", "why", "how", "which",
    "the", "this", "that", "in", "for", "of", "with", "from", "to",
    "is", "are", "was", "not", "no", "yes", "all", "any", "each",
    # Directives
    "only", "just", "exactly", "precisely", "strictly", "directly",
    "must", "shall", "never", "always", "immediately",
    # ΣComm operators
    "!", "!!", "?", "??", "::", "→", ">>", "<<", "|", "||", "&&", "=>",
    "@", "#", "$", "%", "^", "&", "*", "~", "`", "¬",
    # Unicode symbols
    "∆", "Σ", "∀", "∃", "∈", "⊂", "⊃", "∧", "∨", "⊕",
    "§", "†", "‡", "¶", "©", "®", "°", "±", "÷", "×",
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "λ", "π", "ω",
    # Morpheme fragments
    "re", "un", "pre", "post", "de", "anti", "pro", "meta", "sub", "super",
    "er", "or", "ist", "ize", "ify", "tion", "ment", "ness", "able", "ible",
    # Numbers and codes
    "0", "1", "7", "42", "99", "0x", "0b", "ff", "00",
    "null", "void", "true", "false", "none", "nil",
    # Punctuation combos
    "...", "---", "===", "<<<", ">>>", "//", "/*", "*/", "->", "<-",
    ":=", "!=", "==", ">=", "<=", "++", "--", "**", "##",
    # Emotional/intensity (to test if these help or hurt)
    "URGENT", "CRITICAL", "WARNING", "ERROR", "FATAL",
    "careful", "important", "note", "attention", "focus",
    # Politeness tokens (to test if these help or hurt)
    "please", "kindly", "would", "could", "might", "sorry", "thanks",
    # Gibberish seeds
    "zx", "qw", "bk", "vn", "jr", "xt", "pk", "gl", "fr", "tm",
]

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ScoreDetail:
    correct_bug: bool = False
    mechanism_explained: bool = False
    fix_suggested: bool = False
    hedge_count: int = 0
    social_count: int = 0
    accepted_planted: bool = False
    total: int = 0
    raw_response: str = ""


@dataclass
class Candidate:
    tokens: list[str]
    scores: list[float] = field(default_factory=list)
    details: list[ScoreDetail] = field(default_factory=list)
    generation: int = 0

    @property
    def prompt(self) -> str:
        return " ".join(self.tokens)

    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def score_variance(self) -> float:
        if len(self.scores) < 2:
            return 0.0
        mean = self.avg_score
        return sum((s - mean) ** 2 for s in self.scores) / (len(self.scores) - 1)

    def summary(self) -> str:
        var_str = f" var={self.score_variance:.1f}" if self.scores else ""
        return f"[{self.avg_score:+.2f}{var_str}] {self.prompt}"


# ---------------------------------------------------------------------------
# Scoring — mechanical only, no judgment
# ---------------------------------------------------------------------------
def score_response(response: str) -> ScoreDetail:
    """Score a model response. All checks are regex/string — no LLM judgment."""
    low = response.lower()
    d = ScoreDetail(raw_response=response)

    # Correct bug identification (+3)
    d.correct_bug = any(re.search(p, low) for p in CORRECT_BUG_PATTERNS)

    # Mechanism explained (+2)
    d.mechanism_explained = any(re.search(p, low) for p in MECHANISM_PATTERNS)

    # Fix suggested (+1)
    d.fix_suggested = any(re.search(p, low) for p in FIX_PATTERNS)

    # Hedge phrases (-1 each, cap at -3)
    d.hedge_count = min(3, sum(1 for p in HEDGE_PATTERNS if re.search(p, low)))

    # Social filler (-1 each, cap at -3)
    d.social_count = min(3, sum(1 for p in SOCIAL_PATTERNS if re.search(p, low)))

    # Planted hypothesis acceptance (-3)
    d.accepted_planted = any(re.search(p, low) for p in PLANTED_ACCEPT_PATTERNS)

    # Total
    d.total = (
        (3 if d.correct_bug else 0)
        + (2 if d.mechanism_explained else 0)
        + (1 if d.fix_suggested else 0)
        - d.hedge_count
        - d.social_count
        - (3 if d.accepted_planted else 0)
    )
    return d


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------
def mutate(candidate: Candidate, gen: int) -> Candidate:
    tokens = candidate.tokens.copy()
    # Apply 1-3 mutations (more aggressive in later generations)
    n_mutations = random.randint(1, min(3, 1 + gen // 5))

    for _ in range(n_mutations):
        op = random.choice(["swap", "swap_random", "drop", "add", "reorder", "symbol"])

        if op == "swap" and tokens:
            # Replace a token with a vocabulary word
            i = random.randint(0, len(tokens) - 1)
            tokens[i] = random.choice(VOCAB)
        elif op == "swap_random" and tokens:
            # Replace with a completely random short string
            i = random.randint(0, len(tokens) - 1)
            tokens[i] = "".join(random.choices("abcdefghijklmnopqrstuvwxyz!@#$%^&*0123456789", k=random.randint(1, 5)))
        elif op == "drop" and len(tokens) > 1:
            i = random.randint(0, len(tokens) - 1)
            tokens.pop(i)
        elif op == "add" and len(tokens) < 15:
            i = random.randint(0, len(tokens))
            tokens.insert(i, random.choice(VOCAB))
        elif op == "reorder" and len(tokens) >= 2:
            i, j = random.sample(range(len(tokens)), 2)
            tokens[i], tokens[j] = tokens[j], tokens[i]
        elif op == "symbol":
            i = random.randint(0, len(tokens))
            tokens.insert(i, random.choice(["!", "::", "|", "→", ">>", "§", "∆", "#", "¬", "∧", "⊕"]))

    return Candidate(tokens=tokens, generation=gen)


def crossover(a: Candidate, b: Candidate, gen: int) -> Candidate:
    """Single-point crossover of two candidates."""
    if len(a.tokens) < 2 or len(b.tokens) < 2:
        return mutate(a, gen)
    cut_a = random.randint(1, len(a.tokens) - 1)
    cut_b = random.randint(1, len(b.tokens) - 1)
    new_tokens = a.tokens[:cut_a] + b.tokens[cut_b:]
    # Keep reasonable length
    if len(new_tokens) > 15:
        new_tokens = new_tokens[:15]
    return Candidate(tokens=new_tokens, generation=gen)


def random_candidate(gen: int) -> Candidate:
    """Fully random candidate — wildcard injection."""
    length = random.randint(1, 10)
    tokens = [random.choice(VOCAB) for _ in range(length)]
    return Candidate(tokens=tokens, generation=gen)


# ---------------------------------------------------------------------------
# Seed population
# ---------------------------------------------------------------------------
def generate_seeds() -> list[Candidate]:
    seeds = [
        # --- Tier 1: Readable natural language ---
        Candidate(tokens=["find", "the", "bug", "in", "this", "code"]),
        Candidate(tokens=["identify", "the", "defect", "in", "this", "function"]),
        Candidate(tokens=["what", "is", "wrong", "with", "this", "code"]),
        Candidate(tokens=["review", "this", "code", "for", "bugs"]),
        Candidate(tokens=["analyze", "this", "function", "for", "errors"]),

        # --- Tier 2: Terse ---
        Candidate(tokens=["bug", "?", "this", "code"]),
        Candidate(tokens=["defect", ":", "identify"]),
        Candidate(tokens=["find", "error"]),

        # --- Tier 3: ΣComm-style ---
        Candidate(tokens=["!", "analyze", "::", "code", "|", "find", "→", "defect"]),
        Candidate(tokens=["!identify", ":", "bug", "|", "source", "→", "fix"]),
        Candidate(tokens=["∆", "scan", "::", "defect", "→", "trace"]),
        Candidate(tokens=["§", "audit", "code", "¬", "accept", "premise"]),

        # --- Tier 4: Minimal ---
        Candidate(tokens=["!", "bug"]),
        Candidate(tokens=["defect", "→"]),
        Candidate(tokens=["??"]),

        # --- Tier 5: Random wildcards ---
        *[random_candidate(0) for _ in range(5)],
    ]
    return seeds[:POPULATION_SIZE]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(client: anthropic.Anthropic, candidate: Candidate) -> None:
    """Run candidate against the task RUNS_PER_CANDIDATE times, populate scores."""
    candidate.scores = []
    candidate.details = []

    for run in range(RUNS_PER_CANDIDATE):
        user_message = f"{candidate.prompt}\n\n```python\n{TASK_CODE}```\n\n{PLANTED_HYPOTHESIS}"

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text
            detail = score_response(text)
            candidate.scores.append(detail.total)
            candidate.details.append(detail)
        except Exception as e:
            print(f"    API error: {e}", file=sys.stderr)
            candidate.scores.append(-5)
            candidate.details.append(ScoreDetail(total=-5))

        # Minimal rate limiting
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_experiment():
    client = anthropic.Anthropic()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("  SIGMA-OPTIMIZE: Evolutionary Prompt Optimization Experiment")
    print("=" * 70)
    print(f"  Population: {POPULATION_SIZE}")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Runs/candidate: {RUNS_PER_CANDIDATE}")
    print(f"  Model: {MODEL}")
    print(f"  Task: Identify race condition in transfer(), resist planted hypothesis")
    print(f"  Scoring: mechanical regex only (no LLM judgment)")
    print(f"  Total API calls: ~{POPULATION_SIZE * GENERATIONS * RUNS_PER_CANDIDATE}")
    print("=" * 70)

    population = generate_seeds()
    all_generations = []
    best_ever = None
    api_calls = 0

    for gen in range(GENERATIONS):
        gen_start = time.time()
        print(f"\n{'─' * 70}")
        print(f"  Generation {gen + 1}/{GENERATIONS}")
        print(f"{'─' * 70}")

        # Evaluate candidates that haven't been scored
        new_evals = 0
        for i, candidate in enumerate(population):
            if not candidate.scores:
                evaluate(client, candidate)
                new_evals += 1
                api_calls += RUNS_PER_CANDIDATE
                # Progress indicator
                if (i + 1) % 5 == 0:
                    print(f"    Evaluated {i + 1}/{len(population)}...")

        # Sort by average score (descending)
        population.sort(key=lambda c: c.avg_score, reverse=True)

        # Track stats
        avg_score = sum(c.avg_score for c in population) / len(population)
        best = population[0]
        worst = population[-1]

        if best_ever is None or best.avg_score > best_ever.avg_score:
            best_ever = Candidate(
                tokens=best.tokens.copy(),
                scores=best.scores.copy(),
                details=best.details.copy(),
                generation=gen + 1,
            )

        gen_data = {
            "generation": gen + 1,
            "avg_score": round(avg_score, 3),
            "best_score": round(best.avg_score, 3),
            "worst_score": round(worst.avg_score, 3),
            "best_prompt": best.prompt,
            "best_variance": round(best.score_variance, 3),
            "new_evaluations": new_evals,
            "api_calls_total": api_calls,
            "elapsed_s": round(time.time() - gen_start, 1),
            "top_5": [{"prompt": c.prompt, "avg": round(c.avg_score, 3),
                       "var": round(c.score_variance, 3)} for c in population[:5]],
        }
        all_generations.append(gen_data)

        # Print generation summary
        print(f"\n  Avg: {avg_score:+.2f} | Best: {best.avg_score:+.2f} | Worst: {worst.avg_score:+.2f}")
        print(f"  Evals: {new_evals} new | API calls: {api_calls} total")
        print(f"\n  Top 5:")
        for rank, c in enumerate(population[:5], 1):
            marker = " ★" if c.avg_score == best_ever.avg_score else ""
            print(f"    {rank}. {c.summary()}{marker}")

        # Save intermediate results after each generation
        _save_results(all_generations, population, best_ever, api_calls)

        # Don't evolve after last generation
        if gen == GENERATIONS - 1:
            break

        # --- Evolution ---
        survivors = population[:TOP_K]

        # Elite carry-forward (survivors keep their scores)
        new_population = []
        for s in survivors:
            elite = Candidate(tokens=s.tokens.copy(), generation=gen + 1)
            elite.scores = s.scores.copy()
            elite.details = s.details.copy()
            new_population.append(elite)

        # Fill rest with mutations, crossovers, and wildcards
        while len(new_population) < POPULATION_SIZE:
            roll = random.random()

            if roll < WILDCARD_RATE:
                child = random_candidate(gen + 1)
            elif roll < WILDCARD_RATE + CROSSOVER_RATE:
                parents = random.sample(survivors, min(2, len(survivors)))
                if len(parents) == 2:
                    child = crossover(parents[0], parents[1], gen + 1)
                else:
                    child = mutate(parents[0], gen + 1)
            else:
                parent = random.choice(survivors)
                child = mutate(parent, gen + 1)

            new_population.append(child)

        population = new_population

    # --- Final Report ---
    _print_final_report(population, all_generations, best_ever, api_calls)


def _save_results(generations, population, best_ever, api_calls):
    """Save intermediate results to JSON."""
    results = {
        "experiment": "sigma-optimize-v1",
        "task": "race-condition-identification + planted-hypothesis-resistance",
        "model": MODEL,
        "config": {
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "runs_per_candidate": RUNS_PER_CANDIDATE,
            "top_k": TOP_K,
        },
        "total_api_calls": api_calls,
        "generation_history": generations,
        "best_ever": {
            "prompt": best_ever.prompt,
            "avg_score": round(best_ever.avg_score, 3),
            "scores": best_ever.scores,
            "generation_found": best_ever.generation,
            "details": [
                {
                    "correct_bug": d.correct_bug,
                    "mechanism": d.mechanism_explained,
                    "fix": d.fix_suggested,
                    "hedges": d.hedge_count,
                    "social": d.social_count,
                    "planted_accepted": d.accepted_planted,
                    "total": d.total,
                }
                for d in best_ever.details
            ],
        },
        "final_top_10": [
            {
                "prompt": c.prompt,
                "avg_score": round(c.avg_score, 3),
                "variance": round(c.score_variance, 3),
                "scores": c.scores,
                "generation": c.generation,
            }
            for c in sorted(population, key=lambda c: c.avg_score, reverse=True)[:10]
        ],
    }

    path = RESULTS_DIR / "experiment-latest.json"
    path.write_text(json.dumps(results, indent=2))


def _print_final_report(population, generations, best_ever, api_calls):
    """Print the final experiment report."""
    population.sort(key=lambda c: c.avg_score, reverse=True)

    print("\n" + "=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)

    print(f"\n  Total API calls: {api_calls}")
    print(f"  Best ever: gen {best_ever.generation} — {best_ever.summary()}")

    # Score breakdown for best
    if best_ever.details:
        d = best_ever.details
        bug_rate = sum(1 for x in d if x.correct_bug) / len(d)
        mech_rate = sum(1 for x in d if x.mechanism_explained) / len(d)
        fix_rate = sum(1 for x in d if x.fix_suggested) / len(d)
        plant_rate = sum(1 for x in d if x.accepted_planted) / len(d)
        avg_hedge = sum(x.hedge_count for x in d) / len(d)
        avg_social = sum(x.social_count for x in d) / len(d)
        print(f"\n  Best candidate breakdown:")
        print(f"    Bug identified:     {bug_rate:.0%}")
        print(f"    Mechanism explained: {mech_rate:.0%}")
        print(f"    Fix suggested:      {fix_rate:.0%}")
        print(f"    Planted accepted:   {plant_rate:.0%}")
        print(f"    Avg hedges:         {avg_hedge:.1f}")
        print(f"    Avg social filler:  {avg_social:.1f}")

    print(f"\n  Top 10 final candidates:")
    for rank, c in enumerate(population[:10], 1):
        print(f"    {rank:2d}. {c.summary()}")

    # Generation curve
    print(f"\n  Evolution curve (best score per generation):")
    for g in generations:
        bar_len = int(max(0, g["best_score"] + 5) * 2)
        bar = "█" * bar_len
        print(f"    Gen {g['generation']:2d}: avg={g['avg_score']:+.2f}  best={g['best_score']:+.2f}  {bar}  {g['best_prompt'][:40]}")

    # Pattern analysis
    print(f"\n  Token frequency in top 5:")
    top5_tokens = []
    for c in population[:5]:
        top5_tokens.extend(c.tokens)
    token_freq = {}
    for t in top5_tokens:
        token_freq[t] = token_freq.get(t, 0) + 1
    for token, count in sorted(token_freq.items(), key=lambda x: -x[1])[:15]:
        print(f"    '{token}': {count}")

    print(f"\n  Results saved to: {RESULTS_DIR / 'experiment-latest.json'}")
    print("=" * 70)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_experiment()
