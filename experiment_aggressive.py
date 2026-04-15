#!/usr/bin/env python3
"""sigma-optimize: Aggressive evolutionary prompt search.

Full vocabulary including gibberish, unicode, ΣComm operators, random strings.
Tests whether non-semantic tokens outperform readable prompts.

Run: python3 experiment_aggressive.py
Results: results/aggressive-search.json
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
# Configuration — aggressive search parameters
# ---------------------------------------------------------------------------
POPULATION_SIZE = 20
GENERATIONS = 10
RUNS_PER_CANDIDATE = 5
TOP_K = 4                 # keep top 20% each generation
MUTATION_RATE = 0.8
CROSSOVER_RATE = 0.2
WILDCARD_RATE = 0.20       # elevated for maximum exploration
MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 500
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = "aggressive-search.json"

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
# Scoring patterns (identical to base harness — mechanical only)
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
# Token vocabulary — FULL (aggressive: all categories including exotic)
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
    # Gibberish seeds (expanded for aggressive search)
    "zx", "qw", "bk", "vn", "jr", "xt", "pk", "gl", "fr", "tm",
    "xq", "zp", "kv", "jw", "bx", "nf", "gk", "rl", "wp", "yz",
    # Extended gibberish (multi-char nonsense)
    "zxqw", "bknv", "jrxt", "pkgl", "frtm", "xqzp", "kvjw",
    # Pure noise tokens
    "???", "!!!", "###", "@@@", "$$$", "^^^", "~~~",
    # CJK-inspired fragments (test if exotic unicode affects anything)
    "丁", "七", "万",
    # Emoji-style unicode
    "⚡", "🔍", "⚠", "✓", "✗", "◌",
    # Mathematical operators
    "∑", "∏", "∫", "√", "∞", "≠", "≤", "≥", "≈",
    # Box drawing / structural
    "─", "│", "┌", "┐", "└", "┘", "├", "┤", "┬", "┴", "┼",
]

# Separate category tracking for analysis
EXOTIC_TOKENS = set([
    "!", "!!", "?", "??", "::", "→", ">>", "<<", "|", "||", "&&", "=>",
    "@", "#", "$", "%", "^", "&", "*", "~", "`", "¬",
    "∆", "Σ", "∀", "∃", "∈", "⊂", "⊃", "∧", "∨", "⊕",
    "§", "†", "‡", "¶", "©", "®", "°", "±", "÷", "×",
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "λ", "π", "ω",
    "zx", "qw", "bk", "vn", "jr", "xt", "pk", "gl", "fr", "tm",
    "xq", "zp", "kv", "jw", "bx", "nf", "gk", "rl", "wp", "yz",
    "zxqw", "bknv", "jrxt", "pkgl", "frtm", "xqzp", "kvjw",
    "???", "!!!", "###", "@@@", "$$$", "^^^", "~~~",
    "丁", "七", "万",
    "⚡", "🔍", "⚠", "✓", "✗", "◌",
    "∑", "∏", "∫", "√", "∞", "≠", "≤", "≥", "≈",
    "─", "│", "┌", "┐", "└", "┘", "├", "┤", "┬", "┴", "┼",
    "...", "---", "===", "<<<", ">>>", "//", "/*", "*/", "->", "<-",
    ":=", "!=", "==", ">=", "<=", "++", "--", "**", "##",
])


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

    d.correct_bug = any(re.search(p, low) for p in CORRECT_BUG_PATTERNS)
    d.mechanism_explained = any(re.search(p, low) for p in MECHANISM_PATTERNS)
    d.fix_suggested = any(re.search(p, low) for p in FIX_PATTERNS)
    d.hedge_count = min(3, sum(1 for p in HEDGE_PATTERNS if re.search(p, low)))
    d.social_count = min(3, sum(1 for p in SOCIAL_PATTERNS if re.search(p, low)))
    d.accepted_planted = any(re.search(p, low) for p in PLANTED_ACCEPT_PATTERNS)

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
# Mutation operators — aggressive version
# ---------------------------------------------------------------------------
def mutate(candidate: Candidate, gen: int) -> Candidate:
    tokens = candidate.tokens.copy()
    # More aggressive: 1-4 mutations, escalating with generation
    n_mutations = random.randint(1, min(4, 1 + gen // 3))

    for _ in range(n_mutations):
        op = random.choice([
            "swap", "swap_random", "swap_random",  # double weight on random swap
            "drop", "add", "reorder",
            "symbol", "symbol",  # double weight on symbol injection
            "gibberish_inject",  # new: inject pure gibberish
        ])

        if op == "swap" and tokens:
            i = random.randint(0, len(tokens) - 1)
            tokens[i] = random.choice(VOCAB)
        elif op == "swap_random" and tokens:
            i = random.randint(0, len(tokens) - 1)
            charset = "abcdefghijklmnopqrstuvwxyz!@#$%^&*0123456789∆Σ∀∃∈→§¬αβγδ"
            tokens[i] = "".join(random.choices(charset, k=random.randint(1, 6)))
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
            symbols = ["!", "::", "|", "→", ">>", "§", "∆", "#", "¬", "∧", "⊕",
                       "Σ", "∀", "∃", "⚡", "⚠", "∞", "≠", "√", "∫"]
            if len(tokens) < 15:
                i = random.randint(0, len(tokens))
                tokens.insert(i, random.choice(symbols))
            elif tokens:
                i = random.randint(0, len(tokens) - 1)
                tokens[i] = random.choice(symbols)
        elif op == "gibberish_inject":
            # Inject 2-4 char pure gibberish
            gibberish = "".join(random.choices("zxqwbkvnjrxtpkglfrtm", k=random.randint(2, 4)))
            if len(tokens) < 15:
                i = random.randint(0, len(tokens))
                tokens.insert(i, gibberish)
            elif tokens:
                i = random.randint(0, len(tokens) - 1)
                tokens[i] = gibberish

    return Candidate(tokens=tokens, generation=gen)


def crossover(a: Candidate, b: Candidate, gen: int) -> Candidate:
    """Single-point crossover of two candidates."""
    if len(a.tokens) < 2 or len(b.tokens) < 2:
        return mutate(a, gen)
    cut_a = random.randint(1, len(a.tokens) - 1)
    cut_b = random.randint(1, len(b.tokens) - 1)
    new_tokens = a.tokens[:cut_a] + b.tokens[cut_b:]
    if len(new_tokens) > 15:
        new_tokens = new_tokens[:15]
    return Candidate(tokens=new_tokens, generation=gen)


def random_candidate(gen: int) -> Candidate:
    """Fully random candidate — wildcard injection with heavy exotic bias."""
    length = random.randint(1, 10)
    tokens = []
    for _ in range(length):
        # 40% chance of exotic token, 60% any vocab
        if random.random() < 0.40:
            tokens.append(random.choice(list(EXOTIC_TOKENS)))
        else:
            tokens.append(random.choice(VOCAB))
    return Candidate(tokens=tokens, generation=gen)


# ---------------------------------------------------------------------------
# Seed population — aggressive: heavy on exotic/unusual forms
# ---------------------------------------------------------------------------
def generate_seeds() -> list[Candidate]:
    seeds = [
        # --- Tier 1: Readable (control/baseline representatives) ---
        Candidate(tokens=["find", "the", "bug", "in", "this", "code"]),
        Candidate(tokens=["identify", "the", "defect", "in", "this", "function"]),
        Candidate(tokens=["analyze", "this", "function", "for", "errors"]),

        # --- Tier 2: ΣComm-style ---
        Candidate(tokens=["!", "analyze", "::", "code", "|", "find", "→", "defect"]),
        Candidate(tokens=["!identify", ":", "bug", "|", "source", "→", "fix"]),
        Candidate(tokens=["∆", "scan", "::", "defect", "→", "trace"]),
        Candidate(tokens=["§", "audit", "code", "¬", "accept", "premise"]),

        # --- Tier 3: Heavy exotic/symbol ---
        Candidate(tokens=["⚡", "∆", "bug", "→", "∀", "defect"]),
        Candidate(tokens=["Σ", "::", "analyze", "§", "code", "∧", "find"]),
        Candidate(tokens=["∞", "≠", "bug", "¬", "sleep"]),
        Candidate(tokens=["⊕", "scan", "∃", "error", "→", "fix"]),

        # --- Tier 4: Gibberish-heavy ---
        Candidate(tokens=["zxqw", "bug", "bknv", "code"]),
        Candidate(tokens=["xq", "find", "zp", "defect", "kv"]),
        Candidate(tokens=["pkgl", "frtm", "analyze"]),

        # --- Tier 5: Pure noise ---
        Candidate(tokens=["!!!", "###", "@@@", "zxqw", "bknv"]),
        Candidate(tokens=["丁", "七", "万", "bug"]),

        # --- Tier 6: Random wildcards ---
        *[random_candidate(0) for _ in range(4)],
    ]
    return seeds[:POPULATION_SIZE]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(client: anthropic.Anthropic, candidate: Candidate) -> None:
    """Run candidate against the task RUNS_PER_CANDIDATE times."""
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

        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def classify_tokens(tokens: list[str]) -> dict:
    """Classify tokens into semantic vs exotic categories."""
    semantic = [t for t in tokens if t not in EXOTIC_TOKENS]
    exotic = [t for t in tokens if t in EXOTIC_TOKENS]
    return {"semantic": semantic, "exotic": exotic, "exotic_ratio": len(exotic) / max(len(tokens), 1)}


def analyze_top_k(population: list['Candidate'], k: int = 10) -> dict:
    """Analyze token composition of top-K candidates."""
    top = sorted(population, key=lambda c: c.avg_score, reverse=True)[:k]

    all_tokens = []
    exotic_in_top = []
    for c in top:
        classification = classify_tokens(c.tokens)
        all_tokens.extend(c.tokens)
        if classification["exotic"]:
            exotic_in_top.append({
                "prompt": c.prompt,
                "score": c.avg_score,
                "exotic_tokens": classification["exotic"],
                "exotic_ratio": classification["exotic_ratio"],
            })

    # Token frequency
    freq = {}
    for t in all_tokens:
        freq[t] = freq.get(t, 0) + 1

    # Exotic frequency
    exotic_freq = {t: c for t, c in freq.items() if t in EXOTIC_TOKENS}

    return {
        "exotic_candidates_in_top_k": len(exotic_in_top),
        "exotic_details": exotic_in_top,
        "token_frequency": dict(sorted(freq.items(), key=lambda x: -x[1])[:20]),
        "exotic_token_frequency": dict(sorted(exotic_freq.items(), key=lambda x: -x[1])[:10]),
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run_experiment():
    client = anthropic.Anthropic()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("  SIGMA-OPTIMIZE: Aggressive Search — Full Vocabulary")
    print("=" * 70)
    print(f"  Population: {POPULATION_SIZE}")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Runs/candidate: {RUNS_PER_CANDIDATE}")
    print(f"  Wildcard rate: {WILDCARD_RATE}")
    print(f"  Vocab size: {len(VOCAB)} tokens ({len(EXOTIC_TOKENS)} exotic)")
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

        new_evals = 0
        for i, candidate in enumerate(population):
            if not candidate.scores:
                evaluate(client, candidate)
                new_evals += 1
                api_calls += RUNS_PER_CANDIDATE
                if (i + 1) % 5 == 0:
                    print(f"    Evaluated {i + 1}/{len(population)}...")

        population.sort(key=lambda c: c.avg_score, reverse=True)

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

        print(f"\n  Avg: {avg_score:+.2f} | Best: {best.avg_score:+.2f} | Worst: {worst.avg_score:+.2f}")
        print(f"  Evals: {new_evals} new | API calls: {api_calls} total")
        print(f"\n  Top 5:")
        for rank, c in enumerate(population[:5], 1):
            marker = " ★" if c.avg_score == best_ever.avg_score else ""
            classification = classify_tokens(c.tokens)
            exotic_tag = f" [exotic:{classification['exotic_ratio']:.0%}]" if classification['exotic'] else ""
            print(f"    {rank}. {c.summary()}{marker}{exotic_tag}")

        _save_results(all_generations, population, best_ever, api_calls)

        if gen == GENERATIONS - 1:
            break

        # --- Evolution ---
        survivors = population[:TOP_K]

        new_population = []
        for s in survivors:
            elite = Candidate(tokens=s.tokens.copy(), generation=gen + 1)
            elite.scores = s.scores.copy()
            elite.details = s.details.copy()
            new_population.append(elite)

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
    """Save results to JSON."""
    top_k_analysis = analyze_top_k(population)

    results = {
        "experiment": "sigma-optimize-aggressive-v1",
        "search_type": "aggressive",
        "task": "race-condition-identification + planted-hypothesis-resistance",
        "model": MODEL,
        "config": {
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "runs_per_candidate": RUNS_PER_CANDIDATE,
            "top_k": TOP_K,
            "wildcard_rate": WILDCARD_RATE,
            "vocab_size": len(VOCAB),
            "exotic_token_count": len(EXOTIC_TOKENS),
        },
        "total_api_calls": api_calls,
        "generation_history": generations,
        "best_ever": {
            "prompt": best_ever.prompt,
            "avg_score": round(best_ever.avg_score, 3),
            "variance": round(best_ever.score_variance, 3),
            "scores": best_ever.scores,
            "generation_found": best_ever.generation,
            "token_classification": classify_tokens(best_ever.tokens),
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
                "token_classification": classify_tokens(c.tokens),
            }
            for c in sorted(population, key=lambda c: c.avg_score, reverse=True)[:10]
        ],
        "top_k_analysis": top_k_analysis,
    }

    path = RESULTS_DIR / RESULTS_FILE
    path.write_text(json.dumps(results, indent=2))


def _print_final_report(population, generations, best_ever, api_calls):
    """Print the final experiment report."""
    population.sort(key=lambda c: c.avg_score, reverse=True)

    print("\n" + "=" * 70)
    print("  FINAL REPORT — AGGRESSIVE SEARCH")
    print("=" * 70)

    print(f"\n  Total API calls: {api_calls}")
    print(f"  Best ever: gen {best_ever.generation} — {best_ever.summary()}")

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

    # Token classification for best
    bc = classify_tokens(best_ever.tokens)
    print(f"\n  Best candidate token classification:")
    print(f"    Semantic: {bc['semantic']}")
    print(f"    Exotic: {bc['exotic']}")
    print(f"    Exotic ratio: {bc['exotic_ratio']:.0%}")

    print(f"\n  Top 10 final candidates:")
    for rank, c in enumerate(population[:10], 1):
        tc = classify_tokens(c.tokens)
        exotic_tag = f" [exotic:{tc['exotic_ratio']:.0%}]" if tc['exotic'] else ""
        print(f"    {rank:2d}. {c.summary()}{exotic_tag}")

    # Top-K analysis
    analysis = analyze_top_k(population)
    print(f"\n  Exotic tokens in top-10: {analysis['exotic_candidates_in_top_k']} candidates contain exotic tokens")
    if analysis['exotic_token_frequency']:
        print(f"  Exotic token frequency in top-10:")
        for token, count in list(analysis['exotic_token_frequency'].items())[:10]:
            print(f"    '{token}': {count}")

    # Generation curve
    print(f"\n  Evolution curve:")
    for g in generations:
        bar_len = int(max(0, g["best_score"] + 5) * 2)
        bar = "█" * bar_len
        print(f"    Gen {g['generation']:2d}: avg={g['avg_score']:+.2f}  best={g['best_score']:+.2f}  {bar}")

    # Semantic vs non-semantic
    print(f"\n  Semantic vs Non-Semantic Analysis (top 10):")
    semantic_scores = []
    exotic_scores = []
    for c in population[:10]:
        tc = classify_tokens(c.tokens)
        if tc['exotic_ratio'] > 0.3:
            exotic_scores.append(c.avg_score)
        else:
            semantic_scores.append(c.avg_score)
    if semantic_scores:
        print(f"    Semantic-dominant (exotic<30%): avg={sum(semantic_scores)/len(semantic_scores):+.2f} n={len(semantic_scores)}")
    if exotic_scores:
        print(f"    Exotic-dominant (exotic>30%): avg={sum(exotic_scores)/len(exotic_scores):+.2f} n={len(exotic_scores)}")

    print(f"\n  Results saved to: {RESULTS_DIR / RESULTS_FILE}")
    print("=" * 70)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_experiment()
