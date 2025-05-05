#!/usr/bin/env python3

import csv
import logging
import random
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI

from config import OPENAI_API_KEY

# ——— Configuration ———
LEXICON_PATH = Path("/usr/share/dict/words")
LEXICON_LIMIT = 2000
RULE_COUNTS = [5, 10, 20, 40, 80, 100, 200, 400, 800]
TRIALS = 5
OUTPUT_CSV = Path("gpt41results.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ——— Logging setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ——— OpenAI client ———
client = OpenAI(api_key=OPENAI_API_KEY)

def load_lexicon(path: Path, limit: int) -> List[str]:
    """
    Load up to `limit` alphabetic words from `path` for rule sampling.
    """
    words: List[str] = []
    with path.open() as f:
        for line in f:
            word = line.strip().lower()
            if word.isalpha():
                words.append(word)
            if len(words) >= limit:
                break
    if len(words) < limit:
        raise FileNotFoundError(
            f"Found only {len(words)} words in {path}"
        )
    logger.info("Loaded %d unique words", len(words))
    return words


def build_prompt(rules: List[str]) -> str:
    """
    Build a story prompt listing rules for the LLM to include.
    """
    header = "Create a short story subject to the following rules:\n"
    body = "\n".join(
        f"{i}. Include the word '{word}'"
        for i, word in enumerate(rules, start=1)
    )
    return f"{header}{body}\n"


def is_transient_error(exc: Exception) -> bool:
    """
    Detect rate-limit or timeout errors from OpenAI.
    """
    status = getattr(exc, "http_status", None)
    if status == 429:
        return True
    code = getattr(exc, "code", "")
    if code in ("rate_limit_exceeded", "timeout"):
        return True
    msg = str(exc).lower()
    return "timeout" in msg or "timed out" in msg


def call_llm_with_retries(
    prompt: str,
    model: str = "gpt-4.1",
    timeout: float = 30.0,
    max_retries: int = 5,
    backoff_base: float = 1.0,
) -> str:
    """
    Send `prompt` to the LLM, retrying on transient errors with
    exponential backoff.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout,
            )
            return response.choices[0].message.content

        except Exception as exc:
            if is_transient_error(exc):
                wait = backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    "Transient error (%d/%d): %s — retrying in %.1fs",
                    attempt, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"LLM call failed after {max_retries} retries")


def run_experiment(
    lexicon: List[str],
    rule_counts: List[int],
    trials: int,
    csv_path: Path,
) -> None:
    """
    For each R in rule_counts, run `trials` stories asking the LLM
    to include R words. Record results to CSV and save the first
    prompt & story of each R.
    """
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["R", "trial", "passed", "failed", "adherence"] )

        for R in rule_counts:
            for t in range(trials):
                rules = random.sample(lexicon, R)
                prompt = build_prompt(rules)
                logger.info(
                    "R=%d trial=%d prompt len=%d", R, t, len(prompt)
                )
                story = call_llm_with_retries(prompt)

                if t == 0:
                    (OUTPUT_DIR / f"R{R:03d}_prompt.txt").write_text(
                        prompt, encoding="utf-8"
                    )
                    (OUTPUT_DIR / f"R{R:03d}_story.txt").write_text(
                        story, encoding="utf-8"
                    )
                    logger.info("Saved prompt & story for R=%d", R)

                passed = sum(1 for w in rules if w in story.lower())
                adherence = passed / R if R else 1.0
                writer.writerow([R, t, passed, R - passed, adherence])
                logger.info(
                    "→ passed %d/%d (%.1f%%)", passed, R, adherence * 100
                )


def analyze(csv_path: Path) -> None:
    """
    Load CSV results, compute mean ± std by R, and plot adherence curve.
    """
    df = pd.read_csv(csv_path)
    df["adherence"] = pd.to_numeric(df["adherence"], errors="coerce")

    summary = (
        df.groupby("R")["adherence"]
          .agg(["mean", "std"] )
          .reset_index()
    )
    logger.info("Summary stats:\n%s", summary)

    plt.errorbar(
        summary["R"], summary["mean"], yerr=summary["std"], marker="o"
    )
    plt.xlabel("Number of Rules")
    plt.ylabel("Adherence Rate")
    plt.title("Rule-Fatigue Curve")
    plt.tight_layout()
    plt.show()


def main() -> None:
    lexicon = load_lexicon(LEXICON_PATH, LEXICON_LIMIT)
    run_experiment(lexicon, RULE_COUNTS, TRIALS, OUTPUT_CSV)
    analyze(OUTPUT_CSV)


if __name__ == "__main__":
    main()
