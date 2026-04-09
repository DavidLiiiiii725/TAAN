"""
data/prepare_datasets.py
========================
Download GSM8K and MATH from HuggingFace, classify each problem by type
using the TAAN ProblemClassifier, and write the JSONL files expected by the
SLURM training scripts.

Output files (written to the same directory as this script)
-----------------------------------------------------------
gsm8k_math_all.jsonl
    All problems from both datasets.  Each line is a JSON object with keys:
        problem       – raw problem text
        solution      – reference solution / answer
        problem_type  – label from problem_classifier.py
        dataset       – "gsm8k" or "math"

gsm8k_math_seen.jsonl
    Subset containing only problems whose ``problem_type`` belongs to
    SEEN_TYPES (arithmetic, equation, ratio, algebra, number_theory).
    Used for the GRPO baseline training run.

gsm8k_math_unseen.jsonl
    Subset of problems whose ``problem_type`` belongs to UNSEEN_TYPES
    (geometry, probability, counting_and_prob, precalculus).
    Useful for held-out evaluation.

Usage
-----
    # Basic (downloads both datasets, uses HuggingFace cache)
    python data/prepare_datasets.py

    # Only GSM8K
    python data/prepare_datasets.py --datasets gsm8k

    # Only MATH
    python data/prepare_datasets.py --datasets math

    # Custom output directory
    python data/prepare_datasets.py --output-dir /path/to/data

    # Limit number of samples per dataset (useful for quick tests)
    python data/prepare_datasets.py --max-samples 1000

Dependencies
------------
    pip install datasets tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Ensure project root is on path so problem_classifier can be imported
# regardless of where this script is invoked from.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from problem_classifier import classify_problem  # noqa: E402

# ---------------------------------------------------------------------------
# Seen / unseen type splits (matching README experiment setup)
# ---------------------------------------------------------------------------

SEEN_TYPES: set[str] = {"arithmetic", "equation", "ratio", "algebra", "number_theory"}
UNSEEN_TYPES: set[str] = {"geometry", "probability", "counting_and_prob", "precalculus"}

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def _iter_gsm8k(max_samples: int | None = None) -> Iterator[dict]:
    """Yield normalised records from openai/gsm8k (train + test splits)."""
    from datasets import load_dataset  # type: ignore

    print("Loading GSM8K …", flush=True)
    ds = load_dataset("openai/gsm8k", "main", trust_remote_code=False)

    count = 0
    for split in ("train", "test"):
        for row in ds[split]:
            if max_samples is not None and count >= max_samples:
                return
            yield {
                "problem": row["question"].strip(),
                "solution": row["answer"].strip(),
                "dataset": "gsm8k",
            }
            count += 1


def _iter_math(max_samples: int | None = None) -> Iterator[dict]:
    """Yield normalised records from lighteval/MATH (train + test splits).

    Falls back to ``hendrycks/competition_math`` if ``lighteval/MATH`` is
    unavailable, then to ``competition_math`` (no namespace).
    """
    from datasets import load_dataset  # type: ignore

    candidates = [
        ("lighteval/MATH", "all"),
        ("hendrycks/competition_math", None),
        ("competition_math", None),
    ]

    ds = None
    for repo_id, config in candidates:
        try:
            print(f"Loading MATH from '{repo_id}' …", flush=True)
            kwargs: dict = {"trust_remote_code": False}
            if config:
                kwargs["name"] = config
            ds = load_dataset(repo_id, **kwargs)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  Could not load '{repo_id}': {exc}", flush=True)

    if ds is None:
        print(
            "WARNING: Could not load any MATH dataset variant. "
            "Skipping MATH.",
            flush=True,
        )
        return

    count = 0
    for split in ds.keys():
        for row in ds[split]:
            if max_samples is not None and count >= max_samples:
                return
            # Different versions use different field names.
            problem = (
                row.get("problem")
                or row.get("question")
                or row.get("input")
                or ""
            ).strip()
            solution = (
                row.get("solution")
                or row.get("answer")
                or row.get("output")
                or ""
            ).strip()
            if not problem:
                continue
            yield {
                "problem": problem,
                "solution": solution,
                "dataset": "math",
            }
            count += 1


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare(
    datasets: list[str],
    output_dir: Path,
    max_samples: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_path = output_dir / "gsm8k_math_all.jsonl"
    seen_path = output_dir / "gsm8k_math_seen.jsonl"
    unseen_path = output_dir / "gsm8k_math_unseen.jsonl"

    stats: dict[str, dict[str, int]] = {}  # dataset → {problem_type: count}
    total = 0

    with (
        open(all_path, "w", encoding="utf-8") as f_all,
        open(seen_path, "w", encoding="utf-8") as f_seen,
        open(unseen_path, "w", encoding="utf-8") as f_unseen,
    ):
        iterators: list[Iterator[dict]] = []
        if "gsm8k" in datasets:
            iterators.append(_iter_gsm8k(max_samples))
        if "math" in datasets:
            iterators.append(_iter_math(max_samples))

        for it in iterators:
            for record in it:
                ptype = classify_problem(record["problem"])
                record["problem_type"] = ptype

                line = json.dumps(record, ensure_ascii=False)
                f_all.write(line + "\n")

                ds_name = record["dataset"]
                stats.setdefault(ds_name, {})
                stats[ds_name][ptype] = stats[ds_name].get(ptype, 0) + 1
                total += 1

                if ptype in SEEN_TYPES:
                    f_seen.write(line + "\n")
                elif ptype in UNSEEN_TYPES:
                    f_unseen.write(line + "\n")

                if total % 1000 == 0:
                    print(f"  Processed {total} problems …", flush=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Dataset preparation complete")
    print(f"{'=' * 60}")
    print(f"  Output directory : {output_dir}")
    print(f"  Total problems   : {total}")
    print()

    for ds_name, type_counts in sorted(stats.items()):
        ds_total = sum(type_counts.values())
        print(f"  [{ds_name.upper()}] {ds_total} problems")
        for ptype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
            tag = ""
            if ptype in SEEN_TYPES:
                tag = "(seen)"
            elif ptype in UNSEEN_TYPES:
                tag = "(unseen)"
            print(f"    {ptype:>20s}: {cnt:>5d}  {tag}")
        print()

    seen_total = sum(
        cnt
        for tc in stats.values()
        for pt, cnt in tc.items()
        if pt in SEEN_TYPES
    )
    unseen_total = sum(
        cnt
        for tc in stats.values()
        for pt, cnt in tc.items()
        if pt in UNSEEN_TYPES
    )
    print(f"  Written to {all_path.name}    : {total}")
    print(f"  Written to {seen_path.name}   : {seen_total}")
    print(f"  Written to {unseen_path.name} : {unseen_total}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare GSM8K + MATH datasets for TAAN training."
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["gsm8k", "math"],
        default=["gsm8k", "math"],
        help="Which dataset(s) to include (default: both)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR,
        help="Directory to write JSONL files (default: same as this script)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (omit for full dataset)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare(
        datasets=args.datasets,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )
