"""
openrlhf_integration/apply_patch.py
=====================================
One-shot script that applies the TAAN integration patches to an installed
OpenRLHF package.

Usage
-----
    python openrlhf_integration/apply_patch.py [--dry-run]

The script:
1. Locates the installed openrlhf package.
2. Patches ``openrlhf/datasets/prompts_dataset.py`` to add problem_type.
3. Patches ``openrlhf/trainer/grpo_trainer.py`` (or equivalent) to use
   TAAN normalization.
4. Creates backup files (``*.bak``) before modifying.

The patches are idempotent: running twice is safe.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_openrlhf_root() -> Path:
    """Return the root directory of the installed openrlhf package."""
    spec = importlib.util.find_spec("openrlhf")
    if spec is None or spec.origin is None:
        raise ImportError(
            "openrlhf package not found. "
            "Install it with: pip install openrlhf"
        )
    return Path(spec.origin).parent


def backup(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  Backed up: {bak}")


def patch_file(path: Path, old: str, new: str, dry_run: bool = False) -> bool:
    """Replace ``old`` with ``new`` in ``path``. Returns True if patched."""
    text = path.read_text(encoding="utf-8")
    if old not in text:
        if new in text:
            print(f"  Already patched: {path}")
        else:
            print(f"  [WARN] Pattern not found in {path}. Manual patch may be needed.")
        return False

    new_text = text.replace(old, new, 1)
    if dry_run:
        print(f"  [DRY-RUN] Would patch: {path}")
        return True

    backup(path)
    path.write_text(new_text, encoding="utf-8")
    print(f"  Patched: {path}")
    return True


def patch_file_regex(path: Path, pattern: str, replacement: str, dry_run: bool = False) -> bool:
    """Replace the first regex match of ``pattern`` with ``replacement`` in
    ``path``.  Returns True if a replacement was made."""
    text = path.read_text(encoding="utf-8")
    # Check if already patched (idempotency marker)
    if "# TAAN:" in text:
        print(f"  Already patched: {path}")
        return False
    new_text, n = re.subn(pattern, replacement, text, count=1)
    if n == 0:
        print(f"  [WARN] Regex pattern not found in {path}. Manual patch may be needed.")
        return False
    if dry_run:
        print(f"  [DRY-RUN] Would patch: {path}")
        return True
    backup(path)
    path.write_text(new_text, encoding="utf-8")
    print(f"  Patched: {path}")
    return True


# ---------------------------------------------------------------------------
# Patch definitions
# ---------------------------------------------------------------------------

# ---- Dataset patch -------------------------------------------------------
# We insert a problem_type field into the dataset's __getitem__ return value.
#
# The return statement in prompts_dataset.py may vary by version:
#   return {"input_ids": input_ids, "attention_mask": attention_mask, "info": info}
#   return {"input_ids": input_ids, "attention_mask": attention_mask, "info": info,}
# We try exact strings first, then fall back to a flexible regex.

DATASET_RETURN_PATTERNS = [
    # Exact match (most common)
    'return {"input_ids": input_ids, "attention_mask": attention_mask, "info": info}',
    # Trailing comma variant
    'return {"input_ids": input_ids, "attention_mask": attention_mask, "info": info,}',
]

# Regex fallback: match any return-dict that ends with "info": <name>}
DATASET_RETURN_REGEX = (
    r'(return \{"input_ids": \w+, "attention_mask": \w+, "info": \w+,?\})'
)

DATASET_RETURN_NEW = '''\
# TAAN: add problem_type for within-type advantage normalization
        try:
            from problem_classifier import classify_problem as _clf
            _problem_text = info.get("input", info.get("problem", info.get("question", "")))
            _problem_type = _clf(_problem_text) if isinstance(_problem_text, str) else "unknown"
        except Exception:
            _problem_type = "unknown"
        return {"input_ids": input_ids, "attention_mask": attention_mask, "info": info, "problem_type": _problem_type}'''


# ---- Trainer patch -------------------------------------------------------
# Replace the inline normalization expression in the trainer.
# OpenRLHF may store the GRPOTrainer in different locations depending on version:
#   - openrlhf/trainer/grpo_trainer.py          (v0.3+)
#   - openrlhf/trainer/ppo_trainer.py            (older, mixed PPO/GRPO)
#   - openrlhf/trainer/ray/grpo_trainer.py       (Ray distributed)

TRAINER_CANDIDATE_PATHS = [
    ("trainer", "grpo_trainer.py"),
    ("trainer", "ppo_trainer.py"),
    ("trainer", "ray", "grpo_trainer.py"),
    ("trainer", "ray", "ppo_actor.py"),
]

TRAINER_NORM_OLD_PATTERNS = [
    # Pattern 1: most common form in OpenRLHF
    "advantages = (advantages - advantages.mean()) / (advantages.std() + eps)",
    # Pattern 2: with explicit unbiased=False
    "advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)",
    # Pattern 3: inline eps
    "advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)",
    # Pattern 4: with self.eps
    "advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)",
]

TRAINER_NORM_NEW_TEMPLATE = """\
# TAAN: within-type advantage normalization
            _problem_types = batch.get("problem_type", None) if isinstance(batch, dict) else None
            if _problem_types is not None:
                try:
                    import sys as _sys, os as _os
                    _taan_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
                    if _taan_dir not in _sys.path:
                        _sys.path.insert(0, _taan_dir)
                    from taan_advantage import compute_taan_advantages as _taan_fn
                    advantages = _taan_fn(advantages, _problem_types, eps={eps})
                except Exception as _e:
                    # Fallback to global normalization if TAAN is unavailable
                    advantages = (advantages - advantages.mean()) / (advantages.std() + {eps})
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + {eps})"""

TRAINER_NORM_NEW = TRAINER_NORM_NEW_TEMPLATE.format(eps="1e-8")

TRAINER_COLLATE_NOTE = (
    "# TAAN NOTE: collate_fn must preserve 'problem_type' as list[str].\n"
    "# If using a custom collate_fn, ensure problem_type is not stacked as tensor.\n"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _find_trainer_file(root: Path) -> Path | None:
    """Return the first existing candidate trainer file, or None."""
    for parts in TRAINER_CANDIDATE_PATHS:
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return candidate
    return None


def _patch_trainer(trainer_file: Path, dry_run: bool) -> bool:
    """Try each known normalization pattern; return True if any matched."""
    text = trainer_file.read_text(encoding="utf-8")
    for old_pattern in TRAINER_NORM_OLD_PATTERNS:
        if old_pattern in text:
            return patch_file(trainer_file, old_pattern, TRAINER_NORM_NEW, dry_run=dry_run)
    return False


def _patch_dataset(dataset_file: Path, dry_run: bool) -> bool:
    """Try exact strings then regex to patch the dataset return statement."""
    for old in DATASET_RETURN_PATTERNS:
        if patch_file(dataset_file, old, DATASET_RETURN_NEW, dry_run=dry_run):
            return True
        # patch_file prints "Already patched" if new is already present
        text = dataset_file.read_text(encoding="utf-8")
        if DATASET_RETURN_NEW in text:
            return True  # already patched by a previous run

    # Regex fallback
    print("  Trying regex fallback for dataset patch …")
    return patch_file_regex(dataset_file, DATASET_RETURN_REGEX, DATASET_RETURN_NEW, dry_run=dry_run)


def apply_patches(dry_run: bool = False) -> None:
    print("Locating openrlhf package …")
    try:
        root = find_openrlhf_root()
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Found: {root}\n")

    # ---- Dataset file -------------------------------------------------------
    dataset_file = root / "datasets" / "prompts_dataset.py"
    if dataset_file.exists():
        print(f"--- Patching dataset: {dataset_file}")
        patched = _patch_dataset(dataset_file, dry_run)
        if not patched:
            print(
                "  Note: dataset patch could not be applied automatically.\n"
                "  Please inspect the file and apply the patch from PATCH_DIFF "
                "in trainer_patch.py manually."
            )
    else:
        print(f"  [WARN] Dataset file not found: {dataset_file}")

    print()

    # ---- Trainer file -------------------------------------------------------
    trainer_file = _find_trainer_file(root)
    if trainer_file is None:
        searched = ", ".join(str(root.joinpath(*p)) for p in TRAINER_CANDIDATE_PATHS)
        print(
            "  [WARN] No GRPO/PPO trainer file found. Searched:\n"
            f"    {searched}\n"
            "  Please apply the normalization patch manually "
            "(see PATCH_DIFF in trainer_patch.py)."
        )
    else:
        print(f"--- Patching trainer: {trainer_file}")
        patched_any = _patch_trainer(trainer_file, dry_run)
        if not patched_any:
            print(
                "  [WARN] Normalization expression not found in trainer file.\n"
                "         Please apply the patch manually (see PATCH_DIFF in trainer_patch.py)."
            )

    print()
    print("Done." + (" (dry-run — no files modified)" if dry_run else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply TAAN patches to OpenRLHF")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be patched without modifying files")
    args = parser.parse_args()
    apply_patches(dry_run=args.dry_run)
