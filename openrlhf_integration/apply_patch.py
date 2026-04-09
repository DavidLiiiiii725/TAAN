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
3. Patches ``openrlhf/trainer/grpo_trainer.py`` to use TAAN normalization.
4. Creates backup files (``*.bak``) before modifying.

The patches are idempotent: running twice is safe.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
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


# ---------------------------------------------------------------------------
# Patch definitions
# ---------------------------------------------------------------------------

# ---- Dataset patch -------------------------------------------------------
# We insert a problem_type field into the dataset's __getitem__ return value.

DATASET_IMPORT_OLD = "from openrlhf.datasets.utils import"
DATASET_IMPORT_NEW = (
    "from openrlhf.datasets.utils import"
)

# The __getitem__ return statement typically looks like:
#   return {"input_ids": ..., "attention_mask": ..., "info": ...}
# We wrap the return to add problem_type.

DATASET_RETURN_OLD = 'return {"input_ids": input_ids, "attention_mask": attention_mask, "info": info}'
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

TRAINER_NORM_OLD_PATTERNS = [
    # Pattern 1: most common form in OpenRLHF
    "advantages = (advantages - advantages.mean()) / (advantages.std() + eps)",
    # Pattern 2: with explicit unbiased=False
    "advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)",
    # Pattern 3: inline eps
    "advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)",
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

TRAINER_COLLATE_SEARCH = "def collate_fn("
TRAINER_COLLATE_NOTE = (
    "# TAAN NOTE: collate_fn must preserve 'problem_type' as list[str].\n"
    "# If using a custom collate_fn, ensure problem_type is not stacked as tensor.\n"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        patched = patch_file(dataset_file, DATASET_RETURN_OLD, DATASET_RETURN_NEW, dry_run=dry_run)
        if not patched:
            print("  Note: dataset patch may need manual application (see PATCH_DIFF in trainer_patch.py)")
    else:
        print(f"  [WARN] Dataset file not found: {dataset_file}")

    print()

    # ---- Trainer file -------------------------------------------------------
    trainer_file = root / "trainer" / "grpo_trainer.py"
    if trainer_file.exists():
        print(f"--- Patching trainer: {trainer_file}")
        text = trainer_file.read_text(encoding="utf-8")
        patched_any = False
        for old_pattern in TRAINER_NORM_OLD_PATTERNS:
            if old_pattern in text:
                patched_any = patch_file(trainer_file, old_pattern, TRAINER_NORM_NEW, dry_run=dry_run)
                break
        if not patched_any:
            print(
                "  [WARN] Normalization expression not found in grpo_trainer.py.\n"
                "         Please apply the patch manually (see PATCH_DIFF in trainer_patch.py)."
            )
    else:
        print(f"  [WARN] Trainer file not found: {trainer_file}")

    print()
    print("Done." + (" (dry-run — no files modified)" if dry_run else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply TAAN patches to OpenRLHF")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be patched without modifying files")
    args = parser.parse_args()
    apply_patches(dry_run=args.dry_run)
