"""
openrlhf_integration/dataset_patch.py
======================================
Patch for OpenRLHF's PromptDataset (or equivalent dataset class) to add
a ``problem_type`` field to every sample.

How to apply
------------
Option A — drop-in replacement
    Copy/import ``TAANPromptDataset`` instead of ``PromptDataset`` in your
    training script.

Option B — monkey-patch
    from openrlhf_integration.dataset_patch import patch_dataset_class
    patch_dataset_class()

The actual OpenRLHF class is located in
    openrlhf/datasets/prompts_dataset.py
This patch expects the dataset items to be dicts with at least an
``"input"`` key containing the problem text.
"""

from __future__ import annotations

from typing import Any, Dict

from problem_classifier import classify_problem


# ---------------------------------------------------------------------------
# Drop-in replacement dataset class
# ---------------------------------------------------------------------------


class TAANPromptDatasetMixin:
    """Mixin that adds ``problem_type`` classification to any OpenRLHF
    prompt dataset.

    Usage
    -----
    ::

        from openrlhf.datasets.prompts_dataset import PromptDataset
        from openrlhf_integration.dataset_patch import TAANPromptDatasetMixin

        class TAANPromptDataset(TAANPromptDatasetMixin, PromptDataset):
            pass
    """

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item: Dict[str, Any] = super().__getitem__(index)  # type: ignore[misc]

        # Determine which key holds the raw problem text.
        problem_text: str = ""
        for key in ("input", "problem", "question", "prompt"):
            if key in item and isinstance(item[key], str):
                problem_text = item[key]
                break

        item["problem_type"] = classify_problem(problem_text)
        return item


# ---------------------------------------------------------------------------
# Collate function patch
# ---------------------------------------------------------------------------


def taan_collate_fn(batch: list[Dict[str, Any]], base_collate_fn=None) -> Dict[str, Any]:
    """Collate function wrapper that preserves ``problem_type`` as a list.

    Parameters
    ----------
    batch:
        List of dataset items (each is a dict from ``__getitem__``).
    base_collate_fn:
        The original collate function to call for all other fields.
        If ``None``, a simple default is used.

    Returns
    -------
    dict
        Collated batch with an extra ``"problem_type"`` key (list of str).
    """
    import torch
    from torch.utils.data import default_collate

    # Extract problem_type strings before passing to default collate
    # (default_collate cannot handle lists of strings in all versions).
    problem_types = [item.pop("problem_type", "unknown") for item in batch]

    collate = base_collate_fn if base_collate_fn is not None else default_collate
    collated = collate(batch)

    # Restore problem_types as a plain list (not a tensor).
    collated["problem_type"] = problem_types

    # Also restore into original items for safety.
    for item, pt in zip(batch, problem_types):
        item["problem_type"] = pt

    return collated


# ---------------------------------------------------------------------------
# Monkey-patch helper
# ---------------------------------------------------------------------------


def patch_dataset_class() -> None:
    """Monkey-patch OpenRLHF's PromptDataset to include problem_type.

    Call this once at the start of your training script **before** any
    dataset is instantiated.
    """
    try:
        import openrlhf.datasets.prompts_dataset as _mod
        OrigClass = _mod.PromptDataset

        class PatchedPromptDataset(TAANPromptDatasetMixin, OrigClass):  # type: ignore
            pass

        _mod.PromptDataset = PatchedPromptDataset
        print("[TAAN] PromptDataset patched with problem_type classification.")
    except ImportError:
        print(
            "[TAAN] WARNING: openrlhf.datasets.prompts_dataset not found. "
            "Dataset patch not applied."
        )
