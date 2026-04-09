# openrlhf_integration/__init__.py
"""
TAAN — OpenRLHF integration package.

Provides patches for OpenRLHF's PromptDataset and GRPOTrainer to support
Type-Aware Advantage Normalization (TAAN).

Quick start
-----------
::

    # At the top of your training script, before any OpenRLHF imports:
    from openrlhf_integration import apply_taan_patches
    apply_taan_patches()

    # Then proceed with normal OpenRLHF training.
"""

from openrlhf_integration.dataset_patch import patch_dataset_class
from openrlhf_integration.trainer_patch import patch_grpo_trainer


def apply_taan_patches(use_taan: bool = True, taan_eps: float = 1e-8) -> None:
    """Apply all TAAN patches to the installed OpenRLHF package.

    Parameters
    ----------
    use_taan:
        Enable TAAN normalization. If ``False``, applies patches but
        keeps global normalization behavior.
    taan_eps:
        Epsilon for numerical stability.
    """
    patch_dataset_class()
    patch_grpo_trainer(use_taan=use_taan, taan_eps=taan_eps)
