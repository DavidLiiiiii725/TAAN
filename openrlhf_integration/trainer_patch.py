"""
openrlhf_integration/trainer_patch.py
=======================================
Patch for OpenRLHF's GRPO trainer to replace global advantage
normalization with TAAN (within-type normalization).

Target location in OpenRLHF
----------------------------
The advantage normalization in OpenRLHF's GRPO implementation is in
``openrlhf/trainer/grpo_trainer.py``.  The relevant method is typically
``GRPOTrainer.training_step()`` or a helper called ``_compute_advantages()``.

The patch replaces the single call::

    advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

with::

    from taan_advantage import compute_taan_advantages
    advantages = compute_taan_advantages(advantages, batch["problem_type"])

How to apply
------------
Option A — manual edit
    Follow the inline diff shown in the ``PATCH_DIFF`` constant below.

Option B — programmatic patch
    from openrlhf_integration.trainer_patch import patch_grpo_trainer
    patch_grpo_trainer()

Option C — use apply_patch.py
    python openrlhf_integration/apply_patch.py
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Union

import torch

# Bring in the TAAN implementation (assumes TAAN repo is on sys.path).
try:
    from taan_advantage import compute_taan_advantages, grpo_global_normalize
except ImportError:
    # Allow the patch module to be imported without the rest of TAAN installed.
    compute_taan_advantages = None  # type: ignore
    grpo_global_normalize = None  # type: ignore


# ---------------------------------------------------------------------------
# Inline diff for reference
# ---------------------------------------------------------------------------

PATCH_DIFF = """\
--- a/openrlhf/trainer/grpo_trainer.py
+++ b/openrlhf/trainer/grpo_trainer.py
@@ -1,6 +1,11 @@
+import sys, os
+# Allow importing TAAN modules when TAAN repo is on PYTHONPATH
+# or when running from the TAAN project directory.
+from taan_advantage import compute_taan_advantages

 class GRPOTrainer:
     ...
     def training_step(self, batch):
         ...
-        # Original global normalization
-        advantages = (advantages - advantages.mean()) / (advantages.std() + self.args.eps)
+        # TAAN: within-type normalization
+        if self.args.use_taan and "problem_type" in batch:
+            advantages = compute_taan_advantages(
+                advantages,
+                batch["problem_type"],
+                eps=getattr(self.args, "taan_eps", 1e-8),
+            )
+        else:
+            # Fallback: original global normalization
+            advantages = (advantages - advantages.mean()) / (advantages.std() + self.args.eps)
         ...
"""


# ---------------------------------------------------------------------------
# Patched advantage function (drop-in replacement)
# ---------------------------------------------------------------------------


def taan_normalize_advantages(
    advantages: torch.Tensor,
    batch: Dict[str, Any],
    use_taan: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize advantages using TAAN if ``use_taan`` and ``problem_type``
    are available, otherwise fall back to global normalization.

    This function is intended to be called inside ``GRPOTrainer.training_step``
    as a drop-in replacement for the inline normalization expression.

    Parameters
    ----------
    advantages:
        Raw per-sample advantages, shape ``(N,)``.
    batch:
        The current training batch dict.  Must contain ``"problem_type"``
        (list of str, length N) when TAAN is enabled.
    use_taan:
        If ``False``, falls back to standard global normalization.
    eps:
        Numerical stability constant.

    Returns
    -------
    torch.Tensor
        Normalized advantages.
    """
    if use_taan and "problem_type" in batch and compute_taan_advantages is not None:
        return compute_taan_advantages(advantages, batch["problem_type"], eps=eps)

    # Fallback: global normalization
    adv = advantages.float()
    return (adv - adv.mean()) / (adv.std(unbiased=True) + eps)


# ---------------------------------------------------------------------------
# Monkey-patch helper
# ---------------------------------------------------------------------------

# Candidate module paths for GRPOTrainer, tried in order.
_GRPO_MODULE_CANDIDATES = [
    ("openrlhf.trainer.grpo_trainer", "GRPOTrainer"),
    ("openrlhf.trainer.ppo_trainer", "GRPOTrainer"),
    ("openrlhf.trainer.ray.grpo_trainer", "GRPOTrainer"),
    ("openrlhf.trainer.ray.ppo_actor", "GRPOTrainer"),
]


def _import_grpo_trainer():
    """Return (module, class_name, OrigClass) for the first importable
    GRPOTrainer candidate.  Returns ``(None, None, None)`` if not found."""
    for mod_path, cls_name in _GRPO_MODULE_CANDIDATES:
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                return mod, cls_name, cls
        except ImportError:
            continue
    return None, None, None


def patch_grpo_trainer(use_taan: bool = True, taan_eps: float = 1e-8) -> None:
    """Monkey-patch OpenRLHF's GRPOTrainer to use TAAN normalization.

    Call this once **before** the trainer is instantiated.

    Automatically discovers the GRPOTrainer class across multiple possible
    module paths to support different OpenRLHF versions.

    Parameters
    ----------
    use_taan:
        Whether to enable TAAN. If ``False``, training is identical to
        the original GRPO.
    taan_eps:
        Epsilon for numerical stability in TAAN normalization.
    """
    grpo_mod, cls_name, OrigTrainer = _import_grpo_trainer()
    if OrigTrainer is None:
        print(
            "[TAAN] WARNING: GRPOTrainer not found in any of the expected "
            "OpenRLHF module paths. Trainer patch not applied.",
            file=sys.stderr,
        )
        return

    class TAANGRPOTrainer(OrigTrainer):  # type: ignore
        """GRPOTrainer subclass with TAAN advantage normalization."""

        def training_step(self, batch: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            # We intercept the batch and override how advantages are normalized.
            # Store TAAN config on the instance so the overridden method can
            # read it without modifying the original signature.
            self._taan_use_taan = use_taan
            self._taan_eps = taan_eps
            return super().training_step(batch, *args, **kwargs)

        def _normalize_advantages(
            self,
            advantages: torch.Tensor,
            problem_types: Optional[List[str]] = None,
        ) -> torch.Tensor:
            """Override the advantage normalization with TAAN."""
            taan_on = getattr(self, "_taan_use_taan", use_taan)
            eps = getattr(self, "_taan_eps", taan_eps)

            if taan_on and problem_types is not None and compute_taan_advantages is not None:
                return compute_taan_advantages(advantages, problem_types, eps=eps)

            adv = advantages.float()
            return (adv - adv.mean()) / (adv.std(unbiased=True) + eps)

    setattr(grpo_mod, cls_name, TAANGRPOTrainer)
    print(
        f"[TAAN] {cls_name} patched in {grpo_mod.__name__} "
        f"(use_taan={use_taan}, taan_eps={taan_eps})."
    )
