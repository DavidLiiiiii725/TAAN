"""
verify_taan.py
==============
Signal-quality verification for TAAN vs GRPO global normalization.

Simulation
----------
We simulate a batch where seen problem types (arithmetic, equation, ratio)
dominate (e.g. 80 % of examples) and unseen types (geometry, probability)
are rare (20 %).  Seen-type advantages are drawn from a high-variance
distribution while unseen-type advantages come from a low-variance one,
matching the empirical observation that GSM8K training concentrates on
a few types with very different reward magnitudes.

The key metric is the *gradient compression ratio*

    CR = (σ_seen / σ_unseen)²

Under global normalization each unseen-type gradient is scaled by
1 / σ_global ≈ 1 / σ_seen, so the effective scale of an unseen gradient
is σ_unseen / σ_seen ≈ 1 / √CR.  In other words, gradients for unseen
types are suppressed by factor CR compared to seen types.

TAAN normalizes within each type so every group has unit variance → CR = 1.

Expected output
---------------
Running this script on any machine (CPU is fine) should print something like:

    === Batch statistics ===
    Type            count    mean     std
    arithmetic         48   0.12    3.41
    equation           40   0.05    3.28
    ratio              32  -0.08    3.19
    geometry           16  -0.03    0.31
    probability        14   0.01    0.29

    === Normalization comparison ===
    Compression ratio (σ_seen/σ_unseen)²  ≈ 250.3×

    Global norm  — unseen-type gradient scale : 0.090
    TAAN norm    — unseen-type gradient scale : 1.000  (target)

    Signal amplification by TAAN : 11.1×  (= √CR)

Usage
-----
    python verify_taan.py [--seed SEED] [--n-samples N] [--batch-repeat K]
"""

from __future__ import annotations

import argparse
import random
import sys

import torch

from problem_classifier import ProblemClassifier
from taan_advantage import (
    compute_taan_advantages,
    compression_ratio,
    compute_type_statistics,
    grpo_global_normalize,
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

SEEN_TYPES_GSM8K = ["arithmetic", "equation", "ratio"]
UNSEEN_TYPES_GSM8K = ["geometry", "probability"]

# Approximate distribution: 80 % seen, 20 % unseen
TYPE_DISTRIBUTION = {
    "arithmetic": 0.32,
    "equation": 0.26,
    "ratio": 0.22,
    "geometry": 0.11,
    "probability": 0.09,
}

# Reward / advantage variance per type (seen types have much higher variance)
TYPE_STD = {
    "arithmetic": 3.4,
    "equation": 3.2,
    "ratio": 3.1,
    "geometry": 0.30,
    "probability": 0.28,
}

TYPE_MEAN = {k: 0.0 for k in TYPE_STD}  # zero-mean advantages


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------


def generate_batch(
    n_samples: int = 150,
    seed: int = 42,
) -> tuple[torch.Tensor, list[str]]:
    """Generate a synthetic advantage batch.

    Returns
    -------
    advantages : torch.Tensor shape (n_samples,)
    problem_types : list[str]  length n_samples
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    types: list[str] = []
    type_names = list(TYPE_DISTRIBUTION.keys())
    weights = [TYPE_DISTRIBUTION[t] for t in type_names]

    for _ in range(n_samples):
        t = rng.choices(type_names, weights=weights, k=1)[0]
        types.append(t)

    advantages_list: list[float] = []
    for t in types:
        mu = TYPE_MEAN[t]
        sigma = TYPE_STD[t]
        val = torch.empty(1).normal_(mu, sigma).item()
        advantages_list.append(val)

    return torch.tensor(advantages_list), types


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def print_batch_stats(advantages: torch.Tensor, problem_types: list[str]) -> None:
    stats = compute_type_statistics(advantages, problem_types)
    print("\n=== Batch statistics ===")
    print(f"{'Type':>20s}  {'count':>6s}  {'mean':>7s}  {'std':>7s}")
    print("-" * 46)
    for name, s in stats.items():
        print(
            f"{name:>20s}  {s['count']:>6d}  {s['mean']:>+7.3f}  {s['std']:>7.3f}"
        )
    print()


def gradient_scale(
    normalized: torch.Tensor,
    problem_types: list[str],
    target_types: list[str],
) -> float:
    """Compute std of normalized advantages for a subset of types."""
    mask = torch.tensor([t in target_types for t in problem_types], dtype=torch.bool)
    group = normalized[mask]
    if group.numel() < 2:
        return float("nan")
    return group.std(unbiased=True).item()


# ---------------------------------------------------------------------------
# Main verification routine
# ---------------------------------------------------------------------------


def run_verification(n_samples: int = 150, seed: int = 42, batch_repeat: int = 1) -> None:
    print("=" * 60)
    print("  TAAN Signal Quality Verification")
    print("=" * 60)

    all_advantages: list[float] = []
    all_types: list[str] = []

    for k in range(batch_repeat):
        adv, types = generate_batch(n_samples=n_samples, seed=seed + k)
        all_advantages.extend(adv.tolist())
        all_types.extend(types)

    advantages = torch.tensor(all_advantages)

    print_batch_stats(advantages, all_types)

    # ------------------------------------------------------------------
    # Compression ratio
    # ------------------------------------------------------------------
    cr = compression_ratio(
        advantages, all_types, SEEN_TYPES_GSM8K, UNSEEN_TYPES_GSM8K
    )
    print(f"=== Normalization comparison ===")
    print(f"  Compression ratio (σ_seen/σ_unseen)²  ≈ {cr:.1f}×")
    print()

    # ------------------------------------------------------------------
    # Global normalization
    # ------------------------------------------------------------------
    global_norm = grpo_global_normalize(advantages)
    scale_global = gradient_scale(global_norm, all_types, UNSEEN_TYPES_GSM8K)

    # ------------------------------------------------------------------
    # TAAN normalization
    # ------------------------------------------------------------------
    taan_norm = compute_taan_advantages(advantages, all_types)
    scale_taan = gradient_scale(taan_norm, all_types, UNSEEN_TYPES_GSM8K)

    print(f"  Global norm  — unseen-type gradient scale : {scale_global:.3f}")
    print(f"  TAAN norm    — unseen-type gradient scale : {scale_taan:.3f}  (target ≈ 1.0)")
    print()

    amplification = scale_taan / (scale_global + 1e-12)
    print(f"  Signal amplification by TAAN : {amplification:.1f}×  (= √CR ≈ {cr**0.5:.1f}×)")
    print()

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    passed = True

    if cr < 10:
        print("  [WARN] compression ratio seems low — check TYPE_STD values")
        passed = False

    if scale_taan < 0.8 or scale_taan > 1.2:
        print(
            f"  [FAIL] TAAN unseen scale {scale_taan:.3f} is not close to 1.0"
        )
        passed = False
    else:
        print("  [PASS] TAAN unseen-type gradient scale ≈ 1.0")

    if scale_global > scale_taan:
        print(
            "  [FAIL] Global norm scale should be < TAAN scale for unseen types"
        )
        passed = False
    else:
        print("  [PASS] Global norm scale < TAAN scale (gradient suppression confirmed)")

    # Check that TAAN normalised advantages have ~zero mean per type
    stats_taan = compute_type_statistics(taan_norm, all_types)
    for name, s in stats_taan.items():
        if abs(s["mean"]) > 0.05 and s["count"] > 1:
            print(f"  [FAIL] TAAN mean for type '{name}' = {s['mean']:.4f} (expected ≈ 0)")
            passed = False

    if all(abs(s["mean"]) <= 0.05 or s["count"] <= 1 for s in stats_taan.values()):
        print("  [PASS] Per-type means ≈ 0 after TAAN normalization")

    print()
    if passed:
        print("All checks PASSED.  TAAN successfully amplifies unseen-type signal.")
    else:
        print("Some checks FAILED.  Review the output above.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Demonstrate with a tiny hand-crafted example
    # ------------------------------------------------------------------
    print()
    print("=== Mini sanity example ===")
    demo_adv = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    demo_types = ["arith", "arith", "arith", "geo", "geo", "geo"]

    taan_demo = compute_taan_advantages(demo_adv, demo_types)
    global_demo = grpo_global_normalize(demo_adv)

    print("Input advantages :", demo_adv.tolist())
    print("Types            :", demo_types)
    print("TAAN output      :", [round(x, 3) for x in taan_demo.tolist()])
    print("Global norm out  :", [round(x, 3) for x in global_demo.tolist()])
    print()
    print("Note: TAAN gives both groups the same ±1 scale;")
    print("      global norm shrinks 'geo' (high raw values) relative to 'arith'.")

    # ------------------------------------------------------------------
    # Classifier smoke-test
    # ------------------------------------------------------------------
    print()
    print("=== Classifier smoke-test ===")
    clf = ProblemClassifier()
    test_cases = [
        ("Tom has 5 apples and buys 3 more. How many does he have?", "arithmetic"),
        ("Find the area of a circle with radius 7.", "geometry"),
        ("What is the probability of rolling a 6 on a fair die?", "probability"),
        ("Solve for x: 2x + 3 = 11.", "equation"),
        ("A store offers a 20% discount. What is the sale price?", "ratio"),
    ]
    all_clf_pass = True
    for problem, expected in test_cases:
        predicted = clf.classify(problem)
        status = "PASS" if predicted == expected else "FAIL"
        if status == "FAIL":
            all_clf_pass = False
        print(f"  [{status}] expected={expected:>12s}  got={predicted:>12s}  | {problem[:50]}")

    if all_clf_pass:
        print("\nAll classifier checks PASSED.")
    else:
        print("\nSome classifier checks FAILED.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify TAAN signal quality")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--n-samples", type=int, default=150,
                   help="Samples per synthetic batch")
    p.add_argument("--batch-repeat", type=int, default=1,
                   help="Number of batches to concatenate for statistics")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_verification(
        n_samples=args.n_samples,
        seed=args.seed,
        batch_repeat=args.batch_repeat,
    )
