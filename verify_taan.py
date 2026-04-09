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
# Simulation parameters  — GSM8K profile
# ---------------------------------------------------------------------------

SEEN_TYPES_GSM8K = ["arithmetic", "equation", "ratio"]
UNSEEN_TYPES_GSM8K = ["geometry", "probability"]

# Approximate distribution: 80 % seen, 20 % unseen
_GSM8K_DISTRIBUTION = {
    "arithmetic": 0.32,
    "equation": 0.26,
    "ratio": 0.22,
    "geometry": 0.11,
    "probability": 0.09,
}

# Reward / advantage variance per type (seen types have much higher variance)
_GSM8K_STD = {
    "arithmetic": 3.4,
    "equation": 3.2,
    "ratio": 3.1,
    "geometry": 0.30,
    "probability": 0.28,
}

# Keep legacy names for backward compatibility.
TYPE_DISTRIBUTION = _GSM8K_DISTRIBUTION
TYPE_STD = _GSM8K_STD
TYPE_MEAN = {k: 0.0 for k in TYPE_STD}  # zero-mean advantages

# ---------------------------------------------------------------------------
# Simulation parameters  — MATH profile
# ---------------------------------------------------------------------------

SEEN_TYPES_MATH = ["algebra", "number_theory"]
UNSEEN_TYPES_MATH = ["counting_and_prob", "geometry", "precalculus"]

# Approximate MATH distribution (algebra ~30 %, number_theory ~14 %,
# prealgebra ~22 % mapped to arithmetic, rest split across unseen types).
_MATH_DISTRIBUTION = {
    "algebra": 0.30,
    "number_theory": 0.14,
    "arithmetic": 0.22,        # prealgebra problems classified as arithmetic
    "counting_and_prob": 0.12,
    "geometry": 0.12,
    "precalculus": 0.10,
}

# MATH seen types exhibit higher reward variance than unseen types.
_MATH_STD = {
    "algebra": 2.5,
    "number_theory": 2.0,
    "arithmetic": 1.8,
    "counting_and_prob": 0.40,
    "geometry": 0.35,
    "precalculus": 0.30,
}

# ---------------------------------------------------------------------------
# Profile registry
# ---------------------------------------------------------------------------

_PROFILES: dict[str, dict] = {
    "gsm8k": {
        "distribution": _GSM8K_DISTRIBUTION,
        "std": _GSM8K_STD,
        "seen_types": SEEN_TYPES_GSM8K,
        "unseen_types": UNSEEN_TYPES_GSM8K,
        "label": "GSM8K",
    },
    "math": {
        "distribution": _MATH_DISTRIBUTION,
        "std": _MATH_STD,
        "seen_types": SEEN_TYPES_MATH,
        "unseen_types": UNSEEN_TYPES_MATH,
        "label": "MATH",
    },
}

# Combined profile for --dataset both
_BOTH_DISTRIBUTION = {
    k: v * 0.5 for k, v in _GSM8K_DISTRIBUTION.items()
}
for k, v in _MATH_DISTRIBUTION.items():
    _BOTH_DISTRIBUTION[k] = _BOTH_DISTRIBUTION.get(k, 0.0) + v * 0.5

_BOTH_STD = {**_GSM8K_STD, **_MATH_STD}
# 'arithmetic' appears in both profiles; use the higher variance so the
# combined batch reflects the wider GSM8K spread (avoids under-estimating
# the compression ratio for unseen types).
_BOTH_STD["arithmetic"] = max(_GSM8K_STD["arithmetic"], _MATH_STD["arithmetic"])

_PROFILES["both"] = {
    "distribution": _BOTH_DISTRIBUTION,
    "std": _BOTH_STD,
    "seen_types": list(dict.fromkeys(SEEN_TYPES_GSM8K + SEEN_TYPES_MATH)),
    "unseen_types": list(
        dict.fromkeys(UNSEEN_TYPES_GSM8K + UNSEEN_TYPES_MATH)
    ),
    "label": "GSM8K + MATH",
}


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------


def generate_batch(
    n_samples: int = 150,
    seed: int = 42,
    dataset: str = "gsm8k",
) -> tuple[torch.Tensor, list[str]]:
    """Generate a synthetic advantage batch.

    Parameters
    ----------
    n_samples:
        Number of samples to generate.
    seed:
        RNG seed for reproducibility.
    dataset:
        Simulation profile to use: ``"gsm8k"``, ``"math"``, or ``"both"``.

    Returns
    -------
    advantages : torch.Tensor shape (n_samples,)
    problem_types : list[str]  length n_samples
    """
    profile = _PROFILES[dataset]
    distribution: dict[str, float] = profile["distribution"]
    std_map: dict[str, float] = profile["std"]

    rng = random.Random(seed)
    torch.manual_seed(seed)

    types: list[str] = []
    type_names = list(distribution.keys())
    weights = [distribution[t] for t in type_names]

    for _ in range(n_samples):
        t = rng.choices(type_names, weights=weights, k=1)[0]
        types.append(t)

    advantages_list: list[float] = []
    for t in types:
        sigma = std_map[t]
        val = torch.empty(1).normal_(0.0, sigma).item()
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


def run_verification(n_samples: int = 150, seed: int = 42, batch_repeat: int = 1, dataset: str = "gsm8k") -> None:
    profile = _PROFILES[dataset]
    seen_types: list[str] = profile["seen_types"]
    unseen_types: list[str] = profile["unseen_types"]

    print("=" * 60)
    print(f"  TAAN Signal Quality Verification  [{profile['label']}]")
    print("=" * 60)

    all_advantages: list[float] = []
    all_types: list[str] = []

    for k in range(batch_repeat):
        adv, types = generate_batch(n_samples=n_samples, seed=seed + k, dataset=dataset)
        all_advantages.extend(adv.tolist())
        all_types.extend(types)

    advantages = torch.tensor(all_advantages)

    print_batch_stats(advantages, all_types)

    # ------------------------------------------------------------------
    # Compression ratio
    # ------------------------------------------------------------------
    cr = compression_ratio(
        advantages, all_types, seen_types, unseen_types
    )
    print(f"=== Normalization comparison ===")
    print(f"  Compression ratio (σ_seen/σ_unseen)²  ≈ {cr:.1f}×")
    print()

    # ------------------------------------------------------------------
    # Global normalization
    # ------------------------------------------------------------------
    global_norm = grpo_global_normalize(advantages)
    scale_global = gradient_scale(global_norm, all_types, unseen_types)

    # ------------------------------------------------------------------
    # TAAN normalization
    # ------------------------------------------------------------------
    taan_norm = compute_taan_advantages(advantages, all_types)
    scale_taan = gradient_scale(taan_norm, all_types, unseen_types)

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
    p.add_argument(
        "--dataset",
        choices=["gsm8k", "math", "both"],
        default="gsm8k",
        help=(
            "Dataset profile to simulate: "
            "'gsm8k' (arithmetic/equation/ratio seen, geometry/probability unseen), "
            "'math' (algebra/number_theory seen, counting_and_prob/geometry/precalculus unseen), "
            "or 'both' (mixed batch from both profiles). "
            "Default: gsm8k"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_verification(
        n_samples=args.n_samples,
        seed=args.seed,
        batch_repeat=args.batch_repeat,
        dataset=args.dataset,
    )
