"""
tests/test_taan.py
==================
Unit tests for TAAN core modules.

Run with:
    python -m pytest tests/test_taan.py -v
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import torch

# Ensure the project root is on sys.path so imports work both from the root
# and from the tests/ subdirectory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from taan_advantage import (
    compute_taan_advantages,
    compression_ratio,
    compute_type_statistics,
    grpo_global_normalize,
)
from problem_classifier import (
    ALL_TYPES,
    ProblemClassifier,
    classify_problem,
    classify_problems,
)


# ===========================================================================
# taan_advantage tests
# ===========================================================================


class TestComputeTAANAdvantages:
    """Tests for compute_taan_advantages()."""

    def test_two_groups_zero_mean(self):
        """Each group should have zero mean after normalization."""
        adv = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        types = ["arith", "arith", "arith", "geo", "geo", "geo"]
        out = compute_taan_advantages(adv, types)

        arith_mean = out[:3].mean().item()
        geo_mean = out[3:].mean().item()
        assert abs(arith_mean) < 1e-5, f"arith mean {arith_mean} != 0"
        assert abs(geo_mean) < 1e-5, f"geo mean {geo_mean} != 0"

    def test_two_groups_unit_std(self):
        """Each group should have std ≈ 1 after normalization."""
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        types = ["a", "a", "a", "a", "b", "b", "b", "b"]
        out = compute_taan_advantages(adv, types)

        std_a = out[:4].std(unbiased=True).item()
        std_b = out[4:].std(unbiased=True).item()
        assert abs(std_a - 1.0) < 1e-4, f"std_a {std_a} != 1"
        assert abs(std_b - 1.0) < 1e-4, f"std_b {std_b} != 1"

    def test_single_sample_per_type_returns_zero(self):
        """A type with only one sample should produce 0 (neutral gradient)."""
        adv = torch.tensor([5.0, 1.0, 2.0, 3.0])
        types = ["rare", "common", "common", "common"]
        out = compute_taan_advantages(adv, types)
        assert out[0].item() == 0.0, "Single-sample type should give 0"

    def test_list_input_works(self):
        """Should accept Python lists as well as tensors."""
        adv = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
        types = ["a", "a", "a", "b", "b", "b"]
        out = compute_taan_advantages(adv, types)
        assert isinstance(out, torch.Tensor)
        assert out.shape == torch.Size([6])

    def test_single_type_same_as_global(self):
        """With one type, TAAN should equal global normalization."""
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        types = ["x"] * 5
        taan = compute_taan_advantages(adv, types)
        global_ = grpo_global_normalize(adv)
        assert torch.allclose(taan, global_, atol=1e-5), "Single type: TAAN != global"

    def test_output_shape_preserved(self):
        """Output shape must equal input shape."""
        n = 20
        adv = torch.randn(n)
        types = ["a"] * 10 + ["b"] * 10
        out = compute_taan_advantages(adv, types)
        assert out.shape == adv.shape

    def test_equal_scale_across_types(self):
        """
        When seen-type advantages have high variance and unseen-type
        advantages have low variance, TAAN should give both ~unit std,
        unlike global normalization which suppresses the low-variance type.
        """
        torch.manual_seed(0)
        seen_adv = torch.randn(100) * 10.0      # σ ≈ 10
        unseen_adv = torch.randn(20) * 0.1       # σ ≈ 0.1
        adv = torch.cat([seen_adv, unseen_adv])
        types = ["seen"] * 100 + ["unseen"] * 20

        taan = compute_taan_advantages(adv, types)
        global_ = grpo_global_normalize(adv)

        taan_unseen_std = taan[100:].std(unbiased=True).item()
        global_unseen_std = global_[100:].std(unbiased=True).item()

        # TAAN should give ~1.0 std for unseen
        assert abs(taan_unseen_std - 1.0) < 0.1, (
            f"TAAN unseen std {taan_unseen_std:.3f} not close to 1.0"
        )
        # Global norm should give << 1.0 std for unseen (compression)
        assert global_unseen_std < 0.2, (
            f"Global norm unseen std {global_unseen_std:.3f} should be suppressed"
        )


class TestGRPOGlobalNormalize:
    """Tests for grpo_global_normalize()."""

    def test_zero_mean(self):
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        out = grpo_global_normalize(adv)
        assert abs(out.mean().item()) < 1e-5

    def test_unit_std(self):
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        out = grpo_global_normalize(adv)
        assert abs(out.std(unbiased=True).item() - 1.0) < 1e-4

    def test_list_input(self):
        out = grpo_global_normalize([1.0, 2.0, 3.0])
        assert isinstance(out, torch.Tensor)


class TestComputeTypeStatistics:
    def test_returns_correct_keys(self):
        adv = torch.tensor([1.0, 2.0, 10.0, 20.0])
        types = ["a", "a", "b", "b"]
        stats = compute_type_statistics(adv, types)
        assert set(stats.keys()) == {"a", "b"}

    def test_mean_values(self):
        adv = torch.tensor([1.0, 3.0, 10.0, 20.0])
        types = ["a", "a", "b", "b"]
        stats = compute_type_statistics(adv, types)
        assert abs(stats["a"]["mean"] - 2.0) < 1e-5
        assert abs(stats["b"]["mean"] - 15.0) < 1e-5

    def test_count_values(self):
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        types = ["x", "x", "x", "y", "y"]
        stats = compute_type_statistics(adv, types)
        assert stats["x"]["count"] == 3
        assert stats["y"]["count"] == 2


class TestCompressionRatio:
    def test_high_compression(self):
        torch.manual_seed(1)
        seen = torch.randn(100) * 10.0
        unseen = torch.randn(20) * 0.1
        adv = torch.cat([seen, unseen])
        types = ["seen"] * 100 + ["unseen"] * 20
        cr = compression_ratio(adv, types, ["seen"], ["unseen"])
        # σ_seen ≈ 10, σ_unseen ≈ 0.1 → CR ≈ (10/0.1)² = 10000
        assert cr > 100, f"Expected high compression ratio, got {cr:.1f}"

    def test_equal_variance_gives_cr_one(self):
        torch.manual_seed(2)
        adv = torch.cat([torch.randn(50), torch.randn(50)])
        types = ["a"] * 50 + ["b"] * 50
        cr = compression_ratio(adv, types, ["a"], ["b"])
        # Both drawn from N(0,1) → CR ≈ 1
        assert 0.5 < cr < 2.0, f"Expected CR ≈ 1, got {cr:.3f}"

    def test_nan_with_single_sample(self):
        adv = torch.tensor([1.0, 2.0])
        types = ["seen", "unseen"]
        cr = compression_ratio(adv, types, ["seen"], ["unseen"])
        assert math.isnan(cr)


# ===========================================================================
# problem_classifier tests
# ===========================================================================


class TestProblemClassifier:
    """Tests for the rule-based classifier."""

    clf = ProblemClassifier()

    @pytest.mark.parametrize("problem,expected", [
        # arithmetic
        ("Tom has 5 apples and buys 3 more. How many apples does he have?", "arithmetic"),
        ("A store sells 120 items per day. How many items in a week?", "arithmetic"),
        # equation
        ("Solve for x: 2x + 3 = 11.", "equation"),
        ("Find the value of y if 3y - 9 = 0.", "equation"),
        # ratio / percentage
        ("A store offers a 20% discount. What is the sale price of a $50 item?", "ratio"),
        ("The ratio of boys to girls is 3:2. How many boys are there if there are 20 girls?", "ratio"),
        # geometry
        ("Find the area of a circle with radius 7.", "geometry"),
        ("What is the perimeter of a rectangle with length 5 and width 3?", "geometry"),
        # probability
        ("What is the probability of rolling a 6 on a fair die?", "probability"),
        ("A bag contains 3 red and 5 blue balls. What is the chance of drawing red?", "probability"),
        # number theory
        ("Find all prime factors of 360.", "number_theory"),
        ("What is the remainder when 17 is divided by 5?", "number_theory"),
        # algebra
        ("Simplify the polynomial: x^2 + 5x + 6.", "algebra"),
        ("Solve the quadratic equation: x^2 - 4 = 0.", "algebra"),
        # precalculus
        ("If sin(θ) = 0.5, find θ in degrees.", "precalculus"),
        ("Find the limit as x → 0 of sin(x)/x.", "precalculus"),
        # counting_and_prob
        ("In how many ways can you choose 3 items from 10?", "counting_and_prob"),
        ("How many permutations of the letters in 'MATH' are there?", "counting_and_prob"),
    ])
    def test_classify_known_types(self, problem, expected):
        result = self.clf.classify(problem)
        assert result == expected, (
            f"Expected '{expected}', got '{result}' for: {problem[:60]}"
        )

    def test_unknown_returns_default(self):
        result = self.clf.classify("xyz abc 123 @@@")
        assert result == "unknown"

    def test_classify_batch(self):
        problems = [
            "How many apples does she have?",
            "Find the area of the triangle.",
        ]
        results = self.clf.classify_batch(problems)
        assert len(results) == 2
        assert results[0] == "arithmetic"
        assert results[1] == "geometry"

    def test_custom_default_type(self):
        clf = ProblemClassifier(default_type="other")
        result = clf.classify("zzzz")
        assert result == "other"

    def test_all_types_are_valid(self):
        """Every type returned by classify() should be in ALL_TYPES."""
        test_problems = [
            "how many", "find area", "probability of", "solve for x",
            "percent discount", "prime factor", "polynomial", "sin theta",
            "permutations of", "zzzz"
        ]
        for p in test_problems:
            t = classify_problem(p)
            assert t in ALL_TYPES, f"Unknown type '{t}' returned for: {p}"

    def test_module_level_classify_problem(self):
        result = classify_problem("What is the area of a square with side 4?")
        assert result == "geometry"

    def test_module_level_classify_problems(self):
        results = classify_problems(["how many apples", "find the area"])
        assert results == ["arithmetic", "geometry"]


# ===========================================================================
# Integration: classifier + TAAN
# ===========================================================================


class TestClassifierWithTAAN:
    """End-to-end: classify problems then apply TAAN normalization."""

    def test_full_pipeline(self):
        problems = [
            "Tom has 5 apples and 3 oranges. How many fruits total?",
            "She bought 10 items and gave away 4. How many remain?",
            "Calculate 15% of 200.",
            "Find the area of a rectangle 6 × 4.",
            "What is the probability of getting heads twice in a row?",
        ]
        clf = ProblemClassifier()
        types = clf.classify_batch(problems)
        adv = torch.tensor([0.5, -0.3, 1.2, 5.0, 6.0])

        out = compute_taan_advantages(adv, types)
        assert out.shape == adv.shape
        # Output should be finite for all elements
        assert torch.all(torch.isfinite(out)), "TAAN output contains NaN/Inf"

    def test_all_same_type(self):
        """When all problems are the same type, TAAN == global normalization."""
        problems = [
            "Tom has 3 apples and buys 2 more. How many does he have?",
            "A farmer has 10 cows and 5 pigs. How many animals in total?",
            "She spent $4 on Monday and $6 on Tuesday. How much did she spend?",
            "There were 20 students and 3 left. How many remain?",
            "He collected 15 stamps. How many more does he need to reach 50?",
        ]
        clf = ProblemClassifier()
        types = clf.classify_batch(problems)
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        taan = compute_taan_advantages(adv, types)
        global_ = grpo_global_normalize(adv)
        assert torch.allclose(taan, global_, atol=1e-4), (
            "All-same-type TAAN should match global normalization"
        )
