"""
problem_classifier.py
=====================
Keyword-based math problem type classifier for TAAN.

Supported datasets
------------------
GSM8K  : arithmetic, equation, ratio, geometry, probability
MATH   : algebra, number_theory, counting_and_prob, geometry, precalculus

The classifier first tries a lightweight rule-based approach (fast, no GPU
needed). If you later want to swap in an LLM-based classifier you can
subclass ProblemClassifier and override `classify()`.
"""

from __future__ import annotations

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Type registry
# ---------------------------------------------------------------------------

#: All recognised problem types across both datasets.
ALL_TYPES = [
    "arithmetic",
    "equation",
    "ratio",
    "geometry",
    "probability",
    "algebra",
    "number_theory",
    "counting_and_prob",
    "precalculus",
    "unknown",
]

# ---------------------------------------------------------------------------
# Keyword rules  (ordered from most-specific to least-specific)
# ---------------------------------------------------------------------------

# Each entry is (type_name, list_of_keyword_patterns).
# A problem is assigned the FIRST type whose keywords fire.

_RULES: list[tuple[str, list[str]]] = [
    # ------------------------------------------------------------------
    # geometry  (must come before ratio/arithmetic so "area" wins)
    # ------------------------------------------------------------------
    (
        "geometry",
        [
            r"\barea\b",
            r"\bperimeter\b",
            r"\bcircumference\b",
            r"\bradius\b",
            r"\bdiameter\b",
            r"\bvolume\b",
            r"\btriangle\b",
            r"\brectangle\b",
            r"\bcircle\b",
            r"\bsquare\b",
            r"\bpolygon\b",
            r"\bangle\b",
            r"\bcoordinate\b",
            r"\bline segment\b",
            r"\bhypotenuse\b",
            r"\bpythagorean\b",
            r"\bsurface area\b",
            r"\bparabola\b",
            r"\bellipse\b",
            r"\bhyperbola\b",
            r"\bslope\b",
            r"\bmidpoint\b",
        ],
    ),
    # ------------------------------------------------------------------
    # counting_and_prob  (before probability so "permutation" wins)
    # ------------------------------------------------------------------
    (
        "counting_and_prob",
        [
            r"\bpermutation",
            r"\bcombination",
            r"\bin how many ways\b",
            r"\bc\(\s*\d",
            r"\bp\(\s*\d",
            r"\bbinomial\b",
            r"\bfactorial\b",
            r"\barrangement\b",
            r"\bselection\b",
            r"\bways? to (choose|select|arrange)\b",
        ],
    ),
    # ------------------------------------------------------------------
    # probability
    # ------------------------------------------------------------------
    (
        "probability",
        [
            r"\bprobabilit\b",
            r"\blikelihood\b",
            r"\bchance\b",
            r"\bprobable\b",
            r"\bfair coin\b",
            r"\bdie\b",
            r"\bdice\b",
            r"\bdraw.*card\b",
            r"\bcard.*drawn\b",
            r"\brandom(ly)?\b",
        ],
    ),
    # ------------------------------------------------------------------
    # precalculus
    # ------------------------------------------------------------------
    (
        "precalculus",
        [
            r"\btrigonometr\b",
            r"\bsine?\b",
            r"\bcosine?\b",
            r"\btangent\b",
            r"\bsec(ant)?\b",
            r"\bcosec(ant)?\b",
            r"\bcotangent\b",
            r"\bvector\b",
            r"\bmatri(x|ces)\b",
            r"\bdeterminant\b",
            r"\bcomplex number\b",
            r"\bimaginary\b",
            r"\bpolar (form|coordinate)\b",
            r"\bsequence\b",
            r"\bseries\b",
            r"\blimit\b",
            r"\bderivative\b",
            r"\bintegral\b",
        ],
    ),
    # ------------------------------------------------------------------
    # number theory
    # ------------------------------------------------------------------
    (
        "number_theory",
        [
            r"\bprime\b",
            r"\bdivisib\b",
            r"\bfactor\b",
            r"\bgcd\b",
            r"\blcm\b",
            r"\bmod(ulo)?\b",
            r"\bremainder\b",
            r"\binteger\b",
            r"\bdigit\b",
            r"\beven\b",
            r"\bodd\b",
            r"\bdivisor\b",
            r"\bmultiple\b",
        ],
    ),
    # ------------------------------------------------------------------
    # algebra  (equations / expressions / polynomials)
    # Checked BEFORE equation so "quadratic", "polynomial", etc. take
    # priority over generic "solve" / "equation" keywords.
    # Note: "solve for" is NOT listed here; it belongs to equation type.
    # ------------------------------------------------------------------
    (
        "algebra",
        [
            r"\bpolynomial\b",
            r"\bquadratic\b",
            r"\bfunction\b",
            r"\bf\s*\(\s*x\s*\)\b",
            r"\broot(s)?\b",
            r"\bzero(s)?\b",
            r"\bfactor(ize|ization)?\b",
            r"\bexpand\b",
            r"\bsimplif\b",
            r"\bsystem of equation\b",
            r"\blinear equation\b",
            r"\binequality\b",
            r"\binequalit\b",
            r"\bexponent\b",
            r"\blogarithm\b",
        ],
    ),
    # ------------------------------------------------------------------
    # equation  (word-problem style: "solve for x", "find value of y")
    # ------------------------------------------------------------------
    (
        "equation",
        [
            r"\bsolve\b",
            r"\bequation\b",
            r"\bfind (the )?(value|number|x|y|n)\b",
            r"\bwhat (is|was|are|were) (the )?(value|number)\b",
            r"\bunknown\b",
            r"\bvariable\b",
            r"\blet x\b",
            r"\bif x\b",
        ],
    ),
    # ------------------------------------------------------------------
    # ratio / proportion / percentage
    # Note: "\bper\b" is intentionally omitted — it is too broad and
    # fires on arithmetic phrases like "items per day".
    # ------------------------------------------------------------------
    (
        "ratio",
        [
            r"\bratio\b",
            r"\bproportion\b",
            r"\bpercent(age)?\b",
            r"\bfraction\b",
            r"\brate\b",
            r"\bdiscount\b",
            r"\bmarkup\b",
            r"\btax\b",
            r"\binterest\b",
            r"\bsale price\b",
            r"\bscale\b",
        ],
    ),
    # ------------------------------------------------------------------
    # arithmetic  (catch-all for basic word problems)
    # ------------------------------------------------------------------
    (
        "arithmetic",
        [
            r"\badd\b",
            r"\bsubtract\b",
            r"\bmultipl\b",
            r"\bdivid\b",
            r"\btotal\b",
            r"\bsum\b",
            r"\bdifference\b",
            r"\bproduct\b",
            r"\bquotient\b",
            r"\bhow many\b",
            r"\bhow much\b",
            r"\bhow (far|long|old|tall|heavy|fast)\b",
            r"\bmore than\b",
            r"\bless than\b",
            r"\btimes\b",
            r"\bspend\b",
            r"\bbuy\b",
            r"\bsell\b",
            r"\beach\b",
            r"\btogether\b",
        ],
    ),
]

# Compile all patterns once for speed.
_COMPILED_RULES: list[tuple[str, list[re.Pattern]]] = [
    (name, [re.compile(p, re.IGNORECASE) for p in patterns])
    for name, patterns in _RULES
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class ProblemClassifier:
    """Rule-based math problem type classifier.

    Parameters
    ----------
    default_type:
        Type label to return when no rule fires. Defaults to ``"unknown"``.
    """

    def __init__(self, default_type: str = "unknown") -> None:
        self.default_type = default_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, problem: str) -> str:
        """Return the problem type for a single problem string.

        Parameters
        ----------
        problem:
            The raw problem text (question only; answer not required).

        Returns
        -------
        str
            One of the labels in ``ALL_TYPES``.
        """
        for type_name, patterns in _COMPILED_RULES:
            for pat in patterns:
                if pat.search(problem):
                    return type_name
        return self.default_type

    def classify_batch(self, problems: List[str]) -> List[str]:
        """Classify a list of problems.

        Parameters
        ----------
        problems:
            List of raw problem strings.

        Returns
        -------
        List[str]
            A list of type labels, one per problem.
        """
        return [self.classify(p) for p in problems]


# ---------------------------------------------------------------------------
# Convenience singleton
# ---------------------------------------------------------------------------

_default_classifier: Optional[ProblemClassifier] = None


def get_default_classifier() -> ProblemClassifier:
    """Return a module-level singleton :class:`ProblemClassifier`."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = ProblemClassifier()
    return _default_classifier


def classify_problem(problem: str) -> str:
    """Classify a single problem using the default classifier."""
    return get_default_classifier().classify(problem)


def classify_problems(problems: List[str]) -> List[str]:
    """Classify a list of problems using the default classifier."""
    return get_default_classifier().classify_batch(problems)


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    examples = [
        "Tom has 5 apples and buys 3 more. How many apples does he have?",
        "Find the area of a circle with radius 7.",
        "What is the probability of rolling a 6 on a fair die?",
        "Solve for x: 2x + 3 = 11.",
        "A store offers a 20% discount. What is the sale price of a $50 item?",
        "Find all prime factors of 360.",
        "Simplify the polynomial: x^2 + 5x + 6.",
        "If sin(θ) = 0.5, find θ in degrees.",
        "In how many ways can you choose 3 items from 10?",
    ]

    clf = ProblemClassifier()
    for ex in examples:
        print(f"[{clf.classify(ex):>20s}]  {ex}")
