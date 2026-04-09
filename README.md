# TAAN — Type-Aware Advantage Normalization

**TAAN** is a drop-in replacement for the global advantage normalization used in GRPO / REINFORCE++.
It normalizes advantages *within* each problem type instead of globally, preventing unseen problem
types from having their gradient signal compressed by dominant seen types.

---

## The Problem

Standard GRPO applies global batch-level advantage normalization:

$$\hat{A}_i = \frac{A_i - \mu_{\text{batch}}}{\sigma_{\text{batch}} + \varepsilon}$$

When a training batch contains a mix of **seen** types (high reward variance, many samples) and
**unseen** types (low variance, few samples), the global σ is dominated by the seen types.
The effective gradient scale for unseen types is shrunk by a factor of:

$$\text{CR} = \left(\frac{\sigma_{\text{seen}}}{\sigma_{\text{unseen}}}\right)^2$$

In GSM8K simulations this ratio is approximately **250×**, effectively silencing learning on unseen
problem types (geometry, probability) while seen types (arithmetic, equation, ratio) dominate.

## The Solution

TAAN normalizes within each problem type:

$$\hat{A}_i = \frac{A_i - \mu_\tau}{\sigma_\tau + \varepsilon}, \quad \tau = \text{problem\_type}(i)$$

This gives every type equal gradient signal strength regardless of how many examples appear in the
batch or how large their raw advantage variance is.

## Comparison with Related Methods

| Method | Normalization scope |
|---|---|
| GRPO | Within-question (same prompt, multiple rollouts) |
| REINFORCE++ | Global batch |
| Dr. GRPO / DAPO | No std normalization |
| **TAAN** | **Within problem-type** |

---

## Repository Structure

```
TAAN/
├── problem_classifier.py        # Keyword-based math problem type classifier
├── taan_advantage.py            # Core TAAN module: compute_taan_advantages()
├── verify_taan.py               # Signal quality verification / simulation
├── tests/
│   └── test_taan.py             # Unit & integration tests (pytest)
├── openrlhf_integration/
│   ├── __init__.py              # apply_taan_patches() convenience entry point
│   ├── dataset_patch.py         # PromptDataset patch: adds problem_type field
│   ├── trainer_patch.py         # GRPOTrainer patch: TAAN normalization
│   └── apply_patch.py           # One-shot patcher for installed OpenRLHF
├── verify.slurm                 # SLURM job: run verify_taan.py on NYU HPC
├── train_baseline.slurm         # SLURM job: GRPO baseline training
└── train_taan.slurm             # SLURM job: TAAN training
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch
```

### 2. Verify signal quality

```bash
python verify_taan.py --seed 42 --n-samples 150 --batch-repeat 10
```

Expected output (abridged):

```
Compression ratio (σ_seen/σ_unseen)²  ≈ 113×

Global norm  — unseen-type gradient scale : 0.105
TAAN norm    — unseen-type gradient scale : 0.983  (target ≈ 1.0)

Signal amplification by TAAN : 9.3×
```

### 3. Run tests

```bash
python -m pytest tests/ -v
```

---

## Integrating TAAN into OpenRLHF

### Option A — programmatic patch (recommended)

Add these two lines to the **top** of your training script, before any OpenRLHF imports:

```python
from openrlhf_integration import apply_taan_patches
apply_taan_patches()
```

### Option B — one-shot file patch

```bash
python openrlhf_integration/apply_patch.py
```

This modifies the installed OpenRLHF package in-place (backs up originals as `.bak`).

### What the patch does

| Location | Change |
|---|---|
| `openrlhf/datasets/prompts_dataset.py` | Adds `"problem_type"` key to every `__getitem__` return value |
| `openrlhf/trainer/grpo_trainer.py` | Replaces `(A - mean) / std` with `compute_taan_advantages(A, batch["problem_type"])` |

The `problem_type` field is populated by `problem_classifier.ProblemClassifier`, a fast
rule-based classifier that requires no GPU.

---

## Experiment Setup

| Setting | Value |
|---|---|
| Base model | Qwen3-4B |
| Framework | OpenRLHF |
| Compute | NYU Torch HPC, H200 GPU, SLURM |
| Dataset | GSM8K + MATH |
| Seen types (GSM8K) | arithmetic, equation, ratio |
| Unseen types (GSM8K) | geometry, probability |
| Seen types (MATH) | algebra, number_theory |
| Unseen types (MATH) | counting_and_prob, geometry, precalculus |

### Running on NYU HPC

```bash
# Verify signal quality (CPU, fast)
sbatch verify.slurm

# Train baseline (GRPO, global norm)
sbatch train_baseline.slurm

# Train TAAN
sbatch train_taan.slurm
```

---

## API Reference

### `taan_advantage.py`

```python
from taan_advantage import compute_taan_advantages, grpo_global_normalize

# TAAN: within-type normalization
normalized = compute_taan_advantages(advantages, problem_types, eps=1e-8)

# Baseline: global normalization
normalized = grpo_global_normalize(advantages, eps=1e-8)

# Diagnostics
from taan_advantage import compute_type_statistics, compression_ratio
stats = compute_type_statistics(advantages, problem_types)
cr = compression_ratio(advantages, problem_types, seen_types, unseen_types)
```

### `problem_classifier.py`

```python
from problem_classifier import classify_problem, classify_problems, ProblemClassifier

# Single problem
ptype = classify_problem("Find the area of a circle with radius 7.")
# → "geometry"

# Batch
ptypes = classify_problems(["How many apples?", "Solve for x: 2x = 6."])
# → ["arithmetic", "equation"]

# Custom classifier
clf = ProblemClassifier(default_type="unknown")
ptype = clf.classify("...")
```

**Supported types:** `arithmetic`, `equation`, `ratio`, `geometry`, `probability`,
`algebra`, `number_theory`, `counting_and_prob`, `precalculus`, `unknown`

---

## Citation

If you use TAAN in your research, please cite:

```bibtex
@misc{taan2025,
  title  = {Type-Aware Advantage Normalization for Generalizable RLVR},
  author = {David Li},
  year   = {2025},
  note   = {NYU DURF Research Project}
}
```
