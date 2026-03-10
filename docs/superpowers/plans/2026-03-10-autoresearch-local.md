# autoresearch-local Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local M3 Max adaptation of Karpathy's autoresearch — an autonomous ML experiment loop where Claude Code (Opus) proposes changes to a small GPT, trains it on TinyShakespeare, and keeps improvements via a git ratchet.

**Architecture:** Three-file design: `prepare.py` (frozen data pipeline + evaluation metric), `train.py` (mutable model + training loop), `program.md` (agent instructions). The agent edits only `train.py`, runs 2-minute experiments on MPS, and uses git commits as a ratchet to keep only improvements.

**Tech Stack:** Python 3.11+, PyTorch (MPS backend), numpy, requests. No CUDA dependencies.

**Spec:** `docs/superpowers/specs/2026-03-10-autoresearch-local-design.md`

---

## Chunk 1: Project Setup and Data Pipeline

### Task 1: Project configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "autoresearch-local"
version = "0.1.0"
description = "Local M3 Max adaptation of Karpathy's autoresearch"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.1",
    "numpy>=1.24",
    "requests>=2.28",
]

[project.optional-dependencies]
analysis = [
    "matplotlib>=3.7",
    "pandas>=2.0",
]
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
run.log
results.tsv
*.pt
*.pth
```

Note: `run.log` and `results.tsv` are untracked working files the agent creates during experiments. They should not be committed.

- [ ] **Step 3: Install dependencies and verify**

Run: `cd /Users/fourcolors/Projects/1_active/autoresearch-local && uv sync`
Expected: Dependencies install successfully, `.venv` created.

- [ ] **Step 4: Verify PyTorch MPS is available**

Run: `cd /Users/fourcolors/Projects/1_active/autoresearch-local && uv run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'PyTorch version: {torch.__version__}')"`
Expected: `MPS available: True` and PyTorch version >= 2.1

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore
git commit -m "feat: add project configuration and dependencies"
```

---

### Task 2: Data pipeline (`prepare.py`)

**Files:**
- Create: `prepare.py`

This file is FROZEN after creation — the agent must never modify it. It contains:
1. Constants shared between prepare and train
2. TinyShakespeare download + train/val split
3. Character-level tokenizer
4. Dataloader that produces batches of token sequences
5. `evaluate_bpb()` — the fixed evaluation metric

- [ ] **Step 1: Create `prepare.py` with constants, data download, tokenizer, dataloader, and evaluation**

Key components:
- `MAX_SEQ_LEN = 512`, `TIME_BUDGET = 120` (2 min), `EVAL_TOKENS = 512 * 256`
- `download_data()`: Downloads TinyShakespeare to `~/.cache/autoresearch-local/`
- `Tokenizer`: Character-level encoder/decoder with `vocab_size`, `encode()`, `decode()`
- `make_dataloader()`: Yields random `(x, y)` batches of token sequences on the target device
- `evaluate_bpb()`: Walks validation data in non-overlapping chunks, computes cross-entropy loss, converts nats-per-token to bits-per-byte via `nats / log(2)`. For char-level tokenizer, 1 token = 1 byte.
- `prepare()`: Downloads data, builds tokenizer, saves 90/10 train/val splits as `.npy` files and vocab as `.json`

See spec for full implementation details.

- [ ] **Step 2: Run `prepare.py` to download data and verify**

Run: `cd /Users/fourcolors/Projects/1_active/autoresearch-local && uv run python prepare.py`

Expected output (approximately):
```
Downloading TinyShakespeare to ~/.cache/autoresearch-local/input.txt...
Downloaded 1115394 characters
Vocab size: 65 characters
Dataset size: 1115394 characters
Train tokens: 1003854
Val tokens: 111540
Saved to ~/.cache/autoresearch-local
```

- [ ] **Step 3: Verify cached files exist**

Run: `ls -la ~/.cache/autoresearch-local/`

Expected: `input.txt`, `train.npy`, `val.npy`, `vocab.json` all present.

- [ ] **Step 4: Commit**

```bash
git add prepare.py
git commit -m "feat: add frozen data pipeline with TinyShakespeare and char-level tokenizer"
```

---

## Chunk 2: Model and Training Loop

### Task 3: GPT model and training loop (`train.py`)

**Files:**
- Create: `train.py`

This is the file the agent will mutate during experiments. The baseline implementation provides a working starting point.

- [ ] **Step 1: Create `train.py` with the GPT model**

Model components:
- `RMSNorm`: Uses `F.rms_norm` for root mean square normalization
- `RotaryEmbedding`: Precomputes RoPE cos/sin frequencies for `MAX_SEQ_LEN`
- `apply_rotary_emb()`: Applies rotary embeddings by splitting tensor into halves, rotating, and combining
- `Attention`: Multi-head attention with RoPE applied to Q/K, then `F.scaled_dot_product_attention` with `is_causal=True`
- `MLP`: SwiGLU feed-forward — `w2(silu(w1(x)) * w3(x))` with dropout
- `Block`: Pre-norm attention + MLP with residual connections
- `GPT`: Token embedding + blocks + RMSNorm + LM head with weight tying

Hyperparameters:
- `DEPTH=6`, `MODEL_DIM=384`, `N_HEADS=6`, `HEAD_DIM=64`, `MLP_RATIO=4`
- `BATCH_SIZE=64`, `LEARNING_RATE=3e-4`, `WEIGHT_DECAY=0.1`, `DROPOUT=0.1`
- `DEVICE="mps"`, `WARMUP_STEPS=100`, `GRAD_CLIP=1.0`

Training loop:
- Loads cached tokenized data from `~/.cache/autoresearch-local/`
- Reconstructs tokenizer from saved vocab
- Trains with AdamW (betas=0.9/0.95) using warmup LR schedule
- Runs until `TIME_BUDGET` (120s) is exhausted
- Exits with code 1 if loss exceeds 100 (divergence detection)
- Logs every 50 steps: step, loss, lr, elapsed, remaining
- At end: runs `evaluate_bpb()`, prints results in grep-friendly `key: value` format
- Reports MPS memory via `torch.mps.driver_allocated_size()`

- [ ] **Step 2: Verify model parameter count is ~20M**

Run: `cd /Users/fourcolors/Projects/1_active/autoresearch-local && uv run python -c "
from train import GPT
import torch
model = GPT(vocab_size=65, dim=384, depth=6, n_heads=6, head_dim=64, mlp_ratio=4)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parameters: {n:,} ({n/1e6:.1f}M)')
"`

Expected: Approximately 15-25M parameters.

- [ ] **Step 3: Run a full training + evaluation cycle**

Run: `cd /Users/fourcolors/Projects/1_active/autoresearch-local && uv run python train.py`

Expected: Training runs for ~120 seconds, prints step logs, then outputs:
```
val_bpb: <some number, likely 1.5-2.5>
peak_vram_mb: <some number>
total_steps: <some number>
model_params: <some number>
```

This is the baseline result the agent will try to improve.

- [ ] **Step 4: Verify results are grep-extractable**

Run: `cd /Users/fourcolors/Projects/1_active/autoresearch-local && uv run python train.py 2>&1 | grep "^val_bpb:"`

Expected: A single line like `val_bpb: 1.832451`

- [ ] **Step 5: Commit**

```bash
git add train.py
git commit -m "feat: add baseline GPT model and training loop for MPS"
```

---

## Chunk 3: Agent Instructions and Final Verification

### Task 4: Agent program (`program.md`)

**Files:**
- Create: `program.md`

This is the natural-language program that tells Claude Code how to run the experiment loop.

- [ ] **Step 1: Create `program.md`**

The program defines:

**Setup phase (once):**
1. Read README.md, prepare.py, train.py for context
2. Propose a run tag (e.g., `mar10`), create branch `autoresearch/<tag>`
3. Create `results.tsv` with header: `commit`, `val_bpb`, `vram_mb`, `status`, `description`
4. Run baseline experiment and log result

**Experiment loop (repeat forever):**
1. Propose ONE change to train.py (architecture, optimizer, hyperparams, etc.)
2. `git add train.py && git commit -m "experiment: <description>"`
3. `python train.py > run.log 2>&1`
4. `grep "^val_bpb:\|^peak_vram_mb:\|^total_steps:" run.log`
5. If crashed: read `tail -30 run.log`, log as `crash`, revert with `git reset --hard HEAD~1`
6. Append to `results.tsv`: `<hash>\t<bpb>\t<vram>\t<status>\t<desc>`
7. If improved (lower bpb): keep commit, print "IMPROVED"
8. If equal or worse: `git reset --hard HEAD~1`, print "DISCARDED"
9. Go to step 1. Never stop to ask the human.

**Scope rules:**
- May ONLY edit `train.py`
- Must NEVER edit `prepare.py` or `program.md`
- Make ONE change per experiment for clear attribution
- If 3 crashes in a row, try more conservative changes
- Don't game the metric — improve actual model quality

- [ ] **Step 2: Commit**

```bash
git add program.md
git commit -m "feat: add agent program with experiment loop instructions"
```

---

### Task 5: Project README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

Contents:
- Project description (local M3 Max adaptation of autoresearch)
- Setup instructions: install uv, `uv sync`, `uv run python prepare.py`, `uv run python train.py`
- How to launch: open Claude Code, prompt with "Have a look at program.md and let's kick off a new experiment!"
- Architecture table: which files do what, who edits them
- Experiment loop summary
- How to review results (`results.tsv`, `git log --oneline`)

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "feat: add project README with setup and usage instructions"
```

---

### Task 6: End-to-end verification

- [ ] **Step 1: Verify full pipeline**

Run the complete pipeline from scratch to confirm everything works:

```bash
cd /Users/fourcolors/Projects/1_active/autoresearch-local
uv run python prepare.py
uv run python train.py 2>&1 | tee run.log
grep "^val_bpb:" run.log
```

Expected: Data downloads (or uses cache), training runs for ~2 minutes on MPS, `val_bpb` is printed and extractable via grep.

- [ ] **Step 2: Verify model is editable by simulating an agent change**

Temporarily change a hyperparameter in `train.py` (e.g., `LEARNING_RATE = 1e-3`), run training, verify a different `val_bpb` is produced, then revert:

```bash
cd /Users/fourcolors/Projects/1_active/autoresearch-local
# Change LR
sed -i '' 's/LEARNING_RATE = 3e-4/LEARNING_RATE = 1e-3/' train.py
# Run
uv run python train.py 2>&1 | grep "^val_bpb:"
# Revert
git checkout train.py
```

Expected: A different `val_bpb` value than the baseline, confirming the agent's edits have measurable impact.

- [ ] **Step 3: Final commit (if any cleanup needed)**

```bash
git status
# Should be clean — nothing to commit
```
