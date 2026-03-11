# autoresearch-small

A tiny autonomous ML experiment loop where **Claude Code** acts as the researcher —
proposing changes to a small GPT model, training it on TinyShakespeare, and keeping
only improvements via a git ratchet.

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
to run locally on Apple Silicon Macs (M1/M2/M3/M4) using MPS.

## What it does

Claude Code reads a set of standing orders (`program.md`), then enters an autonomous
loop: propose a change → train for 2 minutes → evaluate → keep or revert. Over dozens
of experiments it discovers an optimized architecture and training recipe, all tracked
in git history.

**Starting point:** 6-layer, 384-dim GPT → val_bpb **2.71**
**Best result after 72 experiments:** 2-layer, 256-dim GPT → val_bpb **2.13**

## Requirements

- macOS with Apple Silicon (M1, M2, M3, or M4)
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI

## Quick start

```zsh
# Clone the repo
git clone https://github.com/fourcolors/autoresearch-small.git
cd autoresearch-small

# Install dependencies
uv sync

# Download TinyShakespeare and prepare tokenized data
uv run python prepare.py

# Verify training works (runs for 2 minutes)
uv run python train.py
```

## Run the experiment loop

Open Claude Code in this directory and prompt:

> Have a look at program.md and let's kick off a new experiment!
> Let's do the setup first.

Claude Code will:
1. Read the codebase to understand the current model
2. Create an experiment branch
3. Run a baseline experiment
4. Start proposing and testing changes autonomously

Each experiment trains for exactly 2 minutes on MPS, evaluates validation
bits-per-byte (val_bpb), and either keeps the improvement or reverts.

## How it works

| File | Purpose | Who edits |
|---|---|---|
| `prepare.py` | Data pipeline + evaluation metric | Frozen (human only) |
| `train.py` | Model + optimizer + training loop | Claude Code (agent) |
| `program.md` | Agent's standing orders | Human (between runs) |

### The git ratchet

Every experiment is a commit. If val_bpb improves, the commit stays. If not,
`git reset --hard HEAD~1` reverts it. The branch HEAD is always the best
configuration found so far.

### The evaluation metric

**Bits per byte (BPB)** measures how well the model predicts the next character.
Lower is better. It's independent of vocabulary size, so architectural changes
that alter the tokenizer still compare fairly.

## What Claude discovered (72 experiments)

The biggest wins came from **right-sizing the model** for the 2-minute training
budget and **learning rate scheduling**:

| Change | val_bpb | Improvement |
|---|---|---|
| Baseline (6-layer, 384-dim) | 2.712 | — |
| LR 3e-4 → 1e-3 | 2.654 | -2.1% |
| Warmup 100 → 20 steps | 2.582 | -2.7% |
| Depth 6 → 4 layers | 2.309 | -10.6% |
| Depth 4 → 3 layers | 2.227 | -3.5% |
| Depth 3 → 2 layers | 2.205 | -1.0% |
| Dim 384 → 256 | 2.172 | -1.5% |
| ReLU² MLP (replace SwiGLU) | 2.164 | -0.4% |
| MLP ratio 4 → 6 | 2.157 | -0.3% |
| Warmdown (linear decay last 50%) | 2.153 | -0.2% |
| LR → 1.5e-3 | 2.150 | -0.1% |
| LR → 2e-3 | 2.136 | -0.6% |
| Warmup → 10 steps | 2.131 | -0.2% |
| Grad clip 1.0 → 2.0 | **2.126** | -0.2% |

Key insight: the original model was too large to train well in 2 minutes (~140 steps).
Making it smaller allowed ~650 steps, which was the single biggest improvement.

## Project structure

```
autoresearch-small/
├── prepare.py       # Frozen data pipeline and evaluation
├── train.py         # Model and training loop (the mutation target)
├── program.md       # Agent instructions
├── pyproject.toml   # Dependencies
└── results.tsv      # Full experiment log (72 experiments)
```

## Customization

- **Training time:** Change `TIME_BUDGET` in `prepare.py` (default: 120 seconds)
- **Sequence length:** Change `MAX_SEQ_LEN` in `prepare.py` (default: 512)
- **Agent behavior:** Edit `program.md` to change the experiment strategy

## License

MIT
