# autoresearch-small

An autonomous ML experiment loop where **Claude Code** acts as the researcher —
proposing changes to a small GPT model, training it on TinyShakespeare, and keeping
only improvements via a git ratchet.

Adapted from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
to run locally on Apple Silicon Macs (M1/M2/M3/M4) using MPS.

## What it does

You give Claude Code a starting GPT model and a set of standing orders. It then
runs autonomously: propose a change → commit → train for 2 minutes → evaluate →
keep or revert. Over dozens of experiments it discovers an optimized architecture
and training recipe, all tracked in git.

The baseline model starts at **~2.71 val_bpb**. See how low you can go!

## Prerequisites

- **Mac with Apple Silicon** (M1, M2, M3, or M4)
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **[Claude Code](https://docs.anthropic.com/en/docs/claude-code)** — Anthropic's CLI for Claude

If you don't have uv installed:

```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```zsh
# 1. Clone the repo
git clone https://github.com/fourcolors/autoresearch-small.git
cd autoresearch-small

# 2. Install Python dependencies (torch, numpy, requests)
uv sync

# 3. Download TinyShakespeare and prepare tokenized train/val splits
uv run python prepare.py

# 4. Verify training works — this runs for 2 minutes and prints val_bpb at the end
uv run python train.py
```

After step 4 you should see output like:

```
model_params: 10039296 (10.0M)
Training for 120s on mps...
step 50 | loss 2.8432 | lr 1.50e-04 | elapsed 18s | remaining 102s
...
val_bpb: 2.712210
```

If you see a `val_bpb` score, you're ready to go.

## Running the experiment loop

Open Claude Code in the project directory:

```zsh
claude
```

Then paste this prompt:

> Have a look at program.md and let's kick off a new experiment!
> Let's do the setup first.

Claude Code will:

1. **Read the codebase** — `prepare.py` (frozen eval), `train.py` (mutation target), `program.md` (its instructions)
2. **Create an experiment branch** — e.g. `autoresearch/mar11`
3. **Run a baseline** — train the current model and record the starting val_bpb
4. **Enter the experiment loop** — propose a change to `train.py`, commit it, train for 2 minutes, evaluate, keep if improved or revert if not

You can sit back and watch, or interact along the way. Each experiment takes
about 2 minutes. Claude will keep running experiments until you stop it.

## What you'll see

Each experiment produces output like:

```
step 50 | loss 2.6957 | lr 2.00e-03 | elapsed 9s | remaining 111s
step 100 | loss 2.0315 | lr 2.00e-03 | elapsed 19s | remaining 101s
...
val_bpb: 2.152516
DISCARDED: 2.152516 vs baseline 2.125836 (cosine warmdown)
```

Results are logged to `results.tsv`:

```
commit    val_bpb    vram_mb    status    description
ff94eef   2.712210   16899.2    keep      baseline: 6-layer GPT
9d72d65   2.653655   16899.2    keep      increase LR from 3e-4 to 1e-3
```

The git log shows only successful experiments (failures are reverted):

```zsh
git log --oneline
```

## How it works

| File | Purpose | Who edits |
|---|---|---|
| `prepare.py` | Data pipeline + evaluation metric | Frozen — don't touch |
| `train.py` | Model architecture + training loop | Claude Code |
| `program.md` | Agent's standing orders | You (between runs) |

### The git ratchet

Every experiment is a git commit. If val_bpb improves (lower = better), the
commit stays. If not, `git reset --hard HEAD~1` reverts it. The branch HEAD
is always the best configuration found so far.

### The metric: bits per byte (BPB)

BPB measures how efficiently the model predicts the next character. Lower is
better. A score of 2.0 means the model needs 2 bits to encode each byte of
Shakespeare. It's independent of vocabulary size, so architectural changes
compare fairly.

## The baseline model

The starting `train.py` is a 6-layer GPT with:

- 384-dim embeddings, 6 attention heads
- SwiGLU feed-forward network
- Rotary position embeddings (RoPE)
- AdamW optimizer (LR 3e-4)
- ~10M parameters

This is intentionally oversized for the 2-minute training budget — discovering
the right model size for the compute budget is part of the experiment.

## Customization

You can tweak the experiment setup:

- **Training time per experiment:** Change `TIME_BUDGET` in `prepare.py` (default: 120s)
- **Context window:** Change `MAX_SEQ_LEN` in `prepare.py` (default: 512)
- **Agent strategy:** Edit `program.md` to guide what Claude tries

## Project structure

```
autoresearch-small/
├── prepare.py       # Data pipeline and eval metric (frozen)
├── train.py         # Model and training loop (Claude edits this)
├── program.md       # Agent instructions
├── pyproject.toml   # Python dependencies
└── uv.lock          # Locked dependency versions
```

Files created during experiments (gitignored):

```
├── run.log          # Output from the latest training run
├── results.tsv      # Full experiment log with all results
└── model.pt         # Best model checkpoint
```

## License

MIT
