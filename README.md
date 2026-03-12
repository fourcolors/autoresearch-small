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

The baseline model starts at **~2.71 val_bpb**. See how low you can go!

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

## The baseline model

The starting `train.py` is a 6-layer GPT with:
- 384-dim embeddings, 6 attention heads
- SwiGLU feed-forward network
- Rotary position embeddings (RoPE)
- ~10M parameters

This is intentionally oversized for the 2-minute training budget — discovering
that is part of the experiment!

## Customization

- **Training time:** Change `TIME_BUDGET` in `prepare.py` (default: 120 seconds)
- **Sequence length:** Change `MAX_SEQ_LEN` in `prepare.py` (default: 512)
- **Agent behavior:** Edit `program.md` to change the experiment strategy

## License

MIT
