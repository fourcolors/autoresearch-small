# autoresearch-local

Local M3 Max adaptation of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

An autonomous ML experiment loop where Claude Code (Opus) acts as the
researcher — proposing changes to a small GPT model, training it on
TinyShakespeare, and keeping only improvements via a git ratchet.

## Setup

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Download TinyShakespeare and prepare data
uv run python prepare.py

# Verify training works (2 minute run)
uv run python train.py
```

## Run the Experiment Loop

Open Claude Code in this directory and prompt:

> Have a look at program.md and let's kick off a new experiment!
> Let's do the setup first.

Claude Code will:
1. Read the codebase to understand the setup
2. Create an experiment branch
3. Run a baseline experiment
4. Start proposing and testing changes autonomously

## How It Works

| File | Purpose | Who edits |
|---|---|---|
| `prepare.py` | Data pipeline + evaluation metric | Frozen (human only) |
| `train.py` | Model + optimizer + training loop | Claude Code (agent) |
| `program.md` | Agent's standing orders | Human (between runs) |

Each experiment:
1. Agent proposes a change to `train.py`
2. Commits the change
3. Trains for 2 minutes on MPS
4. Evaluates val_bpb (bits per byte)
5. Keeps the commit if improved, reverts if not
6. Logs result to `results.tsv`

## Review Results

After a run, check `results.tsv` for all experiments. The branch HEAD
is always the best configuration found.

The git log shows only successful experiments:
```bash
git log --oneline
```
