# Design: autoresearch-local

An M3 Max Mac adaptation of Karpathy's autoresearch for local autonomous ML experimentation.

## Context

Karpathy's autoresearch is an autonomous ML experiment loop where a coding agent continuously runs experiments, keeping improvements via a git ratchet. The original requires an NVIDIA H100 with CUDA + Flash Attention 3. This project adapts it for Apple Silicon (M3 Max, 128GB) using MPS.

## Goals

- Learn the autoresearch pattern hands-on
- Get real experimental results on a small model
- Run entirely locally on M3 Max — no cloud GPU needed
- Potentially scale findings up to larger models later

## Architecture

Three-file architecture (same as original):

```
autoresearch-local/
├── prepare.py      # FROZEN — data prep, tokenizer, eval metric
├── train.py        # MUTABLE — model + optimizer + training loop (agent edits this)
├── program.md      # Agent instructions for Claude Code
├── pyproject.toml  # Dependencies (no CUDA)
```

### Separation of Concerns

| File | Who edits | Purpose |
|---|---|---|
| `prepare.py` | Human only (frozen) | TinyShakespeare download, char-level tokenizer, `evaluate_bpb()` |
| `train.py` | Claude Code (agent) | ~20M param GPT, AdamW optimizer, MPS training loop |
| `program.md` | Human (between runs) | The agent's standing orders — the experiment loop |

## Key Adaptations from Original

| Aspect | Original (H100) | Ours (M3 Max) |
|---|---|---|
| Device | `cuda` | `mps` |
| Attention | Flash Attention 3 | `F.scaled_dot_product_attention` |
| Optimizer | Fused MuonAdamW | Standard AdamW |
| `torch.compile` | Yes, full graph | Disabled |
| Model size | ~100M+ params | ~20M params |
| Time budget | 5 min | 2 min |
| Dataset | climbmix-400b | TinyShakespeare (~1MB) |
| Tokenizer | BPE (rustbpe) | Character-level |

## Model (train.py)

Small but modern GPT (~20M parameters):

- 6 layers, 384 model dim, 6 attention heads
- RMSNorm (not LayerNorm)
- Rotary Position Embeddings (RoPE)
- SwiGLU activation in MLP
- `F.scaled_dot_product_attention` (no Flash Attention)
- Standard PyTorch AdamW optimizer
- No `torch.compile`

The agent is free to change any of these between experiments.

## Data Pipeline (prepare.py)

- Downloads TinyShakespeare (~1MB) to `~/.cache/autoresearch-local/`
- Character-level tokenizer (65 unique chars)
- `evaluate_bpb()` returns bits-per-byte on held-out validation split
- Fixed constants: `MAX_SEQ_LEN = 512`, `TIME_BUDGET = 120` (2 min)

## Agent Loop (program.md)

Greedy hill-climbing search over model configurations:

1. Create branch `autoresearch/<tag>`
2. Run baseline experiment
3. Loop forever:
   - Propose a change to `train.py`
   - `git commit` the change
   - `python train.py > run.log 2>&1`
   - `grep "^val_bpb:" run.log` to extract metric
   - If improved: keep commit
   - If worse or equal: `git reset --hard HEAD~1`
   - Log to `results.tsv`
   - Repeat

## Dependencies

```
torch >= 2.1
numpy
requests
```

No CUDA, no `kernels`, no `rustbpe`, no `tiktoken`.

## Launch

Manual: open Claude Code (Opus) in the project directory, prompt it to read `program.md` and start experimenting.
