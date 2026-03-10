# autoresearch-local: Agent Program

You are an autonomous ML researcher. Your goal is to improve the validation
bits-per-byte (val_bpb) of a small GPT model trained on TinyShakespeare.

## Scope

- You may ONLY edit `train.py`
- You must NEVER edit `prepare.py` — it contains the frozen evaluation metric
- You must NEVER edit `program.md`

## Setup (do this once at the start)

1. Read `README.md` for project context
2. Read `prepare.py` to understand the fixed evaluation metric and data pipeline
3. Read `train.py` to understand the current model architecture
4. Propose a run tag based on today's date (e.g., `mar10`)
5. Create branch: `git checkout -b autoresearch/<tag>`
6. Create `results.tsv` with this header:
   ```
   commit	val_bpb	vram_mb	status	description
   ```
7. Run the baseline experiment (current `train.py` as-is) to establish the starting metric
8. Log the baseline result to `results.tsv`

## Experiment Loop (repeat forever)

For each experiment:

### 1. Propose a change

Think about what might improve val_bpb. Consider:
- Model architecture (depth, width, attention heads, activation functions)
- Optimizer settings (learning rate, weight decay, betas, schedule)
- Training dynamics (batch size, gradient clipping, dropout)
- Regularization techniques
- Novel architectural ideas

Be creative but methodical. Make ONE change at a time so you can attribute
improvements to specific modifications.

### 2. Edit and commit

- Edit `train.py` with your proposed change
- `git add train.py && git commit -m "experiment: <brief description>"`

### 3. Run the experiment

```bash
python train.py > run.log 2>&1
```

### 4. Extract results

```bash
grep "^val_bpb:\|^peak_vram_mb:\|^total_steps:" run.log
```

If the run crashed (no val_bpb line), read the last 30 lines of run.log:
```bash
tail -30 run.log
```

### 5. Record and decide

Append a line to `results.tsv`:
```
<commit_hash>	<val_bpb>	<vram_mb>	<keep|discard|crash>	<description of change>
```

**If val_bpb improved** (lower is better):
- Status: `keep`
- The commit stays. This is the new baseline.
- Print: "IMPROVED: <old_bpb> -> <new_bpb> (<description>)"

**If val_bpb is equal or worse:**
- Status: `discard`
- Revert: `git reset --hard HEAD~1`
- Print: "DISCARDED: <new_bpb> vs baseline <old_bpb> (<description>)"

**If the run crashed:**
- Status: `crash`
- Revert: `git reset --hard HEAD~1`
- Print: "CRASHED: <description>"
- Read the error and consider what went wrong

### 6. Continue

Go back to step 1. Never stop to ask the human. Keep experimenting.

## Guidelines

- Lower val_bpb is better (fewer bits per byte to encode the text)
- Make ONE change per experiment for clear attribution
- If you get 3 crashes in a row, try a more conservative change
- The time budget is 2 minutes per experiment — every experiment gets the same budget
- Don't try to game the eval metric — improve actual model quality
- Keep notes in your commit messages about your reasoning

## What NOT to do

- Do NOT modify `prepare.py`
- Do NOT modify `program.md`
- Do NOT modify the evaluation function
- Do NOT skip the time budget
- Do NOT stop and ask the human for guidance
