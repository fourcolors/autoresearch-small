"""
autoresearch-local: Model definition and training loop.

THIS FILE IS THE MUTATION TARGET. The agent freely edits this
between experiments to explore architectural and hyperparameter changes.
"""

import os
import sys
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import frozen constants and utilities from prepare.py
from prepare import (
    MAX_SEQ_LEN,
    TIME_BUDGET,
    CACHE_DIR,
    Tokenizer,
    make_dataloader,
    evaluate_bpb,
)

# ---- Hyperparameters ----
DEPTH = 2
MODEL_DIM = 256
N_HEADS = 4
HEAD_DIM = MODEL_DIM // N_HEADS  # 64
MLP_RATIO = 6
DROPOUT = 0.1

BATCH_SIZE = 64
LEARNING_RATE = 1.5e-3
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 20
GRAD_CLIP = 1.0

DEVICE = "mps"


# ---- Model Components ----


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)
        # Store full-dim cos/sin by repeating each frequency pair
        cos = freqs.cos().repeat(1, 2)  # (seq_len, dim)
        sin = freqs.sin().repeat(1, 2)  # (seq_len, dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        seq_len = x.shape[-2]
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to input tensor."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, dim, n_heads, head_dim, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * n_heads * head_dim, bias=False)
        self.out = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x, rope_cos, rope_sin):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # (B, n_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q = apply_rotary_emb(q, rope_cos, rope_sin)
        k = apply_rotary_emb(k, rope_cos, rope_sin)

        # Scaled dot-product attention (causal)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.out(out)


class MLP(nn.Module):
    """ReLU² feed-forward network (from Karpathy's autoresearch)."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.relu(self.fc1(x)).square()))


class Block(nn.Module):
    def __init__(self, dim, n_heads, head_dim, mlp_ratio, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads, head_dim, dropout)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dropout)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, dim, depth, n_heads, head_dim, mlp_ratio, dropout=0.0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.rope = RotaryEmbedding(head_dim)
        self.blocks = nn.ModuleList(
            [Block(dim, n_heads, head_dim, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x)
        rope_cos, rope_sin = self.rope(h)

        for block in self.blocks:
            h = block(h, rope_cos, rope_sin)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits


# ---- Training Loop ----


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Load data
    train_tokens = np.load(os.path.join(CACHE_DIR, "train.npy")).tolist()
    val_tokens = np.load(os.path.join(CACHE_DIR, "val.npy")).tolist()

    with open(os.path.join(CACHE_DIR, "vocab.json"), "r") as f:
        char_to_idx = json.load(f)
    tokenizer = Tokenizer.__new__(Tokenizer)
    tokenizer.vocab_size = len(char_to_idx)
    tokenizer.char_to_idx = char_to_idx
    tokenizer.idx_to_char = {int(v): k for k, v in char_to_idx.items()}

    # Build model
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        dim=MODEL_DIM,
        depth=DEPTH,
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = count_parameters(model)
    print(f"model_params: {n_params} ({n_params / 1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # Dataloader
    loader = make_dataloader(train_tokens, BATCH_SIZE, MAX_SEQ_LEN, DEVICE)

    # Training loop with time budget
    model.train()
    step = 0
    total_loss = 0.0
    start_time = time.time()

    # Warmup + warmdown schedule (cosine decay in last 50% of time)
    WARMDOWN_RATIO = 0.5
    def get_lr(step, elapsed):
        if step < WARMUP_STEPS:
            return LEARNING_RATE * (step + 1) / WARMUP_STEPS
        progress = elapsed / TIME_BUDGET
        if progress > (1.0 - WARMDOWN_RATIO):
            cooldown = (1.0 - progress) / WARMDOWN_RATIO
            return LEARNING_RATE * cooldown
        return LEARNING_RATE

    print(f"Training for {TIME_BUDGET}s on {DEVICE}...")
    print(f"batch_size: {BATCH_SIZE}")
    print(f"seq_len: {MAX_SEQ_LEN}")
    print(f"lr: {LEARNING_RATE}")

    while True:
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        # LR schedule
        lr = get_lr(step, elapsed)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        x, y = next(loader)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Check for divergence
        if loss.item() > 100:
            print("FAIL: loss diverged")
            sys.exit(1)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        step += 1

        # Log every 50 steps
        if step % 50 == 0:
            avg_loss = total_loss / 50
            elapsed = time.time() - start_time
            remaining = TIME_BUDGET - elapsed
            print(
                f"step {step} | loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | elapsed {elapsed:.0f}s | remaining {remaining:.0f}s"
            )
            total_loss = 0.0

    # Final evaluation
    elapsed = time.time() - start_time
    print(f"\nTraining complete: {step} steps in {elapsed:.1f}s")

    val_bpb = evaluate_bpb(model, val_tokens, tokenizer, DEVICE)
    peak_vram_mb = torch.mps.driver_allocated_memory() / 1024 / 1024

    # Print results in grep-friendly format
    print(f"\nval_bpb: {val_bpb:.6f}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")
    print(f"total_steps: {step}")
    print(f"model_params: {n_params}")

    # Save checkpoint only if this is the best model so far
    ckpt_path = os.path.join(os.path.dirname(__file__), "model.pt")
    best_bpb = float("inf")
    if os.path.exists(ckpt_path):
        prev_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        best_bpb = prev_ckpt.get("val_bpb", float("inf"))
        print(f"previous_best_bpb: {best_bpb:.6f}")

    if val_bpb < best_bpb:
        torch.save({
            "model_state_dict": model.state_dict(),
            "val_bpb": val_bpb,
            "step": step,
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "dim": MODEL_DIM,
                "depth": DEPTH,
                "n_heads": N_HEADS,
                "head_dim": HEAD_DIM,
                "mlp_ratio": MLP_RATIO,
                "dropout": DROPOUT,
            },
        }, ckpt_path)
        print(f"checkpoint_saved: {ckpt_path} (NEW BEST: {val_bpb:.6f})")
    else:
        print(f"checkpoint_kept: previous model is better ({best_bpb:.6f} vs {val_bpb:.6f})")

    # Generate a sample from the current model to see what it learned
    generate_sample(model, tokenizer, DEVICE)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="ROMEO:", length=500, temperature=0.8):
    """Generate text from the trained model by sampling one character at a time."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(length):
        # Crop to max sequence length if needed
        x_cond = x[:, -MAX_SEQ_LEN:]
        logits = model(x_cond)
        # Take logits at the last position and apply temperature
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

    generated = tokenizer.decode(x[0].tolist())
    print(f"\n{'='*60}")
    print("GENERATED SAMPLE:")
    print(f"{'='*60}")
    print(generated)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
