"""
autoresearch-local: Data preparation, tokenizer, and evaluation.

THIS FILE IS FROZEN. The agent must not modify it.
All experiments are evaluated by the same evaluate_bpb() function
to ensure fair comparison across architectural changes.
"""

import os
import json
import requests
import numpy as np
import torch

# ---- Constants (shared with train.py) ----
MAX_SEQ_LEN = 512
TIME_BUDGET = 120  # seconds (2 minutes)
EVAL_TOKENS = 512 * 256  # ~131K tokens for evaluation

CACHE_DIR = os.path.expanduser("~/.cache/autoresearch-local")
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_data():
    """Download TinyShakespeare if not already cached."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    data_path = os.path.join(CACHE_DIR, "input.txt")

    if os.path.exists(data_path):
        print(f"Data already cached at {data_path}")
        with open(data_path, "r") as f:
            return f.read()

    print(f"Downloading TinyShakespeare to {data_path}...")
    response = requests.get(DATA_URL)
    response.raise_for_status()
    text = response.text

    with open(data_path, "w") as f:
        f.write(text)

    print(f"Downloaded {len(text)} characters")
    return text


# ---- Character-level Tokenizer ----


class Tokenizer:
    """Simple character-level tokenizer for TinyShakespeare."""

    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return "".join(self.idx_to_char[i] for i in indices)


# ---- Dataloader ----


def make_dataloader(data_tokens, batch_size, seq_len, device):
    """
    Yields batches of (x, y) token sequences from the data.

    Each call yields a new random batch. x is the input sequence,
    y is the target (shifted by 1).
    """
    data = torch.tensor(data_tokens, dtype=torch.long)
    n = len(data)

    while True:
        starts = torch.randint(0, n - seq_len - 1, (batch_size,))
        x = torch.stack([data[s : s + seq_len] for s in starts]).to(device)
        y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts]).to(device)
        yield x, y


# ---- Evaluation ----


@torch.no_grad()
def evaluate_bpb(model, val_tokens, tokenizer, device):
    """
    Evaluate model on validation data. Returns bits-per-byte (BPB).

    BPB = total_nats / (ln(2) * total_bytes)

    This metric is vocab-size-independent, so architectural changes
    (different vocab sizes, tokenizer changes) compare fairly.
    """
    model.eval()

    seq_len = MAX_SEQ_LEN
    total_loss = 0.0
    total_tokens = 0

    data = torch.tensor(val_tokens, dtype=torch.long)
    n = len(data)

    # Walk through validation data in non-overlapping chunks
    for start in range(0, min(n - seq_len - 1, EVAL_TOKENS), seq_len):
        end = start + seq_len
        if end + 1 > n:
            break

        x = data[start:end].unsqueeze(0).to(device)
        y = data[start + 1 : end + 1].unsqueeze(0).to(device)

        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum"
        )
        total_loss += loss.item()
        total_tokens += seq_len

    # Convert from nats-per-token to bits-per-byte
    # For char-level tokenizer: 1 token = 1 byte (ASCII text)
    nats_per_token = total_loss / total_tokens
    bpb = nats_per_token / np.log(2)

    model.train()
    return bpb


# ---- Main: prepare data ----


def prepare():
    """Download data, build tokenizer, create train/val splits."""
    text = download_data()
    tokenizer = Tokenizer(text)

    print(f"Vocab size: {tokenizer.vocab_size} characters")
    print(f"Dataset size: {len(text)} characters")

    # 90/10 train/val split
    tokens = tokenizer.encode(text)
    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    # Save tokenized splits
    np.save(
        os.path.join(CACHE_DIR, "train.npy"), np.array(train_tokens, dtype=np.int16)
    )
    np.save(os.path.join(CACHE_DIR, "val.npy"), np.array(val_tokens, dtype=np.int16))

    # Save tokenizer vocab
    with open(os.path.join(CACHE_DIR, "vocab.json"), "w") as f:
        json.dump(tokenizer.char_to_idx, f)

    print(f"Train tokens: {len(train_tokens)}")
    print(f"Val tokens: {len(val_tokens)}")
    print(f"Saved to {CACHE_DIR}")

    return tokenizer, train_tokens, val_tokens


if __name__ == "__main__":
    prepare()
