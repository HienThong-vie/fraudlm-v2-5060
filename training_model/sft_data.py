"""
Instruction fine-tuning data for FraudLM v2: tokenise, mask the prompt, batch by length.

Purpose: turn the Alpaca JSON rows of `dataset/fraudlm_sft_*.json` into (input_ids, labels) pairs
for next-token training where the model learns ONLY the response. Prompt tokens and padding get
label -100, which `F.cross_entropy(..., ignore_index=-100)` skips; the response and its final EOS
keep their real ids. Batches group examples of similar length so little compute is spent on padding.

The prompt template is never re-written here: every example goes through
`utils.prompt.format_example`, the one template shared with the chat demo (PLAN Section 3).

Author: FraudLM v2 team (COS30049.2-1).
Masked instruction fine-tuning follows Raschka, S. (2024) *Build a Large Language Model (From
Scratch)*, Manning, chapter 7 (custom collate with -100 targets); code repo Apache-2.0.

"""

import json
import random
from pathlib import Path

import tiktoken
import torch

from utils.prompt import EOS_TEXT, format_example

EOS_ID = 50256        # "<|endoftext|>" in the GPT-2 vocabulary (PLAN D4)
IGNORE_INDEX = -100   # label value that cross_entropy ignores
PAD_ID = EOS_ID       # padded input positions are never learned from, so any id works; EOS is the usual choice

_ENC = tiktoken.get_encoding("gpt2")


def load_split(path):
    """Read one dataset split (a JSON list of {"id","instruction","input","output","task","source"})."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty JSON list of examples")
    for r in rows[:5]:
        missing = {"instruction", "output"} - set(r)
        if missing:
            raise ValueError(f"{path}: example {r.get('id')} misses fields {sorted(missing)}")
    return rows


def encode_example(ex, max_len):
    """Tokenise one row into model inputs and masked targets.

    Returns a dict with
      input_ids: tokens 0..n-2 of (prompt + response + EOS)
      labels:    tokens 1..n-1, but -100 wherever the target is still part of the prompt
      n_prompt, n_response, truncated
    or None if the prompt alone does not fit in max_len (the example cannot teach anything).

    Prompt and response are tokenised separately so we know exactly where the response starts.
    The prompt ends with "\\n" and the response is stripped, and GPT-2's tokenizer always splits a
    newline from the next word, so this gives the same tokens as tokenising the joined text.
    """
    prompt, response = format_example(ex)
    assert response.endswith(EOS_TEXT)
    body = response[: -len(EOS_TEXT)]
    # disallowed_special=() treats any "<|endoftext|>" written inside the text as plain characters;
    # the only real EOS is the one id we append ourselves.
    prompt_ids = _ENC.encode(prompt, disallowed_special=())
    response_ids = _ENC.encode(body, disallowed_special=()) + [EOS_ID]

    if len(prompt_ids) >= max_len:
        return None
    seq = prompt_ids + response_ids
    truncated = len(seq) > max_len + 1
    seq = seq[: max_len + 1]  # +1 because inputs and targets are the sequence shifted by one

    input_ids = seq[:-1]
    targets = seq[1:]
    # Target i is token i+1. It belongs to the prompt while i+1 < len(prompt_ids).
    n_masked = len(prompt_ids) - 1
    labels = [IGNORE_INDEX] * n_masked + targets[n_masked:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "n_prompt": len(prompt_ids),
        "n_response": len(seq) - len(prompt_ids),
        "truncated": truncated,
    }


class SFTDataset:
    """All examples of one split, tokenised once up front (12k rows take a few seconds)."""

    def __init__(self, rows, max_len=1024, name="train"):
        self.rows, self.items = [], []
        self.n_truncated = self.n_dropped = 0
        for ex in rows:
            item = encode_example(ex, max_len)
            if item is None:
                self.n_dropped += 1
                continue
            self.n_truncated += item["truncated"]
            self.rows.append(ex)
            self.items.append(item)
        lengths = [len(it["input_ids"]) for it in self.items]
        self.n_tokens = sum(lengths)
        msg = (f"[data] {name}: {len(self.items):,} examples, mean {self.n_tokens / max(1, len(lengths)):.1f} "
               f"tokens, max {max(lengths) if lengths else 0}, cap {max_len}")
        if self.n_truncated or self.n_dropped:
            # Loud on purpose: a truncated response loses its EOS, so the model would not learn to stop.
            msg += f" | WARNING: {self.n_truncated} truncated, {self.n_dropped} dropped (prompt longer than cap)"
        print(msg)

    def __len__(self):
        return len(self.items)


def length_bucketed_batches(lengths, batch_size, seed, chunk_batches=50, shuffle=True):
    """Lists of example indices, one list per batch, grouping similar lengths.

    Shuffle everything, cut into chunks of `chunk_batches` batches, sort each chunk by length, cut
    it into batches, then shuffle the batch order. Short and long examples rarely share a batch
    (little padding), but each epoch still sees a different random order. The result depends only on
    (lengths, batch_size, seed), so a resumed run rebuilds exactly the same batches.
    """
    idx = list(range(len(lengths)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idx)
    chunk = batch_size * chunk_batches
    batches = []
    for start in range(0, len(idx), chunk):
        part = sorted(idx[start:start + chunk], key=lambda i: lengths[i])
        batches += [part[j:j + batch_size] for j in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(items, device="cpu"):
    """Right-pad a list of encoded examples into tensors [B, T].

    Right padding is safe with a causal model: real tokens never attend to the pads after them,
    and the pads' labels are -100 so they add nothing to the loss.
    """
    t = max(len(it["input_ids"]) for it in items)
    x = torch.full((len(items), t), PAD_ID, dtype=torch.long)
    y = torch.full((len(items), t), IGNORE_INDEX, dtype=torch.long)
    for b, it in enumerate(items):
        n = len(it["input_ids"])
        x[b, :n] = torch.tensor(it["input_ids"], dtype=torch.long)
        y[b, :n] = torch.tensor(it["labels"], dtype=torch.long)
    return x.to(device), y.to(device)


def decode_labels(labels):
    """Text of the learned (non -100) label tokens of one example: used to verify the masking."""
    ids = [int(t) for t in labels if int(t) != IGNORE_INDEX]
    return _ENC.decode(ids)
