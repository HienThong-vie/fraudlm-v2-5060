"""
Tests for training_model/sft_data.py: masking, truncation, padding and length-bucketed batches.

The key check: decoding the learned label tokens of an example gives back exactly the response
text (+ EOS) from utils.prompt.format_example, so the model learns the answer and never the prompt.

Author: FraudLM v2 team (COS30049.2-1).

"""

from pathlib import Path

import pytest
import tiktoken

from training_model.sft_data import (
    EOS_ID, IGNORE_INDEX, PAD_ID, SFTDataset, collate, decode_labels, encode_example,
    length_bucketed_batches, load_split,
)
from utils.prompt import format_example

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dataset" / "fraudlm_sft_train.json"
ENC = tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def rows():
    if not TRAIN.exists():
        pytest.skip("dataset not built")
    data = load_split(TRAIN)
    # A few rows of every task, so templates with and without an Input block are both covered.
    picked = {}
    for r in data:
        picked.setdefault(r["task"], [])
        if len(picked[r["task"]]) < 10:
            picked[r["task"]].append(r)
    return [r for task_rows in picked.values() for r in task_rows]


def test_labels_decode_to_exact_response(rows):
    for ex in rows:
        prompt, response = format_example(ex)
        it = encode_example(ex, max_len=1024)
        assert not it["truncated"]
        # Learned tokens == response text + EOS, nothing more, nothing less.
        assert decode_labels(it["labels"]) == response, ex["id"]
        assert it["labels"][-1] == EOS_ID
        # Every prompt target is masked; the first learned target is the first response token.
        n_masked = sum(1 for t in it["labels"] if t == IGNORE_INDEX)
        assert n_masked == it["n_prompt"] - 1
        # Inputs are prompt + response shifted by one; the prompt part equals the prompt tokens.
        assert it["input_ids"][: it["n_prompt"]] == ENC.encode(prompt)
        assert len(it["input_ids"]) == len(it["labels"])


def test_separate_tokenising_equals_joined(rows):
    """Prompt and response are tokenised apart; that must give the same ids as the joined text."""
    for ex in rows:
        prompt, response = format_example(ex)
        it = encode_example(ex, max_len=1024)
        joined = ENC.encode(prompt + response, allowed_special={"<|endoftext|>"})
        assert it["input_ids"] + [it["labels"][-1]] == joined, ex["id"]


def test_truncation_is_flagged_and_counted(rows):
    long_ex = max(rows, key=lambda r: len(r["output"]))
    it = encode_example(long_ex, max_len=it_len(long_ex) - 10)
    assert it["truncated"] and len(it["input_ids"]) == it_len(long_ex) - 10
    assert it["labels"][-1] != EOS_ID  # the cut response lost its EOS: that is why we warn
    ds = SFTDataset([long_ex, rows[0]], max_len=it_len(long_ex) - 10, name="t")
    assert ds.n_truncated >= 1
    # A prompt longer than the cap cannot teach anything: dropped, not crashed.
    assert encode_example(long_ex, max_len=5) is None


def it_len(ex):
    return len(encode_example(ex, max_len=1024)["input_ids"])


def test_collate_pads_inputs_and_masks_labels(rows):
    items = [encode_example(r, 1024) for r in rows[:3]]
    x, y = collate(items)
    t = max(len(i["input_ids"]) for i in items)
    assert x.shape == y.shape == (3, t)
    for b, it in enumerate(items):
        n = len(it["input_ids"])
        assert x[b, :n].tolist() == it["input_ids"]
        assert (x[b, n:] == PAD_ID).all() and (y[b, n:] == IGNORE_INDEX).all()


def test_length_bucketed_batches_cover_all_once_and_are_deterministic():
    lengths = [((i * 37) % 300) + 20 for i in range(1001)]
    a = length_bucketed_batches(lengths, 4, seed=42)
    b = length_bucketed_batches(lengths, 4, seed=42)
    c = length_bucketed_batches(lengths, 4, seed=43)
    assert a == b and a != c
    flat = sorted(i for batch in a for i in batch)
    assert flat == list(range(1001))
    # Bucketing works: the mean within-batch length spread is far below the global spread.
    spread = sum(max(lengths[i] for i in bt) - min(lengths[i] for i in bt) for bt in a) / len(a)
    assert spread < 30
