"""
Tests for utils/generate.py: tokeniser round trip, greedy determinism, EOS stop, context crop, top-k.

Author: FraudLM v2 team (COS30049.2-1).

"""

import pytest
import torch
import torch.nn as nn

from training_model.gpt_model import GPTModel
from utils.generate import EOS_ID, generate, ids_to_text, ids_to_text_until_eos, text_to_ids

TINY = {"vocab_size": 50257, "context_length": 8, "emb_dim": 32, "n_heads": 2,
        "n_layers": 1, "drop_rate": 0.0, "qkv_bias": True}


class FixedNextToken(nn.Module):
    """Stub model whose logits always favour one token (to test stopping logic)."""

    def __init__(self, token_per_row):
        super().__init__()
        self.token_per_row = token_per_row  # list: one forced token per batch row

    def forward(self, idx):
        logits = torch.zeros(idx.shape[0], idx.shape[1], 50257)
        for row, tok in enumerate(self.token_per_row):
            logits[row, :, tok] = 10.0
        return logits


def test_tokenizer_round_trip_and_eos():
    text = "Fraud analyst review.<|endoftext|>"
    ids = text_to_ids(text)
    assert ids.shape[0] == 1 and ids.dtype == torch.long
    assert ids[0, -1].item() == EOS_ID == 50256
    assert ids_to_text(ids) == text
    assert ids_to_text(ids[0].tolist()) == text


def test_ids_to_text_rejects_batches():
    batch = torch.tensor([[464, 3290], [464, 3290]])
    with pytest.raises(ValueError):
        ids_to_text(batch)
    with pytest.raises(ValueError):
        ids_to_text_until_eos(batch)


def test_ids_to_text_until_eos_cuts_padding():
    row = text_to_ids("The dog")[0].tolist() + [EOS_ID, EOS_ID, 464]
    assert ids_to_text_until_eos(row) == "The dog"
    assert ids_to_text_until_eos(torch.tensor([row])) == "The dog"
    assert ids_to_text_until_eos(text_to_ids("No end token")) == "No end token"
    assert ids_to_text(torch.tensor(row)).count("<|endoftext|>") == 2  # plain decode keeps EOS


def test_greedy_is_deterministic_and_shape():
    torch.manual_seed(42)
    model = GPTModel(TINY)
    start = text_to_ids("Hello")
    a = generate(model, start, max_new_tokens=5, eos_id=None)
    b = generate(model, start, max_new_tokens=5, eos_id=None)
    assert a.shape == (1, start.shape[1] + 5)
    assert torch.equal(a, b)
    assert model.training  # generate restores the caller's train/eval mode


def test_context_is_cropped():
    # Context length 8 but 20 new tokens: would crash without cropping.
    torch.manual_seed(42)
    model = GPTModel(TINY).eval()
    out = generate(model, torch.zeros(1, 3, dtype=torch.long), max_new_tokens=20,
                   context_size=TINY["context_length"], eos_id=None)
    assert out.shape == (1, 23)


def test_stops_at_eos_without_appending_it():
    model = FixedNextToken([EOS_ID])
    start = torch.tensor([[1, 2, 3]])
    out = generate(model, start, max_new_tokens=10)
    assert torch.equal(out, start)


def test_batch_rows_finish_at_different_times():
    # Row 0 always says EOS, row 1 never does: row 0 is padded with EOS, all 4 steps run.
    model = FixedNextToken([EOS_ID, 7])
    start = torch.tensor([[1, 2], [3, 4]])
    out = generate(model, start, max_new_tokens=4)
    assert out.shape == (2, 6)
    assert (out[0, 2:] == EOS_ID).all() and (out[1, 2:] == 7).all()


def test_top_k_sampling_only_picks_top_tokens():
    torch.manual_seed(42)

    class TwoGoodTokens(nn.Module):
        def forward(self, idx):
            logits = torch.zeros(idx.shape[0], idx.shape[1], 50257)
            logits[..., 100] = 5.0
            logits[..., 200] = 4.9
            return logits

    out = generate(TwoGoodTokens(), torch.zeros(1, 1, dtype=torch.long), max_new_tokens=30,
                   temperature=1.0, top_k=2, eos_id=None)
    assert set(out[0, 1:].tolist()) <= {100, 200}


# ---------------------------------------------------------------- KV cache (G5 speed-up)

def _tiny_model(context_length=32):
    torch.manual_seed(42)
    cfg = dict(TINY, context_length=context_length, n_layers=2, emb_dim=64, n_heads=4)
    return GPTModel(cfg).eval()


def test_cached_logits_match_full_forward_tiny():
    model = _tiny_model()
    idx = torch.randint(0, 50257, (2, 12), generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        full = model(idx)
        # Prefill 5 tokens, then a 3-token chunk (explicit-mask path), then 4 single tokens.
        logits_a, past = model(idx[:, :5], use_cache=True)
        logits_b, past = model(idx[:, 5:8], past_kv=past, use_cache=True)
        parts = [logits_a, logits_b]
        for t in range(8, 12):
            step, past = model(idx[:, t:t + 1], past_kv=past, use_cache=True)
            parts.append(step)
    assert past[0][0].shape[2] == 12  # cache holds every token
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-5)


def test_default_forward_returns_logits_only():
    model = _tiny_model()
    out = model(torch.zeros(1, 4, dtype=torch.long))
    assert isinstance(out, torch.Tensor) and out.shape == (1, 4, 50257)


def test_cache_cannot_exceed_context():
    model = _tiny_model(context_length=8)
    with torch.no_grad():
        _, past = model(torch.zeros(1, 8, dtype=torch.long), use_cache=True)
        with pytest.raises(ValueError):
            model(torch.zeros(1, 1, dtype=torch.long), past_kv=past, use_cache=True)


def test_cached_generate_equals_uncached_with_sliding_window():
    # Context 8, prompt 3, 20 new tokens: the cache is used up to 8 tokens, then generate falls back
    # to recomputing the sliding window. Tokens must be identical to the no-cache path.
    model = _tiny_model(context_length=8)
    start = torch.tensor([[11, 22, 33]])
    a = generate(model, start, max_new_tokens=20, context_size=8, eos_id=None, use_cache=True)
    b = generate(model, start, max_new_tokens=20, context_size=8, eos_id=None, use_cache=False)
    assert torch.equal(a, b)


def test_cached_generate_eos_padding_unchanged():
    model = _tiny_model()
    start = torch.tensor([[1, 2, 3], [4, 5, 6]])
    free = generate(model, start, max_new_tokens=10, eos_id=None, use_cache=False)
    fake_eos = free[0, 5].item()  # a token row 0 really produces -> acts as "EOS" for this test
    a = generate(model, start, max_new_tokens=10, eos_id=fake_eos, use_cache=True)
    b = generate(model, start, max_new_tokens=10, eos_id=fake_eos, use_cache=False)
    assert torch.equal(a, b)


def test_cached_generate_with_lora():
    from training_model.lora import LinearWithLoRA, apply_lora

    model = _tiny_model()
    apply_lora(model, rank=4, alpha=8)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LinearWithLoRA):
                m.lora.B.normal_(0.0, 0.02)
    start = torch.tensor([[7, 8, 9, 10]])
    a = generate(model, start, max_new_tokens=15, eos_id=None, use_cache=True)
    b = generate(model, start, max_new_tokens=15, eos_id=None, use_cache=False)
    assert torch.equal(a, b)


def test_stop_fn_stops_and_keeps_last_token():
    model = _tiny_model()
    start = torch.tensor([[1, 2, 3]])
    calls = []

    def stop_after_four(ids):
        calls.append(ids.shape[1])
        return ids.shape[1] - start.shape[1] >= 4

    out = generate(model, start, max_new_tokens=20, eos_id=None, stop_fn=stop_after_four)
    assert out.shape == (1, 7)
    assert calls == [4, 5, 6, 7]
    full = generate(model, start, max_new_tokens=20, eos_id=None)
    assert torch.equal(out, full[:, :7])
