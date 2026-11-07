"""
Tests for utils/load_gpt2_weights.py: OpenAI GPT-2 small loads into our GPTModel and behaves like GPT-2.

Author: FraudLM v2 team (COS30049.2-1).

Downloads gpt2-small (~550 MB) once into checkpoint/pretrained/gpt2-small/.
"""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from training_model.gpt_model import GPT_CONFIGS, GPTModel
from utils.generate import generate, ids_to_text, text_to_ids
from utils.load_gpt2_weights import load_pretrained_gpt2

CACHE_DIR = str(Path(__file__).resolve().parents[1] / "checkpoint" / "pretrained")

# Fixed ~190-token English paragraph (written by the team) used to check the language-model loss.
PARAGRAPH = (
    "Banks have a legal duty to know who their customers are before they open an account. "
    "When a person applies online, the bank checks the details they give, such as their name, "
    "address, phone number and email, against other records. Most applications are genuine, but a "
    "small share are made by criminals who use stolen or invented identities. These fraudsters often "
    "open many accounts in a short time, use free email services, or give phone numbers that do not "
    "match their address. Because fraud is rare, a model that simply says every application is genuine "
    "would be right more than ninety-nine times out of a hundred, yet it would catch no fraud at all. "
    "For this reason, analysts measure how many fraud cases a system finds while keeping the number of "
    "false alarms low. The final decision always stays with a trained person, who can ask for more "
    "documents or call the applicant before the account is opened."
)


def lm_loss(model, text):
    """Mean next-token cross-entropy over the text (the pretraining objective)."""
    ids = text_to_ids(text)
    with torch.no_grad():
        logits = model(ids[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten()).item()


@pytest.fixture(scope="module")
def gpt2_small():
    torch.manual_seed(42)
    return load_pretrained_gpt2("gpt2-small", cache_dir=CACHE_DIR)


def test_all_parameters_loaded_and_tied(gpt2_small):
    assert not any(p.is_meta for p in gpt2_small.parameters())
    assert gpt2_small.out_head.weight is gpt2_small.tok_emb.weight
    assert sum(p.numel() for p in gpt2_small.parameters()) == 124_439_808


def test_pretrained_loss_in_range(gpt2_small):
    n_tokens = text_to_ids(PARAGRAPH).shape[1]
    loss = lm_loss(gpt2_small, PARAGRAPH)

    torch.manual_seed(42)
    random_model = GPTModel(GPT_CONFIGS["gpt2-small"]).eval()
    random_loss = lm_loss(random_model, PARAGRAPH)

    print(f"\n[G1] paragraph tokens={n_tokens}  pretrained loss={loss:.4f}  random-init loss={random_loss:.4f}")
    assert 150 <= n_tokens <= 260
    assert 3.0 <= loss <= 4.5
    assert random_loss > 10.0


def test_fp16_base_loss_close_to_fp32(gpt2_small):
    """Review R8: with weights stored in fp16 the loss must stay close to fp32 (no silent overflow)."""
    loss32 = lm_loss(gpt2_small, PARAGRAPH)
    model16 = copy.deepcopy(gpt2_small).half()
    ids = text_to_ids(PARAGRAPH)
    with torch.no_grad():
        logits16 = model16(ids[:, :-1])
    assert torch.isfinite(logits16).all()
    loss16 = F.cross_entropy(logits16.float().flatten(0, 1), ids[:, 1:].flatten()).item()
    print(f"\n[G1] gpt2-small loss fp32 {loss32:.4f} vs fp16 base {loss16:.4f}")
    assert abs(loss16 - loss32) < 0.05


def test_greedy_continuation_is_english(gpt2_small):
    out = generate(gpt2_small, text_to_ids("Every effort moves you"), max_new_tokens=25, temperature=0.0)
    text = ids_to_text(out)
    print(f"\n[G1] greedy: {text!r}")
    assert text.startswith("Every effort moves you")
    new_part = text[len("Every effort moves you"):]
    words = new_part.split()
    assert len(words) >= 10
    # Fluent English: mostly plain ASCII words, no garbage characters.
    assert sum(w.strip(".,;:!?'\"").isalpha() for w in words) / len(words) > 0.7


def test_matches_huggingface_reference(gpt2_small):
    """Optional: our logits equal the HF GPT-2 logits. transformers is used only inside this test."""
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import load_file

    # Build HF's GPT-2 small from the same safetensors file (no second download).
    hf_model = transformers.GPT2LMHeadModel(transformers.GPT2Config())
    state = load_file(str(Path(CACHE_DIR) / "gpt2-small" / "model.safetensors"))
    state = {(k if k.startswith("transformer.") else "transformer." + k): v for k, v in state.items()}
    result = hf_model.load_state_dict(state, strict=False)
    # Only the tied lm_head (and non-persistent mask buffers in old versions) may be "missing".
    assert all(k == "lm_head.weight" or k.endswith("attn.bias") for k in result.missing_keys)
    hf_model.tie_weights()
    hf_model.eval()

    ids = text_to_ids(PARAGRAPH[:300])
    with torch.no_grad():
        diff32 = (gpt2_small(ids) - hf_model(ids).logits).abs().max().item()
        # GPT-2 logits are around -100, so float32 rounding alone (different kernels / summation
        # order) gives ~1e-4 differences. In float64 that noise vanishes and any real mapping
        # error would still show up, so the strict 1e-4 check is done in float64.
        ours64 = copy.deepcopy(gpt2_small).double()(ids)
        ref64 = hf_model.double()(ids).logits
    diff64 = (ours64 - ref64).abs().max().item()
    print(f"\n[G1] max |ours - HF| logit difference: float32 {diff32:.2e}, float64 {diff64:.2e}")
    assert diff32 < 1e-3
    assert torch.allclose(ours64, ref64, atol=1e-4, rtol=0)


# ---------------------------------------------------------------- KV cache on real weights

def test_cached_incremental_logits_match_full_forward(gpt2_small):
    ids = text_to_ids(PARAGRAPH)[:, :60]
    with torch.no_grad():
        full = gpt2_small(ids)
        logits, past = gpt2_small(ids[:, :40], use_cache=True)  # prefill
        parts = [logits]
        for t in range(40, 60):  # then one token at a time
            step, past = gpt2_small(ids[:, t:t + 1], past_kv=past, use_cache=True)
            parts.append(step)
    diff = (torch.cat(parts, dim=1) - full).abs().max().item()
    # fp32: GPT-2 logits are ~100 in size, so kernel rounding gives ~1e-4 absolute (~1e-6 relative),
    # as in the HF reference test. Allow that here, then check strictly in float64.
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-4, rtol=1e-5)

    model64 = copy.deepcopy(gpt2_small).double()
    with torch.no_grad():
        full64 = model64(ids)
        logits, past = model64(ids[:, :40], use_cache=True)
        parts = [logits]
        for t in range(40, 60):
            step, past = model64(ids[:, t:t + 1], past_kv=past, use_cache=True)
            parts.append(step)
    diff64 = (torch.cat(parts, dim=1) - full64).abs().max().item()
    print(f"\n[G5] cached vs full-forward max logit difference: float32 {diff:.2e}, float64 {diff64:.2e}")
    assert diff64 < 1e-8


def test_cached_greedy_equals_uncached_batch1_and_batch2(gpt2_small):
    import time

    one = text_to_ids("Every effort moves you")
    # Two prompts with the same token length so they form one batch.
    a_ids = text_to_ids("The bank checked the new application form")
    b_ids = text_to_ids("A fraud analyst reviews the account opening request")
    n = min(a_ids.shape[1], b_ids.shape[1])
    two = torch.cat([a_ids[:, :n], b_ids[:, :n]], dim=0)
    assert two.shape[0] == 2
    for start in (one, two):
        a = generate(gpt2_small, start, max_new_tokens=30, temperature=0.0, use_cache=True)
        b = generate(gpt2_small, start, max_new_tokens=30, temperature=0.0, use_cache=False)
        assert torch.equal(a, b), (ids_to_text(a[0]), ids_to_text(b[0]))

    # Speed on CPU: ~180-token prompt, 40 new tokens (no early EOS so both do the same work).
    prompt = text_to_ids(PARAGRAPH)
    timings = {}
    for flag in (False, True):
        t0 = time.perf_counter()
        out = generate(gpt2_small, prompt, max_new_tokens=40, eos_id=None, use_cache=flag)
        timings[flag] = time.perf_counter() - t0
        assert out.shape[1] == prompt.shape[1] + 40
    print(f"\n[G5] gpt2-small CPU, prompt {prompt.shape[1]} tok + 40 new: "
          f"uncached {40 / timings[False]:.1f} tok/s, cached {40 / timings[True]:.1f} tok/s, "
          f"speed-up x{timings[False] / timings[True]:.1f}")
