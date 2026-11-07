"""
Tests for training_model/gpt_model.py: output shapes, causal masking, parameter counts,
weight tying and gradient checkpointing.

Author: FraudLM v2 team (COS30049.2-1).

"""

import copy

import pytest
import torch

from training_model.gpt_model import GPT_CONFIGS, GPTModel, LayerNorm, count_parameters

# A tiny config so the logic tests run in milliseconds on CPU.
TINY = {"vocab_size": 50257, "context_length": 32, "emb_dim": 64, "n_heads": 4,
        "n_layers": 2, "drop_rate": 0.0, "qkv_bias": True}

# GPT-2 parameter counts with the output head tied to the token embedding.
EXPECTED_PARAMS = {
    "gpt2-small": 124_439_808,
    "gpt2-medium": 354_823_168,
    "gpt2-large": 774_030_080,
    "gpt2-xl": 1_557_611_200,
}


def test_configs_have_contract_keys():
    for name, cfg in GPT_CONFIGS.items():
        for key in ("vocab_size", "context_length", "emb_dim", "n_heads", "n_layers", "drop_rate", "qkv_bias"):
            assert key in cfg, (name, key)
        assert cfg["vocab_size"] == 50257 and cfg["context_length"] == 1024 and cfg["qkv_bias"] is True


def test_output_shape_small():
    torch.manual_seed(42)
    model = GPTModel(GPT_CONFIGS["gpt2-small"]).eval()
    idx = torch.randint(0, 50257, (2, 10))
    with torch.no_grad():
        logits = model(idx)
    assert logits.shape == (2, 10, 50257)


def test_causal_mask_future_tokens_do_not_change_past_logits():
    torch.manual_seed(42)
    model = GPTModel(TINY).eval()
    idx = torch.randint(0, 50257, (1, 12))
    changed = idx.clone()
    changed[0, 8:] = (changed[0, 8:] + 123) % 50257  # change only positions 8..11
    with torch.no_grad():
        a = model(idx)
        b = model(changed)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-6)  # earlier positions unchanged
    assert not torch.allclose(a[:, 8:], b[:, 8:])        # later positions do change


def test_rejects_too_long_input():
    model = GPTModel(TINY)
    with pytest.raises(ValueError):
        model(torch.zeros(1, TINY["context_length"] + 1, dtype=torch.long))


def test_head_is_tied_to_token_embedding():
    model = GPTModel(TINY)
    assert model.out_head.weight is model.tok_emb.weight


@pytest.mark.parametrize("name", list(GPT_CONFIGS))
def test_parameter_counts(name):
    # Build on the meta device: shapes only, no memory, so even XL is instant.
    with torch.device("meta"):
        model = GPTModel(GPT_CONFIGS[name])
    n = count_parameters(model)
    print(f"\n[G1] {name}: {n:,} parameters")
    assert n == EXPECTED_PARAMS[name]


def test_layernorm_fp16_large_rows_stay_finite():
    """Review R8: an fp16 residual stream reaches ~3,000; LayerNorm must not overflow to inf."""
    torch.manual_seed(42)
    x = torch.randn(4, 768) * 1000
    x[:, 0] = 3000.0  # variance ~1e6, far above fp16's max of 65,504
    ln32 = LayerNorm(768)
    with torch.no_grad():
        ln32.scale.uniform_(0.5, 1.5)
        ln32.shift.uniform_(-0.1, 0.1)
    ln16 = copy.deepcopy(ln32).half()

    ref = ln32(x)
    out = ln16(x.half())
    assert out.dtype == torch.float16  # returns the caller's dtype
    assert torch.isfinite(out).all()
    assert out.float().abs().sum() > 0  # the old fp16 maths gave inf variance and all-zero rows
    assert torch.allclose(out.float(), ref, atol=1e-2, rtol=1e-2)


def test_layernorm_fp32_unchanged():
    """The fp32 path is exactly the textbook formula (so pretrained results do not move)."""
    torch.manual_seed(42)
    x = torch.randn(3, 5, 64)
    ln = LayerNorm(64)
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    expected = ln.scale * ((x - mean) / torch.sqrt(var + ln.eps)) + ln.shift
    assert torch.equal(ln(x), expected)


def test_gradient_checkpointing_same_result():
    torch.manual_seed(42)
    model = GPTModel(TINY).train()
    idx = torch.randint(0, 50257, (2, 16))

    model.gradient_checkpointing = False
    loss_plain = model(idx).float().mean()
    loss_plain.backward()
    grad_plain = model.trf_blocks[0].att.W_query.weight.grad.clone()
    model.zero_grad()

    model.gradient_checkpointing = True
    loss_ckpt = model(idx).float().mean()
    loss_ckpt.backward()
    grad_ckpt = model.trf_blocks[0].att.W_query.weight.grad

    assert torch.allclose(loss_plain, loss_ckpt, atol=1e-6)
    assert torch.allclose(grad_plain, grad_ckpt, atol=1e-6)
