"""
Tests for training_model/lora.py: base frozen, B = 0 keeps the base output, save/load round trip,
merge gives the same logits, trainable counts for gpt2-small and gpt2-xl.

Author: FraudLM v2 team (COS30049.2-1).

"""

import copy

import pytest
import torch

from training_model.gpt_model import GPT_CONFIGS, GPTModel
from training_model.lora import LinearWithLoRA, apply_lora, load_lora, lora_state_dict, merge_lora

TINY = {"vocab_size": 50257, "context_length": 32, "emb_dim": 64, "n_heads": 4,
        "n_layers": 2, "drop_rate": 0.0, "qkv_bias": True}


def expected_lora_count(cfg, rank, targets=("attn", "ff")):
    """rank * (in + out) per wrapped Linear, summed over all blocks."""
    d = cfg["emb_dim"]
    per_block = 0
    if "attn" in targets:
        per_block += 4 * rank * (d + d)
    if "ff" in targets:
        per_block += 2 * rank * (d + 4 * d)
    return per_block * cfg["n_layers"]


def _random_update(model):
    """Pretend training happened: give every B a non-zero value."""
    torch.manual_seed(42)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LinearWithLoRA):
                m.lora.B.normal_(0.0, 0.02)


def test_only_lora_params_trainable():
    torch.manual_seed(42)
    model = GPTModel(TINY)
    n = apply_lora(model, rank=4, alpha=8)
    assert n == expected_lora_count(TINY, 4)
    for name, p in model.named_parameters():
        assert p.requires_grad == (".lora." in name), name
    assert sum(isinstance(m, LinearWithLoRA) for m in model.modules()) == 6 * TINY["n_layers"]


def test_targets_attn_only_and_bad_target():
    model = GPTModel(TINY)
    assert apply_lora(model, rank=2, alpha=4, targets=("attn",)) == expected_lora_count(TINY, 2, ("attn",))
    with pytest.raises(RuntimeError):
        apply_lora(model, rank=2, alpha=4)  # applying twice is refused
    with pytest.raises(ValueError):
        apply_lora(GPTModel(TINY), rank=2, alpha=4, targets=("embedding",))


def test_b_zero_gives_base_output():
    torch.manual_seed(42)
    model = GPTModel(TINY).eval()
    idx = torch.randint(0, 50257, (2, 10))
    with torch.no_grad():
        base = model(idx)
    apply_lora(model, rank=4, alpha=8)
    with torch.no_grad():
        wrapped = model(idx)
    assert torch.equal(base, wrapped) or torch.allclose(base, wrapped, atol=1e-6)


def test_gradients_reach_only_lora():
    torch.manual_seed(42)
    model = GPTModel(TINY).train()
    apply_lora(model, rank=4, alpha=8)
    _random_update(model)  # with B = 0 the gradient of A would be exactly zero
    model(torch.randint(0, 50257, (2, 10))).mean().backward()
    for name, p in model.named_parameters():
        if ".lora." in name:
            assert p.grad is not None and p.grad.abs().sum() > 0, name
        else:
            assert p.grad is None, name


def test_gradient_checkpointing_with_lora():
    """Frozen base + LoRA + checkpointing (the XL-on-T4 setup): same loss and grads as without it,
    every LoRA param gets a gradient and no base param does."""
    torch.manual_seed(42)
    model = GPTModel(TINY)
    apply_lora(model, rank=4, alpha=8)
    _random_update(model)  # non-zero B so A also gets a non-zero gradient
    model.train()
    idx = torch.randint(0, 50257, (2, 16), generator=torch.Generator().manual_seed(42))
    targets = torch.randint(0, 50257, (2, 16), generator=torch.Generator().manual_seed(7))

    def loss_and_grads(use_ckpt):
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing = use_ckpt
        logits = model(idx)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        for n, p in model.named_parameters():
            if ".lora." in n:
                assert p.grad is not None and p.grad.abs().sum() > 0, n
            else:
                assert p.grad is None, n
        return loss.item(), grads

    loss_plain, grads_plain = loss_and_grads(False)
    loss_ckpt, grads_ckpt = loss_and_grads(True)
    assert abs(loss_plain - loss_ckpt) < 1e-6
    assert grads_plain.keys() == grads_ckpt.keys()
    for n in grads_plain:
        assert torch.allclose(grads_plain[n], grads_ckpt[n], atol=1e-6), n


def test_save_and_load_adapter_round_trip():
    torch.manual_seed(42)
    trained = GPTModel(TINY).eval()
    base_state = {k: v.clone() for k, v in trained.state_dict().items()}
    apply_lora(trained, rank=4, alpha=8)
    _random_update(trained)
    adapter = lora_state_dict(trained)
    assert adapter and all(".lora." in k for k in adapter)

    fresh = GPTModel(TINY).eval()
    fresh.load_state_dict(base_state)
    apply_lora(fresh, rank=4, alpha=8)
    load_lora(fresh, adapter)
    idx = torch.randint(0, 50257, (1, 12))
    with torch.no_grad():
        assert torch.allclose(trained(idx), fresh(idx), atol=1e-6)


def test_merge_gives_same_logits():
    torch.manual_seed(42)
    model = GPTModel(TINY).eval()
    apply_lora(model, rank=4, alpha=8)
    _random_update(model)
    idx = torch.randint(0, 50257, (2, 12))
    with torch.no_grad():
        before = model(idx)
        merge_lora(model)
        after = model(idx)
    assert not any(isinstance(m, LinearWithLoRA) for m in model.modules())
    assert torch.allclose(before, after, atol=1e-5)


def test_pretrained_small_lora_and_merge():
    """Same checks on real gpt2-small weights with the default LoRA settings (rank 16, alpha 32)."""
    from pathlib import Path
    from utils.load_gpt2_weights import load_pretrained_gpt2

    cache = str(Path(__file__).resolve().parents[1] / "checkpoint" / "pretrained")
    model = load_pretrained_gpt2("gpt2-small", cache_dir=cache)
    base_copy = copy.deepcopy(model)  # untouched base, reused for the float64 check below
    idx = torch.randint(0, 50257, (1, 16), generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        base = model(idx)
    n = apply_lora(model, rank=16, alpha=32)
    print(f"\n[G1] gpt2-small LoRA r=16 a=32 trainable = {n:,}")
    assert n == expected_lora_count(GPT_CONFIGS["gpt2-small"], 16) == 2_654_208
    with torch.no_grad():
        assert torch.allclose(model(idx), base, atol=1e-5)
        _random_update(model)
        adapter = lora_state_dict(model)
        before = model(idx)
        merge_lora(model)
        after = model(idx)
    diff = (before - after).abs().max().item()
    # Real GPT-2 logits are ~100 in size, so float32 rounding allows ~1e-5 relative, not absolute.
    assert torch.allclose(before, after, atol=1e-4, rtol=1e-5)

    # Strict check on the same real weights and adapter in float64, where rounding noise vanishes.
    model64 = base_copy.double()
    apply_lora(model64, rank=16, alpha=32)
    model64.double()  # new LoRA layers are float32 by default; the whole check runs in float64
    load_lora(model64, adapter)
    with torch.no_grad():
        before64 = model64(idx)
        merge_lora(model64)
        after64 = model64(idx)
    diff64 = (before64 - after64).abs().max().item()
    print(f"[G1] gpt2-small merge max logit difference: float32 {diff:.2e}, float64 {diff64:.2e}")
    assert torch.allclose(before64, after64, atol=1e-8, rtol=0)  # stricter than the 1e-5 target


def test_xl_trainable_count_on_meta():
    # Meta device: no memory used, so we can count XL on a laptop.
    with torch.device("meta"):
        model = GPTModel(GPT_CONFIGS["gpt2-xl"])
        n = apply_lora(model, rank=16, alpha=32)
    print(f"\n[G1] gpt2-xl LoRA r=16 a=32 trainable = {n:,}")
    assert n == expected_lora_count(GPT_CONFIGS["gpt2-xl"], 16) == 22_118_400
