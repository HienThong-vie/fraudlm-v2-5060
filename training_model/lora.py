"""
LoRA (Low-Rank Adaptation) for our GPTModel (Stage 3 building block).

Purpose: freeze every pretrained GPT-2 weight and learn a small low-rank update
W + scale * (A @ B) for chosen nn.Linear layers. Only A and B are trained, which is what lets
GPT-2 XL (1.56B parameters) be fine-tuned on one 16 GB Kaggle GPU.
Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models".

Author: FraudLM v2 team (COS30049.2-1).
Follows Raschka, S. (2024) *Build a Large Language Model (From Scratch)*, Manning,
appendix E (LoRALayer, LinearWithLoRA); code repo Apache-2.0.

"""

import math

import torch
import torch.nn as nn

VALID_TARGETS = ("attn", "ff")


class LoRALayer(nn.Module):
    """The trainable low-rank pair: x -> scale * (x @ A @ B)."""

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank  # standard LoRA scaling, so changing rank does not change the step size much
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        # A: Kaiming-uniform with a=sqrt(5), as in Raschka appendix E. A has shape (in_dim, rank), so
        # PyTorch takes fan_in = rank and A ~ U(-1/sqrt(rank), +1/sqrt(rank)) (+-0.25 at rank 16).
        # This is larger than PEFT's init (fan_in = in_dim), so use a modest learning rate.
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B = 0, so A @ B = 0 and the wrapped layer starts exactly equal to the pretrained one.
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        # Adapters stay in float32 even if the frozen base is stored in half precision on the GPU,
        # so optimiser updates are not rounded away. Cast in and back out.
        out = (x.to(self.A.dtype) @ self.A @ self.B) * self.scale
        return out.to(x.dtype)


class LinearWithLoRA(nn.Module):
    """A frozen nn.Linear plus a LoRA update: y = linear(x) + lora(x)."""

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
        self.lora.to(device=linear.weight.device)  # same device as the base weight, float32

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def _target_linears(block, targets):
    """(parent module, attribute name) of each nn.Linear in one TransformerBlock to wrap."""
    places = []
    if "attn" in targets:
        # Query, key, value and the output projection of the attention.
        for name in ("W_query", "W_key", "W_value", "out_proj"):
            places.append((block.att, name))
    if "ff" in targets:
        # The two Linear layers of the feed-forward (index 1 is the GELU).
        for name in ("0", "2"):
            places.append((block.ff.layers, name))
    return places


def apply_lora(model, rank, alpha, targets=("attn", "ff")):
    """Freeze the whole model, wrap the target Linear layers with LoRA, return the trainable count.

    targets: any of "attn" (q/k/v/out projections) and "ff" (feed-forward in/out).
    The output head is not wrapped: it is tied to the token embedding and stays frozen.
    """
    bad = [t for t in targets if t not in VALID_TARGETS]
    if bad:
        raise ValueError(f"Unknown LoRA targets {bad}; valid: {VALID_TARGETS}")
    if any(isinstance(m, LinearWithLoRA) for m in model.modules()):
        raise RuntimeError("LoRA is already applied to this model")

    # 1) Freeze every pretrained weight.
    for p in model.parameters():
        p.requires_grad = False

    # 2) Replace each target Linear by LinearWithLoRA (only the new A and B are trainable).
    for block in model.trf_blocks:
        for parent, name in _target_linears(block, targets):
            linear = getattr(parent, name)
            setattr(parent, name, LinearWithLoRA(linear, rank, alpha))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] rank={rank} alpha={alpha} targets={tuple(targets)}: "
          f"{trainable:,} trainable of {total:,} parameters ({100 * trainable / total:.2f}%)")
    return trainable


def lora_state_dict(model):
    """Only the adapter weights (A and B), copied to CPU: a few MB instead of the full model."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if ".lora." in k}


def load_lora(model, state_dict):
    """Copy saved adapter weights into a model that already went through apply_lora (same rank/targets)."""
    own = {k: v for k, v in model.state_dict().items() if ".lora." in k}
    missing = sorted(set(own) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(own))
    if missing or unexpected:
        raise RuntimeError(f"LoRA keys do not match. Missing: {missing[:5]} Unexpected: {unexpected[:5]}")
    with torch.no_grad():
        for k, v in state_dict.items():
            if own[k].shape != v.shape:
                raise RuntimeError(f"LoRA shape mismatch for {k}: {tuple(v.shape)} vs {tuple(own[k].shape)}")
            own[k].copy_(v)  # state_dict tensors share storage with the parameters


@torch.no_grad()
def merge_lora(model):
    """Fold every LoRA update into its base weight and remove the wrappers.

    After merging, the model is a plain GPTModel again (no extra compute at chat time).
    nn.Linear computes x @ W.T, and LoRA adds x @ (A @ B) * scale, so W_new = W + (A @ B * scale).T
    """
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LinearWithLoRA):
                lin, lora = child.linear, child.lora
                delta = (lora.A @ lora.B * lora.scale).T  # float32, shape (out, in) like lin.weight
                lin.weight.add_(delta.to(lin.weight.dtype))
                setattr(module, name, lin)
    return model
