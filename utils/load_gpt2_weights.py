"""
Download the OpenAI GPT-2 pretrained weights and load them into our own GPTModel (Stage 2).

Purpose: fetch `model.safetensors` for one GPT-2 size from the Hugging Face hub repo
`openai-community/<size>`, rename every tensor to our module names, split the fused Q/K/V
matrix, transpose the Conv1D weights, shape-check everything and fail if any parameter is missing.
No pretraining happens in this project (PLAN D1); these weights are the starting point for LoRA.

Author: FraudLM v2 team (COS30049.2-1).
Weight mapping follows Raschka, S. (2024) *Build a Large Language Model (From Scratch)*, Manning,
chapter 5 (`load_weights_into_gpt`); code repo Apache-2.0.
Written with help from Claude Code (Anthropic); reviewed by Thong.

Note: `transformers` is never imported here (PLAN D3). We only read the raw tensors.
"""

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

# Make `training_model` importable when this file is run directly as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training_model.gpt_model import GPT_CONFIGS, GPTModel  # noqa: E402

# Our size name -> Hugging Face repo that hosts OpenAI's released weights.
HF_REPOS = {
    "gpt2-small": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "gpt2-large": "openai-community/gpt2-large",
    "gpt2-xl": "openai-community/gpt2-xl",
}
WEIGHTS_FILE = "model.safetensors"


def download_gpt2_weights(size, cache_dir="checkpoint/pretrained"):
    """Return the local path of `model.safetensors` for `size`, downloading it once if needed.

    huggingface_hub resumes a broken download and shows a progress bar, which matters for
    the 6.4 GB XL file on Kaggle.
    """
    if size not in HF_REPOS:
        raise ValueError(f"Unknown GPT-2 size {size!r}; choose one of {list(HF_REPOS)}")
    target_dir = Path(cache_dir) / size
    target = target_dir / WEIGHTS_FILE
    if target.exists() and target.stat().st_size > 0:
        print(f"[weights] using cached {target}")
        return target

    from huggingface_hub import hf_hub_download  # only needed when we actually download

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[weights] downloading {HF_REPOS[size]}/{WEIGHTS_FILE} -> {target_dir}")
    path = hf_hub_download(repo_id=HF_REPOS[size], filename=WEIGHTS_FILE, local_dir=str(target_dir))
    return Path(path)


def map_hf_to_ours(hf_state, cfg):
    """Translate a GPT-2 safetensors state dict into our GPTModel parameter names.

    HF GPT-2 uses Conv1D layers that store weights as (in_features, out_features);
    nn.Linear stores (out_features, in_features), so every Conv1D weight is transposed.
    """
    # Some exports prefix every key with "transformer."; strip it so both layouts work.
    sd = {k.removeprefix("transformer."): v for k, v in hf_state.items()}
    ours = {}

    ours["tok_emb.weight"] = sd["wte.weight"]
    ours["pos_emb.weight"] = sd["wpe.weight"]

    for i in range(cfg["n_layers"]):
        hf = f"h.{i}."
        me = f"trf_blocks.{i}."

        # c_attn holds Q, K and V side by side: weight (emb, 3*emb), bias (3*emb). Split in 3.
        q_w, k_w, v_w = torch.chunk(sd[hf + "attn.c_attn.weight"], 3, dim=-1)
        q_b, k_b, v_b = torch.chunk(sd[hf + "attn.c_attn.bias"], 3, dim=-1)
        ours[me + "att.W_query.weight"] = q_w.T
        ours[me + "att.W_key.weight"] = k_w.T
        ours[me + "att.W_value.weight"] = v_w.T
        ours[me + "att.W_query.bias"] = q_b
        ours[me + "att.W_key.bias"] = k_b
        ours[me + "att.W_value.bias"] = v_b

        ours[me + "att.out_proj.weight"] = sd[hf + "attn.c_proj.weight"].T
        ours[me + "att.out_proj.bias"] = sd[hf + "attn.c_proj.bias"]

        ours[me + "norm1.scale"] = sd[hf + "ln_1.weight"]
        ours[me + "norm1.shift"] = sd[hf + "ln_1.bias"]
        ours[me + "norm2.scale"] = sd[hf + "ln_2.weight"]
        ours[me + "norm2.shift"] = sd[hf + "ln_2.bias"]

        ours[me + "ff.layers.0.weight"] = sd[hf + "mlp.c_fc.weight"].T
        ours[me + "ff.layers.0.bias"] = sd[hf + "mlp.c_fc.bias"]
        ours[me + "ff.layers.2.weight"] = sd[hf + "mlp.c_proj.weight"].T
        ours[me + "ff.layers.2.bias"] = sd[hf + "mlp.c_proj.bias"]

    ours["final_norm.scale"] = sd["ln_f.weight"]
    ours["final_norm.shift"] = sd["ln_f.bias"]

    # GPT-2 ties the output head to the token embedding, so the head gets the same matrix.
    ours["out_head.weight"] = ours["tok_emb.weight"]
    return ours


def load_pretrained_gpt2(size, cache_dir="checkpoint/pretrained", device="cpu", dtype=torch.float32):
    """Build a GPTModel of `size` and fill it with OpenAI's pretrained GPT-2 weights.

    Returns the model in eval mode on `device` with parameters in `dtype`.
    """
    if size not in GPT_CONFIGS:
        raise ValueError(f"Unknown GPT-2 size {size!r}; choose one of {list(GPT_CONFIGS)}")
    cfg = GPT_CONFIGS[size]
    path = download_gpt2_weights(size, cache_dir)

    # Build an empty skeleton on the "meta" device (shapes only, no memory, no random init).
    # Without this, XL would need 6 GB for random weights we throw away straight after.
    with torch.device("meta"):
        model = GPTModel(cfg)
    expected = model.state_dict()

    hf_state = load_file(str(path))  # plain tensors on CPU
    ours = map_hf_to_ours(hf_state, cfg)
    # `ours` now holds (views of) every tensor we need; drop the dict so unused entries
    # (e.g. old attention-mask buffers) can be freed and peak RAM stays lower for XL.
    del hf_state

    # Every one of our parameters must be filled, with exactly the right shape.
    missing = sorted(set(expected) - set(ours))
    unexpected = sorted(set(ours) - set(expected))
    if missing or unexpected:
        raise RuntimeError(f"Weight mapping mismatch. Missing: {missing[:5]} Unexpected: {unexpected[:5]}")
    for name, tensor in ours.items():
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise RuntimeError(
                f"Shape mismatch for {name}: checkpoint {tuple(tensor.shape)} "
                f"vs model {tuple(expected[name].shape)}"
            )

    # Convert dtype / move to the device first, then .contiguous() (transposed views are not
    # contiguous). In this order, loading onto a GPU does not make a second full fp32 copy in CPU RAM.
    # Converting one tensor at a time and replacing it lets CPU originals be freed along the way.
    for k in list(ours):
        if k != "out_head.weight":
            ours[k] = ours[k].to(device=device, dtype=dtype).contiguous()
            if k == 'tok_emb.weight' and torch.device(device).type == 'cpu':
                # Own the tied output matrix rather than retaining mapped-file storage.
                # This also gives single-token CPU linear operations allocator-aligned storage.
                ours[k] = ours[k].clone()
    ours["out_head.weight"] = ours["tok_emb.weight"]  # tied: reuse, do not convert a second copy

    # assign=True swaps the meta tensors for the real ones (no extra copy).
    model.load_state_dict(ours, strict=True, assign=True)

    # Loading assigns the head and the embedding separately, so tie them again explicitly.
    model.out_head.weight = model.tok_emb.weight

    # Final safety net: nothing may still live on the meta device (= never initialised).
    still_meta = [n for n, p in model.named_parameters() if p.is_meta]
    still_meta += [n for n, b in model.named_buffers() if b.is_meta]
    if still_meta:
        raise RuntimeError(f"Parameters left uninitialised: {still_meta[:5]}")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[weights] loaded {size}: {n_params:,} parameters ({dtype}, {device})")
    return model


if __name__ == "__main__":
    # Usage: python utils/load_gpt2_weights.py gpt2-small
    size_arg = sys.argv[1] if len(sys.argv) > 1 else "gpt2-small"
    os.chdir(_PROJECT_ROOT)  # so the default cache_dir resolves inside llm_ver2/
    load_pretrained_gpt2(size_arg)
