"""
GPT-2 model architecture for FraudLM v2 (Stage 1 of the Raschka pipeline).

Purpose: a plain-PyTorch GPT-2 (decoder-only transformer) that can hold the OpenAI GPT-2
weights (small / medium / large / xl). No `transformers` model classes are used (PLAN D3).

Author: FraudLM v2 team (COS30049.2-1).
Architecture follows Raschka, S. (2024) *Build a Large Language Model (From Scratch)*, Manning,
chapters 3-4; code repo Apache-2.0.

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# Sizes of the four OpenAI GPT-2 releases. Only emb_dim / n_layers / n_heads change between them.
# drop_rate is 0.0 because we fine-tune a pretrained model with LoRA (Raschka ch. 7 also uses 0.0);
# a trainer can raise it in its own copy of the config.
_BASE = {"vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0, "qkv_bias": True}
GPT_CONFIGS = {
    "gpt2-small": {**_BASE, "emb_dim": 768, "n_layers": 12, "n_heads": 12},    # 124M
    "gpt2-medium": {**_BASE, "emb_dim": 1024, "n_layers": 24, "n_heads": 16},  # 355M
    "gpt2-large": {**_BASE, "emb_dim": 1280, "n_layers": 36, "n_heads": 20},   # 774M
    "gpt2-xl": {**_BASE, "emb_dim": 1600, "n_layers": 48, "n_heads": 25},      # 1558M
}


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention (Raschka ch. 3)."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length
        self.dropout = dropout
        # Separate Q, K, V projections (GPT-2 stores them fused as c_attn; the loader splits them).
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # mixes the heads back together

    def forward(self, x, past_kv=None, use_cache=False):
        """x: [b, T, d_in]. Returns the attention output, plus (keys, values) when use_cache=True.

        past_kv / use_cache are only for fast generation (KV cache); training never passes them.
        """
        b, num_tokens, _ = x.shape
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        # [b, T, d_out] -> [b, heads, T, head_dim] so every head attends on its own slice.
        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            # KV cache: keys/values of earlier tokens were saved, so only the new tokens are computed
            # and then appended. The new queries attend to every earlier token plus themselves.
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        dropout_p = self.dropout if self.training else 0.0
        if past_kv is None:
            # PyTorch's fused attention computes softmax(QK^T / sqrt(head_dim)) V.
            # is_causal=True applies the mask so a token never sees tokens after it.
            # It is the same maths as Raschka's explicit mask but uses less memory on the GPU.
            context = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            # With a cache there are more keys than queries, and is_causal would align the mask to the
            # wrong corner. New query i sits at position past_len + i, so it may see keys 0..past_len + i.
            # For a single new token this mask is all True (it may see everything before it).
            q_len, k_len = q.shape[2], k.shape[2]
            mask = torch.ones(q_len, k_len, dtype=torch.bool, device=x.device).tril(diagonal=k_len - q_len)
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)

        # Back to [b, T, d_out].
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        out = self.out_proj(context)
        if use_cache:
            return out, (k, v)
        return out


class LayerNorm(nn.Module):
    """Layer normalisation with a learnable scale and shift (Raschka ch. 4)."""

    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps  # 1e-5 is the value GPT-2 was trained with
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Compute in at least float32. With an fp16 base the residual stream reaches ~3,000, so the
        # variance (and each squared deviation) can pass fp16's max of 65,504 and become inf,
        # silently zeroing whole rows. fp16/bf16 are upcast to fp32; fp32 and fp64 stay as they
        # are, so fp32 results are bit-for-bit unchanged.
        compute_dtype = torch.promote_types(x.dtype, torch.float32)
        xf = x.to(compute_dtype)
        mean = xf.mean(dim=-1, keepdim=True)
        # unbiased=False (divide by n) matches GPT-2's original TensorFlow LayerNorm.
        var = xf.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (xf - mean) / torch.sqrt(var + self.eps)
        out = self.scale.to(compute_dtype) * norm_x + self.shift.to(compute_dtype)
        return out.to(x.dtype)  # back to the caller's dtype (fp16 / bf16 / fp32)


class GELU(nn.Module):
    """GELU activation, tanh approximation (the version GPT-2 used)."""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, GELU, project back (Raschka ch. 4)."""

    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # GPT-2 c_fc
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # GPT-2 c_proj
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """One pre-norm transformer block: x + Attn(LN(x)), then x + FF(LN(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, past_kv=None, use_cache=False):
        # Residual ("shortcut") connections keep gradients flowing through deep stacks.
        shortcut = x
        x = self.norm1(x)
        if use_cache:
            x, new_kv = self.att(x, past_kv=past_kv, use_cache=True)
        else:
            x = self.att(x, past_kv=past_kv)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        if use_cache:
            return x, new_kv
        return x


class GPTModel(nn.Module):
    """GPT-2: token + position embeddings -> n_layers blocks -> final norm -> vocabulary logits."""

    supports_kv_cache = True  # generate() checks this flag before using past_kv / use_cache

    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # learned positions
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Weight tying: GPT-2 reuses the token embedding matrix as the output layer.
        # This saves vocab_size * emb_dim parameters (about 80M for XL) and is how OpenAI trained it.
        self.out_head.weight = self.tok_emb.weight

        # Set True to recompute each block's activations in the backward pass instead of storing
        # them. Slower (about one extra forward) but needs far less GPU memory (16 GB T4 / P100).
        self.gradient_checkpointing = False

        # GPT-2 style init (normal, std 0.02). Only matters for a random model; real weights
        # overwrite it. With it a random model's loss is about ln(50257) = 10.8, as expected.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, past_kv=None, use_cache=False):
        """idx: [B, T] token ids -> logits [B, T, vocab].

        Training and evaluation call model(idx) and get the logits only.
        Generation can pass use_cache=True to also get `new_past_kv` (one (keys, values) pair per
        block) and feed it back as past_kv with only the next token: then it returns
        (logits, new_past_kv). This is the KV cache; see utils/generate.py.
        """
        batch_size, seq_len = idx.shape
        past_len = 0 if past_kv is None else past_kv[0][0].shape[2]  # tokens already in the cache
        if past_len + seq_len > self.cfg["context_length"]:
            raise ValueError(
                f"Sequence length {past_len + seq_len} is longer than the context length "
                f"{self.cfg['context_length']}; crop the input first."
            )
        tok_embeds = self.tok_emb(idx)
        # New tokens continue the positions after the cached ones (past_len = 0 without a cache).
        positions = torch.arange(past_len, past_len + seq_len, device=idx.device)
        pos_embeds = self.pos_emb(positions)
        x = self.drop_emb(tok_embeds + pos_embeds)

        if use_cache or past_kv is not None:
            # Generation path (no gradients needed, so no checkpointing).
            new_past_kv = []
            for i, block in enumerate(self.trf_blocks):
                layer_past = None if past_kv is None else past_kv[i]
                x, layer_kv = block(x, past_kv=layer_past, use_cache=True)
                new_past_kv.append(layer_kv)
        else:
            # Normal path used by training: unchanged.
            for block in self.trf_blocks:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    # use_reentrant=False is the recommended mode and works with frozen base weights.
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        if use_cache:
            return logits, new_past_kv
        return logits


def count_parameters(model, trainable_only=False):
    """Number of (unique) parameters. Tied weights are counted once, as in the GPT-2 paper sizes."""
    params = model.parameters()  # parameters() already skips the duplicate tied tensor
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
