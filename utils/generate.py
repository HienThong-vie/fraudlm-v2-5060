"""
Text generation helpers for FraudLM v2: tokenise, decode, and generate new tokens.

Purpose: greedy decoding (temperature 0) and temperature / top-k sampling, stopping at the GPT-2
end-of-text token (id 50256) and cropping the context to the model's 1,024-token window.

Author: FraudLM v2 team (COS30049.2-1).
Follows Raschka, S. (2024) *Build a Large Language Model (From Scratch)*, Manning,
chapters 4-5 (`generate`, `text_to_token_ids`, `token_ids_to_text`); code repo Apache-2.0.

"""

import tiktoken
import torch

EOS_ID = 50256  # "<|endoftext|>" in the GPT-2 vocabulary (PLAN D4)

# One shared tokenizer; tiktoken encoders are cheap to reuse and thread-safe for encoding.
_TOKENIZER = tiktoken.get_encoding("gpt2")


def get_tokenizer():
    return _TOKENIZER


def text_to_ids(text):
    """Encode text into a LongTensor of shape [1, T] (batch of one, ready for the model).

    "<|endoftext|>" is allowed so training examples can end with the EOS token.
    """
    ids = _TOKENIZER.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def ids_to_text(ids):
    """Decode token ids back to text. Accepts a [1, T] or [T] tensor, or a list of ints.

    A batch [B, T] with B > 1 raises ValueError: decode each row separately.
    EOS tokens are decoded literally as "<|endoftext|>"; use ids_to_text_until_eos to cut them.
    """
    if isinstance(ids, torch.Tensor):
        if ids.dim() == 2 and ids.shape[0] != 1:
            raise ValueError(f"ids_to_text takes one sequence, got a batch of {ids.shape[0]}; decode row by row")
        if ids.dim() > 2:
            raise ValueError(f"ids_to_text expects a [1, T] or [T] tensor, got shape {tuple(ids.shape)}")
        ids = ids.reshape(-1).tolist()
    return _TOKENIZER.decode(ids)


def ids_to_text_until_eos(ids_row, eos_id=EOS_ID):
    """Decode one sequence but stop at its first EOS (the EOS and any padding after it are dropped).

    Use this on generate() output: a row may or may not end with EOS (see generate's docstring).
    Pass only the newly generated part, e.g. out[0, prompt_len:], if the prompt itself may hold an EOS.
    """
    if isinstance(ids_row, torch.Tensor):
        if ids_row.dim() == 2 and ids_row.shape[0] != 1:
            raise ValueError("ids_to_text_until_eos takes one row; loop over the batch")
        ids_row = ids_row.reshape(-1).tolist()
    if eos_id in ids_row:
        ids_row = ids_row[: ids_row.index(eos_id)]
    return _TOKENIZER.decode(ids_row)


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size=1024, temperature=0.0, top_k=None, eos_id=EOS_ID,
             stop_fn=None, use_cache=True):
    """Append up to `max_new_tokens` tokens to `idx` ([B, T] LongTensor) and return [B, T + n].

    temperature == 0 -> greedy (always the most likely token).
    temperature > 0  -> sample; higher = more random. top_k keeps only the k most likely tokens.
    Stops early when every row has produced `eos_id`. The EOS token itself is not appended for
    the row that finishes last; rows that finished earlier are padded with EOS so the batch stays
    rectangular. Set eos_id=None to never stop early.

    stop_fn (optional): called as stop_fn(idx) after each new token is appended (idx = the whole
    [B, T + n] so far). Return True to stop now; the token just added is kept. Example: stop when
    the decoded new text contains "\\n### ". Default None = never called.
    use_cache (default True): reuse the keys/values of earlier tokens (KV cache) so each step runs
    the model on one new token instead of the whole sequence. Same tokens as use_cache=False,
    much faster. Only used for models that support it (GPTModel); others always recompute.
    """
    was_training = model.training
    model.eval()  # no dropout while generating
    use_cache = use_cache and getattr(model, "supports_kv_cache", False)

    past_kv = None
    finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
    for _ in range(max_new_tokens):
        if use_cache and idx.shape[1] <= context_size:
            if past_kv is None:
                # Prefill: run the whole prompt once and keep its keys/values.
                logits, past_kv = model(idx, use_cache=True)
            else:
                # Only the newest token goes through the model; earlier ones come from the cache.
                logits, past_kv = model(idx[:, -1:], past_kv=past_kv, use_cache=True)
        else:
            # The model only has learned positions for `context_size` tokens: keep the most recent
            # ones. Once this window starts sliding every position shifts, so the cache is no longer
            # valid and we recompute the window each step (same result as before the cache existed).
            past_kv = None
            logits = model(idx[:, -context_size:])
        logits = logits[:, -1, :]  # only the last position predicts the next token

        if top_k is not None:
            # Mask everything below the k-th largest logit so it can never be chosen.
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1:]
            logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None:
            # Rows that already ended keep emitting EOS (padding).
            idx_next = torch.where(finished.unsqueeze(-1), torch.full_like(idx_next, eos_id), idx_next)
            finished = finished | (idx_next.squeeze(-1) == eos_id)
            if finished.all():
                break

        idx = torch.cat((idx, idx_next), dim=1)
        if stop_fn is not None and stop_fn(idx):
            break

    model.train(was_training)
    return idx
