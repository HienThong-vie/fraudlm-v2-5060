"""
G2 feasibility check for FraudLM v2: can one free Kaggle GPU LoRA-fine-tune GPT-2 XL in time?

Purpose: load OpenAI GPT-2 weights into our own GPTModel, add LoRA adapters, and time short
training runs on dummy batches for several memory settings (base-weight dtype, gradient
checkpointing, micro-batch size, sequence length). For each setting it logs peak GPU memory,
seconds per step and tokens per second, then projects the time of the full fine-tuning run
(N examples x epochs) and picks the fastest setting that fits in memory.

No real training data is used here: the batches are a fixed public-domain-style paragraph
repeated to the sequence length (enough to check that the loss is finite and falls).

How to run
- In this repo (CPU smoke test):
    .venv/Scripts/python kaggle/feasibility.py --size gpt2-small --device cpu --steps 3
- On Kaggle: do not run this file directly. Run `python kaggle/make_bundle.py` and paste the
  generated `kaggle/feasibility_bundle.py` into one notebook cell (see kaggle/RUNBOOK_G2.md).

Author: FraudLM v2 team (COS30049.2-1).
Uses the model, LoRA and weight-loading code that follows Raschka, S. (2024) *Build a Large
Language Model (From Scratch)*, Manning; code repo Apache-2.0.

"""

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# >>> REPO_ONLY (make_bundle.py removes this block: the bundle writes the modules itself)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# <<< REPO_ONLY

from training_model.gpt_model import GPT_CONFIGS  # noqa: E402
from training_model.lora import apply_lora, lora_state_dict, load_lora  # noqa: E402
from utils.load_gpt2_weights import load_pretrained_gpt2  # noqa: E402


# =====================================================================================
# 1. Settings (edit here, or override with command-line flags)
# =====================================================================================

SIZE = "gpt2-xl"                 # model to test first
# PLAN D2 fallback order: if a size misses the budget, the next smaller one is tried.
SIZE_ORDER = ["gpt2-xl", "gpt2-large", "gpt2-medium"]

# Projection of the real fine-tuning run (data-builder will give the final numbers).
N_TRAIN = 12_000                 # training examples (dataset v1, G3)
MEAN_TOKENS = 200                # mean prompt + response tokens (dataset v1 measured 198.4)
EPOCHS = 2
PAD_OVERHEAD = 1.15              # extra tokens from padding inside length-bucketed batches
EVAL_OVERHEAD = 1.10             # val-loss passes, sample generations and checkpoint saves
BUDGET_HOURS = 5.0               # PLAN G2 rule: projected run <= 5 h on XL, else drop a size

# What to measure.
MODES = ["fp16-base", "fp32-base"]   # dtype of the frozen base weights (LoRA is always fp32)
CHECKPOINTING = [True, False]        # gradient checkpointing on / off
MICRO_BATCHES = [1, 2, 4]
SEQ_LEN = 512                        # main grid
LONG_SEQ_LEN = 1024                  # a few extra points at the full context length
STEPS = 20                           # timed steps per setting
WARMUP = 2                           # untimed steps first (CUDA kernel selection, allocator)

# LoRA and optimiser as planned for G4.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = ("attn", "ff")
LR = 2e-4                        # modest: our A init is ~10x PEFT's (G1 review N6)
GRAD_CLIP = 1.0
SEED = 42

# fp16 base weights are accepted only if the sanity loss with them is this close to the fp32-base loss
# (G4 review R8). Otherwise the frozen base stays fp32 (autocast still runs the matmuls in fp16).
FP16_MAX_LOSS_DIFF = 0.05

# A setting "fits" only if its reserved memory stays under this share of the GPU, leaving room
# for allocator fragmentation and the odd longer batch in the real run.
FIT_FRACTION = 0.90

# Fixed text used for the dummy batches and for the quick loss / generation sanity check.
SAMPLE_TEXT = (
    "A fraud analyst reviews new bank account applications every day. Most applications are "
    "genuine, but a small share are opened with stolen or synthetic identities. The analyst "
    "looks at how quickly the applicant filled in the form, whether the email address looks "
    "new, how many applications came from the same device, and whether the phone number is "
    "valid. A screening model gives each application a score, and only the highest scores "
    "are sent to a person for a closer look. The person, not the model, makes the decision."
)


# =====================================================================================
# 2. Small helpers
# =====================================================================================

def on_kaggle():
    return Path("/kaggle").exists()


def gb(n_bytes):
    return n_bytes / 1024**3


def cpu_peak_rss_gb():
    """Peak resident memory of this process (Linux only; Kaggle runs Linux)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # KB -> GB
    except Exception:
        return float("nan")


def free_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def check_gpu(device):
    """Print the GPU and fail early with a clear message if PyTorch cannot use it."""
    if device.type != "cuda":
        print("[gpu] running on CPU (smoke test only; memory numbers are not measured)")
        return None
    if not torch.cuda.is_available():
        sys.exit("[gpu] CUDA is not available. In Kaggle: Settings -> Accelerator -> GPU T4 x2.")
    props = torch.cuda.get_device_properties(device)
    print(f"[gpu] {torch.cuda.device_count()} GPU(s); using {props.name}, "
          f"{gb(props.total_memory):.1f} GB, compute capability {props.major}.{props.minor}")
    try:
        # A tiny fp16 matmul: some PyTorch builds no longer ship kernels for old GPUs (P100 = 6.0).
        a = torch.randn(64, 64, device=device, dtype=torch.float16)
        (a @ a).sum().item()
    except RuntimeError as e:
        sys.exit(f"[gpu] this PyTorch build cannot run on {props.name}: {e}\n"
                 "Switch Accelerator to 'GPU T4 x2' and run again.")
    return props.total_memory


def dummy_batch(tokens, micro_batch, seq_len, device):
    """Inputs and next-token targets of shape [micro_batch, seq_len] from the repeated sample text."""
    reps = math.ceil((seq_len + 1) / len(tokens))
    row = torch.tensor((tokens * reps)[: seq_len + 1], dtype=torch.long)
    rows = row.unsqueeze(0).repeat(micro_batch, 1).to(device)
    return rows[:, :-1], rows[:, 1:]


def get_tokens():
    """GPT-2 token ids of the sample text (tiktoken is the tokenizer we use everywhere, PLAN D4)."""
    import tiktoken
    return tiktoken.get_encoding("gpt2").encode(SAMPLE_TEXT)


@torch.no_grad()
def sanity_check(model, tokens, device, amp_dtype):
    """Loss on the sample text and a short greedy continuation: proves the weights loaded correctly
    in this dtype (fp16 overflow would show up here as inf/NaN, a higher loss or nonsense text).
    Also returns the dtype of block 1's output under autocast, i.e. whether the residual stream
    runs in fp16 or fp32 (G4 review: this decides how exposed LayerNorm is to fp16 overflow)."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    seen = {}

    def remember_dtype(module, inputs, output):
        seen["dtype"] = output.dtype  # returns None, so the block output is left unchanged

    hook = model.trf_blocks[1].register_forward_hook(remember_dtype)
    x = torch.tensor(tokens, device=device).unsqueeze(0)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits = model(x[:, :-1])
    hook.remove()
    loss = F.cross_entropy(logits.float().flatten(0, 1), x[:, 1:].flatten()).item()

    idx = torch.tensor(enc.encode("Every effort moves you"), device=device).unsqueeze(0)
    for _ in range(15):  # plain greedy decoding, enough to see coherent text
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            nxt = model(idx)[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    text = enc.decode(idx[0].tolist()).replace("\n", " ")
    return loss, text, str(seen.get("dtype")).replace("torch.", "")


# =====================================================================================
# 3. One measurement: STEPS LoRA training steps for one setting
# =====================================================================================

def measure(model, init_lora, tokens, device, amp_dtype, ckpt, micro_batch, seq_len, steps, warmup):
    """Run warmup + steps micro-steps (forward, plain next-token loss, backward, AdamW) and time them.

    Each micro-step is one forward + backward on [micro_batch, seq_len] tokens followed by an
    optimiser step. Gradient accumulation does not change the time per token, so the projection
    below works for any effective batch size.
    """
    load_lora(model, init_lora)          # same fresh adapter for every setting -> comparable losses
    model.gradient_checkpointing = ckpt
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    # GradScaler multiplies the loss so small fp16 gradients do not round to zero; it is only
    # needed (and only enabled) when autocast runs in fp16 on the GPU.
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))
    x, y = dummy_batch(tokens, micro_batch, seq_len, device)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    losses, times = [], []
    for step in range(warmup + steps):
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        # Loss in fp32 (G1 review): fp16 log-softmax over 50,257 classes loses precision.
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)             # back to true gradient size before clipping
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()      # GPU work is asynchronous; wait before reading the clock
        dt = time.perf_counter() - t0
        losses.append(loss.item())
        if step >= warmup:
            times.append(dt)
        del logits, loss

    s_per_step = sum(times) / len(times)
    peak_alloc = gb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else float("nan")
    peak_reserved = gb(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else float("nan")
    del opt, scaler, x, y
    return {
        "s_per_step": s_per_step,
        "tokens_per_s": micro_batch * seq_len / s_per_step,
        "peak_alloc_gb": peak_alloc,
        "peak_reserved_gb": peak_reserved,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "finite": all(math.isfinite(v) for v in losses),
    }


def projected_hours(tokens_per_s, args):
    total_tokens = args.n_train * args.epochs * args.mean_tokens * PAD_OVERHEAD
    return total_tokens / tokens_per_s / 3600 * EVAL_OVERHEAD


def run_setting(model, init_lora, tokens, device, amp_dtype, row, args):
    """Wrap measure() so an out-of-memory error is recorded as a result instead of a crash."""
    print(f"  - {row['mode']:9s} ckpt={'on ' if row['ckpt'] else 'off'} mb={row['micro_batch']} "
          f"seq={row['seq_len']:4d} ... ", end="", flush=True)
    status, result = "ok", {}
    try:
        result = measure(model, init_lora, tokens, device, amp_dtype, row["ckpt"],
                         row["micro_batch"], row["seq_len"], args.steps, args.warmup)
        if not result["finite"]:
            status = "NaN"
    except torch.cuda.OutOfMemoryError:
        status = "OOM"
    except RuntimeError as e:  # some OOMs surface as plain RuntimeError
        if "out of memory" not in str(e).lower():
            raise
        status = "OOM"
    # Leave the except block before freeing: the traceback holds references to the big tensors.
    for p in model.parameters():
        p.grad = None
    free_memory(device)
    row.update(result)
    row["status"] = status
    if status == "ok":
        row["projected_h"] = projected_hours(row["tokens_per_s"], args)
        print(f"{row['s_per_step']:.3f} s/step, {row['tokens_per_s']:,.0f} tok/s, "
              f"peak {row['peak_reserved_gb']:.2f} GB, loss {row['loss_first']:.2f}->{row['loss_last']:.2f}, "
              f"projected {row['projected_h']:.2f} h")
    else:
        print(status)
    return row


# =====================================================================================
# 4. All settings for one model size
# =====================================================================================

def run_size(size, args, device, gpu_total):
    """Sanity-check both base dtypes, then time every setting of the accepted dtype(s).

    fp32 base is always loaded first: its sanity loss is the reference for the fp16 acceptance check
    (G4 review R8). fp16 base is accepted only if its loss differs from fp32's by < FP16_MAX_LOSS_DIFF.
    If fp16 is rejected, fp32-base settings are timed instead.
    """
    print(f"\n=== {size} ===")
    tokens = get_tokens()
    rows = []
    check = {"size": size, "sanity_fp32": None, "sanity_fp16": None, "diff": None, "fp16_ok": None,
             "residual_dtype_fp32": None, "residual_dtype_fp16": None}
    # Autocast fp16 only on the GPU (T4 / P100 have no bf16). On CPU we run plain precision.
    amp_dtype = torch.float16 if device.type == "cuda" else None

    def load(mode):
        free_memory(device)
        t0 = time.perf_counter()
        # Dtype rule (PLAN 3): load the base in its final dtype BEFORE apply_lora, never cast after.
        base_dtype = torch.float16 if mode == "fp16-base" else torch.float32
        model = load_pretrained_gpt2(size, cache_dir=args.cache_dir, device=str(device), dtype=base_dtype)
        load_s = time.perf_counter() - t0
        on_gpu = f"{gb(torch.cuda.memory_allocated(device)):.2f} GB" if device.type == "cuda" else "n/a (CPU)"
        print(f"[load] {size} {mode}: {load_s:.0f} s, weights on GPU {on_gpu}, "
              f"CPU peak RSS so far {cpu_peak_rss_gb():.1f} GB")
        loss, text, res_dtype = sanity_check(model, tokens, device, amp_dtype)
        print(f"[sanity] {mode}: loss on sample text {loss:.4f} | block output dtype {res_dtype} | "
              f"greedy: {text!r}")
        key = "fp16" if mode == "fp16-base" else "fp32"
        check[f"sanity_{key}"], check[f"residual_dtype_{key}"] = loss, res_dtype
        return model, load_s, loss

    def time_mode(model, mode, load_s, loss):
        torch.manual_seed(SEED)
        trainable = apply_lora(model, LORA_RANK, LORA_ALPHA, LORA_TARGETS)
        lora_dtypes = {p.dtype for p in model.parameters() if p.requires_grad}
        assert lora_dtypes == {torch.float32}, f"LoRA params must stay fp32, got {lora_dtypes}"
        init_lora = lora_state_dict(model)
        mode_rows = []
        for ckpt in args.ckpt:
            for mb in args.micro_batches:
                row = {"size": size, "mode": mode, "ckpt": ckpt, "micro_batch": mb, "seq_len": args.seq_len,
                       "load_s": load_s, "sanity_loss": loss, "trainable": trainable}
                mode_rows.append(run_setting(model, init_lora, tokens, device, amp_dtype, row, args))
        # A few points at the full 1,024 context for the best setting of this mode: the real data
        # has some long examples, so the chosen batch must survive them.
        ok = [r for r in mode_rows if r["status"] == "ok" and fits(r, gpu_total)]
        if ok and args.long_seq_len:
            top = max(ok, key=lambda r: r["tokens_per_s"])
            long_points = {(True, 1), (top["ckpt"], max(1, top["micro_batch"] // 2))}
            for ckpt, mb in sorted(long_points):
                row = {"size": size, "mode": mode, "ckpt": ckpt, "micro_batch": mb, "seq_len": args.long_seq_len,
                       "load_s": load_s, "sanity_loss": loss, "trainable": trainable}
                mode_rows.append(run_setting(model, init_lora, tokens, device, amp_dtype, row, args))
        return mode_rows

    # 1) fp32 base: always loaded, it is the reference loss.
    model, load_s, loss32 = load("fp32-base")
    fp32_timed = "fp32-base" in args.modes
    if fp32_timed:
        rows += time_mode(model, "fp32-base", load_s, loss32)
    del model
    free_memory(device)

    # 2) fp16 base: accepted only if its loss matches fp32's.
    if "fp16-base" in args.modes:
        model, load_s, loss16 = load("fp16-base")
        diff = abs(loss16 - loss32) if math.isfinite(loss16) else float("inf")
        check["diff"], check["fp16_ok"] = diff, diff < FP16_MAX_LOSS_DIFF
        verdict = "ACCEPTED" if check["fp16_ok"] else "REJECTED (use base fp32)"
        print(f"[fp16 check] {size}: fp32 base {loss32:.4f} vs fp16 base {loss16:.4f} -> |diff| {diff:.4f} "
              f"(limit {FP16_MAX_LOSS_DIFF}) -> fp16 base {verdict}")
        if check["fp16_ok"]:
            rows += time_mode(model, "fp16-base", load_s, loss16)
        del model
        free_memory(device)
        if not check["fp16_ok"] and not fp32_timed:
            model, load_s, loss32 = load("fp32-base")
            rows += time_mode(model, "fp32-base", load_s, loss32)
            del model
            free_memory(device)
    return rows, check


def fits(row, gpu_total):
    if row["status"] != "ok":
        return False
    if gpu_total is None:  # CPU smoke test: no memory limit to check
        return True
    return row["peak_reserved_gb"] <= FIT_FRACTION * gb(gpu_total)


def pick_best(rows, gpu_total, seq_len):
    ok = [r for r in rows if r["seq_len"] == seq_len and fits(r, gpu_total)]
    return max(ok, key=lambda r: r["tokens_per_s"]) if ok else None


# =====================================================================================
# 5. Report
# =====================================================================================

def print_table(rows, gpu_total):
    head = ("| size | base | ckpt | micro-batch | seq | status | peak GB | s/step | tokens/s | "
            "loss first->last | projected h | fits |")
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for r in rows:
        if r["status"] == "ok":
            print(f"| {r['size']} | {r['mode']} | {'on' if r['ckpt'] else 'off'} | {r['micro_batch']} | "
                  f"{r['seq_len']} | ok | {r['peak_reserved_gb']:.2f} | {r['s_per_step']:.3f} | "
                  f"{r['tokens_per_s']:,.0f} | {r['loss_first']:.2f}->{r['loss_last']:.2f} | "
                  f"{r['projected_h']:.2f} | {'yes' if fits(r, gpu_total) else 'no'} |")
        else:
            print(f"| {r['size']} | {r['mode']} | {'on' if r['ckpt'] else 'off'} | {r['micro_batch']} | "
                  f"{r['seq_len']} | {r['status']} | - | - | - | - | - | no |")


def parse_args():
    p = argparse.ArgumentParser(description="FraudLM v2 G2 Kaggle feasibility check")
    p.add_argument("--size", default=SIZE, choices=list(GPT_CONFIGS))
    p.add_argument("--no_fallback", action="store_true", help="do not try smaller sizes if over budget")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--modes", default=None,
                   help="comma list of fp16-base,fp32-base (default: both on GPU; fp32-base only on CPU, "
                        "because fp16 matmuls on a CPU are ~20x slower and tell us nothing)")
    p.add_argument("--ckpt", default=",".join("on" if c else "off" for c in CHECKPOINTING), help="on,off")
    p.add_argument("--micro_batches", default=",".join(map(str, MICRO_BATCHES)))
    p.add_argument("--seq_len", type=int, default=SEQ_LEN)
    p.add_argument("--long_seq_len", type=int, default=LONG_SEQ_LEN, help="0 = skip the long-context points")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--n_train", type=int, default=N_TRAIN)
    p.add_argument("--mean_tokens", type=int, default=MEAN_TOKENS)
    p.add_argument("--epochs", type=float, default=EPOCHS)
    p.add_argument("--budget_hours", type=float, default=BUDGET_HOURS)
    # On Kaggle keep the multi-GB weight files in /tmp so they never end up in the notebook output.
    p.add_argument("--cache_dir", default="/tmp/fraudlm_pretrained" if on_kaggle() else "checkpoint/pretrained")
    p.add_argument("--out_dir", default="/kaggle/working" if on_kaggle() else ".")
    # parse_known_args: a Jupyter kernel passes its own "-f kernel.json" argument; ignore it.
    args, _ = p.parse_known_args()
    if args.modes is None:
        args.modes = ",".join(MODES) if args.device.startswith("cuda") else "fp32-base"
    args.modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in args.modes if m not in ("fp16-base", "fp32-base")]
    if bad:
        p.error(f"unknown mode(s) {bad}")
    args.ckpt = [c.strip() == "on" for c in args.ckpt.split(",") if c.strip()]
    args.micro_batches = [int(m) for m in args.micro_batches.split(",") if m.strip()]
    return args


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    print(f"[env] python {platform.python_version()}, torch {torch.__version__}, "
          f"CUDA {torch.version.cuda}, device {device}, kaggle={on_kaggle()}")
    print(f"[plan] projection: {args.n_train:,} examples x {args.epochs:g} epochs x {args.mean_tokens} tokens "
          f"(x{PAD_OVERHEAD} padding, x{EVAL_OVERHEAD} eval); budget {args.budget_hours:g} h")
    gpu_total = check_gpu(device)
    t_start = time.perf_counter()

    smaller = SIZE_ORDER[SIZE_ORDER.index(args.size) + 1:] if args.size in SIZE_ORDER else []
    sizes = [args.size] + ([] if args.no_fallback else smaller)
    all_rows, checks, decision = [], [], None
    for size in sizes:
        rows, check = run_size(size, args, device, gpu_total)
        all_rows += rows
        checks.append(check)
        best = pick_best(rows, gpu_total, args.seq_len)
        if best and best["projected_h"] <= args.budget_hours:
            decision = best
            break
        why = "nothing fits in memory" if best is None else f"best projection {best['projected_h']:.2f} h"
        last = size == sizes[-1]
        print(f"[decision] {size}: {why}, over the {args.budget_hours:g} h budget"
              + ("" if last else " -> trying the next smaller size"))
        if last:
            decision = best  # report the best we have, even if it misses the budget
            break

    # ---- Summary for the lead (copy everything from here to the end) ----
    print("\n" + "=" * 100)
    print("G2 FEASIBILITY SUMMARY (copy from here to the end and send it to the lead)")
    print("=" * 100)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}, {gb(props.total_memory):.1f} GB, {torch.cuda.device_count()} visible; "
              f"torch {torch.__version__}, CUDA {torch.version.cuda}")
    else:
        print(f"Device: CPU (smoke test), torch {torch.__version__}")
    print(f"CPU peak RSS: {cpu_peak_rss_gb():.1f} GB; total wall time {(time.perf_counter() - t_start) / 60:.1f} min")
    print(f"LoRA r={LORA_RANK} alpha={LORA_ALPHA} targets={LORA_TARGETS}, lr {LR}, AdamW, clip {GRAD_CLIP}, "
          f"{args.steps} timed steps after {args.warmup} warm-up")
    print(f"Projection: {args.n_train:,} ex x {args.epochs:g} ep x {args.mean_tokens} tok "
          f"x{PAD_OVERHEAD} pad x{EVAL_OVERHEAD} eval\n")
    for c in checks:
        line = f"{c['size']}: sanity loss fp32 base {c['sanity_fp32']:.4f} (block output {c['residual_dtype_fp32']})"
        if c["sanity_fp16"] is not None:
            line += (f", fp16 base {c['sanity_fp16']:.4f} (block output {c['residual_dtype_fp16']}), "
                     f"|diff| {c['diff']:.4f} -> fp16 base {'ACCEPTED' if c['fp16_ok'] else 'REJECTED'} "
                     f"(limit {FP16_MAX_LOSS_DIFF})")
        print(line)
    r0 = next((r for r in all_rows if "trainable" in r), None)
    if r0:
        print(f"LoRA trainable params: {r0['trainable']:,}")
    print()
    print_table(all_rows, gpu_total)
    print()
    if decision:
        d = decision
        verdict = "MEETS" if d["projected_h"] <= args.budget_hours else "MISSES"
        print(f"RECOMMENDED: {d['size']} | base {d['mode']} | grad checkpointing {'on' if d['ckpt'] else 'off'} | "
              f"micro-batch {d['micro_batch']} at seq {d['seq_len']} (token budget ~{d['micro_batch'] * d['seq_len']} "
              f"tokens per micro-batch) | {d['tokens_per_s']:,.0f} tok/s | peak {d['peak_reserved_gb']:.2f} GB | "
              f"projected {d['projected_h']:.2f} h -> {verdict} the {args.budget_hours:g} h budget")
    else:
        print("RECOMMENDED: none - no setting fitted in memory. Send this whole output to the lead.")
    print("=" * 100)

    out = Path(args.out_dir) / "g2_feasibility_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": all_rows, "fp16_checks": checks, "recommended": decision}, indent=2, default=str))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
