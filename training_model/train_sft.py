"""
LoRA instruction fine-tuning of GPT-2 for FraudLM v2 (Stage 3 of the Raschka pipeline).

Purpose: load OpenAI GPT-2 weights into our GPTModel, freeze them, add LoRA adapters and train only
the adapters on the FraudLM Alpaca dataset with a masked loss (the model learns the responses, not
the prompts). Built to finish on one free Kaggle GPU (T4 / P100, 16 GB, no bf16):
- frozen base stored in fp16, LoRA params + AdamW states in fp32, fp16 autocast + GradScaler
- gradient checkpointing, micro-batches + gradient accumulation, length-bucketed batches
- cosine learning rate with warm-up, gradient clipping 1.0, seed 42
- val loss every N steps on a fixed val subset + 3 fixed sample generations -> samples.txt
- checkpoint (LoRA adapter + optimiser/scaler/step state, never the base) every N steps;
  --resume continues exactly where it stopped; --max_hours saves and exits cleanly before Kaggle's
  12 h limit; stops with a clear message if the loss becomes NaN.

Outputs in <out_dir>/<run_id>/ (PLAN Section 3): lora_adapter.pt, run_config.json, train_log.csv
(step, epoch, train_loss, val_loss, lr, tokens_seen, elapsed_s), samples.txt; plus train_state.pt
(resume state), lora_adapter_best.pt (lowest val-subset loss) and run_summary.json at the end.

Usage
  CPU smoke test:  .venv/Scripts/python training_model/train_sft.py --size gpt2-small --max_train_rows 32 \\
                     --max_steps 10 --micro_batch 2 --grad_accum 1 --eval_every 5 --run_id smoke
  Resume:          same command + --resume            (or --resume <checkpoint dir>)
  Kaggle:          kaggle/make_bundle.py -> kaggle/train_bundle.py, see kaggle/RUNBOOK_G4.md
  Config file:     --config my_run.json  (keys = the names in DEFAULTS; command-line flags win)

Author: FraudLM v2 team (COS30049.2-1).
Training loop follows Raschka, S. (2024) *Build a Large Language Model (From Scratch)*, Manning,
chapters 5 and 7 and appendices D-E (warm-up, cosine decay, clipping, LoRA); code repo Apache-2.0.
Written with help from Claude Code (Anthropic); reviewed by Thong.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_PROJECT_ROOT = None  # set below when run from the repo; stays None inside the Kaggle bundle
# >>> REPO_ONLY (make_bundle.py removes this block: the bundle writes the modules itself)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# <<< REPO_ONLY

from training_model.gpt_model import GPT_CONFIGS  # noqa: E402
from training_model.lora import apply_lora, load_lora, lora_state_dict  # noqa: E402
from training_model.sft_data import (  # noqa: E402
    IGNORE_INDEX, SFTDataset, collate, length_bucketed_batches, load_split,
)
from utils.generate import generate, ids_to_text_until_eos, get_tokenizer  # noqa: E402
from utils.load_gpt2_weights import load_pretrained_gpt2  # noqa: E402
from utils.prompt import format_prompt  # noqa: E402


def on_kaggle():
    return Path("/kaggle").exists()


def in_kaggle_kernel():
    """True inside a Kaggle session (script or notebook kernel). Used only for the exit code (see finish)."""
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle/working").exists()


# =====================================================================================
# 1. Default settings. The lead fills the memory/speed values from the G2 feasibility run.
# =====================================================================================

DEFAULTS = {
    # model and LoRA
    "size": "gpt2-xl",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_targets": "attn,ff",
    # precision and memory (G2 decides; T4 / P100 have no bf16)
    "base_dtype": "fp16",          # frozen base weights: fp16 (half the memory) or fp32
    "amp": True,                   # fp16 autocast + GradScaler on the GPU (ignored on CPU)
    "grad_ckpt": True,             # gradient checkpointing
    "micro_batch": 4,
    "grad_accum": 2,               # effective batch = micro_batch x grad_accum = 8 examples
    "max_len": 512,                # token cap per example (dataset v1.1 max is 336)
    # optimisation
    "epochs": 2,
    "lr": 1e-4,                    # modest: our LoRA A init is ~10x PEFT's (G1 review N6)
    "min_lr_ratio": 0.1,           # cosine decays to 10% of the peak
    "warmup_frac": 0.03,
    "weight_decay": 0.0,           # LoRA adapters are small; no decay needed
    "clip": 1.0,
    "seed": 42,
    "deterministic": False,
    # evaluation, logging, checkpoints
    "eval_every": 200,             # optimiser steps
    "eval_rows": 200,              # fixed val subset for the periodic val loss
    "eval_at_start": True,         # val loss + samples of the untrained model (baseline for the report)
    "save_every": 200,
    "log_every": 10,
    "sample_tokens": 120,          # greedy tokens per sample generation
    "max_hours": 11.0,             # stop training, save and exit after this long (Kaggle kills at 12 h)
    "nan_tolerance": 3,            # consecutive non-finite losses before stopping
    # run identity and paths
    "run_id": "",                  # "" = derived from size and rank, e.g. "xl_r16_v1" or "large_r16_v1"
    "data_dir": "auto",            # folder with fraudlm_sft_{train,val}.json; auto = repo or /kaggle/input
    "out_dir": "/kaggle/working/checkpoint" if on_kaggle() else "checkpoint",
    "cache_dir": "/tmp/fraudlm_pretrained" if on_kaggle() else "checkpoint/pretrained",
    "resume": "auto" if on_kaggle() else "",  # "" = fresh run; "auto" = continue if a checkpoint exists
    # Start a NEW run from a trained LoRA adapter. Unlike --resume, this intentionally creates a
    # new optimiser and schedule so the adapter can be continued on a different dataset.
    "init_adapter": "",
    # limits for smoke tests (0 = no limit)
    "max_steps": 0,
    "max_train_rows": 0,
    "max_val_rows": 0,
}

# Settings that must not change when a run is resumed (they change the maths or the batch order).
RESUME_LOCKED = ["size", "lora_rank", "lora_alpha", "lora_targets", "base_dtype", "micro_batch",
                 "grad_accum", "max_len", "epochs", "lr", "min_lr_ratio", "warmup_frac", "weight_decay",
                 "seed", "max_steps", "max_train_rows", "data_sha256", "init_adapter_sha256"]

# Contract columns (PLAN 3) first, then two extra GradScaler columns (G4 review R7).
LOG_COLUMNS = ["step", "epoch", "train_loss", "val_loss", "lr", "tokens_seen", "elapsed_s",
               "grad_scale", "amp_skipped"]

# sha256[:16] of train + val JSON that this code was bundled with. make_bundle.py replaces the value in
# kaggle/train_bundle.py (G4 review R2), so a Kaggle dataset that differs from the repo stops the run.
# Empty in the repo: there the data is the repo's own dataset/ folder.
EXPECTED_DATA_SHA256 = ""  # BUNDLE_FILLS_DATA_SHA

# A pretrained GPT-2 scores about 2-3 on our val set before fine-tuning (gpt2-small smoke: 2.60).
# Higher means the base weights did not load correctly (e.g. fp16 overflow): stop at once (R3).
START_VAL_LOSS_MAX = 5.0
SAMPLE_TASKS = ["explain_score", "policy", "banking_qa"]  # one fixed val prompt per task


class SimulatedKill(Exception):
    """Raised by the test hook --debug_crash_at_step: stops the run like a killed process (no save)."""


# =====================================================================================
# 2. Config, paths and checkpoint helpers
# =====================================================================================

def _to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected on/off, got {v!r}")


def parse_config(argv=None):
    """DEFAULTS <- --config JSON file <- command-line flags (each later one wins)."""
    p = argparse.ArgumentParser(description="FraudLM v2 LoRA instruction fine-tuning")
    p.add_argument("--config", default="", help="JSON file with any of the DEFAULTS keys")
    for key, val in DEFAULTS.items():
        if key == "resume":
            # `--resume` alone means "auto"; `--resume <dir>` resumes from that checkpoint folder.
            p.add_argument("--resume", nargs="?", const="auto", default=None)
            continue
        kind = _to_bool if isinstance(val, bool) else type(val)
        p.add_argument(f"--{key}", type=kind, default=None)
    p.add_argument("--force_resume", action="store_true", help="resume even if locked settings changed")
    p.add_argument("--allow_data_mismatch", action="store_true",
                   help="train even if the data hash differs from the one baked into the bundle")
    p.add_argument("--debug_crash_at_step", type=int, default=0, help=argparse.SUPPRESS)
    # parse_known_args: a Jupyter kernel passes its own "-f kernel.json"; ignore it.
    ns, unknown = p.parse_known_args(argv)
    unknown = [u for u in unknown if u.startswith("--")]
    if unknown:
        print(f"[config] WARNING: ignoring unknown flags {unknown}")

    cfg = dict(DEFAULTS)
    if ns.config:
        file_cfg = json.loads(Path(ns.config).read_text(encoding="utf-8"))
        unknown = sorted(set(file_cfg) - set(DEFAULTS))
        if unknown:
            raise SystemExit(f"[config] unknown keys in {ns.config}: {unknown}")
        cfg.update(file_cfg)
    for key in DEFAULTS:
        v = getattr(ns, key)
        if v is not None:
            cfg[key] = v
    if cfg["size"] not in GPT_CONFIGS:
        raise SystemExit(f"[config] unknown size {cfg['size']!r}; choose from {list(GPT_CONFIGS)}")
    if cfg["base_dtype"] not in ("fp16", "fp32"):
        raise SystemExit("[config] base_dtype must be fp16 or fp32")
    if cfg["resume"] and cfg["init_adapter"]:
        raise SystemExit("[config] --resume and --init_adapter are different operations and cannot be used together")
    if not cfg["run_id"]:
        cfg["run_id"] = f"{cfg['size'].removeprefix('gpt2-')}_r{cfg['lora_rank']}_v1"
    cfg["force_resume"] = ns.force_resume
    cfg["allow_data_mismatch"] = ns.allow_data_mismatch
    cfg["debug_crash_at_step"] = ns.debug_crash_at_step
    return cfg


def find_data_dir(arg):
    """The folder holding fraudlm_sft_train.json. 'auto' looks in the repo, ./dataset, then /kaggle/input."""
    name = "fraudlm_sft_train.json"
    if arg and arg != "auto":
        d = Path(arg)
        if not (d / name).exists():
            raise SystemExit(f"[data] {d / name} not found")
        return d
    candidates = [Path("dataset")]
    if _PROJECT_ROOT is not None:
        candidates.insert(0, _PROJECT_ROOT / "dataset")
    for d in candidates:
        if (d / name).exists():
            return d
    if Path("/kaggle/input").exists():
        hits = sorted(Path("/kaggle/input").rglob(name))
        if hits:
            if len(hits) > 1:
                print(f"[data] several copies found, using the first: {[str(h) for h in hits]}")
            return hits[0].parent
    raise SystemExit("[data] fraudlm_sft_train.json not found. On Kaggle: + Add Input -> Your Datasets -> "
                     "the FraudLM dataset (see kaggle/RUNBOOK_G4.md).")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_initial_adapter(model, adapter_path, cfg):
    """Load a LoRA adapter into a newly initialised run and return its SHA-256 fingerprint.

    The model must already have LoRA applied. A neighbouring run_config.json is checked when present,
    preventing an XL/rank/target mismatch from becoming a hard-to-diagnose training failure.
    """
    path = Path(adapter_path)
    if not path.is_file():
        raise SystemExit(f"[init-adapter] file not found: {path}")
    run_cfg_path = path.parent / "run_config.json"
    if run_cfg_path.exists():
        old = json.loads(run_cfg_path.read_text(encoding="utf-8"))
        expected = {
            "size": cfg["size"],
            "lora_rank": cfg["lora_rank"],
            "lora_alpha": cfg["lora_alpha"],
            "lora_targets": cfg["lora_targets"],
        }
        mismatch = {k: (old.get(k), v) for k, v in expected.items()
                    if old.get(k) is not None and old.get(k) != v}
        if mismatch:
            raise SystemExit(f"[init-adapter] incompatible run_config.json (old, new): {mismatch}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("lora"), dict):
        state = state["lora"]
    if not isinstance(state, dict):
        raise SystemExit(f"[init-adapter] {path} does not contain a LoRA state dictionary")
    load_lora(model, state)
    digest = file_sha256(path)
    print(f"[init-adapter] loaded {path} (sha256 {digest[:16]}) as the starting point for a new run")
    return digest


def data_sha256(train_path, val_path):
    """Short fingerprint of the training data: sha256[:16] of train + "-" + sha256[:16] of val."""
    return file_sha256(train_path)[:16] + "-" + file_sha256(val_path)[:16]


def load_train_state(path):
    """Load train_state.pt on the CPU (G4 review R1).

    torch.set_rng_state / torch.cuda.set_rng_state accept only CPU ByteTensors, so the file must not be
    mapped to the GPU. The optimiser's load_state_dict moves its moments to each parameter's device.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("rng_cpu", "rng_cuda"):
        if state.get(key) is not None:
            state[key] = state[key].cpu().to(torch.uint8)
    return state


def check_start_val_loss(v0):
    """Stop if the untrained model's val loss is not that of a working pretrained GPT-2 (R3)."""
    if not math.isfinite(v0) or v0 > START_VAL_LOSS_MAX:
        raise SystemExit(f"[eval] step-0 val loss {v0} is not sane for pretrained GPT-2 (expected about 2-3, "
                         f"limit {START_VAL_LOSS_MAX}). The base weights did not load correctly in this dtype: "
                         f"try base_dtype fp32. Nothing was trained.")


def atomic_save(obj, path):
    """torch.save to a temp file, then rename: a kill during the write never leaves a broken checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_write_text(text, path):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def find_resume_dir(arg, run_dir, run_id):
    """Where to resume from, or None for a fresh run.

    '' -> fresh. A path -> that folder. 'auto' -> this run's own folder if it holds a checkpoint,
    else (on Kaggle) the most advanced copy of checkpoint/<run_id>/ inside /kaggle/input, i.e. the
    output of an earlier notebook version that was attached as an input.
    """
    if not arg:
        return None
    if arg != "auto":
        d = Path(arg)
        if not (d / "train_state.pt").exists():
            raise SystemExit(f"[resume] no train_state.pt in {d}")
        return d
    if (run_dir / "train_state.pt").exists():
        return run_dir
    best, best_step = None, -1
    if Path("/kaggle/input").exists():
        for state in Path("/kaggle/input").rglob("train_state.pt"):
            if state.parent.name != run_id:
                continue
            prog = state.parent / "progress.json"
            step = json.loads(prog.read_text())["step"] if prog.exists() else 0
            if step > best_step:
                best, best_step = state.parent, step
    return best


def lr_at(step, total_steps, cfg):
    """Linear warm-up to the peak LR, then cosine decay to min_lr_ratio x peak (Raschka appendix D)."""
    peak = cfg["lr"]
    warmup = max(1, round(cfg["warmup_frac"] * total_steps))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return peak * (cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * progress)))


def fmt_time(seconds):
    return str(dt.timedelta(seconds=int(max(0, seconds))))


# =====================================================================================
# 3. Loss, evaluation and samples
# =====================================================================================

def batch_loss(model, x, y, amp_dtype):
    """Mean cross-entropy over the learned (non -100) target tokens of one batch."""
    with torch.autocast(x.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits = model(x)
    # Loss in fp32 (G1 review): a fp16 log-softmax over 50,257 classes loses precision.
    return F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate(model, items, micro_batch, device, amp_dtype):
    """Token-weighted mean loss over `items` (sorted by length so batches have little padding)."""
    was_training = model.training
    model.eval()
    order = sorted(range(len(items)), key=lambda i: len(items[i]["input_ids"]))
    total, n_tok = 0.0, 0
    for s in range(0, len(order), micro_batch):
        x, y = collate([items[i] for i in order[s:s + micro_batch]], device)
        n = int((y != IGNORE_INDEX).sum())
        total += batch_loss(model, x, y, amp_dtype).item() * n
        n_tok += n
    model.train(was_training)
    return total / max(1, n_tok)


@torch.no_grad()
def write_samples(model, sample_rows, path, header, cfg, device, amp_dtype):
    """Greedy answers to the 3 fixed val prompts, appended to samples.txt (reference answer shown too)."""
    enc = get_tokenizer()
    lines = [f"===== {header} ====="]
    for ex in sample_rows:
        prompt = format_prompt(ex["instruction"], ex.get("input", ""))
        idx = torch.tensor([enc.encode(prompt, disallowed_special=())], device=device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = generate(model, idx, max_new_tokens=cfg["sample_tokens"],
                           context_size=model.cfg["context_length"])
        answer = ids_to_text_until_eos(out[0, idx.shape[1]:]).strip()
        lines += [f"--- [{ex['task']}] {ex['id']}",
                  f"INSTRUCTION: {ex['instruction'].strip()[:300]}",
                  f"MODEL: {answer}",
                  f"REFERENCE: {ex['output'].strip()[:400]}", ""]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    model.train()


def pick_samples(val_rows):
    """First val row of each task in SAMPLE_TASKS (fixed, so samples are comparable across evals)."""
    picked = []
    for task in SAMPLE_TASKS:
        row = next((r for r in val_rows if r.get("task") == task), None)
        if row is not None:
            picked.append(row)
    return picked or val_rows[:1]


# =====================================================================================
# 4. Training
# =====================================================================================

def train(cfg):
    """Run (or resume) one fine-tuning run. Returns "done", "time_limit" or "nan"."""
    session_start = time.time()
    if cfg['deterministic']:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    run_dir = Path(cfg["out_dir"]) / cfg["run_id"]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[env] python {platform.python_version()}, torch {torch.__version__}, CUDA {torch.version.cuda}, "
          f"device {device}, kaggle={on_kaggle()}")
    if use_cuda:
        props = torch.cuda.get_device_properties(device)
        print(f"[gpu] {props.name}, {props.total_memory / 1024**3:.1f} GB, capability {props.major}.{props.minor}")

    # ---- data ----
    data_dir = find_data_dir(cfg["data_dir"])
    train_path, val_path = data_dir / "fraudlm_sft_train.json", data_dir / "fraudlm_sft_val.json"
    cfg["data_dir_used"] = str(data_dir)
    cfg["data_sha256"] = data_sha256(train_path, val_path)
    if EXPECTED_DATA_SHA256 and cfg["data_sha256"] != EXPECTED_DATA_SHA256:
        msg = (f"[data] the dataset found in {data_dir} (sha {cfg['data_sha256']}) differs from the one this "
               f"bundle was built with (sha {EXPECTED_DATA_SHA256}). Upload the current dataset/ files as a "
               f"New Version of the Kaggle dataset, or ask the lead to rebuild the bundle.")
        if not cfg["allow_data_mismatch"]:
            raise SystemExit(msg)
        print("WARNING " + msg + " (continuing: --allow_data_mismatch)")
    else:
        print(f"[data] {data_dir}: sha {cfg['data_sha256']}"
              + (" = bundle (OK)" if EXPECTED_DATA_SHA256 else ""))
    train_rows, val_rows = load_split(train_path), load_split(val_path)
    if cfg["max_train_rows"]:
        train_rows = train_rows[: cfg["max_train_rows"]]
    if cfg["max_val_rows"]:
        val_rows = val_rows[: cfg["max_val_rows"]]
    train_ds = SFTDataset(train_rows, cfg["max_len"], "train")
    val_ds = SFTDataset(val_rows, cfg["max_len"], "val")
    # Fixed val subset for the periodic loss: same rows every time, chosen with the run seed.
    sub = sorted(random.Random(cfg["seed"]).sample(range(len(val_ds)), min(cfg["eval_rows"], len(val_ds))))
    val_sub_items = [val_ds.items[i] for i in sub]
    sample_rows = pick_samples(val_ds.rows)
    lengths = [len(it["input_ids"]) for it in train_ds.items]

    batches_per_epoch = math.ceil(len(train_ds) / cfg["micro_batch"])
    steps_per_epoch = math.ceil(batches_per_epoch / cfg["grad_accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    if cfg["max_steps"]:
        total_steps = min(total_steps, cfg["max_steps"])
    cfg["total_steps"] = total_steps
    print(f"[plan] {len(train_ds):,} train examples, micro-batch {cfg['micro_batch']} x accum {cfg['grad_accum']} "
          f"= {cfg['micro_batch'] * cfg['grad_accum']} per step, {steps_per_epoch:,} steps/epoch, "
          f"{total_steps:,} steps total")

    # ---- model: base in its final dtype BEFORE apply_lora (PLAN 3 dtype rule) ----
    base_dtype = torch.float16 if (cfg["base_dtype"] == "fp16" and use_cuda) else torch.float32
    if cfg["base_dtype"] == "fp16" and not use_cuda:
        print("[model] CPU run: base kept in fp32 (fp16 matmuls on a CPU are very slow)")
    amp_dtype = torch.float16 if (cfg["amp"] and use_cuda) else None
    model = load_pretrained_gpt2(cfg["size"], cache_dir=cfg["cache_dir"], device=str(device), dtype=base_dtype)
    torch.manual_seed(cfg["seed"])  # LoRA A init depends on the seed
    targets = tuple(t.strip() for t in cfg["lora_targets"].split(",") if t.strip())
    trainable = apply_lora(model, cfg["lora_rank"], cfg["lora_alpha"], targets)
    if cfg["init_adapter"]:
        cfg["init_adapter_sha256"] = load_initial_adapter(model, cfg["init_adapter"], cfg)
    else:
        cfg["init_adapter_sha256"] = ""
    params = [p for p in model.parameters() if p.requires_grad]
    assert {p.dtype for p in params} == {torch.float32}, "LoRA params must stay fp32"
    model.gradient_checkpointing = cfg["grad_ckpt"]
    cfg["trainable_params"] = trainable

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # GradScaler scales the loss up so small fp16 gradients do not underflow to zero.
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    # ---- fresh run or resume ----
    st = {"step": 0, "epoch": 0, "batch_pos": 0, "tokens_seen": 0, "elapsed_prev": 0.0,
          "best_val": float("inf"), "nan_skipped": 0, "amp_skipped": 0}
    resume_dir = find_resume_dir(cfg["resume"], run_dir, cfg["run_id"])
    if resume_dir is not None:
        old_cfg = json.loads((resume_dir / "run_config.json").read_text(encoding="utf-8"))
        cfg['init_adapter_sha256'] = old_cfg.get('init_adapter_sha256', '')
        cfg['init_adapter'] = old_cfg.get('init_adapter', '')
        changed = {k: (old_cfg.get(k), cfg.get(k)) for k in RESUME_LOCKED if old_cfg.get(k) != cfg.get(k)}
        if changed and not cfg["force_resume"]:
            raise SystemExit(f"[resume] locked settings differ from the saved run (old, new): {changed}. "
                             "Use the same settings, a new --run_id, or --force_resume.")
        if resume_dir.resolve() != run_dir.resolve():
            # e.g. a read-only /kaggle/input copy: continue in our own (writable) output folder
            shutil.copytree(resume_dir, run_dir, dirs_exist_ok=True)
            print(f"[resume] copied {resume_dir} -> {run_dir}")
        state = load_train_state(run_dir / "train_state.pt")
        # The adapter is restored from train_state.pt itself (R5): adapter and optimiser state are written
        # in one file, so they always belong to the same step. lora_adapter.pt is only the deliverable.
        load_lora(model, state["lora"])
        opt.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        st.update(state["progress"])
        torch.set_rng_state(state["rng_cpu"])
        random.setstate(state["rng_py"])
        if use_cuda and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(state["rng_cuda"])
        # Drop log rows written after the checkpoint (they will be recomputed identically).
        _prune_log(run_dir / "train_log.csv", st["step"])
        print(f"[resume] from step {st['step']:,} / {total_steps:,} (epoch {st['epoch']}, batch {st['batch_pos']})")
    else:
        if (run_dir / "train_state.pt").exists():
            # Never overwrite a run by accident: a fresh start needs an empty folder.
            raise SystemExit(f"[run] {run_dir} already holds a checkpoint. Use --resume to continue it, "
                             "or choose a new --run_id.")
        run_dir.mkdir(parents=True, exist_ok=True)
        for f in ("train_log.csv", "samples.txt"):
            (run_dir / f).unlink(missing_ok=True)
    cfg_to_save = {k: v for k, v in cfg.items()
                   if k not in ("force_resume", "allow_data_mismatch", "debug_crash_at_step")}
    atomic_write_text(json.dumps(cfg_to_save, indent=2), run_dir / "run_config.json")

    log_path = run_dir / "train_log.csv"
    new_log = not log_path.exists()
    log_f = open(log_path, "a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(LOG_COLUMNS)

    def elapsed():
        return st["elapsed_prev"] + (time.time() - session_start)

    def save_checkpoint(tag=""):
        # train_state.pt (the resume source) holds the adapter too and is written in one atomic rename,
        # so a kill at any moment leaves a consistent (adapter, optimiser, step) triple (R5).
        lora = lora_state_dict(model)
        atomic_save({
            "lora": lora,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "progress": {**st, "elapsed_prev": elapsed()},
            "rng_cpu": torch.get_rng_state(),
            "rng_py": random.getstate(),
            "rng_cuda": torch.cuda.get_rng_state() if use_cuda else None,
        }, run_dir / "train_state.pt")
        atomic_save(lora, run_dir / "lora_adapter.pt")  # the deliverable (adapter only, a few dozen MB)
        atomic_write_text(json.dumps({"step": st["step"], "total_steps": total_steps,
                                      "saved_at": dt.datetime.now().isoformat(timespec="seconds")}),
                          run_dir / "progress.json")
        print(f"[save] step {st['step']:,} -> {run_dir}{' (' + tag + ')' if tag else ''}")

    def run_eval(header):
        v = evaluate(model, val_sub_items, cfg["micro_batch"], device, amp_dtype)
        write_samples(model, sample_rows, run_dir / "samples.txt", f"{header} | val_subset_loss {v:.4f}",
                      cfg, device, amp_dtype)
        if v < st["best_val"]:
            st["best_val"] = v
            atomic_save(lora_state_dict(model), run_dir / "lora_adapter_best.pt")
        return v

    if st["step"] == 0 and cfg["eval_at_start"]:
        v0 = run_eval("step 0 (before fine-tuning)")
        log.writerow([0, 0.0, "", f"{v0:.4f}", 0.0, 0, f"{elapsed():.1f}", f"{scaler.get_scale():.0f}", 0])
        log_f.flush()
        starting_model = "continued FraudLM adapter" if cfg["init_adapter"] else "untrained LoRA = pretrained GPT-2"
        print(f"[eval] step 0: val subset loss {v0:.4f} ({starting_model})")
        try:
            check_start_val_loss(v0)
        except SystemExit:
            log_f.close()
            raise

    # ---- the loop ----
    model.train()
    batches = length_bucketed_batches(lengths, cfg["micro_batch"], cfg["seed"] + st["epoch"])
    win_loss, win_n, consecutive_nan = 0.0, 0, 0
    session_steps, session_t0 = 0, time.time()
    status = "done"
    while st["step"] < total_steps:
        # Time guard first: leave enough time to save before Kaggle stops the session.
        if time.time() - session_start > cfg["max_hours"] * 3600:
            save_checkpoint("time limit")
            print(f"[time] --max_hours {cfg['max_hours']} reached at step {st['step']:,}/{total_steps:,}. "
                  f"Saved. Run again with --resume (Kaggle: attach this version's output, see RUNBOOK_G4.md).")
            status = "time_limit"
            break
        if st["batch_pos"] >= len(batches):  # next epoch: new (seeded) batch order
            st["epoch"] += 1
            st["batch_pos"] = 0
            batches = length_bucketed_batches(lengths, cfg["micro_batch"], cfg["seed"] + st["epoch"])

        lr = lr_at(st["step"], total_steps, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        group = batches[st["batch_pos"]: st["batch_pos"] + cfg["grad_accum"]]
        group_loss, bad = 0.0, False
        for b in group:
            items = [train_ds.items[i] for i in b]
            x, y = collate(items, device)
            loss = batch_loss(model, x, y, amp_dtype)
            if not torch.isfinite(loss):
                bad = True
                break
            # Accumulation (R6): `loss` is the token-mean of this micro-batch; dividing by the number of
            # micro-batches makes the summed gradient the average of the micro-batch means. With length
            # buckets, a token in a short micro-batch therefore weighs a bit more than one in a long
            # micro-batch (Raschka ch. 7 averages per batch the same way). Accepted for simplicity.
            scaler.scale(loss / len(group)).backward()
            group_loss += loss.item() / len(group)
            st["tokens_seen"] += sum(len(it["input_ids"]) for it in items)

        if bad:
            opt.zero_grad(set_to_none=True)
            consecutive_nan += 1
            st["nan_skipped"] += 1
            print(f"[nan] non-finite loss at step {st['step']:,}; batch skipped ({consecutive_nan} in a row)")
            if consecutive_nan > cfg["nan_tolerance"]:
                print(f"[nan] STOPPED: {consecutive_nan} non-finite losses in a row at step {st['step']:,}. "
                      f"The last good checkpoint (step saved earlier) is kept in {run_dir}. Try base_dtype fp32 "
                      f"or a lower lr.")
                # Return instead of raising: a Kaggle version that ends cleanly keeps /kaggle/working, so the
                # last good checkpoint can still be downloaded. The banner above is the loud signal.
                status = "nan"
                break
        else:
            consecutive_nan = 0
            scaler.unscale_(opt)  # true gradient size before clipping
            torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
            scale_before = scaler.get_scale()
            scaler.step(opt)      # skipped automatically if fp16 gradients overflowed
            scaler.update()
            if scaler.get_scale() < scale_before:  # the scale only drops when the step was skipped
                st["amp_skipped"] += 1
            opt.zero_grad(set_to_none=True)
            win_loss += group_loss
            win_n += 1

        st["step"] += 1
        st["batch_pos"] += len(group)
        session_steps += 1
        epoch_f = st["epoch"] + st["batch_pos"] / len(batches)

        if cfg["debug_crash_at_step"] and st["step"] == cfg["debug_crash_at_step"]:
            log_f.close()
            raise SimulatedKill(f"simulated kill after step {st['step']}")

        is_eval = st["step"] % cfg["eval_every"] == 0 or st["step"] == total_steps
        if st["step"] % cfg["log_every"] == 0 or is_eval:
            train_loss = win_loss / win_n if win_n else float("nan")
            val_loss = run_eval(f"step {st['step']} (epoch {epoch_f:.2f})") if is_eval else None
            log.writerow([st["step"], f"{epoch_f:.4f}", f"{train_loss:.4f}",
                          "" if val_loss is None else f"{val_loss:.4f}", f"{lr:.3e}", st["tokens_seen"],
                          f"{elapsed():.1f}", f"{scaler.get_scale():.0f}", st["amp_skipped"]])
            log_f.flush()
            per_step = (time.time() - session_t0) / session_steps
            msg = (f"step {st['step']:>6,}/{total_steps:,} | ep {epoch_f:.2f} | loss {train_loss:.4f} | "
                   f"lr {lr:.2e} | {per_step:.2f} s/step | elapsed {fmt_time(elapsed())} | "
                   f"ETA {fmt_time(per_step * (total_steps - st['step']))}")
            if val_loss is not None:
                msg += f" | val {val_loss:.4f}"
            if scaler.is_enabled():
                msg += f" | scale {scaler.get_scale():.0f}, skipped {st['amp_skipped']}"
            print(msg, flush=True)
            amp_warning(st, cfg, scaler, total_steps)
            win_loss, win_n = 0.0, 0
        if st["step"] % cfg["save_every"] == 0 or st["step"] == total_steps:
            save_checkpoint()

    log_f.close()
    if status == "done":
        full_val = evaluate(model, val_ds.items, cfg["micro_batch"], device, amp_dtype)
        summary = {"status": "done", "steps": st["step"], "epochs": cfg["epochs"],
                   "final_full_val_loss": round(full_val, 4), "best_val_subset_loss": round(st["best_val"], 4),
                   "val_rows": len(val_ds), "train_rows": len(train_ds), "tokens_seen": st["tokens_seen"],
                   "nan_skipped": st["nan_skipped"], "amp_skipped_steps": st["amp_skipped"],
                   "final_grad_scale": scaler.get_scale(), "elapsed_s": round(elapsed(), 1),
                   "trainable_params": trainable, "size": cfg["size"]}
        atomic_write_text(json.dumps(summary, indent=2), run_dir / "run_summary.json")
        print(f"[done] {st['step']:,} steps in {fmt_time(elapsed())}; full val loss ({len(val_ds)} rows) "
              f"{full_val:.4f}; outputs in {run_dir}")
    else:
        # Early stop: record why in run_summary.json too, so the output folder says it is not a finished run
        # (a later --resume that finishes overwrites this file with the "done" summary).
        summary = {"status": STOP_NAMES[status], "steps": st["step"], "total_steps": total_steps,
                   "nan_skipped": st["nan_skipped"], "amp_skipped_steps": st["amp_skipped"],
                   "elapsed_s": round(elapsed(), 1), "size": cfg["size"]}
        atomic_write_text(json.dumps(summary, indent=2), run_dir / "run_summary.json")
    return status


def amp_warning(st, cfg, scaler, total_steps):
    """Warn when GradScaler skips too many steps (R7): then the adapter is barely being updated."""
    if not scaler.is_enabled():
        return
    warmup = max(1, round(cfg["warmup_frac"] * total_steps))
    after = st["step"] - warmup
    if scaler.get_scale() < 1 or (after >= 50 and st["amp_skipped"] > 0.05 * st["step"]):
        print(f"[amp] WARNING: {st['amp_skipped']} of {st['step']} steps skipped by GradScaler, scale "
              f"{scaler.get_scale():g}. The run is barely learning: consider base_dtype fp32 or a lower lr.")


def _prune_log(path, max_step):
    """Keep the header and the rows with step <= max_step (rows after a checkpoint are redone on resume)."""
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    keep = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= max_step]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(keep)


EXIT_CODES = {"done": 0, "time_limit": 3, "nan": 2}               # local shell exit codes (unchanged)
STOP_NAMES = {"done": "done", "time_limit": "time_limit", "nan": "nan_abort"}


def finish(result, run_dir):
    """Print one final line saying why the run stopped and return the process exit code.

    Locally the exit code is 0 / 3 (time limit) / 2 (NaN abort) so shell scripts can tell them apart.
    On Kaggle it is always 0: a script kernel that exits non-zero is marked as a failed version, and a
    failed version's output may not be offered to the next (resume) version via kernel_sources. The
    stop reason is in this last log line and in run_summary.json / progress.json instead.
    """
    run_dir = Path(run_dir)

    def read(name):
        try:
            return json.loads((run_dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    summary, prog = read("run_summary.json"), read("progress.json")
    code = EXIT_CODES[result]
    kaggle = in_kaggle_kernel()
    if code and kaggle:
        code = 0
    print(f"[stop] reason={STOP_NAMES[result]} | run_summary.json status={summary.get('status', 'missing')}"
          f" | progress.json last saved step {prog.get('step', '?')}/{prog.get('total_steps', '?')}"
          f" | exit {code}" + (" (Kaggle: always 0 so the version keeps its output)" if kaggle else "")
          + (" | resume with --resume" if result != "done" else ""), flush=True)
    return code


def run(argv=None):
    """Parse the config and train. Returns (status, run_dir)."""
    cfg = parse_config(argv)
    print("[config] " + ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "debug_crash_at_step"))
    return train(cfg), Path(cfg["out_dir"]) / cfg["run_id"]


def main(argv=None):
    return run(argv)[0]


if __name__ == "__main__":
    _result, _run_dir = run()
    _code = finish(_result, _run_dir)
    # Inside a notebook cell (ipykernel) any sys.exit would show as an error, so only scripts exit.
    if "ipykernel" not in sys.modules:
        sys.exit(_code)
