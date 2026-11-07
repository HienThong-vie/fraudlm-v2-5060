"""
G5 evaluation job for Kaggle: run the evaluator's harness (evaluation/run_eval.py) for B0, FT and B1.

Purpose: one notebook cell that, in a Kaggle "Save & Run All" commit,
- finds the trained adapter (`checkpoint/<run_id>/lora_adapter.pt` + `run_config.json`) and the dataset
  (`fraudlm_sft_test.json`, `fraudlm_demo_months67.json`) under /kaggle/input/**,
- runs B0 (rule responder), FT (base GPT-2 + LoRA adapter) and B1 (the same base GPT-2 without the adapter)
  on the test split (500 rows) and the months 6-7 demo file (300 rows),
- writes results, the loss curve (from the run's train_log.csv), metric bar charts and example picks to
  /kaggle/working/evaluation/.
Every generated answer is appended to a .partial.jsonl file at once (run_eval does that), so a stopped run
keeps its work. The --max_hours guard stops cleanly between two rows. A later version that gets this version's
output as an input copies the partial files back and skips the rows already done.

Precision follows the PLAN 3 dtype rule: the base is loaded in its final dtype (the run's base_dtype, fp16 on
CUDA by default), the LoRA adapter is applied after it and kept unmerged in fp32 (G1 review).

This file only orchestrates; the metrics and generation are the evaluator's code (evaluation/).

Usage
  repo, CPU:  .venv/Scripts/python kaggle/eval_job.py --adapter checkpoint/<run_id>/lora_adapter.pt --limit 3
  Kaggle:     kaggle/make_bundle.py --only eval -> kaggle/eval_bundle.py, see kaggle/RUNBOOK_G5.md
              (script kernel: the lead prepends `import sys; sys.argv = [...]`; every option is a flag)

Author: FraudLM v2 team (COS30049.2-1).

"""

import argparse
import contextlib
import io
import json
import shutil
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = None  # set below when run from the repo; stays None inside the Kaggle bundle
# >>> REPO_ONLY (make_bundle.py removes this block: the bundle writes the modules itself)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# <<< REPO_ONLY

from evaluation import make_figures, pick_examples, run_eval  # noqa: E402


def on_kaggle():
    return Path("/kaggle").exists()


# =====================================================================================
# 1. Settings (edit here, or override with command-line flags)
# =====================================================================================

SPLITS = ["test", "demo67"]          # test = 500 rows, demo67 = 300 explain_score rows from test months 6-7
MODELS = ["b0", "ft", "b1"]          # FT before B1: if time runs out, the fine-tuned numbers exist first
ADAPTER_FILE = "lora_adapter.pt"     # final adapter of the run (lora_adapter_best.pt = lowest val-subset loss)
MAX_HOURS = 11.0                     # stop cleanly after this long (Kaggle kills a session at 12 h)
MAX_NEW_TOKENS = 200                 # same as the evaluator's default
INPUT_ROOT = Path("/kaggle/input")


class TimeLimit(Exception):
    """Raised between two rows when --max_hours is reached; the rows done so far are already on disk."""


# =====================================================================================
# 2. Finding the inputs
# =====================================================================================

def find_adapter(arg, run_id):
    """Path of the adapter file. 'auto' searches the repo checkpoint/ folder, then /kaggle/input/**."""
    if arg and arg != "auto":
        p = Path(arg)
        if not p.exists():
            raise SystemExit(f"[eval_job] adapter not found: {p}")
        return p
    roots = [Path("checkpoint")] + ([INPUT_ROOT] if INPUT_ROOT.exists() else [])
    hits = []
    for root in roots:
        if root.exists():
            hits += [p for p in root.rglob(ADAPTER_FILE) if (p.parent / "run_config.json").exists()]
    if run_id:
        hits = [p for p in hits if p.parent.name == run_id]
    if not hits:
        raise SystemExit("[eval_job] no checkpoint/<run_id>/lora_adapter.pt with run_config.json found. On Kaggle: "
                         "+ Add Input -> Your Work -> Notebooks -> fraudlm-train (see kaggle/RUNBOOK_G5.md).")
    if len({p.parent.name for p in hits}) > 1:
        raise SystemExit(f"[eval_job] adapters of several runs found {sorted({p.parent.name for p in hits})}; "
                         "pass --run_id to choose one.")
    return hits[0]


def find_data_dir(arg):
    """Folder with fraudlm_sft_test.json (repo dataset/, ./dataset, or /kaggle/input/**)."""
    name = "fraudlm_sft_test.json"
    if arg and arg != "auto":
        return Path(arg)
    for d in [Path("dataset")] + ([] if _PROJECT_ROOT is None else [_PROJECT_ROOT / "dataset"]):
        if (d / name).exists():
            return d
    if INPUT_ROOT.exists():
        hits = sorted(INPUT_ROOT.rglob(name))
        if hits:
            return hits[0].parent
    raise SystemExit("[eval_job] fraudlm_sft_test.json not found: add the fraudlm-sft-v1-1 dataset as an input.")


def restore_previous_results(out_dir):
    """Copy result files of an earlier version (attached as an input) so finished work is not redone."""
    if not INPUT_ROOT.exists():
        return 0
    n = 0
    for f in INPUT_ROOT.rglob("evaluation/results/*"):
        if f.is_file() and f.suffix in (".json", ".jsonl", ".csv") and not (out_dir / f.name).exists():
            shutil.copy2(f, out_dir / f.name)
            n += 1
    if n:
        print(f"[eval_job] restored {n} result files from an earlier version")
    return n


# =====================================================================================
# 3. Time guard and early speed print, wrapped around the evaluator's generate_answer
# =====================================================================================

def install_guard(deadline):
    """Make run_eval.generate_answer check the clock and print the speed of the first rows.

    run_eval writes each answer to its .partial.jsonl as soon as it exists, so raising here loses nothing.
    """
    # Always wrap the evaluator's own function (not an earlier wrapper), so calling main() twice in one
    # notebook session never stacks an old deadline on top of the new one.
    if not hasattr(run_eval, "_unguarded_generate_answer"):
        run_eval._unguarded_generate_answer = run_eval.generate_answer
    original = run_eval._unguarded_generate_answer
    stats = {"n": 0, "t0": None}

    def guarded(*a, **kw):
        if time.time() > deadline:
            raise TimeLimit()
        if stats["t0"] is None:
            stats["t0"] = time.time()
        out = original(*a, **kw)
        stats["n"] += 1
        if stats["n"] in (1, 3):
            print(f"[eval_job] first rows: {(time.time() - stats['t0']) / stats['n']:.1f} s/row "
                  f"({out[3]} new tokens on the last one)", flush=True)
        return out

    def reset():
        stats["n"], stats["t0"] = 0, None

    run_eval.generate_answer = guarded
    return reset


# =====================================================================================
# 4. Main
# =====================================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="FraudLM v2 G5 evaluation job (B0, FT, B1)")
    p.add_argument("--adapter", default="auto", help="path to lora_adapter.pt, or auto")
    p.add_argument("--run_id", default="", help="choose the run when several adapters are attached")
    p.add_argument("--data_dir", default="auto")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--splits", default=",".join(SPLITS))
    p.add_argument("--limit", type=int, default=0, help="seeded subset of N rows per split (0 = all)")
    p.add_argument("--max_hours", type=float, default=MAX_HOURS)
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--base_dtype", default="", help="fp16/fp32; default = the training run's base_dtype")
    # Locally the default is under checkpoint/ (ours), never inside the evaluator's evaluation/results/.
    p.add_argument("--out_root", default="/kaggle/working/evaluation" if on_kaggle() else "checkpoint/eval_job")
    p.add_argument("--cache_dir", default="/tmp/fraudlm_pretrained" if on_kaggle() else "checkpoint/pretrained")
    p.add_argument("--device", default="auto")
    args, unknown = p.parse_known_args(argv)  # ignore Jupyter's "-f kernel.json"
    unknown = [u for u in unknown if u.startswith("--")]
    if unknown:
        print(f"[eval_job] WARNING: ignoring unknown flags {unknown}")
    return args


def main(argv=None):
    args = parse_args(argv)
    start = time.time()
    deadline = start + args.max_hours * 3600
    adapter = find_adapter(args.adapter, args.run_id)
    rcfg = json.loads((adapter.parent / "run_config.json").read_text(encoding="utf-8"))
    run_id, size = adapter.parent.name, rcfg["size"]
    # Same base precision as training. run_eval's "auto" = fp16 on CUDA, fp32 on CPU (fp16 is slow on a CPU).
    base_dtype = args.base_dtype or ("auto" if rcfg.get("base_dtype", "fp16") == "fp16" else "fp32")
    data_dir = find_data_dir(args.data_dir)
    results_dir = Path(args.out_root) / "results"
    figures_dir = Path(args.out_root) / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"
    print(f"[eval_job] run {run_id}: {size}, adapter {adapter}, base_dtype {base_dtype} "
          f"(auto = fp16 on CUDA, fp32 on CPU; adapter fp32, unmerged); GPU {gpu}")
    print(f"[eval_job] data {data_dir}; results -> {results_dir}; limit {args.limit or 'none'}; "
          f"max_hours {args.max_hours}")
    restore_previous_results(results_dir)

    splits = [s for s in args.splits.split(",") if s]
    if "demo67" in splits and not (data_dir / run_eval.SPLIT_FILES["demo67"]).exists():
        print(f"[eval_job] WARNING: {run_eval.SPLIT_FILES['demo67']} is not in {data_dir}; demo67 skipped "
              "(upload it to the Kaggle dataset, RUNBOOK_G4 step 1)")
        splits.remove("demo67")
    reset_speed = install_guard(deadline)

    done, status = [], "done"
    for kind in [m for m in args.models.split(",") if m]:
        for split in splits:
            label = "B0" if kind == "b0" else f"{kind.upper()}-{size}"
            split_name = split + (f"-n{args.limit}" if args.limit else "")
            final = results_dir / f"{run_id}_{label}_{split_name}.json"
            if final.exists():
                print(f"[eval_job] {final.name} already done, skipped")
                done.append(final)
                continue
            if time.time() > deadline:
                status = "time_limit"
                break
            cli = ["--model", kind, "--split", split, "--data_dir", str(data_dir), "--out", str(results_dir),
                   "--run_id", run_id, "--cache_dir", args.cache_dir, "--device", args.device,
                   "--base_dtype", base_dtype, "--max_new_tokens", str(args.max_new_tokens), "--resume"]
            if kind != "b0":
                cli += ["--size", size]
            if kind == "ft":
                cli += ["--adapter", str(adapter)]
            if args.limit:
                cli += ["--limit", str(args.limit)]
            print(f"\n[eval_job] === {label} on {split_name} (elapsed {(time.time() - start) / 60:.1f} min) ===")
            reset_speed()
            try:
                run_eval.main(cli)
            except TimeLimit:
                print(f"[eval_job] --max_hours {args.max_hours} reached during {label} {split_name}. The rows done "
                      f"so far are saved in {results_dir}; attach this version's output and run again to finish.")
                status = "time_limit"
                break
            done.append(final)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if status != "done":
            break

    # Figures and example picks from whatever is finished.
    log = adapter.parent / "train_log.csv"
    if log.exists():
        make_figures.main(["--out_dir", str(figures_dir), "loss", "--log", str(log), "--name", run_id])
    for split in splits:
        split_name = split + (f"-n{args.limit}" if args.limit else "")
        files = [f for f in done if f.name.endswith(f"_{split_name}.json")]
        if len(files) >= 2:
            make_figures.main(["--out_dir", str(figures_dir), "bars", "--results", *map(str, files),
                               "--subset", "all", "--name", f"{run_id}_{split_name}_all"])
        ft = [f for f in files if "_FT-" in f.name]
        if ft:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                pick_examples.main([str(ft[0]), "--n", "3"])
            (results_dir / f"{ft[0].stem}_examples.md").write_text(buf.getvalue(), encoding="utf-8")
    status_info = {"status": status, "run_id": run_id, "size": size, "base_dtype": base_dtype,
                   "finished": [f.name for f in done], "elapsed_min": round((time.time() - start) / 60, 1)}
    (Path(args.out_root) / "eval_status.json").write_text(json.dumps(status_info, indent=2), encoding="utf-8")
    print(f"\n[eval_job] {status.upper()}: {len(done)} result files in {results_dir}, figures in {figures_dir}, "
          f"{status_info['elapsed_min']} min")
    return status


if __name__ == "__main__":
    result = main()
    # On Kaggle always exit 0: a script kernel that exits non-zero is marked as failed, and a failed
    # version may not keep /kaggle/working (the partial results). The status is in eval_status.json.
    if not on_kaggle() and "ipykernel" not in sys.modules:
        sys.exit(0 if result == "done" else 3)
