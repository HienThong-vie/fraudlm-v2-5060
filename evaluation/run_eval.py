"""Run the FraudLM v2 evaluation (gate G5): generate answers, score them, write results files.

Models (same prompt, same test rows, same metrics):
  B0  rule-based template responder, no language model           --model b0
  B1  pretrained GPT-2 of --size, no adapter                       --model b1
  FT  pretrained GPT-2 of --size + LoRA adapter (--adapter path)   --model ft (default when --adapter is given)

Usage (from llm_ver2/):
  python evaluation/run_eval.py --model b0 --split test
  python evaluation/run_eval.py --model b1 --size gpt2-small --split test --limit 50
  python evaluation/run_eval.py --size gpt2-xl --adapter checkpoint/<run_id>/lora_adapter.pt --split test
  python evaluation/run_eval.py --rescore evaluation/results/<file>.json   (re-score stored answers, no model)
Splits: test (dataset/fraudlm_sft_test.json), val, demo67 (dataset/fraudlm_demo_months67.json: explain_score rows
from test months 6-7, read for evaluation only).

Decoding: greedy (temperature 0), at most --max_new_tokens new tokens, stop at EOS. The answer is the generated
part only, decoded up to EOS and cut where the model starts a new "### ..." prompt section (metrics.clean_answer).
Loss: token-weighted cross-entropy on the response tokens only (prompt masked with -100, as in training), and
perplexity = exp(loss). Nothing here is used to choose or tune a model: it only measures.

Outputs in --out (default evaluation/results/):
  <tag>_<model>_<split>.json      config, summary (overall / per task, all rows / -heldout rows / rows whose
                                  reference answer does not occur in the train split) and every row
  <tag>_<model>_<split>_rows.csv  one line per row (id, task, answer, main metrics)
  <tag>_<model>_<split>.partial.jsonl   generation progress; --resume continues from it (long Kaggle runs)
  summary.csv                     long table: one line per tag / model / split / subset / task / metric

Author: FraudLM v2 team (COS30049.2-1).
Generation and masked loss follow Raschka, S. (2024) *Build a Large Language Model (From Scratch)*, Manning,
chapters 5 and 7; code repo Apache-2.0.

"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    # Run as a script: make llm_ver2/ importable (inside a Kaggle bundle the modules are already inlined).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from evaluation.baselines import template_answer
from evaluation.metrics import aggregate, clean_answer, score_row, _TEMPLATE_CONTINUATION
from utils.prompt import format_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = {"test": "fraudlm_sft_test.json", "val": "fraudlm_sft_val.json", "demo67": "fraudlm_demo_months67.json"}
TASKS = ["explain_score", "dataset_fact", "policy", "banking_qa", "general"]
EOS_ID = 50256

# Main metrics copied to the per-row CSV (the JSON has all of them).
CSV_METRICS = ["explain_faithful", "score_correct", "threshold_correct", "flag_consistent", "hallucinated_number",
               "ref_numbers_all_present", "factor_mentions", "factor_mentions_in_context", "factor_recall_ref",
               "policy_category", "refusal_correct", "not_in_context_honest", "policy_compliant", "context_echo", "rougeL_f1",
               "ended_with_eos", "n_new_tokens", "loss_sum", "loss_tokens"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", choices=["auto", "b0", "b1", "ft"], default="auto",
                    help="auto = ft if --adapter is given, else b1")
    ap.add_argument("--size", default="gpt2-small", help="gpt2-small | gpt2-medium | gpt2-large | gpt2-xl")
    ap.add_argument("--adapter", default="", help="path to lora_adapter.pt (FT); run_config.json next to it is read")
    ap.add_argument("--lora_rank", type=int, default=None, help="only if there is no run_config.json")
    ap.add_argument("--lora_alpha", type=float, default=None)
    ap.add_argument("--lora_targets", default=None)
    ap.add_argument("--split", choices=list(SPLIT_FILES), default="test")
    ap.add_argument("--data_dir", default=str(PROJECT_ROOT / "dataset"))
    ap.add_argument("--train_file", default="", help="train split for the repeat check (default data_dir/fraudlm_sft_train.json)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate a seeded random subset of N rows (0 = all)")
    ap.add_argument("--tasks", default="", help="comma list to keep only some tasks")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "evaluation" / "results"))
    ap.add_argument("--run_id", default="", help="tag for file names (default: adapter folder name, or 'base')")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--base_dtype", default="auto", help="auto = fp16 on CUDA, fp32 on CPU")
    ap.add_argument("--cache_dir", default=str(PROJECT_ROOT / "checkpoint" / "pretrained"))
    ap.add_argument("--no_loss", action="store_true", help="skip the response-loss pass")
    ap.add_argument("--loss_batch", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="continue from the .partial.jsonl file")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


# --------------------------------------------------------------------------------------- data

def load_rows(args) -> list[dict]:
    path = Path(args.data_dir) / SPLIT_FILES[args.split]
    rows = json.loads(path.read_text(encoding="utf-8"))
    if args.tasks:
        keep = {t.strip() for t in args.tasks.split(",")}
        rows = [r for r in rows if r["task"] in keep]
    if args.limit and args.limit < len(rows):
        # Seeded random subset, kept in file order, so B0 / B1 / FT with the same --limit see the same rows.
        idx = sorted(random.Random(args.seed).sample(range(len(rows)), args.limit))
        rows = [rows[i] for i in idx]
    return rows


def train_outputs(args) -> set[str] | None:
    """Reference answers of the train split (G3 review F8: some test answers repeat a train answer by design)."""
    p = Path(args.train_file) if args.train_file else Path(args.data_dir) / SPLIT_FILES["test"].replace("test", "train")
    if not p.exists():
        print(f"[eval] WARNING: {p} not found, the 'novel' subset is skipped")
        return None
    return {r["output"].strip() for r in json.loads(p.read_text(encoding="utf-8"))}


# --------------------------------------------------------------------------------------- model

def resolve_model_kind(args) -> str:
    if args.model != "auto":
        return args.model
    return "ft" if args.adapter else "b1"


def load_model(args, kind: str):
    """Base GPT-2 in its final dtype first, then LoRA (PLAN 3 dtype rule: never cast after apply_lora)."""
    from utils.load_gpt2_weights import load_pretrained_gpt2

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.base_dtype == "auto":
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
    else:
        dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.base_dtype]
    model = load_pretrained_gpt2(args.size, cache_dir=args.cache_dir, device=device, dtype=dtype)
    info = {"device": device, "base_dtype": str(dtype).replace("torch.", "")}
    if kind == "ft":
        from training_model.lora import apply_lora, load_lora

        adapter = Path(args.adapter)
        if not adapter.exists():
            raise FileNotFoundError(f"adapter not found: {adapter}")
        cfg_path = adapter.parent / "run_config.json"
        rcfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        if rcfg.get("size") and rcfg["size"] != args.size:
            raise ValueError(f"adapter was trained on {rcfg['size']}, but --size is {args.size}")
        rank = args.lora_rank or int(rcfg.get("lora_rank", 16))
        alpha = args.lora_alpha or float(rcfg.get("lora_alpha", 32))
        targets = tuple((args.lora_targets or rcfg.get("lora_targets", "attn,ff")).split(","))
        apply_lora(model, rank, alpha, targets)
        # Adapters stay fp32 and unmerged (merge would need fp32 base weights).
        load_lora(model, torch.load(adapter, map_location=device, weights_only=True))
        info.update({"adapter": str(adapter), "lora_rank": rank, "lora_alpha": alpha, "lora_targets": list(targets),
                     "run_config_found": cfg_path.exists()})
    model.eval()
    return model, device, info


@torch.no_grad()
def generate_answer(model, prompt: str, device: str, max_new_tokens: int) -> tuple[str, str, bool, int]:
    """Greedy answer to one prompt -> (clean answer, raw generated text, ended_with_eos, new token count).

    One call to utils.generate.generate (KV cache on). stop_fn ends the answer as soon as the model starts a
    new '### ...' section (base GPT-2 does this often); greedy tokens up to that point are unchanged.
    """
    from utils.generate import generate, ids_to_text_until_eos, text_to_ids

    idx = text_to_ids(prompt).to(device)
    n_prompt = idx.shape[1]
    hit = {"template": False}

    def stop_fn(seq):
        if _TEMPLATE_CONTINUATION.search(ids_to_text_until_eos(seq[0, n_prompt:].cpu())):
            hit["template"] = True
            return True
        return False

    out = generate(model, idx, max_new_tokens=max_new_tokens, context_size=1024, temperature=0.0, eos_id=EOS_ID,
                   stop_fn=stop_fn)
    gen = out[0, n_prompt:].cpu()
    # For a single row generate() stops at EOS without appending it, so fewer tokens than allowed and no
    # template stop means the model ended its answer with EOS.
    ended = gen.shape[0] < max_new_tokens and not hit["template"]
    raw = ids_to_text_until_eos(gen)
    return clean_answer(raw), raw, ended, int(gen.shape[0])


@torch.no_grad()
def response_losses(model, rows: list[dict], device: str, batch: int) -> list[tuple[float, int]]:
    """(sum of token losses, number of response tokens) per row; prompt tokens are masked as in training."""
    from training_model.sft_data import IGNORE_INDEX, collate, encode_example

    items = [encode_example(r, 1024) for r in rows]
    out: list[tuple[float, int]] = [(float("nan"), 0)] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(items[i]["input_ids"]) if items[i] else 0)
    order = [i for i in order if items[i] is not None]
    for s in range(0, len(order), batch):
        ids = order[s:s + batch]
        x, y = collate([items[i] for i in ids], device)
        logits = model(x).float()  # loss in fp32 (PLAN 3 dtype rule)
        tok = F.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=IGNORE_INDEX, reduction="none")
        tok = tok.view(y.shape)
        mask = y != IGNORE_INDEX
        for b, i in enumerate(ids):
            out[i] = (float(tok[b][mask[b]].sum()), int(mask[b].sum()))
    return out


# --------------------------------------------------------------------------------------- summary

def subsets(rows: list[dict], train_out: set[str] | None) -> dict[str, list[dict]]:
    """all rows / held-out wordings (id ends with -heldout) / reference answer not seen in train."""
    out = {"all": rows, "heldout": [r for r in rows if r["id"].endswith("-heldout")]}
    if train_out is not None:
        out["novel_output"] = [r for r in rows if not r["output_in_train"]]
    return out


def summarise(scored: list[dict], train_out) -> dict:
    summary = {}
    for sub, rs in subsets(scored, train_out).items():
        summary[sub] = {"overall": aggregate(rs)}
        for t in TASKS:
            tr = [r for r in rs if r["task"] == t]
            if tr:
                summary[sub][t] = aggregate(tr)
        if any(r["task"] == "policy" for r in rs):
            cats = sorted({r.get("policy_category") for r in rs if r["task"] == "policy"})
            for c in cats:
                summary[sub][f"policy:{c}"] = aggregate([r for r in rs if r["task"] == "policy" and r.get("policy_category") == c])
    return summary


def update_summary_csv(path: Path, tag: str, label: str, size: str, split: str, summary: dict):
    cols = ["run_tag", "model", "size", "split", "subset", "task", "metric", "value", "n", "written"]
    old = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            old = [r for r in csv.DictReader(f) if not (r["run_tag"] == tag and r["model"] == label and r["split"] == split)]
    now = dt.datetime.now().isoformat(timespec="minutes")
    new = []
    for sub, by_task in summary.items():
        for task, mets in by_task.items():
            for k, v in mets.items():
                new.append({"run_tag": tag, "model": label, "size": size, "split": split, "subset": sub, "task": task,
                            "metric": k, "value": f"{v['value']:.6g}", "n": v["n"], "written": now})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(old + new)


# --------------------------------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    kind = resolve_model_kind(args)
    if kind == "ft" and not args.adapter:
        raise SystemExit("--model ft needs --adapter checkpoint/<run_id>/lora_adapter.pt")
    label = "B0" if kind == "b0" else f"{kind.upper()}-{args.size}"
    tag = args.run_id or (Path(args.adapter).parent.name if args.adapter else "base")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The split name carries the subset size, so a --limit dry run never overwrites a full run.
    split_name = args.split + (f"-n{args.limit}" if args.limit else "") + (f"-{args.tasks.replace(',', '+')}" if args.tasks else "")
    stem = f"{tag}_{label}_{split_name}"
    partial = out_dir / f"{stem}.partial.jsonl"

    rows = load_rows(args)
    train_out = train_outputs(args)
    print(f"[eval] {label} on {args.split}: {len(rows)} rows (limit {args.limit or 'none'}) -> {out_dir / stem}.json")

    done: dict[str, dict] = {}
    if args.resume and partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
        print(f"[eval] resume: {len(done)} rows already generated")
    elif partial.exists():
        partial.unlink()

    model, device, info = (None, "cpu", {})
    if kind != "b0":
        model, device, info = load_model(args, kind)

    # 1) answers
    t0 = time.time()
    todo = [r for r in rows if r["id"] not in done]
    with partial.open("a", encoding="utf-8") as pf:
        for k, r in enumerate(todo, 1):
            prompt = format_prompt(r["instruction"], r.get("input", ""))
            if kind == "b0":
                ans, raw, eos, n_new = template_answer(r["instruction"], r.get("input", "")), "", None, 0
            else:
                ans, raw, eos, n_new = generate_answer(model, prompt, device, args.max_new_tokens)
            rec = {"id": r["id"], "answer": ans, "raw": raw, "ended_with_eos": eos, "n_new_tokens": n_new}
            done[r["id"]] = rec
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pf.flush()
            if kind != "b0" and (k % 10 == 0 or k == len(todo)):
                el = time.time() - t0
                eta = el / k * (len(todo) - k)
                print(f"[eval] {k}/{len(todo)} rows, {el / k:.1f} s/row, ETA {eta / 60:.1f} min", flush=True)

    # 2) response loss (B1 / FT only)
    losses = None
    if kind != "b0" and not args.no_loss:
        tl = time.time()
        losses = response_losses(model, rows, device, args.loss_batch)
        print(f"[eval] response loss pass: {time.time() - tl:.1f} s")

    # 3) score
    scored = []
    for i, r in enumerate(rows):
        g = done[r["id"]]
        m = score_row(r, g["answer"])
        rec = {"id": r["id"], "task": r["task"], "source": r.get("source", ""),
               "heldout": r["id"].endswith("-heldout"),
               "output_in_train": (r["output"].strip() in train_out) if train_out is not None else None,
               "instruction": r["instruction"], "input": r.get("input", ""), "reference": r["output"],
               "answer": g["answer"], "raw": g["raw"], "ended_with_eos": g["ended_with_eos"],
               "n_new_tokens": g["n_new_tokens"], **m}
        if losses is not None:
            rec["loss_sum"], rec["loss_tokens"] = losses[i]
        scored.append(rec)
    summary = summarise(scored, train_out)

    result = {
        "model": label, "kind": kind, "size": None if kind == "b0" else args.size, "split": split_name,
        "run_tag": tag, "n_rows": len(rows), "limit": args.limit, "tasks_filter": args.tasks,
        "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens, "stop": "EOS or a new '###' section"},
        "model_info": info, "seed": args.seed, "written": dt.datetime.now().isoformat(timespec="seconds"),
        "generation_seconds": round(time.time() - t0, 1),
        "summary": summary, "rows": scored,
    }
    write_outputs(result, out_dir, stem, train_out)
    partial.unlink(missing_ok=True)
    return result


def write_outputs(result: dict, out_dir: Path, stem: str, train_out):
    """Write <stem>.json, <stem>_rows.csv, update summary.csv and print a short table."""
    scored, summary = result["rows"], result["summary"]
    (out_dir / f"{stem}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    with (out_dir / f"{stem}_rows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "task", "heldout", "output_in_train", "answer"] + CSV_METRICS)
        for r in scored:
            w.writerow([r["id"], r["task"], r["heldout"], r["output_in_train"], r["answer"]] + [r.get(k, "") for k in CSV_METRICS])
    update_summary_csv(out_dir / "summary.csv", result["run_tag"], result["model"], result["size"] or "-",
                       result["split"], summary)
    print(f"[eval] done: {out_dir / stem}.json")
    s = summary["all"]
    for t in ["overall"] + TASKS:
        if t in s:
            keys = [k for k in ("explain_faithful", "hallucinated_number", "factor_precision", "ref_numbers_all_present",
                                "refusal_correct", "not_in_context_honest", "policy_compliant", "context_echo",
                                "rougeL_f1", "response_loss") if k in s[t]]
            print(f"  {t:14s} " + "  ".join(f"{k}={s[t][k]['value']:.3f}(n={s[t][k]['n']})" for k in keys))


def rescore(path: str, train_file: str = ""):
    """Re-score the stored answers of a results JSON with the current metrics (no model, no generation).

    Use after a metric rule changes; the answers, raw text and response losses are kept as they were.
    """
    p = Path(path)
    result = json.loads(p.read_text(encoding="utf-8"))
    tf = Path(train_file) if train_file else PROJECT_ROOT / "dataset" / SPLIT_FILES["test"].replace("test", "train")
    train_out = {r["output"].strip() for r in json.loads(tf.read_text(encoding="utf-8"))} if tf.exists() else None
    keep = ("id", "task", "source", "heldout", "output_in_train", "instruction", "input", "reference", "answer", "raw",
            "ended_with_eos", "n_new_tokens", "loss_sum", "loss_tokens")
    scored = []
    for old in result["rows"]:
        row = {"id": old["id"], "task": old["task"], "instruction": old["instruction"], "input": old["input"],
               "output": old["reference"]}
        rec = {k: old[k] for k in keep if k in old}
        rec.update(score_row(row, old["answer"]))
        scored.append(rec)
    result["rows"] = scored
    result["summary"] = summarise(scored, train_out)
    result["rescored"] = dt.datetime.now().isoformat(timespec="seconds")
    write_outputs(result, p.parent, p.stem, train_out)
    return result


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--rescore":
        # python evaluation/run_eval.py --rescore evaluation/results/<file>.json [more files]
        for f in sys.argv[2:]:
            rescore(f)
    else:
        main()
