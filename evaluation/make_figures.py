"""Figures for the FraudLM v2 evaluation: fine-tuning loss curve and a B0 / B1 / FT metrics bar chart.

Usage (from llm_ver2/):
  python evaluation/make_figures.py loss --log checkpoint/<run_id>/train_log.csv --name <run_id>
  python evaluation/make_figures.py bars --results evaluation/results/A.json evaluation/results/B.json ... \
         --subset all --name test_all

Style rules (team protocol): 300 dpi PNG, 16 cm wide or less, readable on A4 (9 pt text), no title inside the
image (the caption goes under the figure in the report; caption facts are in evaluation/EVAL_NOTES.md).
Every plotted value is read from a results file or train_log.csv; nothing is typed in by hand.

Author: FraudLM v2 team (COS30049.2-1).

"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed (Kaggle, CI)
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "evaluation" / "figures"
CM = 1 / 2.54
WIDTH_CM = 16.0

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 300, "figure.dpi": 100})

# (label on the chart, task key in the summary, metric key, transform) - every bar is "higher is better".
BAR_METRICS = [
    ("Explain: score, threshold\nand flag exact", "explain_score", "explain_faithful", None),
    ("Explain: no invented\nnumber", "explain_score", "hallucinated_number", lambda v: 1 - v),
    ("Explain: factor\nprecision", "explain_score", "factor_precision", None),
    ("Dataset facts:\nall numbers right", "dataset_fact", "ref_numbers_all_present", None),
    ("Policy: correct\nrefusal", "policy", "refusal_correct", None),
    ("Policy: 'not in\ncontext' honesty", "policy", "not_in_context_honest", None),
    ("All answers:\npolicy compliant", "overall", "policy_compliant", None),
    ("Banking Q&A:\nROUGE-L F1", "banking_qa", "rougeL_f1", None),
]
MODEL_COLOURS = {"B0": "#9e9e9e", "B1": "#6baed6", "FT": "#08519c"}


def plot_loss(log_path: Path, out_path: Path):
    steps, train, vsteps, val = [], [], [], []
    with log_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = int(float(r["step"]))
            if r.get("train_loss"):
                steps.append(s)
                train.append(float(r["train_loss"]))
            if r.get("val_loss"):
                vsteps.append(s)
                val.append(float(r["val_loss"]))
    fig, ax = plt.subplots(figsize=(WIDTH_CM * CM, 7 * CM))
    if train:
        ax.plot(steps, train, color="#9ecae1", linewidth=0.8, label="training loss (logged steps)")
    if val:
        ax.plot(vsteps, val, color="#08519c", marker="o", markersize=3, linewidth=1.2, label="validation loss")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("cross-entropy on response tokens")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return {"train_points": len(train), "val_points": len(val),
            "val_first": val[0] if val else None, "val_last": val[-1] if val else None,
            "val_min": min(val) if val else None}


def _family(label: str) -> str:
    return label.split("-", 1)[0]


def plot_bars(result_paths: list[Path], subset: str, out_path: Path):
    """Horizontal grouped bars: one group per metric (top to bottom), one bar per model.

    Each bar is labelled with its value and, in brackets, the number of answers (or factor mentions) it is based on,
    so a 1.00 on 2 answers is not read like a 1.00 on 190.
    """
    results = [json.loads(p.read_text(encoding="utf-8")) for p in result_paths]
    labels = [r["model"] for r in results]
    values, counts = [], []  # [model][metric]
    for r in results:
        s = r["summary"][subset]
        row, cnt = [], []
        for _, task, key, fn in BAR_METRICS:
            cell = s.get(task, {}).get(key)
            v = None if cell is None else cell["value"]
            row.append(None if v is None else (fn(v) if fn else v))
            cnt.append(None if cell is None else cell["n"])
        values.append(row)
        counts.append(cnt)
    n_m, n_k = len(results), len(BAR_METRICS)
    height = 0.8 / n_m
    fig, ax = plt.subplots(figsize=(WIDTH_CM * CM, (3 + 1.3 * n_k * max(1, n_m) / 2) * CM))
    for i, (lab, row, cnt) in enumerate(zip(labels, values, counts)):
        # metric k is drawn at y = n_k - 1 - k so the first metric is on top; models in legend order top-down
        ys = [n_k - 1 - k - (i - (n_m - 1) / 2) * height for k in range(n_k)]
        xs = [0 if v is None else v for v in row]
        colour = MODEL_COLOURS.get(_family(lab), None)
        bars = ax.barh(ys, xs, height=height * 0.92, label=lab, color=colour, edgecolor="black", linewidth=0.3)
        for bar, v, n in zip(bars, row, cnt):
            txt = "n/a" if v is None else f"{v:.2f} ({n})"
            ax.text((v or 0) + 0.01, bar.get_y() + bar.get_height() / 2, txt, va="center", ha="left", fontsize=6.5)
    ax.set_yticks([n_k - 1 - k for k in range(n_k)])
    ax.set_yticklabels([m[0].replace("\n", " ") for m in BAR_METRICS], fontsize=7.5)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("share (higher is better); (n) = answers or factor mentions counted")
    ax.legend(frameon=False, ncol=n_m, loc="lower center", bbox_to_anchor=(0.4, 1.0))
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return {"models": labels, "subset": subset, "metrics": [m[2] for m in BAR_METRICS],
            "values": values, "counts": counts, "sources": [str(p) for p in result_paths]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="FraudLM v2 evaluation figures")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("loss")
    a.add_argument("--log", required=True)
    a.add_argument("--name", default="")
    b = sub.add_parser("bars")
    b.add_argument("--results", nargs="+", required=True)
    b.add_argument("--subset", default="all", choices=["all", "heldout", "novel_output"])
    b.add_argument("--name", default="")
    ap.add_argument("--out_dir", default=str(FIG_DIR))
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "loss":
        name = args.name or Path(args.log).parent.name
        out = out_dir / f"loss_curve_{name}.png"
        info = plot_loss(Path(args.log), out)
    else:
        name = args.name or args.subset
        out = out_dir / f"metrics_bars_{name}.png"
        info = plot_bars([Path(p) for p in args.results], args.subset, out)
    # A small JSON next to the PNG records exactly which numbers were drawn (for the G6 audit).
    (out.with_suffix(".json")).write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"[fig] {out}")
    return out


if __name__ == "__main__":
    main()
