"""Build continued-SFT data from balanced predictions made by eight ML projects.

Ground-truth target columns are deliberately ignored. The generated JSON/JSONL stays local because
the source datasets have different licences, including non-commercial and no-derivatives terms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.multisource_prompt import format_prediction_context, prediction_answer  # noqa: E402
from utils.prompt import FEATURE_LABELS  # noqa: E402

# The three question types; each case gets exactly one, chosen by a hash of its key (see make_example).
INTENTS = [
    ("explain", "Explain this classifier result without treating it as proof of fraud."),
    ("next_steps", "What should a fraud analyst do next for this case?"),
    ("limitations", "What are the limitations of this prediction?"),
]


def _stable_int(text: str) -> int:
    """Hash text to a large integer that is the same on every run and machine (unlike Python's hash())."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _clean_value(value):
    """Make a CSV cell JSON-safe: missing -> "not provided", floats rounded to 6 decimals, numpy scalars -> Python."""
    if pd.isna(value):
        return "not provided"
    if isinstance(value, float):
        return round(value, 6)
    return value.item() if hasattr(value, "item") else value


def balanced_sample(frame: pd.DataFrame, threshold: float, limit: int, seed: int) -> pd.DataFrame:
    """Deterministically keep both review outcomes; use all rows when limit is zero."""
    if not limit or len(frame) <= limit:
        return frame.copy()
    work = frame.copy()
    work["_flagged"] = work["_score"] >= threshold
    parts = []
    # Balanced sampling: half the quota from flagged rows and half from not-flagged rows, so the
    # rare flagged outcome is not drowned out; a fixed random_state keeps the draw reproducible.
    per_group = limit // 2
    for flagged in (False, True):
        group = work[work["_flagged"] == flagged]
        take = min(len(group), per_group)
        if take:
            parts.append(group.sample(take, random_state=seed + int(flagged)))
    out = pd.concat(parts) if parts else work.iloc[0:0]
    # If one group was too small to fill its half, top up with random rows from everything not yet taken.
    remaining = limit - len(out)
    if remaining:
        rest = work.drop(index=out.index).sample(min(remaining, len(work) - len(out)), random_state=seed + 7)
        out = pd.concat([out, rest])
    return out.drop(columns="_flagged").sort_index()


def read_source(source: dict, root: Path, limit: int, seed: int) -> list[dict]:
    """Read one classifier's prediction CSV and return up to `limit` stored cases (dicts).

    Only the id column, the score column and the approved extra columns are used; the ground-truth
    target column is never used. `source` is one entry of configs/prediction_sources.json.
    """
    path = (root / source["prediction_file"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{source['id']}: prediction file not found: {path}")
    frame = pd.read_csv(path)
    score_col = source["score_column"]
    if score_col not in frame:
        raise ValueError(f"{source['id']}: missing score column {score_col!r}")
    # Keep only rows with a valid score in [0, 1]; non-numeric scores become NaN and are dropped.
    frame["_score"] = pd.to_numeric(frame[score_col], errors="coerce")
    frame = frame[frame["_score"].between(0, 1, inclusive="both")].copy()
    frame = balanced_sample(frame, float(source["threshold"]), limit, seed)
    rows = []
    for idx, row in frame.iterrows():
        raw_id = idx if source["id_column"] == "__row__" else row[source["id_column"]]
        case_id = str(int(raw_id)) if isinstance(raw_id, float) and raw_id.is_integer() else str(raw_id)
        signals = {col: _clean_value(row[col]) for col in source.get("extra_columns", [])
                   if col in row and col not in {"top_factors"}}
        # BAF cases also carry the three local factors (logistic-regression contributions); keep only
        # known feature names with a raises/lowers direction.
        if source["id"] == "baf_base" and row.get("top_factors"):
            try:
                factors = json.loads(row["top_factors"])
                signals["main model factors"] = "; ".join(
                    f"{FEATURE_LABELS[f['feature']]}={_clean_value(f.get('value'))} ({f['direction']} score)"
                    for f in factors if f.get('feature') in FEATURE_LABELS and f.get('direction') in ('raises', 'lowers'))
            except (TypeError, ValueError, KeyError):
                pass
        rows.append({
            "case_key": f"{source['id']}:{case_id}", "source_id": source["id"], "case_id": case_id,
            "dataset_name": source["name"], "dataset_kind": source["kind"], "model": source["model"],
            "score": float(row["_score"]), "threshold": float(source["threshold"]), "signals": signals,
            'prediction_row_index': int(idx),
            'model_metrics': {k: source[k] for k in ('pr_auc', 'precision', 'recall', 'f1', 'roc_auc', 'actual_test_fpr') if k in source},
            **{key: source[key] for key in ('experiment', 'model_sha256', 'fpr_budget', 'offline_only') if key in source},
        })
    return rows


def make_example(case: dict) -> dict:
    """Turn one stored case into an instruction/input/output training row (the output is a template answer)."""
    intent, instruction = INTENTS[_stable_int(case["case_key"]) % len(INTENTS)]
    return {
        "id": "multi_" + case["case_key"].replace(":", "_"),
        "instruction": instruction,
        "input": format_prediction_context(case),
        "output": prediction_answer(case, intent),
        "task": "multisource_prediction",
        "source": case["source_id"],
    }


def split_name(key: str) -> str:
    """Hash split by case key: buckets 0-89 train, 90-94 val, 95-99 test (about 90/5/5), the same on every run."""
    bucket = _stable_int(key) % 100
    return "train" if bucket < 90 else ("val" if bucket < 95 else "test")


def build(args) -> dict:
    """Build the train/val/test JSON files, the prediction store and a manifest; return the manifest.

    Steps: read each registry source -> hash-split the cases -> optionally mix in the original FraudLM
    rows (screening val/test rows that overlap the prior run's training data) -> shuffle and write.
    """
    registry_path = Path(args.registry).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    root = registry_path.parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_cases, counts = [], {}
    for source in registry["sources"]:
        cases = read_source(source, root, args.max_rows_per_source, args.seed)
        all_cases.extend(cases)
        counts[source["id"]] = len(cases)
        print(f"[source] {source['id']}: {len(cases):,} selected")

    # Split by hashing the case key, so a case never appears in two splits.
    splits = {name: [] for name in ("train", "val", "test")}
    for case in all_cases:
        splits[split_name(case["case_key"])].append(make_example(case))
    include_original = not args.no_original
    overlap_excluded = {}
    if include_original:
        prior = Path(args.prior_training_file)
        if not prior.is_file():
            raise FileNotFoundError('Prior continued-training data are required to screen original validation/test contexts: ' + str(prior))
        prior_rows = json.loads(prior.read_text(encoding='utf-8'))
        from evaluation.compare_original_validation import screen
        for name in splits:
            original = ROOT / "dataset" / f"fraudlm_sft_{name}.json"
            original_rows = json.loads(original.read_text(encoding="utf-8"))
            # Drop original val/test rows that overlap the earlier run's training data, so they stay held out.
            if name in ('val', 'test'):
                original_rows, removed = screen(original_rows, prior_rows)
                overlap_excluded[name] = len(removed)
            splits[name].extend(original_rows)

    # Seeded shuffle (so the mix of sources and original rows is reproducible), then write the three splits.
    rng = random.Random(args.seed)
    for name, rows in splits.items():
        rng.shuffle(rows)
        (output / f"fraudlm_sft_{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with open(output / "prediction_store.jsonl", "w", encoding="utf-8") as fh:
        for case in sorted(all_cases, key=lambda x: x["case_key"]):
            fh.write(json.dumps(case) + "\n")
    # The manifest records seeds, counts and SHA-256 hashes of the inputs and outputs, so a later run
    # can show which data a given adapter was trained on.
    manifest = {
        "seed": args.seed, "max_rows_per_source": args.max_rows_per_source,
        "prediction_rows": len(all_cases), "selected_by_source": counts,
        "split_rows": {k: len(v) for k, v in splits.items()},
        "includes_original_fraudlm_data": include_original,
        'model_fact_training_rows': 0,
        'original_overlap_excluded': overlap_excluded,
        'prior_training_sha256': hashlib.sha256(Path(args.prior_training_file).read_bytes()).hexdigest() if include_original else None,
        'generated_file_sha256': {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                                  for name in ('fraudlm_sft_train.json', 'fraudlm_sft_val.json', 'fraudlm_sft_test.json', 'prediction_store.jsonl')},
        'registry_sha256': hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        'sources': {s['id']: {'prediction_file': s['prediction_file'],
                            'prediction_sha256': hashlib.sha256((root / s['prediction_file']).read_bytes()).hexdigest(),
                            'model_sha256': s.get('model_sha256'),
                            'row_index_definition': 'Zero-based CSV data-row index, excluding header'} for s in registry['sources']},
        "safety": "Target labels were excluded. Generated files are local and must not be published.",
    }
    (output / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def parser() -> argparse.ArgumentParser:
    """Command-line options: registry file, output folder, per-source row cap, seed and original-data mixing."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default=str(ROOT / "configs" / "prediction_sources.json"))
    ap.add_argument("--output", default=str(ROOT / "dataset_final_v2"))
    ap.add_argument("--max-rows-per-source", type=int, default=500,
                    help="balanced cap per source; 0 means every prediction and can be very large")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-original", action="store_true", help="do not mix the original FraudLM instruction rows")
    ap.add_argument('--prior-training-file', default=str(ROOT/'dataset_multisource/fraudlm_sft_train.json'),
                    help='Completed continued-run training rows used for original validation/test overlap screening')
    return ap


if __name__ == "__main__":
    build(parser().parse_args())
