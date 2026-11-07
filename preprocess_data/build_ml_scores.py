"""Build the ML grounding table preprocessed/ml_scores.csv from the team's frozen classifier.

Purpose: give the FraudLM dataset builder real model outputs (score, threshold, flag, top factors)
so every number the chatbot learns to explain comes from the team classifier, not from invention.

Source of the scores
    The team repo (KennyDao1324/ai4cyber-fraud) ships artifacts/production_model.joblib: a scikit-learn
    Pipeline (imputer + scaler + one-hot, then LogisticRegression with class_weight balanced) fitted on
    Base months 0-4, with the review threshold 0.7462 set at a 5% false-positive rate on validation
    month 5. This script loads that artefact and scores BAF Base. If the artefact is missing it falls
    back to a placeholder LR fitted here on months 0-4 (flag --allow-placeholder); the sidecar JSON
    ml_scores.meta.json records which one was used ("frozen" or "placeholder").

Top factors
    The LR is linear, so the contribution of one transformed column is coef * (x - mean), where mean is
    the average applicant of months 0-4. Contributions of one original feature (a numeric column, its
    missing-indicator, or the levels of one categorical column) are summed. Only non-protected model inputs
    are kept (the three protected columns are not model inputs anyway). direction = raises / lowers the score.

Rows
    ml_scores.csv                  development months 0-5 only (training material for the LLM)
    ml_scores_months6_7_demo.csv   test months 6-7, for the chat demo / evaluation ONLY, never for training

Columns: app_id (row number in Base.csv), month, score, threshold, flagged, label (the true BAF label, for
evaluation only, never shown to the model), top_factors (JSON list of {feature, direction, contribution, value}).

Run from llm_ver2/:  .venv/Scripts/python -m preprocess_data.build_ml_scores

Author: FraudLM v2 team (COS30049.2-1).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_data import common as C
from utils.prompt import FEATURE_LABELS

TARGET_FPR = 0.05


def load_frozen():
    import joblib

    art = joblib.load(C.ARTIFACT)
    assert art["model_name"] == "logistic_regression", art["model_name"]
    return art


def prepare_features(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Same input handling as the repo's inference code: model columns only, -1 sentinels -> NaN."""
    X = df[art["feature_columns"]].copy()
    for col in art["missing_sentinel_columns"]:
        if col in X.columns:
            X[col] = X[col].mask(X[col].eq(-1))
    return X


def original_feature(transformed_name: str, features: list[str]) -> str:
    """'num__missingindicator_x' -> x, 'num__x' -> x, 'cat__housing_status_BA' -> housing_status."""
    body = transformed_name.split("__", 1)[1]
    if body.startswith("missingindicator_"):
        body = body[len("missingindicator_"):]
    for f in sorted(features, key=len, reverse=True):
        if body == f or body.startswith(f + "_"):
            return f
    raise ValueError(f"cannot map {transformed_name}")


class Decomposer:
    """Per-original-feature contribution to the logit relative to the average months 0-4 applicant."""

    def __init__(self, art: dict, ref_X: pd.DataFrame) -> None:
        self.pipe = art["pipeline"]
        pre = self.pipe.named_steps["preprocessor"]
        self.coef = self.pipe.named_steps["model"].coef_[0]
        self.intercept = float(self.pipe.named_steps["model"].intercept_[0])
        names = list(pre.get_feature_names_out())
        self.features = list(art["feature_columns"])
        owner = [original_feature(n, self.features) for n in names]
        self.groups = {f: [j for j, o in enumerate(owner) if o == f] for f in self.features}
        ref = pre.transform(ref_X)
        self.ref_mean = np.asarray(ref.mean(axis=0)).ravel()

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_features) array, columns in self.features order."""
        T = self.pipe.named_steps["preprocessor"].transform(X)
        T = T.toarray() if hasattr(T, "toarray") else np.asarray(T)
        per_col = (T - self.ref_mean) * self.coef
        return np.stack([per_col[:, self.groups[f]].sum(axis=1) for f in self.features], axis=1)

    def offset(self) -> float:
        """logit = intercept + sum(coef * ref_mean) + sum of all feature contributions."""
        return self.intercept + float(self.ref_mean @ self.coef)


def top_factors_json(contrib_row: np.ndarray, raw_row: pd.Series, features: list[str], k: int) -> str:
    allowed = [j for j, f in enumerate(features) if f in FEATURE_LABELS and f not in C.EXTRA_EXCLUDED_FACTORS]  # protected / non-inputs never appear
    order = sorted(allowed, key=lambda j: -abs(contrib_row[j]))[:k]
    out = []
    for j in order:
        f = features[j]
        v = raw_row[f]
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = None if np.isnan(v) else float(v)
        out.append({
            "feature": f,
            "direction": "raises" if contrib_row[j] > 0 else "lowers",
            "contribution": round(float(contrib_row[j]), 4),
            "value": v,
        })
    return json.dumps(out, ensure_ascii=False)


def sample_rows(df: pd.DataFrame, n_flagged: int, n_below: int, rng: np.random.Generator, threshold: float) -> pd.DataFrame:
    """Balanced sample: flagged rows, plus below-threshold rows half of them close to the threshold."""
    flagged = df[df["score"] >= threshold]
    below = df[df["score"] < threshold]
    near = below[below["score"] >= threshold - 0.15]
    far = below[below["score"] < threshold - 0.15]
    take = lambda d, n: d.iloc[rng.choice(len(d), size=min(n, len(d)), replace=False)]
    parts = [take(flagged, n_flagged), take(near, n_below // 2), take(far, n_below - n_below // 2)]
    return pd.concat(parts).sort_values("app_id").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev-flagged", type=int, default=7000, help="flagged development rows to keep")
    ap.add_argument("--dev-below", type=int, default=7000, help="below-threshold development rows to keep")
    ap.add_argument("--demo-flagged", type=int, default=300, help="flagged months 6-7 rows for the demo file")
    ap.add_argument("--demo-below", type=int, default=300, help="below-threshold months 6-7 rows for the demo file")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--allow-placeholder", action="store_true", help="fit a placeholder LR if the frozen artefact is missing")
    args = ap.parse_args()

    rng = np.random.default_rng(C.SEED)
    warnings.filterwarnings("ignore", category=UserWarning)
    df = pd.read_csv(C.BASE_ZIP)
    df.insert(0, "app_id", np.arange(len(df), dtype=np.int64))
    print(f"Base rows: {len(df):,}")

    if C.ARTIFACT.is_file():
        art = load_frozen()
        source = "frozen"
        threshold = float(art["threshold"])
    elif args.allow_placeholder:
        raise SystemExit("placeholder fit is not needed: the frozen artefact exists in the team repo")
    else:
        raise SystemExit(f"frozen artefact not found: {C.ARTIFACT}")

    X_all = prepare_features(df, art)
    df["score"] = art["pipeline"].predict_proba(X_all)[:, 1]
    print(f"threshold {threshold:.6f}; flagged share months 0-5: {(df.loc[df.month <= 5, 'score'] >= threshold).mean():.4f}")

    dec = Decomposer(art, X_all[df["month"] <= 4])  # reference = average applicant of the training months
    features = dec.features

    outputs = {}
    for name, mask, nf, nb in (
        ("dev", df["month"] <= C.DEV_MAX_MONTH, args.dev_flagged, args.dev_below),
        ("demo", df["month"] > C.DEV_MAX_MONTH, args.demo_flagged, args.demo_below),
    ):
        part = sample_rows(df[mask], nf, nb, rng, threshold)
        idx = part["app_id"].to_numpy()  # app_id == row index of df
        Xp = X_all.iloc[idx]
        contrib = dec.contributions(Xp)
        # sanity: the decomposition must reproduce the model's own logit
        logit = dec.offset() + contrib.sum(axis=1)
        model_logit = art["pipeline"].decision_function(Xp)
        assert np.allclose(logit, model_logit, atol=1e-6), "decomposition does not match the model logit"
        raw = Xp.reset_index(drop=True)
        # keep the raw (unmasked) intended_balcon_amount so format_value can show 'not provided' for negative codes
        raw["intended_balcon_amount"] = df["intended_balcon_amount"].iloc[idx].to_numpy()
        rows = pd.DataFrame({
            "app_id": part["app_id"],
            "month": part["month"],
            "score": part["score"].round(6),
            "threshold": round(threshold, 6),
            "flagged": (part["score"] >= threshold).astype(int),
            "label": part["fraud_bool"],
            "top_factors": [top_factors_json(contrib[i], raw.iloc[i], features, args.top_k) for i in range(len(part))],
        })
        outputs[name] = rows

    assert outputs["dev"]["month"].max() <= C.DEV_MAX_MONTH, "test months leaked into the development table"
    assert outputs["demo"]["month"].min() > C.DEV_MAX_MONTH
    outputs["dev"].to_csv(C.ML_SCORES, index=False)
    outputs["demo"].to_csv(C.ML_SCORES_DEMO, index=False)

    try:
        commit = subprocess.run(["git", "-C", str(C.ML_REPO), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    card = json.loads(C.MODEL_CARD.read_text(encoding="utf-8"))
    meta = {
        "source": source,
        "source_note": ("scores come from artifacts/production_model.joblib of the team repo (Base champion, logistic "
                        "regression); the team has not yet confirmed this is the frozen A2-14 model (CLAUDE.md open decision 2)"),
        "team_repo": "https://github.com/KennyDao1324/ai4cyber-fraud",
        "team_repo_commit": commit,
        "artifact_sha256": C.sha256_file(C.ARTIFACT),
        "model": art["model_name"],
        "threshold": threshold,
        "target_fpr": TARGET_FPR,
        "model_card_test_threshold": card["datasets"][0]["test_metrics"]["threshold"],
        "reference_population": "Base months 0-4 (average applicant for the factor contributions)",
        "dev_flag_rate_months0_5": round(float((df.loc[df["month"] <= C.DEV_MAX_MONTH, "score"] >= threshold).mean()), 4),
        "dev_rows": len(outputs["dev"]),
        "dev_flagged": int(outputs["dev"]["flagged"].sum()),
        "demo_rows_months6_7": len(outputs["demo"]),
        "dev_months": sorted(int(m) for m in outputs["dev"]["month"].unique()),
        "demo_months": sorted(int(m) for m in outputs["demo"]["month"].unique()),
        "seed": C.SEED,
        "top_k": args.top_k,
        "columns": ["app_id", "month", "score", "threshold", "flagged", "label", "top_factors"],
        "top_factors_keys": ["feature", "direction", "contribution", "value"],
        "contract_extension": "top_factors items carry an extra 'value' key (raw feature value) next to feature/direction/contribution",
    }
    C.ML_SCORES_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("source", "threshold", "dev_rows", "dev_flagged", "demo_rows_months6_7")}, indent=2))
    print("wrote", C.ML_SCORES, C.ML_SCORES_DEMO, C.ML_SCORES_META)


if __name__ == "__main__":
    sys.exit(main())
