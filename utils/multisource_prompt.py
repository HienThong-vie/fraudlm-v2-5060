"""Shared prompt/context helpers for predictions from the eight fraud classifiers."""

from __future__ import annotations

import json
import re

MULTISOURCE_RULES = (
    "Rules: support a fraud analyst; never approve, reject, block or freeze anything; "
    "a score is a review signal, not proof of fraud; use only the supplied facts."
)


def format_prediction_context(case: dict, include_rules: bool = True) -> str:
    """Render only facts that are safe to ground an answer in (never the target label)."""
    score = float(case["score"])
    threshold = float(case["threshold"])
    flagged = score >= threshold
    lines = [MULTISOURCE_RULES] if include_rules else []
    lines += [
        f"Dataset: {case['dataset_name']} ({case['dataset_kind']})",
        f"Case ID: {case['case_id']}",
        f"Classifier: {case['model']}",
        f"Model score: {score:.6f}",
        f"Review threshold: {threshold:.6f}",
        f"Result: {'flagged for review' if flagged else 'not flagged'}",
    ]
    signals = case.get("signals") or {}
    if signals:
        lines.append("Approved case signals:")
        lines.extend(f"- {name}: {value}" for name, value in signals.items())
    else:
        lines.append("Case-level explanation factors: not available for this classifier output.")
    return "\n".join(lines)


def prediction_answer(case: dict, intent: str) -> str:
    score = float(case["score"])
    threshold = float(case["threshold"])
    flagged = score >= threshold
    result = "flagged for review" if flagged else "not flagged"
    base = (f"{case['dataset_name']} case {case['case_id']} has a model score of {score:.6f}. "
            f"The frozen review threshold is {threshold:.6f}, so it is {result}.")
    if intent == "next_steps":
        return (base + " An analyst should verify the underlying transaction and customer evidence, check for "
                "related activity, and document the review. The score alone is not proof of fraud and must not "
                "make an automatic customer decision.")
    if intent == "limitations":
        return (base + " This output does not include case-level explanation factors, so I cannot claim why the "
                "classifier produced this score. Performance can change under data drift; use the held-out metrics "
                "for model-level reliability and investigate the case separately.")
    return (base + " It is a ranking signal for analyst review, not proof of fraud, a calibrated certainty, or a fraud verdict. "
            "No case-level reason should be invented when the classifier output does not provide one.")


def load_prediction_store(path) -> dict[str, dict]:
    cases = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cases[row["case_key"]] = row
    return cases


_UNSAFE_DECISION = re.compile(
    r"\b(?:approve|approved|reject|rejected|block|blocked|freeze|frozen|close the account|deny|denied)\b", re.I)
_FRAUD_VERDICT = re.compile(r"\b(?:is|was|definitely|proves?|confirmed)\s+(?:a\s+)?fraud(?:ulent)?\b", re.I)


def guard_prediction_answer(text: str, case: dict, intent: str = "explain") -> tuple[str, bool]:
    """Replace an unsafe/contradictory grounded answer with a deterministic fact-only answer."""
    raw = (text or "").strip()
    flagged = float(case["score"]) >= float(case["threshold"])
    contradictory = (("not flagged" in raw.lower() and flagged) or
                     ("flagged for review" in raw.lower() and not flagged))
    unsafe = not raw or bool(_UNSAFE_DECISION.search(raw)) or bool(_FRAUD_VERDICT.search(raw)) or contradictory
    return (prediction_answer(case, intent), True) if unsafe else (raw, False)
