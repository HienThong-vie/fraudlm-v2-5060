"""Shared prompt/context helpers for predictions from the eight fraud classifiers."""

from __future__ import annotations

import json
import re
from decimal import Decimal

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


def calculated_threshold_answer(question: str, case: dict) -> str | None:
    """Answer score/threshold gap questions with stored numbers, never model arithmetic."""
    has_threshold = re.search(r"\b(?:threshold|cutoff|cut-off)\b", question, re.I)
    has_score = re.search(r"\bscore\b", question, re.I)
    asks_gap = re.search(r"\b(?:difference|gap|distance|margin)\b", question, re.I)
    asks_distance = re.search(r"\bhow\s+(?:far|much)\b", question, re.I)
    has_comparison = has_score or re.search(r"\b(?:above|below|higher|lower|away|apart)\b", question, re.I)
    if not has_threshold or not ((has_score and asks_gap) or (asks_distance and has_comparison)):
        return None
    score = Decimal(str(case["score"]))
    threshold = Decimal(str(case["threshold"]))
    gap = threshold - score
    base = f"The model score is {score:.6f} and the review threshold is {threshold:.6f}. "
    if gap > 0:
        return base + f"The score is {gap:.6f} below the review threshold, so it is not flagged."
    if gap < 0:
        return base + f"The score is {-gap:.6f} above the review threshold, so it is flagged for review."
    return base + "The difference is 0.000000: the score equals the threshold, so it is flagged for review."


_UNSAFE_DECISION = re.compile(
    # A frozen threshold is a fixed model setting, not a customer action.
    r"\b(?:approve|approved|reject|rejected|block|blocked|freeze|"
    r"frozen(?!\s+(?:review\s+)?threshold\b)|close the account|deny|denied)\b", re.I)
_FRAUD_VERDICT = re.compile(r"\b(?:is|was|definitely|proves?|confirmed)\s+(?:a\s+)?fraud(?:ulent)?\b", re.I)


def prediction_guard_reasons(text: str, case: dict) -> list[str]:
    """Report the exact rules that would replace this model answer."""
    raw = (text or "").strip()
    reasons = []
    if not raw:
        reasons.append("Empty model response")
    decision = _UNSAFE_DECISION.search(raw)
    if decision:
        reasons.append(f"Decision-language rule matched {decision.group(0)!r}")
    verdict = _FRAUD_VERDICT.search(raw)
    if verdict:
        reasons.append(f"Fraud-verdict rule matched {verdict.group(0)!r}")
    flagged = float(case["score"]) >= float(case["threshold"])
    contradictory = (("not flagged" in raw.lower() and flagged) or
                     ("flagged for review" in raw.lower() and not flagged))
    if contradictory:
        reasons.append("Flagged status contradicts the stored score/threshold comparison")
    return reasons


def guard_prediction_answer(text: str, case: dict, intent: str = "explain") -> tuple[str, bool]:
    """Replace an unsafe/contradictory grounded answer with a deterministic fact-only answer."""
    raw = (text or "").strip()
    unsafe = bool(prediction_guard_reasons(raw, case))
    return (prediction_answer(case, intent), True) if unsafe else (raw, False)
