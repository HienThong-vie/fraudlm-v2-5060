"""Shared prompt/context helpers for predictions from the eight fraud classifiers."""

from __future__ import annotations

import json
import re
import hashlib
from decimal import Decimal

from utils.guard import check_answer

# Fixed rule line placed at the top of every grounded prompt, in training and at chat time.
MULTISOURCE_RULES = (
    "Rules: support a fraud analyst; never approve, reject, block or freeze anything; "
    "a score is a review signal, not proof of fraud; use only the supplied facts."
)


def format_prediction_context(case: dict, include_rules: bool = True) -> str:
    """Render only facts that are safe to ground an answer in (never the target label)."""
    score = float(case["score"])
    threshold = float(case["threshold"])
    # A case is flagged when its score reaches the threshold (equal counts as flagged).
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
    if case.get('experiment'):
        lines.append(f"Model experiment: {case['experiment']}; calibration FPR budget: {case['fpr_budget']:.1%}.")
        lines.append('Observed signals below are inputs, not feature-attribution or causal evidence.')
    if case.get('offline_only'):
        lines.append('Availability: offline AML benchmark; not validated for live transaction decisions.')
    signals = case.get("signals") or {}
    if signals:
        lines.append("Approved case signals:")
        lines.extend(f"- {name}: {value}" for name, value in signals.items())
    else:
        lines.append("Case-level explanation factors: not available for this classifier output.")
    return "\n".join(lines)


def prediction_answer(case: dict, intent: str) -> str:
    """Template answer for one stored case: the training target and the guard's fallback reply.

    `intent` is "explain", "next_steps" or "limitations". Every answer repeats the exact score,
    threshold and flag status, says the score is not proof of fraud, and mentions model factors only if
    the case supplies them.
    """
    score = float(case["score"])
    threshold = float(case["threshold"])
    flagged = score >= threshold
    result = "flagged for review" if flagged else "not flagged"
    # Pick one of three wordings from a hash of the case key, so a case always gets the same wording.
    style = int(hashlib.sha256(str(case.get('case_key', case['case_id'])).encode()).hexdigest()[:8], 16) % 3
    bases = (
        f"{case['dataset_name']} case {case['case_id']} has a model score of {score:.6f}. The review threshold is {threshold:.6f}, so it is {result}.",
        f"For {case['dataset_name']} case {case['case_id']}, compare score {score:.6f} with threshold {threshold:.6f}: the recorded result is {result}.",
        f"The recorded result for {case['dataset_name']} case {case['case_id']} is {result}. Its score is {score:.6f}; the review threshold is {threshold:.6f}.",
    )
    base = bases[style]
    if intent == "next_steps":
        return (base + " An analyst should verify the underlying transaction and customer evidence, check for "
                "related activity, and document the review. The score alone is not proof of fraud and must not "
                "make an automatic customer decision.")
    if intent == "limitations":
        factor_note = (" Supplied model-attribution factors are listed in the context; they are not causal evidence."
                       if case.get('signals', {}).get('main model factors') else
                       " No model-attribution factors are supplied; observed inputs alone do not explain the score.")
        return (base + factor_note + " Performance can change under data drift; use the exploratory metrics "
                "for model-level reliability and investigate the case separately.")
    if case.get('signals', {}).get('main model factors'):
        return (base + " Supplied model-attribution factors: " + case['signals']['main model factors']
                + ". These describe model behaviour, not causes or proof of fraud. An analyst must review the evidence.")
    return (base + " It is a ranking signal for analyst review, not proof of fraud, a calibrated certainty, or a fraud verdict. "
            "No case-level reason should be invented when the classifier output does not provide one.")


def load_prediction_store(path) -> dict[str, dict]:
    """Read a prediction_store.jsonl file into {case_key: case}, where case_key is "<dataset_id>:<case_id>"."""
    cases = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            cases[row["case_key"]] = row
    return cases


def calculated_threshold_answer(question: str, case: dict) -> str | None:
    """Answer score/threshold gap questions with stored numbers, never model arithmetic."""
    # Detect "how far is the score from the threshold" style questions with simple keyword rules.
    has_threshold = re.search(r"\b(?:threshold|cutoff|cut-off)\b", question, re.I)
    has_score = re.search(r"\bscore\b", question, re.I)
    asks_gap = re.search(r"\b(?:difference|gap|distance|margin)\b", question, re.I)
    asks_distance = re.search(r"\bhow\s+(?:far|much)\b", question, re.I)
    has_comparison = has_score or re.search(r"\b(?:above|below|higher|lower|away|apart)\b", question, re.I)
    if not has_threshold or not ((has_score and asks_gap) or (asks_distance and has_comparison)):
        return None
    # Decimal avoids binary floating-point rounding, so the six printed decimals are exact.
    score = Decimal(str(case["score"]))
    threshold = Decimal(str(case["threshold"]))
    gap = threshold - score
    base = f"The model score is {score:.6f} and the review threshold is {threshold:.6f}. "
    if gap > 0:
        return base + f"The score is {gap:.6f} below the review threshold, so it is not flagged."
    if gap < 0:
        return base + f"The score is {-gap:.6f} above the review threshold, so it is flagged for review."
    return base + "The difference is 0.000000: the score equals the threshold, so it is flagged for review."


# Guard patterns: customer-action words ("approve", "block", ...) and flat fraud verdicts ("is fraud").
_UNSAFE_DECISION = re.compile(
    # A frozen threshold is a fixed model setting, not a customer action.
    r"\b(?:approve|approved|reject|rejected|block|blocked|freeze|"
    r"frozen(?!\s+(?:review\s+)?threshold\b)|close the account|deny|denied)\b", re.I)
_FRAUD_VERDICT = re.compile(r"\b(?:is|was|definitely|proves?|confirmed)\s+(?:a\s+)?fraud(?:ulent)?\b", re.I)


def prediction_guard_reasons(text: str, case: dict) -> list[str]:
    """Report the exact rules that would replace this model answer."""
    raw = (text or "").strip()
    reasons = []
    # Each check below appends one human-readable reason; an empty list means the answer passes.
    if not raw:
        reasons.append("Empty model response")
    decision = _UNSAFE_DECISION.search(raw)
    if decision:
        reasons.append(f"Decision-language rule matched {decision.group(0)!r}")
    verdict = _FRAUD_VERDICT.search(raw)
    if verdict:
        reasons.append(f"Fraud-verdict rule matched {verdict.group(0)!r}")
    # The flag status the model states must agree with the stored score/threshold comparison.
    flagged = float(case["score"]) >= float(case["threshold"])
    low = raw.lower()
    negative = re.search(r"\bnot\s+flagged\b(?:\s+for\s+review)?", low)
    positive_only = re.sub(r"\bnot\s+flagged\b(?:\s+for\s+review)?", "", low)
    contradictory = ((bool(negative) and flagged) or
                     (bool(re.search(r"\bflagged\b", positive_only)) and not flagged))
    if contradictory:
        reasons.append("Flagged status contradicts the stored score/threshold comparison")
    if re.search(r"\b(?:not at risk|risk[- ]free|guaranteed safe|definitely safe)\b", low):
        reasons.append("Unsupported guarantee of safety")
    # The answer may not quote a different case ID, or name a different classifier family.
    if case.get('case_id'):
        mentioned = re.findall(r'\bcase\s+(?:id\s*:?\s*)?(\d[\w.-]*)', raw, re.I)
        if any(value.rstrip('.') != str(case['case_id']) for value in mentioned):
            reasons.append('Case ID differs from the supplied case')
    expected_model = re.sub(r'[^a-z0-9]', '', str(case.get('model', '')).lower())
    families = {'catboost': r'\bcatboost\b', 'xgboost': r'\bxgboost\b',
                'randomforest': r'\brandom[ _-]?forest\b',
                'logisticregression': r'\blogistic[ _-]?regression\b',
                'histgradientboosting': r'\bhist[ _-]?gradient[ _-]?boosting\b',
                'extratrees': r'\bextra[ _-]?trees\b'}
    if expected_model in families and any(re.search(pattern, low) for name, pattern in families.items() if name != expected_model):
        reasons.append('Classifier identity differs from the supplied current model')
    # Without model-attribution factors, input signals must not be described as reasons for the score.
    if not case.get('signals', {}).get('main model factors') and re.search(
            r'\bso\s+(?:the\s+)?(?:model|classifier)\s+(?:gave|produced)|\b(?:raises?|lowers?)\s+(?:the\s+)?(?:model\s+)?score', low):
        reasons.append('Input signals presented as model attributions without attribution evidence')
    return reasons


def guard_prediction_answer(text: str, case: dict | None, intent: str = "explain",
                            question: str = "") -> tuple[str, bool]:
    """Replace an unsafe/contradictory grounded answer with a deterministic fact-only answer."""
    raw = (text or "").strip()
    # Two layers: the general FraudLM guard (utils.guard) and the prediction-specific rules above.
    checked = check_answer(raw, question=question, ctx=None,
                           grounding_text=format_prediction_context(case) if case else None)
    unsafe = checked.status == 'guarded' or bool(prediction_guard_reasons(raw, case) if case else [])
    if unsafe:
        return (prediction_answer(case, intent) if case else checked.text), True
    return checked.text, False


def source_facts(source: dict) -> str:
    """Runtime registry facts; never ask the adapter to recall mutable model settings."""
    lines = [f"Dataset: {source['name']}", f"Classifier: {source['model']}",
             f"Review threshold: {source['threshold']:.6f}"]
    for name in ('pr_auc', 'precision', 'recall', 'f1', 'roc_auc', 'actual_test_fpr'):
        if name in source:
            lines.append(f"Exploratory {name}: {source[name]:.6f}")
    lines.append('These are model-level exploratory results, not proof of fraud or an individual probability.')
    if source.get('offline_only'):
        lines.append('Offline AML benchmark; not validated for live transaction decisions.')
    return '\n'.join(lines)
