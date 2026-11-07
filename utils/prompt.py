"""Alpaca prompt template for FraudLM v2 (the ONE place the template is defined).

Purpose: every module (dataset builder, trainer, evaluator, chat) imports the template from here,
so the text the model sees in training is byte-identical to the text it sees at chat time.
Author: FraudLM v2 team (COS30049.2-1).
Template format follows the Alpaca instruction format (Taori et al. 2023), as used in
Raschka, S. (2024) *Build a Large Language Model (From Scratch)*, Manning; code repo Apache-2.0.


Contract (PLAN.md Section 3):
    format_prompt(instruction, input="") -> str   ends with "### Response:\\n"
    format_example(ex) -> (prompt, response_with_eos)
    SYSTEM_RULES: str  one policy line placed at the top of the `input` of every grounded task

Additions (logged in TEAM_LOG.md; the contract above is unchanged):
    FEATURE_LABELS, format_value(), format_score_context(): the plain-language score context block.
    The dataset builder and the chat demo must call format_score_context() so a training `input`
    and a live chat `input` have exactly the same layout.
"""

from __future__ import annotations

import math

# GPT-2 end-of-text token as text. tiktoken's "gpt2" encoding maps it to id 50256, but only when
# the caller passes allowed_special={"<|endoftext|>"}; the trainer must do that.
EOS_TEXT = "<|endoftext|>"

HEADER = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)

# Short policy line. Put at the top of `input` for grounded tasks (explain_score, dataset_fact,
# policy) so the model sees the rules next to the facts. It restates OOS07 and D7 from PLAN.md.
SYSTEM_RULES = (
    "Rules: you support a fraud analyst and never approve, reject or block an application; "
    "use only the facts given here; do not cite age, income or employment as a reason."
)


def format_prompt(instruction: str, input: str = "") -> str:
    """Return the prompt text. Ends with '### Response:\\n' so generation continues from there.

    The whole '### Input:' block is left out when `input` is empty (Alpaca convention).
    """
    instruction = instruction.strip()
    input = (input or "").strip()
    parts = [HEADER, "", "### Instruction:", instruction]
    if input:
        parts += ["", "### Input:", input]
    parts += ["", "### Response:", ""]
    return "\n".join(parts)


def format_example(ex: dict) -> tuple[str, str]:
    """Return (prompt, response_with_eos) for one dataset row.

    The trainer masks the prompt tokens (-100) and learns only on the response, which ends with EOS.
    prompt + response is the exact training text.
    """
    prompt = format_prompt(ex["instruction"], ex.get("input", ""))
    response = ex["output"].strip() + EOS_TEXT
    return prompt, response


# ---------------------------------------------------------------------------------------------
# Score context block (the `input` of explain_score examples and of live chat)
# ---------------------------------------------------------------------------------------------

# Plain-language names for the 26 model inputs of the team classifier. Protected attributes
# (customer_age, income, employment_status) and non-inputs (x1, x2, month, device_fraud_count)
# are deliberately absent: they may never be named as a factor (PLAN.md D7).
FEATURE_LABELS: dict[str, str] = {
    "name_email_similarity": "name-email similarity",
    "prev_address_months_count": "months at previous address",
    "current_address_months_count": "months at current address",
    "days_since_request": "days since request",
    "intended_balcon_amount": "intended initial transfer amount",
    "payment_type": "payment plan type",
    "zip_count_4w": "applications from the same ZIP code over 4 weeks",
    "velocity_6h": "application velocity over 6 hours",
    "velocity_24h": "application velocity over 24 hours",
    "velocity_4w": "application velocity over 4 weeks",
    "bank_branch_count_8w": "applications at the same bank branch over 8 weeks",
    "date_of_birth_distinct_emails_4w": "distinct emails sharing the same date of birth over 4 weeks",
    "credit_risk_score": "credit risk score",
    "email_is_free": "free email domain",
    "housing_status": "housing status",
    "phone_home_valid": "home phone valid",
    "phone_mobile_valid": "mobile phone valid",
    "bank_months_count": "months with a previous bank account",
    "has_other_cards": "other cards held",
    "proposed_credit_limit": "proposed credit limit",
    "foreign_request": "foreign request",
    "source": "application channel",
    "session_length_in_minutes": "session length in minutes",
    "device_os": "device operating system",
    "keep_alive_session": "keep-alive session",
    "device_distinct_emails_8w": "distinct emails on the device over 8 weeks",
}

# 0/1 columns that are shown as yes / no
BINARY_FEATURES = frozenset(
    {"email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards",
     "foreign_request", "keep_alive_session"}
)

THRESHOLD_NOTE = "set for a 5% false-positive rate"
# 'raises' / 'lowers' is measured against the average applicant of the classifier's training months (0-4);
# the factor lines and the answers say so (G3 review F1), otherwise a not-flagged result whose factors all
# 'raise' the score reads as a contradiction.
DIRECTION_BASIS = "compared with a typical application"
NOT_PROVIDED = "not provided"


def format_value(feature: str, value) -> str:
    """Show one raw feature value the way the context block does.

    Missing (None / NaN) and the BAF negative missing code of intended_balcon_amount become
    'not provided'. 0/1 flags become yes / no. Whole numbers print without decimals, values of 100
    or more are rounded, other numbers keep 3 significant digits. Never scientific notation.
    """
    if value is None:
        return NOT_PROVIDED
    if isinstance(value, str):
        try:
            x = float(value)
        except ValueError:
            return value
    else:
        x = float(value)
    if not math.isfinite(x):
        return NOT_PROVIDED
    if feature == "intended_balcon_amount" and x < 0:
        return NOT_PROVIDED
    if feature in BINARY_FEATURES:
        return "yes" if x >= 0.5 else "no"
    if x.is_integer():
        return str(int(x))
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.3g}" if abs(x) >= 0.001 else f"{x:.4f}"


def format_score_context(score: float, threshold: float, factors: list[dict], rules: bool = True) -> str:
    """Build the `input` block for one scored application.

    factors: [{"feature": column, "direction": "raises"|"lowers", "value": raw value}, ...]
             non-protected features only, strongest first (the caller guarantees this). The direction is
             relative to a typical application (the average applicant of months 0-4).
    The score and threshold use 4 decimals so a score just above the threshold never prints equal to it.
    """
    flagged = score >= threshold
    lines = []
    if rules:
        lines.append(SYSTEM_RULES)
    lines += [
        f"Model score: {score:.4f}",
        f"Review threshold: {threshold:.4f} ({THRESHOLD_NOTE})",
        f"Result: {'flagged for review' if flagged else 'not flagged'}",
        "Main factors:",
    ]
    for f in factors:
        name = FEATURE_LABELS[f["feature"]]  # KeyError on a protected / unknown column is intended
        lines.append(f"- {name}: {format_value(f['feature'], f.get('value'))} ({f['direction']} the model score {DIRECTION_BASIS})")
    return "\n".join(lines)
