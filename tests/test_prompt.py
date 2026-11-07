"""Tests for utils/prompt.py: the Alpaca template, EOS handling and the score context block.

Author: FraudLM v2 team (COS30049.2-1).
"""

import pytest

from utils import prompt as P

HEADER = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


def test_prompt_with_input_matches_contract():
    out = P.format_prompt("Say hi.", "some context")
    assert out == f"{HEADER}\n\n### Instruction:\nSay hi.\n\n### Input:\nsome context\n\n### Response:\n"


def test_prompt_without_input_omits_input_block():
    out = P.format_prompt("Say hi.")
    assert "### Input:" not in out
    assert out == f"{HEADER}\n\n### Instruction:\nSay hi.\n\n### Response:\n"
    assert P.format_prompt("Say hi.", "   ") == out  # whitespace-only input counts as empty


def test_prompt_ends_with_response_marker():
    assert P.format_prompt("x", "y").endswith("### Response:\n")


def test_format_example_returns_prompt_and_response_with_eos():
    prompt, response = P.format_example({"instruction": "Q?", "input": "", "output": " A. "})
    assert prompt == P.format_prompt("Q?", "")
    assert response == "A.<|endoftext|>"


def test_eos_is_one_gpt2_token():
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(P.EOS_TEXT, allowed_special={"<|endoftext|>"})
    assert ids == [50256]
    _, response = P.format_example({"instruction": "Q?", "input": "", "output": "A"})
    assert enc.encode(response, allowed_special={"<|endoftext|>"})[-1] == 50256


def test_system_rules_content():
    r = P.SYSTEM_RULES
    assert r.startswith("Rules:")
    assert "never approve, reject or block" in r
    assert "do not cite age, income or employment" in r


def test_feature_labels_have_no_protected_or_non_input_columns():
    for col in ("customer_age", "income", "employment_status", "x1", "x2", "month", "device_fraud_count", "fraud_bool"):
        assert col not in P.FEATURE_LABELS
    assert len(P.FEATURE_LABELS) == 26
    assert P.FEATURE_LABELS["velocity_4w"] == "application velocity over 4 weeks"


@pytest.mark.parametrize(
    "feature,value,expected",
    [
        ("housing_status", "BA", "BA"),
        ("email_is_free", 1, "yes"),
        ("keep_alive_session", 0.0, "no"),
        ("current_address_months_count", 12.0, "12"),
        ("velocity_4w", 5230.37, "5230"),
        ("name_email_similarity", 0.5009406, "0.501"),
        ("prev_address_months_count", None, "not provided"),
        ("prev_address_months_count", float("nan"), "not provided"),
        ("intended_balcon_amount", -3.2, "not provided"),
        ("intended_balcon_amount", 32.59, "32.6"),
    ],
)
def test_format_value(feature, value, expected):
    assert P.format_value(feature, value) == expected


def test_score_context_layout_and_flag():
    factors = [
        {"feature": "housing_status", "direction": "raises", "value": "BA"},
        {"feature": "has_other_cards", "direction": "lowers", "value": 1},
        {"feature": "credit_risk_score", "direction": "raises", "value": 338},
    ]
    ctx = P.format_score_context(0.81254, 0.7462441, factors)
    lines = ctx.split("\n")
    assert lines[0] == P.SYSTEM_RULES
    assert lines[1] == "Model score: 0.8125"
    assert lines[2] == "Review threshold: 0.7462 (set for a 5% false-positive rate)"
    assert lines[3] == "Result: flagged for review"
    assert lines[4] == "Main factors:"
    rel = "compared with a typical application"
    assert P.DIRECTION_BASIS == rel
    assert lines[5] == f"- housing status: BA (raises the model score {rel})"
    assert lines[6] == f"- other cards held: yes (lowers the model score {rel})"
    assert lines[7] == f"- credit risk score: 338 (raises the model score {rel})"
    below = P.format_score_context(0.5, 0.7462441, factors, rules=False)
    assert "Result: not flagged" in below and not below.startswith("Rules:")


def test_score_context_refuses_protected_features():
    with pytest.raises(KeyError):
        P.format_score_context(0.9, 0.7, [{"feature": "customer_age", "direction": "raises", "value": 40}])
