"""Regression checks for exact case-grounded arithmetic and honest response provenance."""

import pytest

from chat_multisource import MultiSourceSession
from utils.multisource_prompt import calculated_threshold_answer


QUESTION = "What is the difference between this case's model score and its review threshold?"
CASE = {"case_key": "ieee_cis:3663555", "source_id": "ieee_cis", "case_id": "3663555",
        "dataset_name": "IEEE-CIS Fraud Detection", "dataset_kind": "real, anonymized",
        "model": "test classifier", "score": 0.200272, "threshold": 0.645987, "signals": {}}


@pytest.mark.parametrize("score,threshold,expected", [
    (0.200272, 0.645987, "0.445715 below the review threshold, so it is not flagged"),
    (0.8, 0.6, "0.200000 above the review threshold, so it is flagged for review"),
    (0.6, 0.6, "equals the threshold, so it is flagged for review"),
    (0.1, 0.3, "0.200000 below"),
    (0, 1, "1.000000 below"),
])
def test_difference_uses_stored_values_and_threshold_boundary(score, threshold, expected):
    assert expected in calculated_threshold_answer(QUESTION, dict(CASE, score=score, threshold=threshold))


@pytest.mark.parametrize("question", [QUESTION,
    "Calculate the gap between the score and threshold.",
    "How far below the review threshold is this case?",
    "How much higher is the score than the cutoff?",
    "What is the margin between the score and the cut-off?",
])
def test_common_gap_questions_are_routed(question):
    assert "0.445715 below" in calculated_threshold_answer(question, CASE)


@pytest.mark.parametrize("question", [
    "Explain this classifier result.", "What is a review threshold?",
    "What is the difference between fraud and a false positive?",
    "How far did this customer travel?", "How much data was used to choose the threshold?",
])
def test_other_questions_stay_with_the_model(question):
    assert calculated_threshold_answer(question, CASE) is None


def test_case_gap_bypasses_model_and_reports_calculated_provenance():
    output = []

    def unexpected_generation(*args):
        pytest.fail("A stored-fact calculation must not call the model")

    session = MultiSourceSession({CASE["case_key"]: CASE}, unexpected_generation, output.append, debug=True)
    session.handle("ask ieee_cis:3663555 " + QUESTION)
    assert any("0.445715 below" in line for line in output)
    assert "  [calculated from stored facts]" in output
    assert any("model was not called" in line for line in output)
    assert not any("new tokens" in line or "Raw model answer" in line for line in output)


def test_question_guard_still_blocks_customer_decision_requests():
    output = []
    session = MultiSourceSession({CASE["case_key"]: CASE}, lambda *args: pytest.fail("No generation"),
                                 output.append)
    session.handle("ask ieee_cis:3663555 " + QUESTION + " Should I approve this application?")
    assert not any("calculated from stored facts" in line for line in output)
    assert any("can't approve" in line for line in output)


def test_without_a_case_no_numbers_are_invented():
    calls, output = [], []

    def generate(question, context):
        calls.append((question, context))
        return "Please supply a case.", {}

    session = MultiSourceSession({}, generate, output.append)
    session.handle(QUESTION)
    assert len(calls) == 1 and calls[0][1] == ""
    assert not any("calculated from stored facts" in line for line in output)
