import pytest

from chat_multisource import MultiSourceSession, main, parser
from utils.multisource_prompt import guard_prediction_answer, prediction_guard_reasons


CASE = {"case_key": "demo:7", "source_id": "demo", "case_id": "7",
        "dataset_name": "Demo", "dataset_kind": "synthetic", "model": "test model",
        "score": 0.2, "threshold": 0.6, "signals": {}}


def session_for(raw, debug=False):
    output, calls = [], []

    def generate(question, context):
        calls.append((question, context))
        return raw, {"prompt_tokens": 10, "new_tokens": 8}

    return MultiSourceSession({"demo:7": CASE}, generate, output.append, debug=debug), output, calls


def test_debug_keeps_same_final_reply_and_reveals_unsafe_decision():
    raw = "Block the account."
    normal, normal_output, _ = session_for(raw)
    debug, debug_output, _ = session_for(raw, debug=True)
    normal.handle("ask demo:7 Explain the result.")
    debug.handle("ask demo:7 Explain the result.")
    assert raw not in normal_output
    assert raw in debug_output
    assert any("matched 'Block'" in line for line in debug_output)
    assert [line for line in debug_output if line.startswith("FraudLM>")] == [
        line for line in normal_output if line.startswith("FraudLM>")]
    assert any("[guarded replacement]" in line for line in debug_output)


@pytest.mark.parametrize("raw,reason", [
    ("", "Empty model response"),
    ("Block the account.", "Decision-language rule matched 'Block'"),
    ("This is not confirmed fraud.", "Fraud-verdict rule matched 'confirmed fraud'"),
    ("The case is flagged for review.", "Flagged status contradicts"),
])
def test_debug_identifies_existing_guard_triggers(raw, reason):
    assert any(reason in item for item in prediction_guard_reasons(raw, CASE))
    assert guard_prediction_answer(raw, CASE)[1]


def test_debug_safe_answer_and_toggle_off():
    raw = "This case is not flagged; an analyst can check the available evidence."
    session, output, calls = session_for(raw)
    session.handle("debug on")
    session.handle("ask demo:7 Explain the result.")
    assert any("no rule matched" in line for line in output)
    assert "FraudLM> " + raw in output
    session.handle("debug off")
    output.clear()
    session.handle("ask demo:7 Explain the result.")
    assert not any(line.startswith("[debug]") for line in output)
    assert len(calls) == 2


def test_not_proof_of_fraud_does_not_trigger_a_verdict_rule():
    assert prediction_guard_reasons("This score is not proof of fraud.", CASE) == []


def test_question_guard_reports_that_generation_was_skipped():
    session, output, calls = session_for("SHOULD NOT BE GENERATED", debug=True)
    session.handle("ask demo:7 Should I approve this application?")
    assert not calls
    assert any("Question guard: decision_request; model was not called" in line for line in output)
    assert "SHOULD NOT BE GENERATED" not in output


def test_debug_free_text_reports_missing_case_context():
    session, output, _ = session_for("A general answer.", debug=True)
    session.handle("What is a review threshold?")
    assert any("not applied (no case attached)" in line for line in output)


def test_cli_debug_flag_reaches_session(tmp_path, monkeypatch, capsys):
    import chat_multisource as chat
    adapter, store = tmp_path / "adapter.pt", tmp_path / "store.jsonl"
    adapter.touch()
    store.touch()

    class FakeEngine:
        def __init__(self, *args):
            pass

        def generate(self, question, context):
            return "The frozen threshold is 0.6.", {}

    monkeypatch.setattr(chat, "Engine", FakeEngine)
    monkeypatch.setattr(chat, "load_prediction_store", lambda path: {"demo:7": CASE})
    assert not parser().parse_args(["--adapter", str(adapter)]).debug
    assert main(["--adapter", str(adapter), "--store", str(store), "--size", "gpt2-xl",
                 "--debug", "--once", "ask demo:7 Explain the result."]) == 0
    output = capsys.readouterr().out
    assert "Raw model answer" in output and "no rule matched" in output


@pytest.mark.parametrize("threshold", ["frozen threshold", "frozen review threshold",
                                      "FROZEN REVIEW THRESHOLD", "frozen\nreview threshold"])
def test_fixed_threshold_wording_is_preserved(threshold):
    raw = f"The {threshold} is 0.6; the score is 0.2, so this case is not flagged."
    assert prediction_guard_reasons(raw, CASE) == []
    assert guard_prediction_answer(raw, CASE) == (raw, False)


@pytest.mark.parametrize("action", ["Freeze the account.", "The account should be frozen.",
                                   "Block this customer.", "Reject the transaction."])
def test_fixed_threshold_does_not_hide_a_later_customer_decision(action):
    raw = "The frozen review threshold is 0.6. " + action
    assert guard_prediction_answer(raw, CASE)[1]
    assert any("Decision-language rule" in reason for reason in prediction_guard_reasons(raw, CASE))


def test_reported_ieee_answer_reaches_the_user_without_replacement():
    case = dict(CASE, case_key="ieee_cis:3663555", case_id="3663555",
                dataset_name="IEEE-CIS Fraud Detection", score=0.200272, threshold=0.645987)
    raw = ("IEEE-CIS Fraud Detection case 3663555 has a model score of 0.200272. "
           "The frozen review threshold is 0.645987, so it is not flagged. "
           "This output does not include case-level explanation factors, so I cannot claim why the "
           "classifier produced this score. Performance can change under data drift; use the held-out "
           "metrics for model-level reliability and investigate the case separately.")
    assert guard_prediction_answer(raw, case) == (raw, False)
    output = []
    session = MultiSourceSession({case["case_key"]: case}, lambda q, c: (raw, {}), output.append, debug=True)
    session.handle("ask ieee_cis:3663555 Explain this classifier result.")
    assert "FraudLM> " + raw in output
    assert any("no rule matched" in line for line in output)
    assert not any("[guarded replacement]" in line for line in output)


def test_fixed_threshold_does_not_disable_other_guard_checks():
    for text in ("The frozen review threshold is 0.6. This case is flagged for review.",
                 "The frozen review threshold is 0.6. This is confirmed fraud."):
        assert guard_prediction_answer(text, CASE)[1]
