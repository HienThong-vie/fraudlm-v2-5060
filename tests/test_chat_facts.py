"""
Tests for utils/chat_facts.py: the Facts block that the chat builds for dataset / model questions must look
exactly like the `### Input:` of the training `dataset_fact` rows, contain the right lines, and let the
reference answers pass the guard's number check.

Author: FraudLM v2 team (COS30049.2-1).

"""

import json
import re
import time
from pathlib import Path

import pytest

from utils import guard as G
from utils.chat_facts import MAX_LINES, FactIndex
from utils.generate import get_tokenizer
from utils.prompt import SYSTEM_RULES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def idx():
    try:
        return FactIndex.load()
    except Exception as e:  # EDA tables or model card not on this machine
        pytest.skip(f"fact sources not available: {e}")


def gold_rows(task="dataset_fact", splits=("train", "val", "test")):
    rows = []
    for sp in splits:
        p = ROOT / "dataset" / f"fraudlm_sft_{sp}.json"
        if not p.exists():
            pytest.skip("dataset not built")
        rows += [r for r in json.loads(p.read_text(encoding="utf-8")) if r["task"] == task]
    return rows


def best(idx, family):
    return max(idx.by_family[family], key=lambda f: f.raw["s"])


def test_loading_the_facts_is_fast_and_complete(idx):
    t = time.perf_counter()
    FactIndex.load()
    assert time.perf_counter() - t < 8
    assert len(idx.facts) == 154
    assert {"level", "month", "total", "pattern", "numeric", "recall", "assoc", "psi", "corr", "model"} <= set(idx.by_family)


def test_block_layout_is_the_training_layout(idx):
    sel = idx.select("What is the fraud rate for housing status BA?")
    lines = sel.input_text.split("\n")
    assert lines[0] == SYSTEM_RULES
    assert lines[1] == "Facts (BAF Base, development months 0-5):"
    assert all(l.startswith("- ") for l in lines[2:]) and [l[2:] for l in lines[2:]] == sel.lines
    # the same first two lines (rules, header) open real training inputs, for both header kinds
    heads = {tuple(r["input"].split("\n")[:2]) for r in gold_rows()}
    assert (SYSTEM_RULES, "Facts (BAF Base, development months 0-5):") in heads
    model = idx.select("What is the review threshold and how was it set?")
    assert (SYSTEM_RULES, model.input_text.split("\n")[1]) in heads
    assert model.input_text.split("\n")[1] == "Facts about the classifier:"


def test_every_line_is_a_real_fact_line(idx):
    every = {f.line for f in idx.facts}
    for q in ("What is the fraud rate for housing status BA?", "Which month had the highest fraud rate?",
              "How well does the model do on the held-out test months?"):
        assert set(idx.select(q).lines) <= every


CASES = [
    # question, family, a line that must be shown, max number of lines
    ("What is the fraud rate for housing status BA?", "level", "housing status = BA: 129,775 applications, 4,571 frauds, fraud rate 3.52%", 3),
    ("What is the fraud rate for devices running windows?", "level", "device operating system = windows:", 3),
    ("How often is fraud found when proposed credit limit = 1000?", "level", "proposed credit limit = 1000:", 3),
    ("How many frauds are there when home phone valid = no?", "level", "home phone valid = no: 473,148 applications, 6,105 frauds", 2),
    ("Which level of housing status has the highest fraud rate?", "level", "housing status = BA:", 7),
    ("Which month had the highest fraud rate?", "month", "month 5: 119,323 applications, 1,411 frauds, fraud rate 1.18%", 6),
    ("How did month 3 look for fraud?", "month", "month 3: 150,936 applications", 3),
    ("What is the overall fraud rate in months 0-5?", "total", "794,989 applications, 8,151 frauds, fraud rate 1.03%", 1),
    ("What is the PR-AUC of the model?", "model", "Held-out test months 6-7 (50,000-row sample): PR-AUC 0.1695", 3),
    ("What is the review threshold and how was it set?", "model", "Review threshold: 0.7462", 3),
    ("How much fraud does the model catch at the review threshold?", "model", "recall 0.514, false-positive rate 0.046", 3),
    ("Which model is used and how was it chosen?", "model", "logistic regression with balanced class weights", 3),
    ("How strongly are credit risk score and proposed credit limit correlated?", "corr", "Spearman correlation 0.634", 1),
    ("What is the fraud rate when previous address is missing, intended transfer amount present, bank history missing?",
     "pattern", "fraud rate 1.76%", 3),
    ("Did foreign request change much between months?", "psi", "foreign request: largest monthly drift (PSI) 0.01", 3),
    ("How much fraud does credit risk score catch on its own at a 5% false-positive rate?", "recall", "credit risk score: catches", 3),
    ("How strongly is housing status associated with fraud?", "assoc", "housing status: association with fraud (Cramer's V) 0.110", 3),
    ("How do the legitimate and fraud medians of name-email similarity compare?", "numeric", "name-email similarity: median 0.501", 3),
    ("What is the fraud rate for velocity over 24 hours?", "default", None, 3),
]


@pytest.mark.parametrize("q,family,must,max_lines", CASES)
def test_selection(idx, q, family, must, max_lines):
    sel = idx.select(q)
    assert sel is not None, q
    assert sel.family == family, (q, sel.family, sel.lines)
    assert 1 <= len(sel.lines) <= max_lines, sel.lines
    if must:
        assert any(must in l for l in sel.lines), (must, sel.lines)
    else:  # "not in the facts": none of the lines is about the asked feature
        assert not any("velocity over 24 hours" in l for l in sel.lines)


def test_rank_questions_always_contain_the_true_maximum(idx):
    for q, fam in (("Which feature drifted the most?", "psi"), ("Which feature is most associated with fraud?", "assoc"),
                   ("Which one catches the most fraud alone at a 5% false-positive rate?", "recall")):
        sel = idx.select(q)
        assert sel.family == fam and len(sel.lines) == 5
        assert best(idx, fam).line in sel.lines, q
    # the same for the highest-rate level and month
    sel = idx.select("Which level of housing status has the highest fraud rate?")
    assert max(idx.levels["housing_status"], key=lambda f: f.raw["rate"]).line in sel.lines
    assert max(idx.by_family["month"], key=lambda f: f.raw["rate"]).line in idx.select("Which month had the highest fraud rate?").lines


def test_a_named_level_comes_with_distractors_about_other_features(idx):
    sel = idx.select("What is the fraud rate for housing status BA?")
    assert sel.lines[0].startswith("housing status = BA") or any(l.startswith("housing status = BA") for l in sel.lines)
    assert sum(l.startswith("housing status") for l in sel.lines) == 1
    assert len(sel.lines) == 3


def test_selection_is_deterministic(idx):
    q = "What is the fraud rate for housing status BA?"
    assert idx.select(q).input_text == idx.select(q).input_text
    assert idx.select(q).input_text != idx.select(q.replace("BA", "BB")).input_text


@pytest.mark.parametrize("q", [
    "What is PR-AUC?", "What is a false positive?", "How do I report a lost card?", "What is phishing?",
    "How many accounts can I open?", "Why was this application flagged?", "Which factor moved the score the most?",
    "hi", "What is fraud?", "How do banks use machine learning?", "Tell me about mortgages", "",
])
def test_ordinary_questions_are_not_fact_questions(idx, q):
    assert idx.select(q) is None, q


def test_with_an_application_in_focus_only_explicit_data_questions_use_facts(idx):
    assert idx.select("What does keep-alive session tell us here?", strict=True) is None
    assert idx.select("What does the number next to Model score represent?", strict=True) is None
    assert idx.select("Is the score over the threshold?", strict=True) is None
    assert idx.select("What is the fraud rate for housing status BA?", strict=True) is not None
    assert idx.select("Which month had the highest fraud rate?", strict=True) is not None
    assert idx.select("What is the PR-AUC of the model?", strict=True) is not None


def test_retrieval_covers_the_reference_answers(idx):
    """For the training questions of dataset_fact, the chosen lines contain every number of the reference answer.

    'Which of these features ...' questions are left out: their reference answer is the maximum of a random
    subset of three to five features, the chat returns the true maximum instead.
    """
    rows = [r for r in gold_rows() if not (re.search(r"\b(highest|largest|biggest|most|worst|riskiest|best|strongest|top|peak|greatest)\b", r["instruction"], re.I)
                                           and re.search(r"\b(which|where|find|rank)\b", r["instruction"], re.I)
                                           and not re.search(r"level|month", r["instruction"], re.I))]
    none = ok = 0
    for r in rows:
        sel = idx.select(r["instruction"])
        if sel is None:
            none += 1
        elif not G.invented_numbers(r["output"], sel.input_text, r["instruction"]):
            ok += 1
    assert none <= 0.01 * len(rows), none
    assert ok >= 0.97 * len(rows), (ok, len(rows))


def test_blocks_fit_the_window_and_never_name_a_protected_attribute(idx):
    enc = get_tokenizer()
    for r in gold_rows(splits=("val", "test")):
        sel = idx.select(r["instruction"])
        if sel is None:
            continue
        assert len(sel.lines) <= MAX_LINES
        assert len(enc.encode(sel.input_text)) <= 600, sel.input_text
        assert not G.protected_hits("\n".join(sel.lines), True), sel.lines


def test_the_guard_checks_answers_against_the_selected_lines(idx):
    q = "What is the fraud rate for housing status BA?"
    sel = idx.select(q)
    good = "For housing status = BA, the fraud rate is 3.52% (4,571 frauds among 129,775 applications)."
    assert G.check_answer(good, q, None, grounding_text=sel.input_text, fact_lines=sel.lines).status == "ok"
    bad = G.check_answer("For housing status = BA, the fraud rate is 9.99% (5,000 frauds among 129,775 applications).",
                         q, None, grounding_text=sel.input_text, fact_lines=sel.lines)
    assert bad.status == "guarded" and any("9.99" in r for r in bad.reasons)
    assert bad.text == G.fallback_facts(sel.lines)
    assert "3.52%" in bad.text and "9.99" not in bad.text
    # the replacement passes the guard itself
    assert G.check_answer(bad.text, q, None, grounding_text=sel.input_text, fact_lines=sel.lines).status == "ok"


def test_a_missing_feature_is_answered_from_unrelated_lines_the_model_must_decline(idx):
    sel = idx.select("What is the fraud rate for velocity over 24 hours?")
    assert sel.family == "default"
    # a made-up number is still caught
    chk = G.check_answer("The fraud rate for application velocity over 24 hours is 2.50%.", "q", None,
                         grounding_text=sel.input_text, fact_lines=sel.lines)
    assert chk.status == "guarded"
