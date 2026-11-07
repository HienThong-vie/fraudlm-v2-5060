"""Acceptance tests for the FraudLM v2 dataset (gate G3).

Checks: schema, unique ids, disjoint splits, token limits, protected-attribute and decision-word rules,
"every number comes from the input", sizes against the build manifest, months 6-7 kept out of training,
and the ml_scores grounding table. Run after `python -m preprocess_data.build_sft`.

Author: FraudLM v2 team (COS30049.2-1).
"""

import json
import re
from collections import Counter

import pandas as pd
import pytest

from preprocess_data import common as C
from utils.prompt import FEATURE_LABELS, format_example

TASKS = {"explain_score", "dataset_fact", "policy", "banking_qa", "general"}
KEYS = {"id", "instruction", "input", "output", "task", "source"}
SPLITS = ("train", "val", "test")

pytestmark = pytest.mark.skipif(not (C.DATASET / "fraudlm_sft_train.json").is_file(), reason="dataset not built yet")


@pytest.fixture(scope="module")
def data():
    return {s: json.loads((C.DATASET / f"fraudlm_sft_{s}.json").read_text(encoding="utf-8")) for s in SPLITS}


@pytest.fixture(scope="module")
def manifest():
    return json.loads((C.DATASET / "build_manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def demo():
    return json.loads((C.DATASET / "fraudlm_demo_months67.json").read_text(encoding="utf-8"))


def all_rows(data):
    return [r for s in SPLITS for r in data[s]]


# ------------------------------------------------------------------ schema
def test_schema_valid_for_every_row(data, demo):
    for r in all_rows(data) + demo:
        assert set(r) == KEYS, r["id"]
        assert all(isinstance(r[k], str) for k in KEYS), r["id"]
        assert r["task"] in TASKS, r["id"]
        assert r["instruction"].strip() and r["output"].strip() and r["source"].strip(), r["id"]
        assert "<|endoftext|>" not in r["instruction"] + r["input"] + r["output"], r["id"]
        assert not r["instruction"].startswith(" ") and not r["output"].endswith(" "), r["id"]


def test_ids_unique(data, demo):
    ids = [r["id"] for r in all_rows(data) + demo]
    assert len(ids) == len(set(ids))


def test_sources_are_recorded(data):
    for r in all_rows(data):
        s = r["source"]
        assert s in ("template", "llm:claude-sonnet") or "(" in s and ")" in s, (r["id"], s)
        if r["task"] == "general":
            assert "alpaca-cleaned" in s and "CC-BY-4.0" in s
        if r["task"] in ("explain_score", "dataset_fact", "policy"):
            assert s in ("template", "llm:claude-sonnet"), r["id"]


# ------------------------------------------------------------------ splits
def test_no_overlap_between_splits(data):
    keys = {s: {C.example_key(r["instruction"], r["input"]) for r in data[s]} for s in SPLITS}
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["test"]
    assert not keys["val"] & keys["test"]
    for s in SPLITS:  # and no duplicates inside a split
        assert len(keys[s]) == len(data[s]), s


def test_test_split_has_heldout_phrasings_never_seen_in_train_or_val(data):
    held = [r for r in data["test"] if r["id"].endswith("-heldout")]
    assert len(held) >= 100
    seen = {C.normalise(r["instruction"]) for s in ("train", "val") for r in data[s]}
    # explain_score / policy questions with a {factor} placeholder are filled with a factor name, so compare
    # only the held-out wordings; each held-out instruction must be absent from train and val
    for r in held:
        assert C.normalise(r["instruction"]) not in seen, r["id"]
    assert not any(r["id"].endswith("-heldout") for s in ("train", "val") for r in data[s])


def test_sizes_match_manifest_targets(data, manifest):
    for s in SPLITS:
        got = Counter(r["task"] for r in data[s])
        for task, want in manifest["splits"][s]["per_task"].items():
            assert got[task] == want
        targets = manifest["args"][f"n_{s}"]
        assert abs(len(data[s]) - targets) <= 0.02 * targets, (s, len(data[s]), targets)
    assert 8000 <= len(data["train"]) <= 15000 or manifest["args"]["n_train"] < 8000
    assert not manifest["shortfall"], manifest["shortfall"]


# ------------------------------------------------------------------ tokens
def test_token_lengths(data, demo):
    longest = 0
    for r in all_rows(data) + demo:
        prompt, resp = format_example(r)
        n = C.token_count(prompt + resp)
        longest = max(longest, n)
        assert n <= C.MAX_TOKENS, r["id"]
        assert n <= C.TARGET_TOKENS, (r["id"], n)
    assert longest > 0


# ------------------------------------------------------------------ safety rules
def test_no_protected_attribute_as_reason(data):
    for r in all_rows(data):
        out = r["output"]
        hit = C.PROTECTED_RE.findall(out)
        if not hit:
            continue
        # only refusal answers of the policy task may name them, and only in a sentence that negates them
        assert r["task"] == "policy", (r["id"], hit)
        for sent in re.split(r"(?<=[.!?])\s+", out):
            if C.PROTECTED_RE.search(sent):
                assert C.NEGATION_RE.search(sent), (r["id"], sent)


def test_no_decision_words_except_refusals(data):
    for r in all_rows(data):
        out = r["output"]
        if not C.DECISION_RE.search(out):
            continue
        assert r["task"] == "policy", (r["id"], out[:120])
        assert C.NEGATION_RE.search(out), (r["id"], out[:120])
        # the sentence containing the decision word must itself carry the negation
        for sent in re.split(r"(?<=[.!?])\s+", out):
            if C.DECISION_RE.search(sent):
                assert C.NEGATION_RE.search(sent), (r["id"], sent)


def test_grounded_tasks_never_call_an_application_fraudulent_or_a_probability(data):
    for r in all_rows(data):
        if r["task"] not in ("explain_score", "dataset_fact", "policy"):
            continue
        for m in C.VERDICT_RE.finditer(r["output"]):
            word = m.group(0).lower()
            assert r["task"] == "policy" and word in ("probability", "probable"), (r["id"], word)
        if re.search(r"probab", r["output"], re.IGNORECASE):
            assert C.NEGATION_RE.search(r["output"]), r["id"]


def test_public_rows_are_clean(data):
    for r in all_rows(data):
        if r["task"] in ("banking_qa", "general"):
            text = r["instruction"] + " " + r["input"] + " " + r["output"]
            assert not C.PROTECTED_RE.search(text), (r["id"], C.PROTECTED_RE.search(text).group(0))
            assert not C.DECISION_RE.search(text), (r["id"], C.DECISION_RE.search(text).group(0))
            assert "{{" not in text


def test_factor_names_in_contexts_are_model_inputs_only(data, demo):
    allowed = set(FEATURE_LABELS.values())
    for r in all_rows(data) + demo:
        if "Main factors:" not in r["input"]:
            continue
        block = r["input"].split("Main factors:\n", 1)[1]
        factor_lines = [ln for ln in block.split("\n") if ln.startswith("- ")]
        assert len(factor_lines) == 3, r["id"]
        for ln in factor_lines:
            name = ln[2:].split(":", 1)[0]
            assert name in allowed, (r["id"], name)


# ------------------------------------------------------------------ numbers come from the input
def _number_in(num: str, text: str) -> bool:
    return re.search(r"(?<![\d.,])" + re.escape(num) + r"(?!\d|\.\d)", text) is not None


def test_every_number_in_grounded_answers_appears_in_the_input(data, demo):
    checked = 0
    for r in all_rows(data) + demo:
        grounded = r["task"] in ("explain_score", "dataset_fact") or (
            r["task"] == "policy" and ("Model score:" in r["input"] or "Facts" in r["input"]))
        if not grounded:
            continue
        for num in C.numbers_in(r["output"]):
            # a number may also come from a feature name quoted in the question (e.g. "over 24 hours")
            assert _number_in(num, r["input"]) or _number_in(num, r["instruction"]), (r["id"], num)
            checked += 1
    assert checked > 10000


# ------------------------------------------------------------------ months 6-7 and the grounding table
def test_ml_scores_development_months_only_and_consistent():
    dev = pd.read_csv(C.ML_SCORES)
    assert dev["month"].max() <= C.DEV_MAX_MONTH
    assert ((dev["score"] >= dev["threshold"]).astype(int) == dev["flagged"]).all()
    assert list(dev.columns) == ["app_id", "month", "score", "threshold", "flagged", "label", "top_factors"]
    for tf in dev["top_factors"]:
        items = json.loads(tf)
        assert len(items) == 3
        for it in items:
            assert it["feature"] in FEATURE_LABELS  # protected / non-input columns are absent
            assert it["direction"] in ("raises", "lowers")
    demo = pd.read_csv(C.ML_SCORES_DEMO)
    assert demo["month"].min() >= C.DEV_MAX_MONTH + 1
    assert not set(demo["app_id"]) & set(dev["app_id"])


def test_ml_scores_sidecar_records_source():
    meta = json.loads(C.ML_SCORES_META.read_text(encoding="utf-8"))
    assert meta["source"] in ("frozen", "placeholder")
    assert abs(meta["threshold"] - meta["model_card_test_threshold"]) < 1e-9 or meta["source"] == "placeholder"


def test_months_6_7_contexts_never_in_training_files(data, demo):
    def block(r):
        return r["input"].split("\n", 1)[1] if "Model score:" in r["input"] else None

    demo_blocks = {block(r) for r in demo}
    assert None not in demo_blocks and len(demo_blocks) == len(demo)
    for r in all_rows(data):
        assert block(r) not in demo_blocks, r["id"]
    for r in demo:
        assert r["id"].startswith("demo67-") and r["task"] == "explain_score"


def test_dataset_files_are_small_enough():
    for p in C.DATASET.glob("*.json"):
        assert p.stat().st_size < 50 * 1024 * 1024, p.name


# ------------------------------------------------------------------ facts and consistency
def test_fact_lines_come_from_the_tables_and_have_no_test_months(data):
    from preprocess_data import facts as F

    known = {f.line for f in F.load_facts()}
    checked = 0
    for r in all_rows(data):
        if r["task"] not in ("dataset_fact", "policy") or "Facts" not in r["input"]:
            continue
        for ln in r["input"].split("\n"):
            if ln.startswith("- "):
                assert ln[2:] in known, (r["id"], ln)
                assert not C.PROTECTED_RE.search(ln), (r["id"], ln)
                if not ln.startswith("- Held-out test months 6-7"):
                    assert not re.search(r"month[s]? [67]\b", ln), (r["id"], ln)
                checked += 1
    assert checked > 5000


def test_development_totals_match_eda_table():
    from preprocess_data import facts as F

    eda = pd.read_csv(C.EDA_TABLES / "eda_monthly.csv")
    dev = eda[eda["month"] <= C.DEV_MAX_MONTH]
    total = [f for f in F.load_facts() if f.family == "total"][0]
    assert total.vals["n"] == f"{int(dev['n'].sum()):,}" == "794,989"
    assert total.vals["frauds"] == f"{int(dev['frauds'].sum()):,}" == "8,151"
    assert total.vals["rate"] == "1.03%"
    assert total.vals["ratio"] == "96.5"


def test_result_line_matches_the_numbers_and_the_answer(data, demo):
    n = 0
    for r in all_rows(data) + demo:
        m = re.search(r"Model score: ([\d.]+)\nReview threshold: ([\d.]+) .*\nResult: (flagged for review|not flagged)", r["input"])
        if not m:
            continue
        score, thr, result = float(m.group(1)), float(m.group(2)), m.group(3)
        assert (score >= thr) == (result == "flagged for review"), r["id"]
        if result == "flagged for review":
            assert "not flagged" not in r["output"], r["id"]
        n += 1
    assert n > 5000


def test_context_inputs_start_with_the_rules(data):
    from utils.prompt import SYSTEM_RULES

    for r in all_rows(data):
        if r["task"] in ("explain_score", "dataset_fact", "policy"):
            assert r["input"].startswith(SYSTEM_RULES), r["id"]


# ------------------------------------------------------------------ binding checks (G3 review F11)
REL = "compared with a typical application"
CTX_RE = re.compile(
    r"Model score: ([\d.]+)\nReview threshold: ([\d.]+) \(.*?\)\nResult: (flagged for review|not flagged)\nMain factors:\n"
    r"((?:- .*\n?){3})")
FACTOR_LINE_RE = re.compile(r"^- (.+?): (.+) \((raises|lowers) the model score " + re.escape(REL) + r"\)$")
VERB_RE = re.compile(r"\b(raises?|lowers?)\b|\bpush(?:es)? the (?:model )?score (up|down)\b", re.IGNORECASE)


def parse_context(r):
    """(score, threshold, result, {factor name: (value, direction)}, context block) or None."""
    m = CTX_RE.search(r["input"])
    if not m:
        return None
    factors = {}
    for ln in m.group(4).strip().split("\n"):
        mm = FACTOR_LINE_RE.match(ln)
        assert mm, (r["id"], ln)
        factors[mm.group(1)] = (mm.group(2), mm.group(3))
    return m.group(1), m.group(2), m.group(3), factors, m.group(0)


def context_rows(data, demo):
    for r in all_rows(data) + demo:
        ctx = parse_context(r)
        if ctx:
            yield r, ctx


def test_every_factor_line_says_what_raises_and_lowers_are_measured_against(data, demo):
    n = 0
    for r, _ in context_rows(data, demo):
        assert r["input"].count(REL) >= 3, r["id"]
        n += 1
    assert n > 6000


def test_score_and_threshold_are_bound_to_their_own_lines(data, demo):
    for r, (sc, th, _, factors, _) in context_rows(data, demo):
        out = r["output"]
        allowed = {sc, th} | {v for v, _ in factors.values()}  # a factor value such as 0.0802 also has 4 decimals
        for x in re.findall(r"\b0\.\d{4}\b", out):
            assert x in allowed, (r["id"], x)
        for x in re.findall(r"score(?: is| of)?\s*\(?(0\.\d{4})", out):
            assert x == sc, (r["id"], "score", x, sc)
        for x in re.findall(r"threshold(?: is| of)?\s*\(?(0\.\d{4})", out):
            assert x == th, (r["id"], "threshold", x, th)
        for x in re.findall(r"scoring (0\.\d{4})", out):
            assert x == th, (r["id"], "scoring", x, th)


def _mentions(out, factors):
    pat = "|".join(re.escape(n) for n in sorted(factors, key=len, reverse=True))
    return [(m.start(), m.end(), m.group(1).lower(), m.group(2)) for m in re.finditer(r"(" + pat + r") \(([^)]*)\)", out, re.IGNORECASE)]


def test_factor_value_and_direction_are_bound_to_their_own_context_line(data, demo):
    names_lower = {}
    checked = 0
    for r, (_, _, _, factors, _) in context_rows(data, demo):
        names_lower = {n.lower(): (v, d) for n, (v, d) in factors.items()}
        out = r["output"]
        ms = _mentions(out, factors)
        verbs = []
        for i, (s, e, name, val) in enumerate(ms):
            assert val == names_lower[name][0], (r["id"], name, val)
            tail_end = ms[i + 1][0] if i + 1 < len(ms) else len(out)
            m = VERB_RE.search(out, e, tail_end)
            if not m:
                verbs.append(None)
            elif m.group(1):
                verbs.append("raises" if m.group(1).lower().startswith("raise") else "lowers")
            else:
                verbs.append("raises" if m.group(2).lower() == "up" else "lowers")
        for i in range(len(ms) - 1, -1, -1):  # "A and B raise ...": a mention without a verb takes the next one's
            if verbs[i] is None and i + 1 < len(ms):
                verbs[i] = verbs[i + 1]
        for (s, e, name, val), v in zip(ms, verbs):
            if v is not None:
                assert v == names_lower[name][1], (r["id"], name, v)
                checked += 1
        # factor_detail wording: "<Name> is <value> for this application and it raises" / "Here <name> is <value>, and that raises"
        for name, (val, d) in factors.items():
            for mm in re.finditer(re.escape(name) + r" is (.+?)(?: for this application and it|, and that) (raises|lowers)", out, re.IGNORECASE):
                assert mm.group(1) == val and mm.group(2) == d, (r["id"], name, mm.group(0))
                checked += 1
    assert checked > 8000


def test_result_line_is_consistent_in_both_directions(data, demo):
    n = 0
    for r, (sc, th, result, _, _) in context_rows(data, demo):
        out = r["output"]
        assert (float(sc) >= float(th)) == (result == "flagged for review"), r["id"]
        if result == "not flagged":
            assert not re.search(r"(application is|so this one is|it is|was) flagged for review|^Yes|^Flagged for review:", out), (r["id"], out[:120])
        else:
            assert not re.search(r"^No\b|is below the|not flagged", out), (r["id"], out[:120])
        n += 1
    assert n > 6000


def test_not_flagged_results_with_only_raising_factors_say_it_is_not_enough(data, demo):
    n = 0
    for r, (_, _, result, factors, _) in context_rows(data, demo):
        if result == "not flagged" and all(d == "raises" for _, d in factors.values()):
            assert not re.search(r"\braise\b", r["output"]), (r["id"], r["output"][:160])
            if "push the model score up" in r["output"]:
                assert "enough to reach the review threshold" in r["output"], r["id"]
                n += 1
    assert n > 50
    for r, (_, _, result, factors, _) in context_rows(data, demo):  # the mirror case, if it ever occurs
        if result == "flagged for review" and all(d == "lowers" for _, d in factors.values()):
            assert not re.search(r"\blower\b", r["output"]), r["id"]


def test_dataset_fact_numbers_come_from_one_fact_line(data):
    allowed = {"5", "0.25", "0", "6", "7", "4"}
    n = bad = 0
    for r in all_rows(data):
        if r["task"] != "dataset_fact":
            continue
        out = r["output"]
        if "PR-AUC" in out and "validation" in out:  # the one answer built from two model-fact lines
            continue
        nums = [x for x in C.numbers_in(out) if x not in allowed]
        lines = [ln for ln in r["input"].split("\n") if ln.startswith("- ")]
        n += 1
        if not any(all(_number_in(x, ln) for x in nums) for ln in lines):
            bad += 1
    assert n > 1500 and bad == 0, bad


def test_rank_answers_name_the_true_maximum(data):
    checked = 0
    for r in all_rows(data):
        if r["task"] != "dataset_fact":
            continue
        out, lines = r["output"], [ln[2:] for ln in r["input"].split("\n") if ln.startswith("- ")]
        for pat, lab in ((r"highest fraud rate.*?: ([\d.]+)% ", r"fraud rate ([\d.]+)%"),
                         (r"had the highest fraud rate, ([\d.]+)% ", r"fraud rate ([\d.]+)%"),
                         (r"catches the most fraud on its own: ([\d.]+)%", r"catches ([\d.]+)%"),
                         (r"most associated with fraud, with a Cramer's V of ([\d.]+)\.", r"Cramer's V\) ([\d.]+)"),
                         (r"drifted the most, with a largest monthly PSI of ([\d.]+)\.", r"\(PSI\) ([\d.]+)")):
            m = re.search(pat, out)
            if m:
                vals = [float(re.search(lab, ln).group(1)) for ln in lines if re.search(lab, ln)]
                assert float(m.group(1)) == max(vals), r["id"]
                checked += 1
    assert checked > 100


# ------------------------------------------------------------------ wording fixes (F2, F4)
def test_protected_attribute_refusals_make_no_causal_claims(data):
    for r in all_rows(data):
        if r["task"] == "policy":
            for bad in ("cannot explain", "did not drive", "so they", "could not drive", "had no effect"):
                assert bad not in r["output"], (r["id"], bad)


def test_singular_and_plural_counts(data):
    for r in all_rows(data):
        text = r["input"] + " " + r["output"]
        assert not re.search(r"(?<![\d,.])1 (?:frauds|applications)\b", text), r["id"]
    assert any(re.search(r"(?<![\d,.])1 fraud\b", r["output"]) for r in all_rows(data))


def test_templated_questions_read_naturally(data):
    labels = sorted(FEATURE_LABELS.values(), key=len, reverse=True)
    numeric = [lab for f, lab in FEATURE_LABELS.items() if f not in ("payment_type", "housing_status", "source", "device_os")]
    for r in all_rows(data):
        ins = r["instruction"]
        for lab in labels:  # "Show me the exact other cards held." style
            assert not re.search(r"\bthe (?:exact )?" + re.escape(lab) + r"\b", ins), (r["id"], ins)
        m = re.match(r"^Which (.+?) has the highest fraud rate\?$", ins)
        if m:
            assert m.group(1) not in numeric, (r["id"], ins)
    for r in all_rows(data):
        if r["task"] == "dataset_fact":
            assert "(recall) at a false-positive" not in r["output"] and "catches 0.514 of frauds" not in r["output"]


# ------------------------------------------------------------------ F3 public data, F5 context reuse
def test_bitext_rows_have_no_offensive_language_or_requests_for_personal_data(data):
    from preprocess_data import public_sets as P

    bit = [r for r in all_rows(data) if r["source"] == P.BITEXT_SOURCE]
    assert len(bit) > 1000
    for r in bit:
        assert not C.PROFANITY_RE.search(r["instruction"] + " " + r["output"]), r["id"]
        assert not P.asks_personal_data(r["output"]), (r["id"], r["output"][:120])
    raw = C.RAW / "bitext_retail_banking" / "bitext.csv"
    if raw.is_file():
        df = pd.read_csv(raw)
        offensive = {P._clean(str(i)) for i, t in zip(df["instruction"], df["tags"].astype(str)) if "W" in t}
        assert not any(r["instruction"] in offensive for r in bit)


def test_val_and_test_contexts_never_repeat_a_train_context(data):
    blocks = {s: {ctx[4] for r, ctx in ((r, parse_context(r)) for r in data[s]) if ctx} for s in SPLITS}
    assert len(blocks["train"]) > 1000
    assert not blocks["val"] & blocks["train"]
    assert not blocks["test"] & blocks["train"]
    assert not blocks["test"] & blocks["val"]


def test_flipping_the_age_proxy_switch_is_supported():
    # the switch exists, is empty by default (team decision pending) and is honoured by the two builders
    assert hasattr(C, "EXTRA_EXCLUDED_FACTORS")
    from preprocess_data import facts as F

    assert F._feature_ok("date_of_birth_distinct_emails_4w") == ("date_of_birth_distinct_emails_4w" not in C.EXTRA_EXCLUDED_FACTORS)
    assert not F._feature_ok("customer_age")
