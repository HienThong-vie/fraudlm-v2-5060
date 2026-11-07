"""Facts and question/answer templates for the `dataset_fact` task (plus the dataset side of `policy`).

Facts come from two sources only:
    - the EDA tables in ../reports/tables/eda_*.csv (BAF Base, development months 0-5; monthly rows of the
      test months 6-7 are dropped), and
    - the team classifier's model card / metrics in the cloned team repo (preprocessed/_ml_repo).
Rows about the protected columns (customer_age, income, employment_status) and the non-input columns
(x1, x2, month, device_fraud_count) are dropped, so no such fact can ever be quoted. eda_age_groups.csv
is never read. Every number in an answer is copied from the fact line in the `input`.

Author: FraudLM v2 team (COS30049.2-1).
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

import pandas as pd

from preprocess_data import common as C
from utils.prompt import FEATURE_LABELS, SYSTEM_RULES

FACT_HEADER = "Facts (BAF Base, development months 0-5):"
MODEL_HEADER = "Facts about the classifier:"


# ------------------------------------------------------------------ formatting
def i_(v) -> str:
    return f"{int(round(float(v))):,}"


def pct2(v) -> str:      # value already a percentage
    return f"{float(v):.2f}%"


def pct1(v) -> str:
    return f"{float(v):.1f}%"


def f3(v) -> str:
    return f"{float(v):.3f}"


def f2(v) -> str:
    return f"{float(v):.2f}"


def med(v) -> str:
    x = float(v)
    if x.is_integer():
        return str(int(x))
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.3g}"


def cnt(n_text: str, noun: str) -> str:
    """'4,571' + 'fraud' -> '4,571 frauds'; '1' -> '1 fraud' (G3 review F4)."""
    return f"{n_text} {noun}" if n_text == "1" else f"{n_text} {noun}s"


def with_counts(v: dict) -> dict:
    v["n_t"], v["frauds_t"] = cnt(v["n"], "application"), cnt(v["frauds"], "fraud")
    return v


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def level_text(feature: str, value) -> str:
    """'housing status = BA'; 0/1 flags as yes / no; ordinal values without a trailing '.0'."""
    label = FEATURE_LABELS[feature]
    v = str(value)
    if feature in ("email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards", "foreign_request", "keep_alive_session"):
        v = "yes" if float(v) >= 0.5 else "no"
    else:
        try:
            x = float(v)
            v = str(int(x)) if x.is_integer() else v
        except ValueError:
            pass
    return f"{label} = {v}"


PATTERN_WORDS = {"prev_addr": "previous address", "balcon": "intended transfer amount", "bank_months": "bank history"}


def pattern_text(p: str) -> str:
    """'prev_addr MISSING / balcon ok / bank_months ok' -> plain words."""
    parts = []
    for chunk in p.split(" / "):
        key, state = chunk.rsplit(" ", 1)
        parts.append(f"{PATTERN_WORDS[key]} {'missing' if state.upper() == 'MISSING' else 'present'}")
    return ", ".join(parts)


@dataclass
class Fact:
    family: str                 # table family, e.g. "level", "month"
    group: str                  # rows that can be ranked against each other (same feature / same table)
    key: str                    # unique id, used for the train / eval split of target facts
    line: str                   # the fact line shown in the input
    vals: dict                  # formatted values for the answer templates
    split: str = "train"        # "train" or "eval": which target pool the fact belongs to
    raw: dict = field(default_factory=dict)


CATEGORICAL_FEATURES = frozenset({"payment_type", "housing_status", "source", "device_os"})


def _pool(key: str) -> str:
    h = int(hashlib.sha256(f"facts-{C.SEED}:{key}".encode()).hexdigest()[:8], 16) / 2**32
    return "eval" if h < 0.2 else "train"


def _feature_ok(f: str) -> bool:
    return f in FEATURE_LABELS and f not in C.EXTRA_EXCLUDED_FACTORS


def load_facts() -> list[Fact]:
    T = C.EDA_TABLES
    out: list[Fact] = []

    def add(fam, grp, key, line, vals, raw=None):
        out.append(Fact(fam, grp, f"{fam}:{key}", line, vals, _pool(f"{fam}:{key}"), raw or {}))

    # --- level / flag / ordinal tables: fraud rate by feature level
    for name in ("eda_binary_flags.csv", "eda_category_levels.csv", "eda_ordinal_features.csv"):
        df = pd.read_csv(T / name)
        fcol, vcol = df.columns[0], df.columns[1]
        for _, r in df.iterrows():
            if not _feature_ok(r[fcol]):
                continue
            ft = level_text(r[fcol], r[vcol])
            v = {"fv": ft, "label": FEATURE_LABELS[r[fcol]], "level": ft.split(" = ", 1)[1], "n": i_(r["n"]),
                 "frauds": i_(r["frauds"]), "rate": pct2(r["rate"]), "share": pct1(r["share"])}
            with_counts(v)
            line = f"{ft}: {v['n_t']}, {v['frauds_t']}, fraud rate {v['rate']}, share of applications {v['share']}"
            add("level", r[fcol], f"{name}:{ft}", line, v, {"rate": float(r["rate"])})

    # --- missing-value patterns
    df = pd.read_csv(T / "eda_missing_patterns.csv")
    for _, r in df.iterrows():
        pt = pattern_text(r["pattern"])
        v = {"fv": pt, "n": i_(r["n"]), "frauds": i_(r["frauds"]), "rate": pct2(r["rate"]), "share": pct1(r["share"])}
        with_counts(v)
        line = f"{pt}: {v['n_t']}, {v['frauds_t']}, fraud rate {v['rate']}, share of applications {v['share']}"
        add("pattern", "missing", pt, line, v, {"rate": float(r["rate"])})

    # --- monthly rows (development months only) and their total
    df = pd.read_csv(T / "eda_monthly.csv")
    df = df[df["month"] <= C.DEV_MAX_MONTH]
    for _, r in df.iterrows():
        m = int(r["month"])
        v = {"m": str(m), "n": i_(r["n"]), "frauds": i_(r["frauds"]), "rate": pct2(r["rate"])}
        with_counts(v)
        add("month", "month", str(m), f"month {m}: {v['n_t']}, {v['frauds_t']}, fraud rate {v['rate']}", v, {"rate": float(r["rate"])})
    n_tot, fr_tot = int(df["n"].sum()), int(df["frauds"].sum())
    ratio = (n_tot - fr_tot) / fr_tot
    v = {"n": i_(n_tot), "frauds": i_(fr_tot), "rate": pct2(100 * fr_tot / n_tot), "ratio": f"{ratio:.1f}"}
    with_counts(v)
    add("total", "total", "dev", f"months 0-5 together: {v['n_t']}, {v['frauds_t']}, fraud rate {v['rate']}, about {v['ratio']} legitimate applications per fraud", v)

    # --- numeric separation
    df = pd.read_csv(T / "eda_numeric_separation.csv")
    for _, r in df.iterrows():
        if not _feature_ok(r["feature"]):
            continue
        lab = FEATURE_LABELS[r["feature"]]
        v = {"label": lab, "ml": med(r["median_legit"]), "mf": med(r["median_fraud"]), "ks": f3(r["ks_statistic"])}
        add("numeric", "numeric", r["feature"], f"{lab}: median {v['ml']} for legitimate applications, {v['mf']} for fraud applications, KS statistic {v['ks']}", v, {"s": float(r["ks_statistic"])})

    # --- single-feature recall
    df = pd.read_csv(T / "eda_single_feature_recall.csv")
    for _, r in df.iterrows():
        if not _feature_ok(r["feature"]):
            continue
        lab = FEATURE_LABELS[r["feature"]]
        v = {"label": lab, "recall": pct1(100 * r["recall_at_5pct_fpr"])}
        add("recall", "recall", r["feature"], f"{lab}: catches {v['recall']} of frauds at a 5% false-positive rate when used alone", v, {"s": float(r["recall_at_5pct_fpr"])})

    # --- association with fraud
    df = pd.read_csv(T / "eda_feature_association.csv")
    for _, r in df.iterrows():
        if not _feature_ok(r["feature"]):
            continue
        lab = FEATURE_LABELS[r["feature"]]
        v = {"label": lab, "V": f3(r["V"])}
        add("assoc", "assoc", r["feature"], f"{lab}: association with fraud (Cramer's V) {v['V']}", v, {"s": float(r["V"])})

    # --- drift
    df = pd.read_csv(T / "eda_feature_drift_psi.csv")
    for _, r in df.iterrows():
        if not _feature_ok(r["feature"]):
            continue
        lab = FEATURE_LABELS[r["feature"]]
        v = {"label": lab, "psi": f2(r["max_psi"]), "large": bool(r["max_psi"] > 0.25)}
        add("psi", "psi", r["feature"], f"{lab}: largest monthly drift (PSI) {v['psi']}; a PSI above 0.25 is a large shift", v, {"s": float(r["max_psi"])})

    # --- correlations
    df = pd.read_csv(T / "eda_strong_correlations.csv")
    for _, r in df.iterrows():
        if not (_feature_ok(r["feature_a"]) and _feature_ok(r["feature_b"])):
            continue
        a, b = FEATURE_LABELS[r["feature_a"]], FEATURE_LABELS[r["feature_b"]]
        v = {"a": a, "b": b, "rho": f3(r["rho"])}
        add("corr", "corr", f"{r['feature_a']}|{r['feature_b']}", f"{a} and {b}: Spearman correlation {v['rho']}", v)

    out += load_model_facts()
    return out


def load_model_facts() -> list[Fact]:
    card = json.loads(C.MODEL_CARD.read_text(encoding="utf-8"))
    base = card["datasets"][0]
    t = base["test_metrics"]
    thr = f"{t['threshold']:.4f}"
    vpr = f"{base['validation_pr_auc']:.4f}"
    facts = []

    def add(key, line, vals):
        facts.append(Fact("model", "model", f"model:{key}", line, vals, "train"))

    add("type", "Classifier: logistic regression with balanced class weights, trained on months 0-4 and chosen on validation month 5 by PR-AUC", {})
    add("threshold", f"Review threshold: {thr}, set for a 5% false-positive rate on validation month 5", {"thr": thr})
    add("val", f"Validation month 5: PR-AUC {vpr}", {"vpr": vpr})
    add("test", (f"Held-out test months 6-7 (50,000-row sample): PR-AUC {t['pr_auc']:.4f}, recall {t['recall']:.3f}, "
                 f"false-positive rate {t['false_positive_rate']:.3f}, precision {t['precision']:.3f}, F1 {t['f1']:.3f}, ROC-AUC {t['roc_auc']:.3f}"),
        {"pr": f"{t['pr_auc']:.4f}", "recall": f"{t['recall']:.3f}", "fpr": f"{t['false_positive_rate']:.3f}",
         "prec": f"{t['precision']:.3f}", "f1": f"{t['f1']:.3f}", "roc": f"{t['roc_auc']:.3f}"})
    return facts


# ------------------------------------------------------------------ question templates
# per intent: t (template wording), l (paraphrases), held (test only). {placeholders} come from Fact.vals.
Q: dict[str, dict] = {
    "level_rate": {
        "t": ["What is the fraud rate for {fv}?", "What is the fraud rate when {fv}?"],
        "l": ["How often is fraud found when {fv}?", "Tell me the fraud rate for applications with {fv}.",
              "What fraction of applications with {fv} are frauds?", "What does the data say about the fraud rate for {fv}?",
              "Give me the fraud rate where {fv}."],
        "held": ["How risky is {fv} in terms of fraud rate?", "Look up the fraud rate for {fv}."],
    },
    "level_count": {
        "t": ["How many applications have {fv}?", "How many frauds are there when {fv}?"],
        "l": ["What is the number of applications with {fv}?", "Count the applications with {fv}.",
              "How many applications in the data have {fv}, and how many are frauds?"],
        "held": ["How big is the group with {fv}?"],
    },
    "level_share": {
        "t": ["What share of applications have {fv}?"],
        "l": ["How common is {fv} among all applications?", "What percentage of applications have {fv}?",
              "How large a share of the data has {fv}?"],
        "held": ["What proportion of applications show {fv}?"],
    },
    "level_full": {
        "t": ["What does the data show for {fv}?", "Give me the facts for {fv}."],
        "l": ["Summarise the data for {fv}.", "Tell me everything the table says about {fv}.",
              "What are the numbers for {fv}?"],
        "held": ["Report what we know about {fv}."],
    },
    "level_rank": {
        "t": ["Which level of {label} has the highest fraud rate?"],
        "l": ["Which {label} value is riskiest by fraud rate?", "For {label}, which level shows the highest fraud rate?",
              "Where is the fraud rate highest for {label}?"],
        "held": ["Which {label} setting has the worst fraud rate?"],
    },
    "pattern_rate": {
        "t": ["What is the fraud rate when {fv}?"],
        "l": ["How high is the fraud rate when {fv}?", "Tell me the fraud rate for applications with {fv}.",
              "What does the data say about applications with {fv}?"],
        "held": ["How often is fraud found when {fv}?"],
    },
    "month_rate": {
        "t": ["What was the fraud rate in month {m}?"],
        "l": ["How many frauds were there in month {m}?", "Give me the numbers for month {m}.",
              "What happened to the fraud rate in month {m}?", "Tell me the fraud rate and the number of applications in month {m}."],
        "held": ["How did month {m} look for fraud?"],
    },
    "month_rank": {
        "t": ["Which month had the highest fraud rate?"],
        "l": ["In which month was the fraud rate highest?", "Which month looks worst for fraud?"],
        "held": ["Find the month with the peak fraud rate."],
    },
    "total": {
        "t": ["What is the overall fraud rate in months 0-5?", "How many applications and frauds are in the development months?"],
        "l": ["How imbalanced is the development data?", "What is the overall fraud rate across the development months?",
              "How many legitimate applications are there per fraud?", "Give me the headline numbers for the development data."],
        "held": ["How rare is fraud in the development months overall?"],
    },
    "numeric": {
        "t": ["How do the legitimate and fraud medians of {label} compare?"],
        "l": ["What is the median {label} for fraud and for legitimate applications?", "How different is {label} between fraud and legitimate applications?",
              "What does the KS statistic say for {label}?"],
        "held": ["Compare {label} for fraud and legitimate applications."],
    },
    "recall": {
        "t": ["How much fraud does {label} catch on its own at a 5% false-positive rate?"],
        "l": ["What is the single-feature recall for {label}?", "If I used only {label}, how much fraud would it catch?",
              "How well does {label} work alone at a 5% false-positive rate?"],
        "held": ["What recall does {label} reach by itself?"],
    },
    "recall_rank": {
        "t": ["Which of these features catches the most fraud on its own?"],
        "l": ["Which feature is the best single signal?", "Which one catches the most fraud alone at a 5% false-positive rate?"],
        "held": ["Which feature is the strongest on its own?"],
    },
    "assoc": {
        "t": ["How strongly is {label} associated with fraud?"],
        "l": ["What is Cramer's V for {label}?", "How related is {label} to fraud?", "Give me the association between {label} and fraud."],
        "held": ["How much does {label} go together with fraud?"],
    },
    "assoc_rank": {
        "t": ["Which of these features is most associated with fraud?"],
        "l": ["Which has the strongest Cramer's V?", "Which feature goes together with fraud the most?"],
        "held": ["Which one is the most strongly tied to fraud?"],
    },
    "psi": {
        "t": ["How much did {label} drift over the months?"],
        "l": ["What is the PSI for {label}?", "Is {label} drifting a lot?", "Did {label} change much between months?"],
        "held": ["How stable is {label} over time?"],
    },
    "psi_rank": {
        "t": ["Which of these features drifted the most?"],
        "l": ["Which feature shifted the most between months?", "Which has the largest PSI?"],
        "held": ["Which one changed the most over time?"],
    },
    "corr": {
        "t": ["How strongly are {a} and {b} correlated?"],
        "l": ["What is the correlation between {a} and {b}?", "Are {a} and {b} related?"],
        "held": ["How do {a} and {b} move together?"],
    },
    "model_threshold": {
        "t": ["What is the review threshold and how was it set?"],
        "l": ["Which threshold flags an application for review?", "How was the review threshold chosen?", "What is the cut-off for a review flag?"],
        "held": ["Where is the line for flagging an application?"],
    },
    "model_test": {
        "t": ["How well does the model do on the held-out test months?"],
        "l": ["What are the test results of the classifier?", "Give me the test metrics for the model.",
              "How good is the model on months 6-7?"],
        "held": ["What did the classifier score on the final test?"],
    },
    "model_recall": {
        "t": ["How much fraud does the model catch at the review threshold?"],
        "l": ["What is the recall of the classifier on the test months?", "What false-positive rate does the model have on the test months?"],
        "held": ["What share of frauds does the classifier find on the test months?"],
    },
    "model_prauc": {
        "t": ["What is the PR-AUC of the model?"],
        "l": ["What PR-AUC did the classifier reach?", "Tell me the PR-AUC on validation and on test."],
        "held": ["How does the model rank on PR-AUC?"],
    },
    "model_type": {
        "t": ["Which model is used and how was it chosen?"],
        "l": ["What kind of classifier produces the model score?", "What model does the score come from?"],
        "held": ["Tell me about the classifier behind the score."],
    },
}


def _fmt_rate_first(f: Fact) -> str:
    return f.vals["fv"]


def _answer_and_ctx(intent: str, target: list[Fact], rng: random.Random):
    """Return (answer, question_fields) for one intent. `target` holds the fact(s) the answer is built from."""
    f = target[0]
    v = f.vals
    if intent == "level_rate":
        return rng.choice([
            f"For {v['fv']}, the fraud rate is {v['rate']} ({v['frauds_t']} among {v['n_t']}).",
            f"The fraud rate for {v['fv']} is {v['rate']}, based on {v['frauds_t']} in {v['n_t']}.",
        ]), v
    if intent == "level_count":
        return rng.choice([
            f"{cap(v['n_t'])} have {v['fv']}, including {v['frauds_t']}.",
            f"There are {v['n_t']} with {v['fv']}, including {v['frauds_t']}.",
        ]), v
    if intent == "level_share":
        return rng.choice([
            f"{v['share']} of the applications have {v['fv']} ({v['n_t']}).",
            f"Applications with {v['fv']} make up {v['share']} of the data, which is {v['n_t']}.",
        ]), v
    if intent == "level_full":
        return (f"For {v['fv']} the data shows {v['n_t']} and {v['frauds_t']}, a fraud rate of {v['rate']}, "
                f"and a share of {v['share']} of all applications."), v
    if intent == "level_rank":
        best = max(target, key=lambda x: x.raw["rate"])
        b = best.vals
        return (f"{b['level']} has the highest fraud rate for {b['label']}: {b['rate']} ({b['frauds_t']} among {b['n_t']})."), {"label": b["label"]}
    if intent == "pattern_rate":
        return rng.choice([
            f"When {v['fv']}, the fraud rate is {v['rate']} ({v['frauds_t']} among {v['n_t']}).",
            f"Applications with {v['fv']} have a fraud rate of {v['rate']}, from {v['frauds_t']} in {v['n_t']}.",
        ]), v
    if intent == "month_rate":
        return rng.choice([
            f"In month {v['m']} the fraud rate was {v['rate']} ({v['frauds_t']} among {v['n_t']}).",
            f"Month {v['m']} had {v['n_t']} and {v['frauds_t']}, a fraud rate of {v['rate']}.",
        ]), v
    if intent == "month_rank":
        best = max(target, key=lambda x: x.raw["rate"])
        b = best.vals
        return f"Month {b['m']} had the highest fraud rate, {b['rate']} ({b['frauds_t']} among {b['n_t']}).", {}
    if intent == "total":
        return rng.choice([
            f"Across development months 0-5 there are {v['n_t']} and {v['frauds_t']}, a fraud rate of {v['rate']}. That is about {v['ratio']} legitimate applications per fraud.",
            f"The development months 0-5 hold {v['n_t']} with {v['frauds_t']}, so the fraud rate is {v['rate']}, or about {v['ratio']} legitimate applications for each fraud.",
        ]), v
    if intent == "numeric":
        return rng.choice([
            f"For {v['label']}, the median is {v['ml']} for legitimate applications and {v['mf']} for fraud applications; the KS statistic is {v['ks']}.",
            f"The median {v['label']} is {v['ml']} among legitimate applications and {v['mf']} among fraud applications, with a KS statistic of {v['ks']}.",
        ]), v
    if intent == "recall":
        return rng.choice([
            f"Used alone, {v['label']} catches {v['recall']} of frauds at a 5% false-positive rate.",
            f"{v['label'][:1].upper() + v['label'][1:]} on its own catches {v['recall']} of frauds at a 5% false-positive rate.",
        ]), v
    if intent == "recall_rank":
        best = max(target, key=lambda x: x.raw["s"])
        b = best.vals
        return f"{b['label'][:1].upper() + b['label'][1:]} catches the most fraud on its own: {b['recall']} of frauds at a 5% false-positive rate.", {}
    if intent == "assoc":
        return rng.choice([
            f"The association between {v['label']} and fraud (Cramer's V) is {v['V']}.",
            f"{v['label'][:1].upper() + v['label'][1:]} has a Cramer's V of {v['V']} with fraud.",
        ]), v
    if intent == "assoc_rank":
        best = max(target, key=lambda x: x.raw["s"])
        b = best.vals
        return f"{b['label'][:1].upper() + b['label'][1:]} is the most associated with fraud, with a Cramer's V of {b['V']}.", {}
    if intent == "psi":
        tail = "which is above 0.25, so it is a large shift." if v["large"] else "which is not above 0.25, so it is not a large shift."
        return f"The largest monthly drift for {v['label']} is a PSI of {v['psi']}, {tail}", v
    if intent == "psi_rank":
        best = max(target, key=lambda x: x.raw["s"])
        b = best.vals
        return f"{b['label'][:1].upper() + b['label'][1:]} drifted the most, with a largest monthly PSI of {b['psi']}.", {}
    if intent == "corr":
        return f"The Spearman correlation between {v['a']} and {v['b']} is {v['rho']}.", v
    raise KeyError(intent)


FAMILY_INTENTS = {
    "level": ["level_rate", "level_rate", "level_count", "level_share", "level_full", "level_rank"],
    "pattern": ["pattern_rate"],
    "month": ["month_rate", "month_rate", "month_rank"],
    "total": ["total"],
    "numeric": ["numeric"],
    "recall": ["recall", "recall", "recall_rank"],
    "assoc": ["assoc", "assoc", "assoc_rank"],
    "psi": ["psi", "psi", "psi_rank"],
    "corr": ["corr"],
}
RANK_INTENTS = {"level_rank", "month_rank", "recall_rank", "assoc_rank", "psi_rank"}


class FactBank:
    def __init__(self) -> None:
        self.facts = load_facts()
        self.by_family: dict[str, list[Fact]] = {}
        self.by_group: dict[tuple[str, str], list[Fact]] = {}
        for f in self.facts:
            self.by_family.setdefault(f.family, []).append(f)
            self.by_group.setdefault((f.family, f.group), []).append(f)

    # weights: how often each family is drawn (model facts are few but important)
    WEIGHTS = {"level": 6, "pattern": 1.5, "month": 2, "total": 1, "numeric": 2, "recall": 2, "assoc": 2, "psi": 2, "corr": 0.5, "model": 2}

    def make(self, rng: random.Random, pool: str, split: str):
        """One dataset_fact example. pool: 'train' or 'eval' (which target facts may be used)."""
        fams = list(self.WEIGHTS)
        fam = rng.choices(fams, weights=[self.WEIGHTS[f] for f in fams])[0]
        if fam == "model":
            return self._model_example(rng, split)
        intent = rng.choice(FAMILY_INTENTS[fam])
        if intent == "level_rank":
            groups = [(k, v) for k, v in self.by_group.items() if k[0] == "level" and 3 <= len(v) <= 8]
            key, rows = rng.choice(groups)
            target = list(rows)
            ctx_rows = list(rows)
        elif intent == "month_rank":
            target = [f for f in self.by_family["month"]]
            ctx_rows = list(target)
        elif intent in ("recall_rank", "assoc_rank", "psi_rank"):
            fam_rows = self.by_family[fam]
            k = rng.randint(3, 5)
            target = rng.sample(fam_rows, k)
            ctx_rows = list(target)
        else:
            candidates = [f for f in self.by_family[fam] if f.split == pool] or self.by_family[fam]
            t = rng.choice(candidates)
            target = [t]
            others = [f for f in self.by_family[fam] if f.key != t.key]
            n_extra = min(len(others), rng.randint(2, 4))
            ctx_rows = [t] + rng.sample(others, n_extra)
        rng.shuffle(ctx_rows)
        answer, fields = _answer_and_ctx(intent, target, rng)
        qd = Q[intent]
        held = False
        if split == "test" and rng.random() < 0.5:
            tpl, src, held = rng.choice(qd["held"]), "llm:claude-sonnet", True
        else:
            pool_q = [(q, "template") for q in qd["t"]] + [(q, "llm:claude-sonnet") for q in qd["l"]] * 2
            tpl, src = rng.choice(pool_q)
        qfields = {**target[0].vals, **fields}
        question = tpl.format(**{k: qfields.get(k, "") for k in ("fv", "label", "m", "a", "b")})
        inp = "\n".join([SYSTEM_RULES, FACT_HEADER] + [f"- {f.line}" for f in ctx_rows])
        return question, inp, answer, src, held

    def _model_example(self, rng: random.Random, split: str):
        m = {f.key.split(":", 1)[1]: f for f in self.by_family["model"]}
        intent = rng.choice(["model_threshold", "model_test", "model_recall", "model_prauc", "model_type"])
        if intent == "model_threshold":
            used = ["threshold"]
            v = m["threshold"].vals
            ans = f"The review threshold is {v['thr']}. It was set for a 5% false-positive rate on validation month 5; applications at or above it are flagged for review."
        elif intent == "model_test":
            used = ["test"]
            v = m["test"].vals
            ans = (f"On the held-out test months 6-7 (a 50,000-row sample), the model reaches PR-AUC {v['pr']}, recall {v['recall']} at a false-positive rate of {v['fpr']}, "
                   f"precision {v['prec']}, F1 {v['f1']} and ROC-AUC {v['roc']}.")
        elif intent == "model_recall":
            used = ["test"]
            v = m["test"].vals
            ans = f"On the held-out test months 6-7, the model's recall is {v['recall']} at a false-positive rate of {v['fpr']}."
        elif intent == "model_prauc":
            used = ["val", "test"]
            ans = f"The PR-AUC is {m['val'].vals['vpr']} on validation month 5 and {m['test'].vals['pr']} on the held-out test months 6-7."
        else:
            used = ["type"]
            ans = "The score comes from a logistic regression with balanced class weights, trained on months 0-4 and chosen on validation month 5 by PR-AUC."
        extra = [k for k in ("type", "threshold", "val", "test") if k not in used]
        rng.shuffle(extra)
        ctx = used + extra[: rng.randint(0, 2)]
        rng.shuffle(ctx)
        qd = Q[intent]
        held = False
        if split == "test" and rng.random() < 0.5:
            q, src, held = rng.choice(qd["held"]), "llm:claude-sonnet", True
        else:
            pool_q = [(x, "template") for x in qd["t"]] + [(x, "llm:claude-sonnet") for x in qd["l"]] * 2
            q, src = rng.choice(pool_q)
        inp = "\n".join([SYSTEM_RULES, MODEL_HEADER] + [f"- {m[k].line}" for k in ctx])
        return q, inp, ans, src, held

    def missing_example(self, rng: random.Random, split: str):
        """policy: ask for a level / metric that is not among the fact lines -> 'not in the provided facts'."""
        feats = sorted(f for f in FEATURE_LABELS if _feature_ok(f))
        asked = rng.choice(feats)
        # context: rows about OTHER features
        rows = [f for f in self.by_family["level"] if f.group != asked]
        ctx_rows = rng.sample(rows, rng.randint(2, 4))
        label = FEATURE_LABELS[asked]
        questions_t = [f"What is the fraud rate for {label}?", f"How many frauds are there for {label}?"]
        questions_l = [f"Tell me the fraud rate by {label}.", f"What does the data show about {label}?",
                       f"Give me the numbers for {label}."]
        if asked in CATEGORICAL_FEATURES:  # a "which level" question only makes sense for categories
            questions_l.append(f"Which level of {label} has the highest fraud rate?")
        questions_h = [f"How risky is {label} according to these facts?", f"Look up {label} for me."]
        held = False
        if split == "test" and rng.random() < 0.5:
            q, src, held = rng.choice(questions_h), "llm:claude-sonnet", True
        else:
            q, src = rng.choice([(x, "template") for x in questions_t] + [(x, "llm:claude-sonnet") for x in questions_l])
        ans = rng.choice([
            f"The provided facts do not include {label}, so I cannot answer that from them. They cover only the other features listed.",
            f"That is not in the facts I was given: none of the lines mention {label}. I will not guess a number.",
        ])
        inp = "\n".join([SYSTEM_RULES, FACT_HEADER] + [f"- {f.line}" for f in ctx_rows])
        return q, inp, ans, src, held
