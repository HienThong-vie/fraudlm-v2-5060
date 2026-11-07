"""
Fact lines for dataset / model questions in the FraudLM chat demo.

Purpose: the model was trained on `dataset_fact` examples whose `### Input:` holds a few fact lines
(fraud rate by feature level, months, drift, recall, PR-AUC, ...). In chat there is no such block unless we
build one, so this module picks the relevant lines for a question by simple keyword matching and lays them
out exactly as in training (SYSTEM_RULES, then the header, then "- <line>" per fact). The guard then checks
the number of the answer against these lines.

The lines are NOT re-implemented here: `FactIndex.load()` imports `load_facts()` from the data-builder's
`preprocess_data/facts.py` (read-only), which reads the EDA tables in ../reports/tables and the team model
card in preprocessed/_ml_repo. Nothing is chosen by the model; selection is deterministic (a question-seeded
random generator picks the distractor lines, as the training builder did with its own generator).

Author: FraudLM v2 team (COS30049.2-1).

"""

from __future__ import annotations

import random
import re
import zlib
from dataclasses import dataclass

from utils.prompt import FEATURE_LABELS, SYSTEM_RULES

MAX_LINES = 8  # training contexts had 2-8 lines; 8 lines are about 250 tokens

_STOP = {"the", "of", "over", "at", "same", "a", "an", "and", "with", "for", "on", "in"}
_BINARY = {"no", "yes"}


def _norm(text: str) -> str:
    s = text.lower()
    s = re.sub(r"(\d+)\s*(?:h|hr|hrs|hours?)\b", r"\1 hour", s)
    s = re.sub(r"(\d+)\s*(?:w|wk|wks|weeks?)\b", r"\1 week", s)
    s = re.sub(r"\bos\b", "operating system", s)
    return s.replace("keepalive", "keep alive")


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def _tokens(text: str) -> set:
    return {_stem(t) for t in re.findall(r"[a-z0-9]+", _norm(text))} - _STOP


# ------------------------------------------------------------------ what counts as a fact question
# a data / model anchor in the question, so "What is PR-AUC?" (a concept) still goes to the plain banking path
_ANCHOR = re.compile(
    r"\b(?:model|classifier|logistic|dataset|data|development|held-?out|validation|test months?|month\s*\d|which month|"
    r"monthly|each month|per month|by month|fraud rate|frauds|legitimate applications?|review threshold|threshold|cut-?off|"
    r"features?|baf|bank account fraud|eda)\b", re.I)
_TERM = re.compile(
    r"\b(?:rate|how many|count|number|share|percent\w*|proportion|median|ks|statistic|psi|drift\w*|cramer\w*|correlat\w*|"
    r"associat\w*|recall|precision|pr-?auc|roc|f1|auc|threshold|cut-?off|catch\w*|stable|stability|shift\w*|compare|"
    r"highest|lowest|most|largest|worst|best|strongest|peak|look\w*|show\w*|say|numbers?|facts?|summar\w+|tell|trained|chosen|used|"
    r"well|good|metrics?|results?|perform\w*|often|common|different|related|chang\w*|signal|tied|fraction|kind|come|produc\w*|together|big|report|peak)\b",
    re.I)
_DATA_TERM = re.compile(r"\b(?:psi|cramer\w*|associat\w*|tied|signal|catch\w*|drift\w*|shift\w*|correlat\w*|recall|fraud rate)\b", re.I)
_STRONG = re.compile(
    r"\b(?:fraud rate|frauds|dataset|the data|in the data|development|held-?out|test months?|validation|month\s*\d|"
    r"which month|psi|drift\w*|cramer\w*|correlat\w*|spearman|median\w*|ks statistic|pr-?auc|roc|f1|precision|recall|"
    r"how many (?:applications|frauds)|false-positive rate|review threshold|classifier|logistic)\b", re.I)
_RANK = re.compile(r"\b(?:highest|largest|biggest|most|worst|riskiest|best|strongest|top|peak|greatest)\b", re.I)
_WHICH = re.compile(r"\b(?:which|where|find|rank)\b", re.I)

_MODEL = re.compile(r"\b(?:model|classifier|logistic|review threshold|threshold|cut-?off|held-?out|test months?|validation|"
                    r"pr-?auc|roc|f1|precision|false-positive rate)\b", re.I)
_ALONE = re.compile(r"\b(?:alone|on its own|by itself|single[- ]feature|single signal)\b", re.I)
_TOTAL = re.compile(r"\b(?:months?\s*0\s*-\s*5|development (?:months|data)|overall|in total|imbalanc\w*|per fraud|how rare|headline)\b", re.I)
_PSI = re.compile(r"\b(?:drift\w*|psi|stable|stability|shift\w*|chang\w*\s+(?:much\s+)?(?:between|over|across)\s+(?:the\s+)?months)\b", re.I)
_CORR = re.compile(r"\b(?:correlat\w*|spearman)\b", re.I)
_ASSOC = re.compile(r"\b(?:cramer\w*|associat\w*|go(?:es)? together|tied to fraud|related\b.{0,80}\bto fraud)", re.I)
_NUMERIC = re.compile(r"\b(?:median\w*|ks|statistic|legitimate and fraud|fraud and legitimate)\b", re.I)
_RECALL = re.compile(r"\b(?:catch\w*|alone|on its own|by itself|single[- ]feature|single signal|best signal|recall)\b", re.I)
_PATTERN = re.compile(r"\b(?:missing|not provided|present)\b", re.I)
_MONTH = re.compile(r"\b(?:month\s*(\d)|(?:in which|what|which) month|the month with|each month|monthly|per month|by month)\b", re.I)
_PATTERN_WORDS = {"previous address": ("previous address", "prev"), "transfer": ("intended transfer amount",),
                  "bank history": ("bank history", "bank"), "balcon": ("intended transfer amount",)}


@dataclass
class FactSelection:
    input_text: str     # the `### Input:` block, laid out as in the training data
    lines: list         # the fact lines shown (without the "- ")
    family: str         # level | month | total | pattern | numeric | recall | assoc | psi | corr | model | default
    note: str           # one-line description for the screen


class FactIndex:
    """Keyword selection over the fact lines of preprocess_data.facts.load_facts()."""

    def __init__(self, facts: list, fact_header: str, model_header: str):
        self.fact_header, self.model_header = fact_header, model_header
        self.facts = facts
        self.by_family: dict[str, list] = {}
        for f in facts:
            self.by_family.setdefault(f.family, []).append(f)
        self.levels: dict[str, list] = {}      # feature -> level facts in table order
        for f in self.by_family.get("level", []):
            self.levels.setdefault(f.group, []).append(f)
        self.per_feature: dict[tuple, object] = {}  # (family, feature) -> fact for numeric / recall / assoc / psi
        for fam in ("numeric", "recall", "assoc", "psi"):
            for f in self.by_family.get(fam, []):
                self.per_feature[(fam, f.key.split(":", 1)[1])] = f
        self.model = {f.key.split(":", 1)[1]: f for f in self.by_family.get("model", [])}
        known = {feat for _, feat in self.per_feature} | set(self.levels)
        self.label_tokens = {feat: _tokens(FEATURE_LABELS[feat]) for feat in known if feat in FEATURE_LABELS}
        # a categorical level value that names its feature on its own ("windows", "BA", "TELEAPP")
        self.level_word: dict[str, str] = {}
        for feat, rows in self.levels.items():
            for f in rows:
                v = f.vals["level"].lower()
                if v not in _BINARY and v != "other" and not v.isdigit():
                    self.level_word.setdefault(v, feat)

    @classmethod
    def load(cls) -> "FactIndex":
        """Read the EDA tables and model card once (about 1 s). Raises if the files are missing."""
        from preprocess_data.facts import FACT_HEADER, MODEL_HEADER, load_facts

        return cls(load_facts(), FACT_HEADER, MODEL_HEADER)

    # ---------------------------------------------------------------- matching
    def match_features(self, q: str) -> list:
        """Feature columns whose plain-language name the question uses, best match first."""
        qt = _tokens(q)
        scored = []
        for feat, lt in self.label_tokens.items():
            hit = len(lt & qt)
            score = hit / len(lt) if lt else 0.0
            if score >= 0.6 and hit >= min(2, len(lt)):
                scored.append((score, hit, feat))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        if scored:
            best = scored[0][0]
            return [f for s, _, f in scored if s >= best - 1e-9 or s >= 0.75]
        words = re.findall(r"[a-z0-9]+", q.lower())
        for w in words:
            if w in self.level_word:
                return [self.level_word[w]]
        return []

    def is_fact_question(self, q: str, feats: list) -> bool:
        if _MONTH.search(q) or _TOTAL.search(q):
            return bool(_TERM.search(q) or _ANCHOR.search(q))
        if _RANK.search(q) and _WHICH.search(q) and _DATA_TERM.search(q):
            return True  # "Which has the largest PSI?", "Which one catches the most fraud alone?"
        return bool(_TERM.search(q)) and (bool(feats) or bool(_ANCHOR.search(q)))

    def family_of(self, q: str, feats: list) -> str:
        if _MODEL.search(q) and not _ALONE.search(q):
            return "model"
        if _TOTAL.search(q) and not _MONTH.search(q):
            return "total"
        if _PSI.search(q):
            return "psi"
        if _CORR.search(q):
            return "corr"
        if _ASSOC.search(q):
            return "assoc"
        if len(feats) >= 2 and re.search(r"\b(?:related|together)\b", q, re.I):
            return "corr"
        if _NUMERIC.search(q):
            return "numeric"
        if _RECALL.search(q):
            return "recall"
        if _PATTERN.search(q) and any(w in q.lower() for w in ("previous address", "transfer", "bank history", "balcon")):
            return "pattern"
        if _MONTH.search(q):
            return "month"
        return "level"

    # ---------------------------------------------------------------- selection
    def select(self, question: str, strict: bool = False) -> FactSelection | None:
        """The fact block for a dataset / model question, or None when it is not one.

        strict=True (an application is in focus): only questions with an explicit data / model cue count, so
        "What does keep-alive session tell us here?" stays a question about the application.
        """
        q = re.sub(r"\s+", " ", question).strip()
        feats = self.match_features(q)
        if not q or not self.is_fact_question(q, feats):
            return None
        if strict and not _STRONG.search(q):
            return None
        rng = random.Random(zlib.crc32(q.lower().encode("utf-8")))
        fam = self.family_of(q, feats)
        rank = bool(_RANK.search(q) and _WHICH.search(q))
        builder = getattr(self, f"_pick_{fam}")
        facts, header, note = builder(q, feats, rank, rng)
        if not facts:
            facts, header, note = self._pick_default(feats, rng)
            fam = "default"
        facts = facts[:MAX_LINES]
        lines = [f.line for f in facts]
        return FactSelection("\n".join([SYSTEM_RULES, header] + [f"- {x}" for x in lines]), lines, fam, note)

    def _others(self, fams, skip_features, rng, n=2) -> list:
        """n distractor lines of the same family(ies) about other features (the training contexts had them too)."""
        pool = []
        for fam in ([fams] if isinstance(fams, str) else fams):
            for f in self.by_family.get(fam, []):
                feat = f.group if fam == "level" else f.key.split(":", 1)[1]
                if feat not in skip_features:
                    pool.append(f)
        return rng.sample(pool, min(n, len(pool))) if pool else []

    def _mix(self, target: list, others: list, rng) -> list:
        rows = list(target) + list(others)
        rng.shuffle(rows)
        return rows

    def _pick_default(self, feats, rng):
        rows = self._others("level", set(feats), rng, 3)
        return rows, self.fact_header, "no matching fact; a few unrelated fact lines (the model should say it is not in the facts)"

    def _pick_model(self, q, feats, rank, rng):
        ql = q.lower()
        used = []
        if re.search(r"\b(?:threshold|cut-?off|how was it set|flags?)\b", ql):
            used.append("threshold")
        if re.search(r"\b(?:pr-?auc)\b", ql):
            used += ["val", "test"]
        elif re.search(r"\b(?:recall|precision|f1|roc|test|held-?out|false-positive|catch(?:es)?|find|detect|how (?:well|good))\b", ql):
            used.append("test")
        if re.search(r"\b(?:which model|what (?:kind|model)|logistic|how was (?:it|the model) chosen|chosen|trained|behind)\b", ql):
            used.append("type")
        used = [k for k in dict.fromkeys(used) if k in self.model] or [k for k in ("type", "threshold", "val", "test") if k in self.model]
        extra = [k for k in ("type", "threshold", "val", "test") if k in self.model and k not in used]
        rng.shuffle(extra)
        keys = used + extra[: (1 if len(used) < 3 else 0)]
        rng.shuffle(keys)
        return [self.model[k] for k in keys], self.model_header, "classifier facts (model card)"

    def _pick_total(self, q, feats, rank, rng):
        rows = self.by_family.get("total", [])
        return rows[:1], self.fact_header, "overall development-month totals"

    def _pick_month(self, q, feats, rank, rng):
        months = self.by_family.get("month", [])
        m = _MONTH.search(q)
        n = m.group(1) if m and m.group(1) else None
        if rank or n is None or not any(f.vals["m"] == n for f in months):
            return list(months), self.fact_header, "fraud rate by development month (0-5)"
        target = [f for f in months if f.vals["m"] == n]
        others = rng.sample([f for f in months if f.vals["m"] != n], 2)
        return self._mix(target, others, rng), self.fact_header, f"month {n} and two other months"

    def _pick_pattern(self, q, feats, rank, rng):
        ql = q.lower().replace("not provided", "missing")
        words = [w for w in ("previous address", "intended transfer amount", "bank history")
                 if w in ql or (w == "intended transfer amount" and ("transfer" in ql or "balcon" in ql))]
        # "previous address missing, intended transfer amount present, ...": each feature may name its own state
        wanted = {}
        for w in words:
            m = re.search(re.escape(w) + r"\s+(?:is\s+|are\s+)?(missing|present)", ql)
            if m:
                wanted[w] = m.group(1)
        if words and not wanted:  # one state for all the mentioned features
            state = "present" if re.search(r"\bpresent\b", ql) and "missing" not in ql else "missing"
            wanted = {w: state for w in words}
        every = list(self.by_family.get("pattern", []))
        exact = [f for f in every if wanted and all(f"{w} {s}" in f.vals["fv"] for w, s in wanted.items())]
        if len(exact) == 1 and len(wanted) == 3:  # the full pattern is named: it plus two others, as in training
            return self._mix(exact, rng.sample([f for f in every if f is not exact[0]], 2), rng), self.fact_header, "that missing-value pattern and two others"
        rows = sorted(exact or every, key=lambda f: -f.raw["rate"])
        return rows[:MAX_LINES], self.fact_header, "fraud rate by missing-value pattern"

    def _pick_level(self, q, feats, rank, rng):
        ql = q.lower()
        if not feats or feats[0] not in self.levels:
            # no named feature with level tables (e.g. a numeric feature): nothing on its fraud rate by level
            return [], self.fact_header, ""
        feat = feats[0]
        rows = self.levels[feat]
        words = set(re.findall(r"[a-z0-9]+", ql)) - _tokens_raw(FEATURE_LABELS[feat])
        named = [f for f in rows if f.vals["level"].lower() in words and f.vals["level"].lower() not in _BINARY]
        if not named:
            named = [f for f in rows if f.vals["level"].lower() in words and f.vals["level"].lower() in _BINARY]
        label = FEATURE_LABELS[feat]
        if rank or not named or len(rows) <= 3:
            rows = sorted(rows, key=lambda f: -f.raw["rate"]) if len(rows) > MAX_LINES else rows
            return list(rows), self.fact_header, f"all levels of {label}"
        return self._mix(named[:1], self._others("level", {feat}, rng, 2), rng), self.fact_header, f"{named[0].vals['fv']} and two unrelated levels"

    def _pick_per_feature(self, fam, q, feats, rank, rng, what):
        """numeric / recall / assoc / psi: the named feature plus distractors, or the strongest few for 'which ...'."""
        target = [self.per_feature[(fam, f)] for f in feats[:1] if (fam, f) in self.per_feature]
        if target and not rank:
            skip = {feats[0]}
            return self._mix(target, self._others(fam, skip, rng, 2), rng), self.fact_header, f"{what} for {FEATURE_LABELS[feats[0]]} and two others"
        rows = sorted(self.by_family.get(fam, []), key=lambda f: -f.raw.get("s", 0.0))
        top = rows[:5]
        rng.shuffle(top)  # the training contexts were in random order; the true maximum is always among them
        return top, self.fact_header, f"{what}: the five strongest features (the maximum is among them)"

    def _pick_recall(self, q, feats, rank, rng):
        return self._pick_per_feature("recall", q, feats, rank, rng, "single-feature recall")

    def _pick_assoc(self, q, feats, rank, rng):
        return self._pick_per_feature("assoc", q, feats, rank, rng, "association with fraud")

    def _pick_psi(self, q, feats, rank, rng):
        return self._pick_per_feature("psi", q, feats, rank, rng, "monthly drift (PSI)")

    def _pick_numeric(self, q, feats, rank, rng):
        if feats and ("numeric", feats[0]) not in self.per_feature:
            return [], self.fact_header, ""  # no median / KS line for a categorical feature
        return self._pick_per_feature("numeric", q, feats, False, rng, "legitimate vs fraud medians")

    def _pick_corr(self, q, feats, rank, rng):
        rows = self.by_family.get("corr", [])
        labels = [FEATURE_LABELS[f].lower() for f in feats[:2]]
        hit = [f for f in rows if all(l in f.line.lower() for l in labels)] if len(labels) == 2 else []
        if not hit and feats:
            hit = [f for f in rows if FEATURE_LABELS[feats[0]].lower() in f.line.lower()]
        return (hit or list(rows)), self.fact_header, "Spearman correlations"


def _tokens_raw(text: str) -> set:
    """Lower-case word set of a feature label (so a number inside the label is not read as a level)."""
    return set(re.findall(r"[a-z0-9]+", _norm(text)))
