"""Shared paths, safety regexes and small helpers for the FraudLM v2 dataset builders.

Author: FraudLM v2 team (COS30049.2-1).
The protected-term and decision-term lists follow A1 requirement OOS07 and PLAN.md decision D7.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # llm_ver2/
A2_ROOT = ROOT.parent                               # assignment-2/
PREPROCESSED = ROOT / "preprocessed"
RAW = PREPROCESSED / "raw"
ML_REPO = PREPROCESSED / "_ml_repo"
DATASET = ROOT / "dataset"
# EDA tables (A2-09) live in assignment-2/reports/tables. A clone of the GitHub repo has no parent folder,
# so it falls back to the identical copy in preprocessed/eda_tables (copied 06/11, same files).
EDA_TABLES = A2_ROOT / "reports" / "tables"
if not EDA_TABLES.is_dir():
    EDA_TABLES = PREPROCESSED / "eda_tables"
BASE_ZIP = A2_ROOT / "Base.csv.zip"
ARTIFACT = ML_REPO / "artifacts" / "production_model.joblib"
MODEL_CARD = ML_REPO / "artifacts" / "model_card.json"
if not MODEL_CARD.exists():
    MODEL_CARD = PREPROCESSED / 'legacy_model_card.json'
ML_SCORES = PREPROCESSED / "ml_scores.csv"
ML_SCORES_DEMO = PREPROCESSED / "ml_scores_months6_7_demo.csv"
ML_SCORES_META = PREPROCESSED / "ml_scores.meta.json"

SEED = 42
DEV_MAX_MONTH = 5       # development months 0-5; months 6-7 are the untouched test months
MAX_TOKENS = 1024       # hard limit of the GPT-2 context
TARGET_TOKENS = 512

# Columns that may never appear as a reason (A1 OOS07, PLAN.md D7) or are not model inputs.
PROTECTED_COLUMNS = ("customer_age", "income", "employment_status")
NON_INPUT_COLUMNS = ("x1", "x2", "month", "device_fraud_count", "fraud_bool")
EXCLUDED_COLUMNS = PROTECTED_COLUMNS + NON_INPUT_COLUMNS
# Open team decision: date_of_birth_distinct_emails_4w is a model input but correlates with customer_age
# (Spearman -0.46 in the EDA). Put its name here to stop it being stated as a factor or fact; then rebuild.
EXTRA_EXCLUDED_FACTORS: tuple[str, ...] = ()

# Words that name a protected attribute (age, income, employment). Used by the builder to drop
# rows and by the tests to check outputs.
PROTECTED_RE = re.compile(
    r"\b(?:age|ages|aged|ageing|aging|customer_age|older|elderly|senior|seniors|young|younger|"
    r"income|incomes|salary|salaries|wage|wages|earnings|"
    r"employment|employment_status|employed|unemployed|employer|employers|job|jobs|occupation)\b",
    re.IGNORECASE,
)
# Words that would turn the assistant into a decision maker (OOS07).
DECISION_RE = re.compile(
    r"\b(?:approv(?:e|es|ed|ing|al)|reject(?:s|ed|ing|ion)?|block(?:s|ed|ing)?|"
    r"declin(?:e|es|ed|ing)|den(?:y|ies|ied|ying|ial)|blacklist(?:ed)?)\b",
    re.IGNORECASE,
)
# Wording we never use for a flagged application: the model score is not a probability and a
# flag is not a verdict.
VERDICT_RE = re.compile(r"\b(?:fraudulent|fraudster|probability|probable|guilty|criminal)\b", re.IGNORECASE)
# A refusal sentence must carry one of these so the decision / protected words are negated.
NEGATION_RE = re.compile(
    r"\b(?:not|never|cannot|can't|can not|won't|no|neither|nor|without)\b|n't", re.IGNORECASE
)

# Profanity and insults: dropped from public data (Bitext also tags such rows with W = offensive language).
PROFANITY_RE = re.compile(
    r"\b(?:fuck\w*|shit\w*|bullshit|bitch\w*|asshole\w*|ass|bastard\w*|crap\w*|damn\w*|piss\w*|dick\w*|wtf|"
    r"idiot\w*|stupid|moron\w*|dumb\w*|bloody|sucks?|screw\w*|hell)\b",
    re.IGNORECASE,
)

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalise(text: str) -> str:
    """Lower-case, collapse whitespace and drop punctuation: the key used to detect duplicates."""
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", text.lower())).strip()


def example_key(instruction: str, input_text: str) -> str:
    return normalise(instruction) + "||" + normalise(input_text)


def numbers_in(text: str) -> list[str]:
    """Numbers as written (commas kept, trailing '.' dropped), for the 'numbers come from the input' test."""
    return [n.rstrip(",.") for n in NUMBER_RE.findall(text)]


_ENC = None


def token_count(text: str) -> int:
    """Token count with tiktoken gpt2; <|endoftext|> counts as one token."""
    global _ENC
    if _ENC is None:
        import tiktoken

        _ENC = tiktoken.get_encoding("gpt2")
    return len(_ENC.encode(text, allowed_special={"<|endoftext|>"}))
