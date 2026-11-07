"""Load, filter and convert the public instruction sets to the FraudLM Alpaca schema.

banking_qa : Bitext retail-banking customer-support Q&A (CDLA-Sharing-1.0)
general    : Alpaca-cleaned (CC-BY-4.0)

Filters (both sets): drop rows with template placeholders ({{...}}), URLs, code, too long or too short
answers, profanity, and any row that mentions a protected attribute (age, income, employment ...) or a
decision word (approve, reject, block, decline, deny), so the chatbot cannot pick those up from public data.

Extra Bitext filters (G3 review F3):
    - rows tagged W (offensive language) are dropped;
    - answers that ask the user to provide an account number, password, PIN, card number, name or other
      personal data are dropped (a fraud-analyst tool must not learn to ask for them);
    - intents are used in priority tiers: first the ones close to account opening, security and fraud
      handling, then the other everyday ones, then loans and mortgages. Inside a tier the intents are
      taken round-robin, so the shortfall of a tier is filled from the next tier and nothing is padded.

Author: FraudLM v2 team (COS30049.2-1).
"""

from __future__ import annotations

import json
import random
import re

import pandas as pd

from preprocess_data import common as C

BITEXT_SOURCE = "bitext/Bitext-retail-banking-llm-chatbot-training-dataset (CDLA-Sharing-1.0)"
ALPACA_SOURCE = "yahma/alpaca-cleaned (CC-BY-4.0)"

CODE_RE = re.compile(r"```|\bdef \w+\(|\bimport \w+|<\s*/?\w+\s*>|\bSELECT\b|\bprint\(|=>|\{\s*\"")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
REFUSAL_RE = re.compile(r"\bas an ai\b|i'?m sorry|i cannot|i can't", re.IGNORECASE)

# Personal data an assistant for fraud analysts must not learn to ask for.
PII_RE = re.compile(
    r"account (?:number|details|information|id)|card (?:number|details|information)|credit card number|"
    r"pass ?words?|pass ?codes?|\bPINs?\b|\bCVV\b|security code|security question|"
    r"full name|your name|\bname and\b|social security|\bSSN\b|date of birth|"
    r"personal (?:details|information|data|identification)|identification|\bID\b|"
    r"routing number|\bIBAN\b|\bSWIFT\b|e-?mail address|phone number|(?:home|billing|mailing|residential) address|"
    r"recipient'?s? (?:name|information|details)|beneficiary|username|user name|login (?:details|credentials)|"
    r"transaction (?:id|reference)|reference number|order number",
    re.IGNORECASE,
)
# Mentioned anywhere in an answer: the row is dropped even without an explicit request verb (e.g. a form-like
# list "Cardholder's Name: ... Last 4 digits of the Card Number: ...").
STRICT_PII_RE = re.compile(
    r"account number|card number|\bPINs?\b|\bCVV\b|security code|cardholder|last (?:4|four) digits|social security|"
    r"date of birth|\bSSN\b|\bIBAN\b|routing number|recipient'?s? (?:name|information|details)",
    re.IGNORECASE,
)
ASK_RE = re.compile(
    r"\b(?:provid\w*|enter\w*|giv\w*|shar\w*|submit\w*|suppl\w*|input\w*|typ(?:e|ing)\b|send\w*|requir\w*|need\w*|"
    r"confirm\w*|verif\w*|fill\w*|answer\w*|stat(?:e|ing)\b|specif\w*|includ\w*|mention\w*|have ready|"
    r"prepar\w*|gather\w*|ask\w*|request\w*|tell\w*|reply\w*|let me know)\b",
    re.IGNORECASE,
)

# Bitext intents grouped by closeness to account opening, security and fraud handling.
PRIORITY_TIERS: list[tuple[str, ...]] = [
    ("create_account", "close_account", "customer_service", "human_agent", "check_recent_transactions",
     "dispute_ATM_withdrawal", "activate_card", "cancel_card", "recover_swallowed_card", "get_password",
     "set_up_password", "block_card", "make_transfer", "cancel_transfer", "activate_card_international_usage",
     "check_current_balance_on_card"),
    ("check_fees", "check_card_annual_fee", "find_branch", "find_ATM"),
    ("apply_for_loan", "cancel_loan", "check_loan_payments", "apply_for_mortgage", "cancel_mortgage",
     "check_mortgage_payments"),
]

LAST_STATS: dict = {}


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _unsafe(*texts: str) -> bool:
    joined = " ".join(texts)
    return bool(
        C.PROTECTED_RE.search(joined) or C.DECISION_RE.search(joined) or C.PROFANITY_RE.search(joined)
        or URL_RE.search(joined) or CODE_RE.search(joined)
    )


def asks_personal_data(answer: str) -> bool:
    """True when one sentence both asks the user for something and names personal / account data,
    or when the answer names card / account identifiers at all."""
    if STRICT_PII_RE.search(answer):
        return True
    for sentence in re.split(r"(?<=[.!?:;])\s+|\n+", answer):
        if PII_RE.search(sentence) and ASK_RE.search(sentence):
            return True
    return False


def load_bitext(n: int, seed: int = C.SEED) -> list[dict]:
    """Up to n rows; short, placeholder-free, safe, no profanity and no requests for personal data."""
    df = pd.read_csv(C.RAW / "bitext_retail_banking" / "bitext.csv")
    stats = {"raw": len(df)}
    df["instruction"] = df["instruction"].astype(str).map(_clean)
    df["response"] = df["response"].astype(str).map(_clean)
    df = df[~df["tags"].astype(str).str.contains("W")]
    stats["after_tag_W"] = len(df)
    keep = (
        ~df["response"].str.contains(r"\{\{", regex=True)
        & ~df["instruction"].str.contains(r"\{\{", regex=True)
        & df["response"].str.len().between(150, 900)
        & df["instruction"].str.len().between(15, 160)
    )
    df = df[keep]
    stats["after_length_placeholders"] = len(df)
    df = df[[not _unsafe(i, r) for i, r in zip(df["instruction"], df["response"])]]
    stats["after_unsafe_words"] = len(df)
    df = df[[not asks_personal_data(r) for r in df["response"]]]
    stats["after_personal_data_requests"] = len(df)
    df = df.drop_duplicates(subset="instruction").drop_duplicates(subset="response")
    stats["after_dedupe"] = len(df)

    rng = random.Random(seed)
    by_intent = {}
    for intent, g in df.groupby("intent"):
        idx = list(g.index)
        rng.shuffle(idx)
        by_intent[intent] = idx
    stats["rows_per_intent"] = {k: len(v) for k, v in sorted(by_intent.items())}
    rows = []
    for tier in PRIORITY_TIERS:
        queues = [list(by_intent.get(i, [])) for i in tier]
        while any(queues) and len(rows) < n:
            for q, intent in zip(queues, tier):
                if q and len(rows) < n:
                    i = q.pop()
                    rows.append({"instruction": df.at[i, "instruction"], "input": "", "output": df.at[i, "response"],
                                 "task": "banking_qa", "source": BITEXT_SOURCE, "tag": f"bitext:{intent}"})
        if len(rows) >= n:
            break
    stats["returned"] = len(rows)
    LAST_STATS["bitext"] = stats
    rng.shuffle(rows)
    return rows


def load_alpaca(n: int, seed: int = C.SEED) -> list[dict]:
    """About n general instruction rows with short answers and short (or no) input."""
    data = json.loads((C.RAW / "alpaca_cleaned" / "alpaca_data_cleaned.json").read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(data)
    rows, seen = [], set()
    for r in data:
        ins, inp, out = _clean(r["instruction"]), _clean(r.get("input", "")), _clean(r["output"])
        if not (30 <= len(out) <= 500) or len(inp) > 250 or len(ins) < 12:
            continue
        if _unsafe(ins, inp, out) or REFUSAL_RE.search(out):
            continue
        key = C.normalise(ins + " " + inp)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"instruction": ins, "input": inp, "output": out, "task": "general",
                     "source": ALPACA_SOURCE, "tag": "alpaca"})
        if len(rows) >= n:
            break
    return rows
