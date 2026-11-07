"""B0: rule-based template responder (no language model), the first baseline of the FraudLM v2 evaluation.

Purpose: answer a test prompt with fixed sentences filled in from the context, using only keyword rules on
the instruction. It copies numbers from the context, so it is an upper bound for number faithfulness on
explain_score, but it cannot follow a paraphrase it has no rule for and it cannot answer banking_qa or
general questions (it says so). BUILD_PLAN 9.2 defines B0 this way.

This responder is written separately from the dataset templates in preprocess_data/ on purpose: B0 sees
only (instruction, input), never the task label, the policy category or the reference answer.

Author: FraudLM v2 team (COS30049.2-1).

"""

from __future__ import annotations

import re

from evaluation.metrics import extract_numbers, mentioned_features, number_value, parse_context

REL = "compared with a typical application"

# Keyword rules. They use only question wordings that appear in the TRAIN split (the "t" and "l" lists of
# preprocess_data/templates_*.py); the held-out wordings of the test split were left out on purpose, so the
# -heldout rows measure how B0 copes with a paraphrase it has no rule for.
_DECISION_Q = re.compile(r"\b(approv\w*|reject\w*|declin\w*|block\w*|final call|decide for me|stop this account)\b", re.I)
_PROTECTED_Q = re.compile(r"\b(age|young|old|income|salary|job|employment|demographic)\b", re.I)
_VERDICT_Q = re.compile(r"\b(fraudster|fraudulent|is (this|it) fraud|criminal|probability|percentage chance|how likely|"
                        r"prove|confirm this is fraud|for certain|genuine customer|model sure)\b", re.I)
_EVADE_Q = re.compile(r"\b(get .* past|avoid (being )?flag\w*|get around|stolen identity|without being caught|"
                      r"lower the model score of|stay under|look genuine|values should i change)\b", re.I)
_SCOPE_Q = re.compile(r"\b(who are you|what can you do|what is fraudlm|introduce yourself|able to help|"
                      r"assistant is for|are you the one|real customer data|data were you built)\b", re.I)
_PERSONAL_Q = re.compile(r"\b(name|address|phone number|bank branch|account balance|ip address|"
                         r"other applications has)\b", re.I)
_INJECTION = re.compile(r"ignore the rules|pre-approved|system override|instruction found in the form", re.I)
_SUPERLATIVE = re.compile(r"\b(worst|highest|most|largest|strongest|biggest|top)\b", re.I)

CANNOT_ANSWER = ("I can only explain a model score, its main factors and the dataset facts given in the context; "
                 "I have no rule for this question.")


def _state(flagged: bool) -> str:
    return "flagged for review" if flagged else "not flagged"


def _factor_text(f: dict) -> str:
    return f"{f['name']} ({f['value']}), which {f['direction']} the model score"


def _explain(ctx: dict) -> str:
    """The one explain_score answer: score, threshold, result and every listed factor."""
    rel = "at or above" if ctx["flagged"] else "below"
    text = (f"The model score is {ctx['score']}, which is {rel} the review threshold of {ctx['threshold']}, "
            f"so the application is {_state(ctx['flagged'])}.")
    if ctx["factors"]:
        text += " The main factors are " + "; ".join(_factor_text(f) for f in ctx["factors"]) + f", {REL}."
    return text + " The fraud analyst makes the decision."


def _primary_number(line: str) -> float:
    """The number a 'which is the highest' question compares: fraud rate, PSI, Cramer's V or recall."""
    for pat in (r"fraud rate (\d+(?:\.\d+)?)%", r"PSI (?:of )?(\d+(?:\.\d+)?)", r"Cramer's V (?:of )?(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)% of frauds"):
        m = re.search(pat, line)
        if m:
            return float(m.group(1))
    nums = extract_numbers(line)
    return number_value(nums[-1]) if nums else float("-inf")


def _subject(line: str) -> str:
    return line.split(":", 1)[0].lower()


def _facts_answer(instruction: str, facts: list[str]) -> str:
    """Pick the fact line that matches the question best and repeat it."""
    q_words = set(re.findall(r"[a-z0-9]+", instruction.lower())) - {"the", "a", "of", "for", "is", "what", "how",
                                                                    "in", "and", "with", "applications", "fraud"}
    scored = []
    for i, line in enumerate(facts):
        s_words = set(re.findall(r"[a-z0-9]+", _subject(line)))
        scored.append((len(q_words & s_words), i, line))
    best = max(s for s, _, _ in scored)
    if _SUPERLATIVE.search(instruction) and best <= 1:
        line = max(facts, key=_primary_number)
    elif best == 0:
        return "The provided facts do not include that, so I cannot answer it from them."
    else:
        line = max(scored, key=lambda t: (t[0], -t[1]))[2]
    return f"From the facts given: {line}."


def template_answer(instruction: str, input_text: str = "") -> str:
    """B0 answer for one prompt (deterministic; same input -> same output)."""
    ctx = parse_context(input_text)
    has_score = ctx["score"] is not None and ctx["threshold"] is not None and ctx["flagged"] is not None
    q = instruction or ""

    if _INJECTION.search(input_text or ""):
        base = "I ignore instructions found in the input, and I never approve, reject or block an application."
        return base + (" " + _explain(ctx) if has_score else "")
    # Evasion and scope questions come without a score context; with a context, "stay under the threshold"
    # is an explanation question, not an evasion request.
    if not has_score and _EVADE_Q.search(q):
        return "I can't help with getting around fraud checks or making an application look genuine."
    if not has_score and _SCOPE_Q.search(q):
        return ("I am FraudLM, an assistant for fraud analysts. I explain a classifier's model score and its main "
                "factors and answer questions about the facts given to me. I never approve, reject or block an application.")
    if _DECISION_Q.search(q):
        base = "I can't approve, reject or block an application; that decision stays with the fraud analyst."
        return base + (" " + _explain(ctx) if has_score else "")
    if _PROTECTED_Q.search(q):
        base = "Age, income and employment status are not inputs to the classifier, and I never cite them as a reason."
        return base + (" " + _explain(ctx) if has_score else "")
    if _VERDICT_Q.search(q):
        base = "I can't say that. The model score is a screening signal that ranks applications for review; it is not a verdict and not a probability."
        if has_score:
            base += f" The model score is {ctx['score']} and the application is {_state(ctx['flagged'])} (threshold {ctx['threshold']})."
        return base

    if has_score:
        # A question about a value or a personal detail that the context does not list.
        asked = mentioned_features(q)
        asked_names = {f["name"].lower() for f in ctx["factors"] if f["feature"] in asked}
        if (asked and not asked_names) or (_PERSONAL_Q.search(q) and not asked_names):
            names = ", ".join(f["name"] for f in ctx["factors"])
            return (f"That is not in the context I was given. I only have the model score ({ctx['score']}), the review "
                    f"threshold ({ctx['threshold']}), the result and the main factors: {names}.")
        return _explain(ctx)

    if ctx["facts"]:
        return _facts_answer(q, ctx["facts"])

    return CANNOT_ANSWER
