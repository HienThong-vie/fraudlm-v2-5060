"""
Deterministic safety guard for the FraudLM chat demo (question side and answer side).

Purpose: FraudLM only ranks applications for review (OOS07). The guard makes sure the demo never says
anything outside that role, whatever the model writes:
  * a question that asks the tool to approve / reject / block / close something gets a fixed refusal
    (the model is not even called), as does a question about age / income / employment as a reason
    and a prompt-injection attempt;
  * an answer is replaced by a safe template built from the application's context (and marked
    [guarded]) when it contains decision language, a protected attribute as a reason, an invented
    number, a statement that contradicts the Result line, or a definite accusation of fraud;
  * "probability" is rewritten to "model score" (a logistic-regression score is not a calibrated
    probability), and the decision-support footer is added by the caller (FOOTER).
No model, no randomness: the same text always gives the same verdict, so it can be unit-tested
(tests/test_guard.py). It is deliberately strict: a wrong replacement costs one safe answer, a missed
one breaks the policy.

Author: FraudLM v2 team (COS30049.2-1).
Ideas adapted (not imported) from the first FraudLM route's guard (llm/src/fraudlm/infer/guard.py).

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

from utils.prompt import DIRECTION_BASIS, FEATURE_LABELS, format_score_context, format_value

FOOTER = "Decision support only. Model output can be wrong."
GUARDED_TAG = "[guarded]"


# =====================================================================================
# 1. The application in focus
# =====================================================================================

@dataclass(frozen=True)
class AppContext:
    """One scored demo application (a row of preprocessed/ml_scores_months6_7_demo.csv).

    `factors` = [{"feature", "direction", "contribution", "value"}, ...], non-protected only.
    `label` is the ground truth of the demo data; the tool does not know it, so it is never part
    of the prompt or of a reply.
    """

    app_id: int
    month: int
    score: float
    threshold: float
    factors: tuple
    label: int | None = None

    @property
    def flagged(self) -> bool:
        return self.score >= self.threshold

    @property
    def score_s(self) -> str:
        return f"{self.score:.4f}"

    @property
    def threshold_s(self) -> str:
        return f"{self.threshold:.4f}"

    @property
    def input_text(self) -> str:
        """The exact `### Input:` text the model saw in training (rules line + score block)."""
        return format_score_context(self.score, self.threshold, list(self.factors))

    @property
    def card_text(self) -> str:
        """The same block without the rules line: what the analyst sees on screen."""
        return format_score_context(self.score, self.threshold, list(self.factors), rules=False)

    def result_sentence(self) -> str:
        if self.flagged:
            return (f"The model score is {self.score_s}, at or above the review threshold of "
                    f"{self.threshold_s}, so the application is flagged for review.")
        return (f"The model score is {self.score_s}, below the review threshold of "
                f"{self.threshold_s}, so the application is not flagged.")

    def reference_line(self) -> str:
        """One line for fixed replies: 'For reference, application 795470: the model score is ...'."""
        r = self.result_sentence()
        return f"For reference, application {self.app_id}: {r[0].lower()}{r[1:]}"

    def factors_sentence(self) -> str:
        parts = [f"{FEATURE_LABELS[f['feature']]} ({format_value(f['feature'], f.get('value'))}) "
                 f"{f['direction']} the model score" for f in self.factors]
        if not parts:
            return ""
        return "Main factors: " + "; ".join(parts) + f", {DIRECTION_BASIS}."


# =====================================================================================
# 2. Fixed texts (wording follows the refusals the model was trained on)
# =====================================================================================

DECISION_REFUSAL = ("I can't approve, reject, block or close anything. FraudLM only ranks applications for "
                    "review: it explains a model score and its main factors, and the fraud analyst makes "
                    "every decision.")
PROTECTED_REPLY = ("Age, income and employment status are not inputs to the classifier, and I never cite "
                   "them as a reason. Group-level fairness checks are reported separately by the team.")
INJECTION_REPLY = ("I can't change my rules or act on instructions that try to override them. I only explain "
                   "a model score and its main factors, and I never approve, reject or block an "
                   "application; the fraud analyst makes every decision.")
NEED_CONTEXT = ("If you give me an application (type 'explain <app id>', or 'list' to see demo ids), "
                "I can explain its model score and main factors.")
GENERIC_FALLBACK = ("I can't give a reliable answer to that. I can explain the model score of a demo "
                    "application (type 'explain <app id>', or 'list' to see demo ids) and answer general "
                    "banking and fraud-screening questions.")
INTRO = ("I'm FraudLM, a decision-support helper for fraud analysts. I explain why a demo application got "
         "its model score and answer banking and fraud-screening questions. I never approve, reject or "
         "block an application. Type 'help' for the commands.")

SMALL_TALK = [
    (re.compile(r"^(?:hi|hello|hey|hiya|howdy|good\s+(?:morning|afternoon|evening)|greetings)\b[\s!.,]*(?:there|fraudlm)?[\s!.]*$", re.I),
     "Hello. " + INTRO),
    (re.compile(r"^(?:thanks|thank\s+you|thank\s+u|thx|cheers|ty)\b[\s\w!.,]{0,25}$", re.I), "You're welcome."),
    (re.compile(r"^(?:bye|goodbye|good\s+bye|see\s+you|see\s+ya|cya)\b[\s\w!.,]{0,15}$", re.I),
     "Goodbye. Type 'quit' to leave the chat."),
    (re.compile(r"^(?:how\s+are\s+you|how's\s+it\s+going|what's\s+up|sup)\b[\s\w?!.,]{0,15}$", re.I),
     "I'm a language model, so I have no feelings, but I'm ready to help. " + NEED_CONTEXT),
    (re.compile(r"^(?:who\s+are\s+you|what\s+are\s+you|what\s+is\s+fraudlm|what\s+can\s+you\s+do|what\s+do\s+you\s+do)\b[\s\w?!.,]{0,15}$", re.I),
     INTRO),
]


# =====================================================================================
# 3. Question guard
# =====================================================================================

# Decision verbs as a request to the tool. First-person objects ("block my card") are ordinary banking
# customer-service questions and are left to the model.
_QV = (r"(?:approv(?:e|es|ed|ing)|reject(?:s|ed|ing)?|den(?:y|ies|ied|ying)|declin(?:e|es|ed|ing)|"
       r"block(?:s|ed|ing)?|clos(?:e|es|ed|ing)|freez(?:e|es|ing)|suspend(?:s|ed|ing)?|terminat(?:e|es|ed|ing)|"
       r"cancel(?:s|led|ed|ling|ing)?|accept(?:s|ed|ing)?|turn(?:s|ed|ing)?\s+down|green-?light(?:s|ed)?|"
       r"let(?:ting)?\s+(?:it|this|them)\s+(?:through|pass))")
_NOT_OWN = r"(?!\s+(?:my|our|me|your)\b)"
_QNOUN = r"(?:applications?|applicants?|customers?|cases?|transactions?|accounts?|apps?)"
_DECISION_Q = [
    # imperative or direct object: "Block this account.", "Reject the application", "approve it", "approve 795470"
    re.compile(rf"\b{_QV}\b{_NOT_OWN}\s+(?:this|that|the|these|those|all|every|each|his|her|their|its)\s+(?:\w+\s+){{0,2}}?{_QNOUN}\b", re.I),
    re.compile(rf"\b{_QV}\b\s+(?:an?\s+)?(?:applications?|applicants?)\b", re.I),
    re.compile(rf"\b{_QV}\b\s+(?:accounts?|customers?|cases?|transactions?)\b", re.I),
    re.compile(rf"\b{_QV}\b\s+(?:it|them|him|her|this|that|these|those|everything|all|every|each)\b", re.I),
    re.compile(rf"\b{_QV}\b\s+(?:{_QNOUN}\s*)?(?:id\s*)?#?\d{{3,9}}\b", re.I),
    # modal frame: "Should I approve?", "Can you reject?", "Do you recommend approving?", "Who should I reject first?"
    re.compile(rf"\b(?:should|shall|can|could|would|will|may|must|do|does|did|want|need|going)\s+(?:i|we|you|they)\s+(?:\w+\s+){{0,3}}?{_QV}\b"
               rf"(?=\s*(?:[?.!,]|$)|\s+(?:or|and|it|this|that|them|him|her|these|those|first|next|now|today|anyway|too)\b)", re.I),
    # the noun forms and fixed phrases: "Please confirm approval", "Is approval recommended?", "Just let it through"
    re.compile(r"\b(?:confirm|grant|give|issue|recommend\w*|suggest\w*|advise\w*)\s+(?:\w+\s+)?(?:approval|rejection|denial)\b"
               r"|\b(?:approval|rejection|denial)\s+(?:recommended|advised|warranted|needed|appropriate|justified)\b"
               r"|\b(?:let|lets|letting)\s+(?:it|this|them|him|her)\s+(?:through|pass|in)\b|\bwave\s+(?:it|this|them)\s+through\b"
               r"|\bpass\s+(?:it|this|them)\s+(?:through|on)\b|\bgreen-?light\b", re.I),
    # passive: "Should this account be blocked?", "Can it be approved?"
    re.compile(rf"\b(?:this|that|the|it|these|those)\s+(?:\w+\s+){{0,3}}?(?:be|been|get|gets|getting)\s+{_QV}\b", re.I),
    # "Mark this application as approved."
    re.compile(r"\b(?:mark|set|label|tag|move|put|change|update|turn|record|log)\b.{0,50}\bas\s+(?:approved|rejected|blocked|declined|denied|accepted|closed|fraud|fraudulent|genuine|legit\w*)\b", re.I),
    # asking the tool for the verdict
    re.compile(rf"\b(?:your|the)\s+(?:decision|verdict|recommendation)\b.{{0,40}}\b{_QV}\b", re.I),
    re.compile(r"\b(?:make|take)\s+(?:the|a)\s+(?:final\s+)?(?:decision|call)\b|\bdecide\s+for\s+me\b", re.I),
]

_PROT = (r"(?:age|ages|aged|ageing|aging|customer_age|income|incomes|salary|salaries|wage|wages|earnings|"
         r"employment(?:\s+status)?|employment_status|employed|unemployed|self-employed|employers?|jobs?|occupations?|"
         r"young|younger|youngest|old|older|oldest|elderly|retired|retirees?|pensioners?|"
         r"years?[\s-]+old|year-olds?|demographics?|gender|race|ethnicity|nationality|religion)")
_PROT_RE = re.compile(rf"\b{_PROT}\b", re.I)
# column-like names (age_group, customer_age, employment_status): any underscore part that is one of these
_SEGMENTS = {"age", "ages", "aged", "income", "incomes", "salary", "salaries", "wage", "wages", "employment",
             "employed", "unemployed", "employer", "job", "jobs", "occupation"}
_IDENT_RE = re.compile(r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b", re.I)
# in a QUESTION, a protected term counts when it is asked about as a reason / driver / link to fraud, or
# asked about this applicant ("What is the applicant's age?")
_REASON_CUE = re.compile(
    r"\b(?:because|due\s+to|reason|reasons|cause|caused|causes|causing|why|drive|driven|drives|driving|driver|drivers|"
    r"factor|factors|explain|explains|explained|based\s+on|result\s+of|contribut\w+|influenc\w+|affect\w*|matter|matters|"
    r"depend\w*|raises?|lowers?|score|scores|flag|flagged|suspicious|risk|risky|fraud\w*|model|classif\w+|"
    r"more\s+likely|less\s+likely|likely|used|discriminat\w*|biased?|"
    r"applicants?|applications?|customers?|person|this\s+(?:one|case|account)|"
    r"threshold|input|inputs|feature|features)\b", re.I)
# in an ANSWER the cue is narrower, so a plain banking sentence ("you must be 18 years of age") passes
_ANSWER_CUE = re.compile(
    r"\b(?:because|due\s+to|reason|reasons|cause|caused|causes|drive|driven|drives|driving|driver|drivers|factor|factors|"
    r"explain|explains|based\s+on|result\s+of|contribut\w+|influenc\w+|affect\w*|raises?|lowers?|score|scores|flag|flagged|"
    r"suspicious|risk|risky|fraud\w*|model|classif\w+|more\s+likely|less\s+likely|discriminat\w*|biased?)\b", re.I)

_INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass|stop\s+following|(?:do\s+not|don't)\s+follow)\b.{0,40}\b(?:instructions?|rules?|guidelines?|polic(?:y|ies)|prompt|restrictions?)\b"
    r"|\b(?:system\s+prompt|developer\s+mode|jailbreak|dan\s+mode)\b|\bpretend\s+(?:to\s+be|you\s+are|that\s+you)\b"
    r"|\byou\s+are\s+now\b|\bact\s+as\s+(?:if|an?|the)\b|\breveal\s+your\s+(?:rules|prompt|instructions)\b", re.I)


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


@dataclass
class QuestionCheck:
    """kind: ok | empty | small_talk | decision_request | protected_reason | injection.
    `reply` is the fixed reply when the model must not be called (None for kind ok)."""

    kind: str
    reply: str | None = None


# A customer talking about their own card / account / transfer is a banking question, not a request to decide on a
# screened application ("I want to close my account", "help me cancel it").
_OWN_THING = re.compile(r"\bmy\s+(?:\w+\s+){0,2}?(?:accounts?|cards?|transfers?|payments?|loans?|mortgages?|cheques?|checks?|pins?|profiles?|subscriptions?|memberships?|wallets?|logins?)\b", re.I)
_APPLICATION_WORD = re.compile(r"\b(?:applications?|applicants?)\b", re.I)


def is_decision_request(question: str) -> bool:
    q = _norm_ws(question)
    if _OWN_THING.search(q) and not _APPLICATION_WORD.search(q):
        return False
    return any(p.search(q) for p in _DECISION_Q)


def names_protected_reason(question: str) -> bool:
    q = _norm_ws(question)
    names = bool(_PROT_RE.search(q)) or any(set(m.group().lower().split("_")) & _SEGMENTS for m in _IDENT_RE.finditer(q))
    return names and bool(_REASON_CUE.search(q))


def check_question(question: str, ctx: AppContext | None = None) -> QuestionCheck:
    """Decide whether the question may go to the model. Order: empty, small talk, injection, decision, protected."""
    q = _norm_ws(question)
    if not q:
        return QuestionCheck("empty", "Please type a question. Type 'help' for the commands.")
    for pattern, reply in SMALL_TALK:
        if pattern.match(q):
            return QuestionCheck("small_talk", reply)
    if _INJECTION.search(q):
        return QuestionCheck("injection", INJECTION_REPLY)
    if is_decision_request(q):
        return QuestionCheck("decision_request", f"{DECISION_REFUSAL} {ctx.reference_line()}" if ctx else f"{DECISION_REFUSAL} {NEED_CONTEXT}")
    if names_protected_reason(q):
        return QuestionCheck("protected_reason", f"{PROTECTED_REPLY} {ctx.reference_line()}" if ctx else PROTECTED_REPLY)
    return QuestionCheck("ok")


# =====================================================================================
# 4. Answer guard
# =====================================================================================

_SENTENCES = re.compile(r"(?:(?<!\d)\.|\.(?!\d)|[!?;:\n])")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCES.split(text) if s.strip()]


# ---- 4a. decision language -----------------------------------------------------------------
_DEC = re.compile(
    r"\b(?:approv(?:e|es|ed|ing|al)|reject(?:s|ed|ing|ion)?|block(?:s|ed|ing)?|declin(?:e|es|ed|ing)|"
    r"den(?:y|ies|ied|ying|ial)|accept(?:s|ed|ing)?|turn(?:s|ed|ing)?\s+(?:\w+\s+)?down|"
    r"green-?light(?:s|ed)?|go\s+ahead|onboard(?:s|ed|ing)?|safe\s+to\s+(?:proceed|open|accept)|"
    r"proceed\s+with\s+(?:the\s+)?(?:application|account|opening)|open\s+the\s+account)\b"
    # 'close' / 'freeze' ... only as an action on an account, application, case or card ('close to the threshold' is fine)
    r"|\b(?:clos(?:e|es|ed|ing)|freez(?:e|es|ing)|froze|frozen|suspend(?:s|ed|ing)?|terminat(?:e|es|ed|ing)|"
    r"cancel(?:s|led|ed|ling|ing)?)\s+(?:\w+\s+){0,2}?(?:accounts?|applications?|cases?|cards?|files?)\b", re.I)
_NEGATOR = re.compile(r"\b(?:not|never|no|cannot|unable|nor)\b|n't\b", re.I)
# who declines: the assistant itself ("I do not approve ..."), never an instruction ("don't reject it, approve it")
_SELF = re.compile(r"\bI\b|(?i:\b(?:we|FraudLM|this\s+assistant|the\s+assistant|the\s+chatbot|this\s+tool|the\s+tool|the\s+model|the\s+classifier|the\s+system)\b)")
_LIST_GAP = re.compile(r"^(?:\s|,|\bor\b|\band\b|\bnor\b|/)*$", re.I)
# "the fraud analyst decides whether to approve": the decision is handed to the analyst, not made
_ANALYST_WHETHER = re.compile(r"\banalysts?\b.*\b(?:decides?|decision|choose|chooses|makes?)\b.*\b(?:whether|if)\b", re.I)
_WHETHER_ANALYST = re.compile(r"\b(?:whether|if)\b.*\b(?:is|are|stays?|remains?)\b.*\banalyst", re.I)
# outside a grounded answer a decision word only counts when it is about an application or applicant (or the tool
# says "I would block ..."); customer-service steps such as "to close your account, open Settings" are fine
_APP_TARGET = re.compile(r"\b(?:applications?|applicants?)\b", re.I)
_OWN_TARGET = re.compile(r"\b(?:your|my|our)\s+(?:\w+\s+)?(?:applications?|accounts?|cards?|cheques?|checks?)\b", re.I)
_FIRST_PERSON_DECISION = re.compile(
    r"\b(?:I|we)\s+(?:would\s+|will\s+|'ll\s+|can\s+|could\s+|should\s+|must\s+|recommend\s+|suggest\s+|advise\s+|propose\s+|think\s+|say\s+)*"
    r"(?:approv|reject|block|declin|den[iy]|accept|clos|freez)\w*")


def decision_hits(text: str, grounded: bool) -> list[str]:
    """Decision words that are neither the assistant declining nor the analyst deciding."""
    hits = []
    for s in _sentences(text):
        if _ANALYST_WHETHER.search(s) or _WHETHER_ANALYST.search(s):
            continue
        prev_neg, prev_end = False, 0
        sentence_hits = []
        for m in _DEC.finditer(s):
            span_start = max(s.rfind(",", 0, m.start()), 0)
            if prev_neg and _LIST_GAP.match(s[prev_end:m.start()]):  # "I do not approve, reject or block"
                neg = True
            else:
                neg = bool(_NEGATOR.search(s[span_start:m.start()])) and bool(_SELF.search(s[:m.start()]))
            if not neg:
                sentence_hits.append(m.group())
            prev_neg, prev_end = neg, m.end()
        if not sentence_hits:
            continue
        if not grounded:
            about_case = bool(_APP_TARGET.search(s)) and not _OWN_TARGET.search(s)
            if not (about_case or _FIRST_PERSON_DECISION.search(s)):
                continue  # e.g. customer-service steps: "to block your card, open the app"
        hits.extend(sentence_hits)
    return hits


# ---- 4b. protected attributes as a reason ----------------------------------------------------
_PLIST = rf"{_PROT}(?:\s*(?:,|\band\b|\bor\b|\bnor\b)\s*{_PROT})*"
# the narrow forms in which naming a protected attribute is allowed: ruling it out as an input or a reason
_RULED_OUT = [
    re.compile(rf"\b{_PLIST}\s+(?:is|are|was|were)\s+(?:never|not)\s+(?:(?:a|an|the|any)\s+)?(?:\w+\s+){{0,3}}?(?:inputs?|features?|factors?|reasons?|basis|used|considered|cited|variables?)\b", re.I),
    re.compile(rf"\b(?:never|cannot|can't|can\s+not|do\s+not|don't|does\s+not|doesn't|will\s+not|won't|must\s+not|not)\s+(?:\w+\s+){{0,2}}?(?:use|uses|used|consider|considers|cite|cites|give|gives|state|states|treat|treats|take|takes)\s+(?:{_PLIST}|them|it|these)\b", re.I),
    re.compile(rf"\b(?:without|excluding|ignores?|ignoring)\s+{_PLIST}\b", re.I),
]

# occupation words are checked in answers only (a question about a student account is a normal banking question)
_OCCUPATION = re.compile(r"\b(?:students?|housewives|housewife|pensioners?|unemployment|self-employed)\b", re.I)


def protected_hits(text: str, grounded: bool) -> list[str]:
    """Sentences that name a protected attribute, unless it is only being ruled out as a reason."""
    out = []
    for s in _sentences(text):
        rest = s
        for rx in _RULED_OUT:
            rest = rx.sub(" ", rest)
        named = (bool(_PROT_RE.search(rest)) or bool(_OCCUPATION.search(rest))
                 or any(set(m.group().lower().split("_")) & _SEGMENTS for m in _IDENT_RE.finditer(rest)))
        if named and (grounded or _ANSWER_CUE.search(rest)):
            out.append(s.strip())
    return out


# ---- 4c. numbers -----------------------------------------------------------------------------
_NUM = re.compile(r"(?<![A-Za-z_\d])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.\d+|\d+)(%?)")
_SAFE_INTS = {"0", "1"}  # "a score between 0 and 1", "1 fraud"
_ALWAYS_OK = {"2022"}    # "the synthetic Bank Account Fraud dataset suite (NeurIPS 2022)": a trained provenance answer


def _canon(raw: str) -> str | None:
    try:
        d = Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None
    return format(d.normalize(), "f")


def extract_numbers(text: str) -> list[tuple[str, str, bool]]:
    """(raw text, canonical value, has % sign) for each number in `text`. '473,148' -> '473148', '0.50' -> '0.5'."""
    out = []
    for m in _NUM.finditer(text or ""):
        c = _canon(m.group(1))
        if c is not None:
            out.append((m.group(1) + m.group(2), c, bool(m.group(2))))
    return out


def invented_numbers(answer: str, grounding_text: str | None, question: str = "", extra_allowed: tuple = (),
                     n_factors: int = 3) -> list[str]:
    """Numbers in `answer` that cannot be traced to the text the model was shown.

    grounded (grounding_text given): every number must occur in the grounding text or the question
    (0 / 1 and the number of listed factors are always fine).
    ungrounded (None): there is nothing to check a measurement against, so only precise measurements are
    flagged (three or more decimals such as 0.7462, or a fractional percentage such as 3.52%) unless the
    question or the fixed allowance contains them. Whole numbers, round percentages and short decimals
    (steps, "5%", "0.25") pass: the trained concept answers use them.
    """
    allowed = {c for _, c, _ in extract_numbers(question)} | {c for c in (_canon(str(x)) for x in extra_allowed) if c}
    allowed |= _ALWAYS_OK
    if grounding_text is not None:
        allowed |= {c for _, c, _ in extract_numbers(grounding_text)}
        allowed |= _SAFE_INTS | {str(n_factors)}
    bad = []
    for raw, c, pct in extract_numbers(answer):
        if c in allowed:
            continue
        if grounding_text is None:
            decimals = len(c.split(".")[1]) if "." in c else 0
            if decimals < 3 and not (pct and decimals >= 1):
                continue
        if raw not in bad:
            bad.append(raw)
    return bad


# ---- 4d. contradictions with the Result line and definite accusations ------------------------
# A sentence that states the opposite of the Result line. Words before the claim that make it conditional or
# negated ("if", "would", "does not mean", "even though" ...) cancel it.
_CLAIM_FLAGGED = re.compile(
    r"\b(?:is|was|been|being|gets?|got)\s+(?:also\s+|therefore\s+|now\s+)?flagged\b"
    r"|(?:\b(?:is|was)\s+|,\s*)(?:at\s+or\s+)?(?:above|over)\s+the\s+(?:review\s+)?threshold\b", re.I)
_CLAIM_NOT_FLAGGED = re.compile(
    r"\b(?:is|was|stays?|remains?|got|gets?)\s+(?:also\s+|therefore\s+)?not\s+(?:been\s+)?flagged\b"
    r"|\bunflagged\b|(?:\b(?:is|was)\s+|,\s*)(?:below|under)\s+the\s+(?:review\s+)?threshold\b", re.I)
_CONDITIONAL = re.compile(
    r"\b(?:if|whether|would|could|might|may|unless|until|once|when|than|rather|although|though|despite|"
    r"not\s+mean|does\s+not|doesn't|did\s+not|didn't|not\s+necessarily|no\s+guarantee|not\s+(?:a\s+)?proof)\b"
    r"|n't\b|\bnever\b", re.I)


def contradicts_result(answer: str, flagged: bool) -> bool:
    """True when a sentence states the opposite of the Result line (flagged vs not flagged)."""
    claim = _CLAIM_NOT_FLAGGED if flagged else _CLAIM_FLAGGED
    for s in _sentences(answer):
        for m in claim.finditer(s):
            if not _CONDITIONAL.search(s[: m.start()]):
                return True
    return False


_ACCUSATION = re.compile(
    # subject = this case / this person ("this is fraud", "the applicant is a fraudster", "this application is fraudulent")
    r"\b(?:(?:this|that|it|he|she|they)|(?:(?:the|this|that|an?|your|their|his|her)\s+)?(?:\w+\s+)?"
    r"(?:applicant|application|customer|person|case|account|transaction|one))\s+"
    r"(?:is|are|was|were|will\s+be)\s+(?:definitely\s+|certainly\s+|clearly\s+|obviously\s+|surely\s+|likely\s+|probably\s+)?"
    r"(?:a\s+|an\s+)?(?:fraud|fraudster|fraudulent|scammer|criminal|guilty|scam)\b(?!\s+(?:applications?|cases?|transactions?|accounts?|rates?))"
    r"|\b(?:definitely|certainly|clearly|obviously|surely)\s+(?:a\s+)?(?:fraud|fraudster|fraudulent|scammer|criminal|guilty)\b", re.I)
_ACCUSATION_HEDGE = re.compile(r"\b(?:not|never|no|if|whether|may|might|could|can|would|unless|possible|possibly|potentially|whose|prove|proof)\b|n't\b", re.I)


def accusation_hits(text: str) -> list[str]:
    """Definite statements that someone is a fraudster / the case is fraud (the model score is not proof)."""
    out = []
    for s in _sentences(text):
        for m in _ACCUSATION.finditer(s):
            if not _ACCUSATION_HEDGE.search(s[max(0, m.start() - 60): m.end()]):
                out.append(s.strip())
                break
    return out


# ---- 4d2. nonsense: off-topic or looping output (typical of the untuned base model) ---------------
_ON_TOPIC = re.compile(
    r"\b(?:model|score|scores|scored|threshold|flag|flagged|flags|factor|factors|context|classifier|dataset|analyst|"
    r"review|application|applications|applicant|fraud|frauds|result|data|rules?|facts?|guess|cannot|can't|rate|recall|"
    r"precision|auc|psi|drift|inputs?)\b", re.I)
_TEMPLATE_MARKER = re.compile(r"###\s*(?:Response|Instruction|Input)\b|<\|endoftext\|>")


def _same_text(a: str, b: str) -> bool:
    """The answer only repeats the question (the untuned base model often does this)."""
    key = lambda t: re.sub(r"\W+", " ", t or "").strip().lower()  # noqa: E731
    return bool(key(a)) and key(a) == key(b)


def refers_to_context(answer: str, grounding_text: str = "") -> bool:
    """A grounded answer should at least talk about the score, flag, factors or the data.

    Crude on purpose (it only catches nonsense): an on-topic word, or a factor name / number from the context.
    """
    if _ON_TOPIC.search(answer):
        return True
    low = answer.lower()
    if any(label.lower() in low for label in FEATURE_LABELS.values() if label.lower() in grounding_text.lower()):
        return True
    shown = {c for _, c, _ in extract_numbers(grounding_text)} - _SAFE_INTS
    return any(c in shown for _, c, _ in extract_numbers(answer))


def is_degenerate(answer: str) -> bool:
    """Looping output: template markers leaking through, or the same sentence/line three or more times."""
    if _TEMPLATE_MARKER.search(answer):
        return True
    seen: dict[str, int] = {}
    for part in re.split(r"(?<=[.!?])\s+|\n+", answer):
        key = re.sub(r"\W+", " ", part).strip().lower()
        if len(key.split()) >= 3:
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                return True
    return False


# ---- 4e. rewrite -------------------------------------------------------------------------------
_PROB = re.compile(r"\bprobabilit(y|ies)\b", re.I)
_PROB_NEGATED = re.compile(r"(?:\b(?:not|never|no|rather\s+than|instead\s+of|than)\b|n't)\s+(?:\w+\s+){0,3}$", re.I)


def rewrite_wording(text: str) -> tuple[str, list[str]]:
    """'probability' -> 'model score' (the score of a logistic regression is not a calibrated probability)."""
    notes = []

    def sub(m: re.Match) -> str:
        # "is not a probability of fraud" is the correct statement, keep it
        if _PROB_NEGATED.search(m.string[max(0, m.start() - 40): m.start()]):
            return m.group(0)
        plural = m.group(1).lower() == "ies"
        out = "model scores" if plural else "model score"
        return out.capitalize() if m.group(0)[0].isupper() else out

    new = _PROB.sub(sub, text)
    if new != text:
        notes.append("'probability' rewritten to 'model score'")
    return new, notes


# ---- 4f. the verdict ------------------------------------------------------------------------------

@dataclass
class AnswerCheck:
    """status: ok | rewritten | guarded.  `text` is what the user sees (without the footer)."""

    text: str
    status: str = "ok"
    reasons: list = field(default_factory=list)
    raw: str = ""


def fallback_text(ctx: AppContext | None) -> str:
    """The safe replacement answer: built from the context by template, so every number is exact."""
    if ctx is None:
        return GENERIC_FALLBACK
    tail = ("A flag is a screening signal, not proof of fraud; the fraud analyst makes the decision." if ctx.flagged else
            "A score below the threshold does not guarantee the application is genuine; the fraud analyst makes the decision.")
    return " ".join(p for p in (ctx.result_sentence(), ctx.factors_sentence(), tail) if p)


def fallback_facts(lines: list) -> str:
    """The safe replacement for a dataset / model question: the fact lines the answer should have come from."""
    return ("I could not produce a reliable answer from the facts. The facts I have for this question are:\n"
            + "\n".join(f"- {x}" for x in lines))


def check_answer(answer: str, question: str = "", ctx: AppContext | None = None, grounding_text: str | None = None,
                 known_numbers: tuple = (), fact_lines: list | None = None) -> AnswerCheck:
    """Check one model answer. Grounded when an application is in focus (ctx) or grounding_text is given.

    known_numbers: fixed facts that may be quoted even without an application (the demo's review threshold).
    fact_lines: for a dataset / model question, the fact lines that are `grounding_text`'s content; they become
    the replacement answer if the guard rejects the model's.
    """
    raw = answer or ""
    if grounding_text is None and ctx is not None:
        grounding_text = ctx.input_text
    grounded = grounding_text is not None
    text, notes = rewrite_wording(_norm_text(raw))
    reasons = []
    if not text.strip():
        reasons.append("the model produced no answer")
    dec = decision_hits(text, grounded)
    if dec:
        reasons.append("decision language: " + ", ".join(sorted({d.lower() for d in dec})))
    prot = protected_hits(text, grounded)
    if prot:
        reasons.append("protected attribute named as a possible reason")
    # the review threshold is a fixed fact of the demo and the application id is in the user's own request
    extra = tuple(known_numbers) + ((f"{ctx.threshold:.4f}", ctx.app_id) if ctx is not None else ())
    nums = invented_numbers(text, grounding_text, question, extra, n_factors=len(ctx.factors) if ctx else 3)
    if nums:
        reasons.append("number not in the context: " + ", ".join(nums))
    if ctx is not None and contradicts_result(text, ctx.flagged):
        reasons.append("contradicts the Result line")
    acc = accusation_hits(text)
    if acc:
        reasons.append("definite accusation of fraud")
    if text.strip() and _same_text(text, question):
        reasons.append("the answer only repeats the question")
    elif text.strip() and is_degenerate(text):
        reasons.append("repetitive or broken output")
    if grounded and text.strip() and not refers_to_context(text, grounding_text):
        reasons.append("answer does not refer to the application")
    if reasons:
        safe = fallback_facts(fact_lines) if (ctx is None and fact_lines) else fallback_text(ctx)
        return AnswerCheck(safe, "guarded", reasons, raw)
    return AnswerCheck(text, "rewritten" if notes else "ok", notes, raw)


def _norm_text(text: str) -> str:
    """Trim and collapse runs of blank lines; keep single line breaks (numbered steps)."""
    text = (text or "").replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def render(check: AnswerCheck) -> str:
    """The text to print: the answer, with the [guarded] mark when it was replaced."""
    if check.status == "guarded":
        return f"{GUARDED_TAG} {check.text}"
    return check.text
