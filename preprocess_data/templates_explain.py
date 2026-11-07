"""Question phrasings and answer templates for the `explain_score` task.

Each intent has
    t   : template questions (hand-written, source "template")
    l   : paraphrases seen in training (written by Claude, source "llm:claude-sonnet")
    held: paraphrases used ONLY in the test split (never in train or val)
    applies: "flagged" | "not_flagged" | "any"
Answers are built from the context numbers only (score, threshold, factor names and values), so every
number in an answer is copied from the `input`. Answers never use the words approve / reject / block,
"fraudulent" or "probability", and never name age, income or employment.

Author: FraudLM v2 team (COS30049.2-1).
"""

from __future__ import annotations

import random

from preprocess_data.text_utils import REL, cap, factors_sentence, join_list

# {factor} in a question is replaced by the plain-language name of one of the listed factors.
INTENTS: dict[str, dict] = {
    "why_flagged": {
        "applies": "flagged",
        "t": ["Why was this application flagged for review?", "Explain why this application was flagged."],
        "l": [
            "What made the model flag this application?",
            "Can you tell me why this one got flagged?",
            "Which signals pushed this application over the review line?",
            "What is behind the flag on this application?",
            "Why is this application in my review queue?",
            "Walk me through why the model flagged this case.",
            "Why did the classifier flag this application?",
            "I see a flag on this application. What caused it?",
            "What triggered the review flag here?",
            "Give me the reasons this application was flagged.",
        ],
        "held": [
            "Help me understand why this application triggered a review flag.",
            "What led the classifier to put this application forward for review?",
            "Why did this case end up on the review list?",
        ],
    },
    "why_not_flagged": {
        "applies": "not_flagged",
        "t": ["Why was this application not flagged for review?", "Explain why this application was not flagged."],
        "l": [
            "What kept this application below the review line?",
            "Why didn't the model flag this one?",
            "Why is this application not in the review queue?",
            "Why did this application stay under the threshold?",
            "Can you explain why there is no flag on this application?",
            "What explains the low result for this application?",
            "Why was this case left unflagged?",
            "Tell me why this application did not reach the review threshold.",
            "Which signals kept the score down for this application?",
            "Why does the classifier not flag this application?",
        ],
        "held": [
            "Help me see why this application did not trigger a flag.",
            "What stopped this case from being put forward for review?",
            "Why is there no review flag on this one?",
        ],
    },
    "score_meaning": {
        "applies": "any",
        "t": ["What does this model score mean?", "How should I read the model score?"],
        "l": [
            "How do I interpret this score?",
            "What does a score like this tell me?",
            "Can you explain the score for this application?",
            "What is the model score telling me here?",
            "Help me read this result.",
            "What does the number next to Model score represent?",
            "How is the review threshold related to this score?",
            "What should I take from this score?",
        ],
        "held": [
            "Could you make sense of this score for me?",
            "What am I looking at with this score and threshold?",
        ],
    },
    "above_threshold": {
        "applies": "any",
        "t": ["Is the model score at or above the review threshold?", "Did this application reach the review threshold?"],
        "l": [
            "Is the score over the threshold?",
            "Does this application meet the review threshold?",
            "Did the score cross the threshold?",
            "Is this application above or below the line?",
            "Where does this score sit compared with the threshold?",
            "Is this score high enough to be flagged?",
            "Check whether the score reaches the review threshold.",
        ],
        "held": [
            "Does the score get this application past the review threshold?",
            "Is this one above the cut-off?",
        ],
    },
    "top_factor": {
        "applies": "any",
        "t": ["What is the main factor behind this score?", "Which factor matters most for this application?"],
        "l": [
            "What is the strongest factor here?",
            "Which factor moved the score the most?",
            "What is the biggest driver of this score?",
            "Name the top factor for this application.",
            "What stands out most in this application?",
            "Which single factor has the largest effect on the score?",
            "What is the leading factor in this result?",
        ],
        "held": [
            "Which input pulled the score most?",
            "What is the number one factor in this case?",
        ],
    },
    "list_factors": {
        "applies": "any",
        "t": ["List the main factors for this application.", "What are the main factors in this result?"],
        "l": [
            "Which factors were used to explain this score?",
            "Show me all the main factors and which way they push the score.",
            "Give me the factor list for this application.",
            "What are the top factors and their values?",
            "Can you list what is driving the score up or down?",
            "Summarise the main factors for me.",
            "Which factors does the model point to?",
        ],
        "held": [
            "Break down the main factors behind this score.",
            "Tell me the factors listed for this case and what they do to the score.",
        ],
    },
    "factor_detail": {
        "applies": "any",
        "t": ["What does {factor} tell us here?", "How does {factor} affect the score?"],
        "l": [
            "Why does {factor} matter in this application?",
            "Can you explain the role of {factor}?",
            "What is the effect of {factor} on this result?",
            "Tell me more about {factor} for this case.",
            "Does {factor} push the score up or down?",
            "How much should I read into {factor}?",
            "What does the value of {factor} mean for the score?",
        ],
        "held": [
            "What is {factor} doing to the score in this case?",
            "Could you explain what {factor} means here?",
        ],
    },
    "summary": {
        "applies": "any",
        "t": ["Summarise this result in one sentence.", "Give me a one-line summary of this application."],
        "l": [
            "Sum up the result in a sentence.",
            "Can I get a quick summary of this application?",
            "In one line, what is the outcome here?",
            "TL;DR of this result?",
            "Give me the short version.",
            "Briefly, what does the model say about this application?",
        ],
        "held": [
            "Summarise what the classifier says in one short sentence.",
            "What is the bottom line for this application?",
        ],
    },
    "plain_language": {
        "applies": "any",
        "t": ["Explain this result in plain language.", "Explain this result to someone who is new to fraud screening."],
        "l": [
            "Explain this in simple words.",
            "Can you put this result in everyday language?",
            "I am new here. What does this result mean?",
            "Break this result down for a new colleague.",
            "Explain it like I have never seen a fraud score before.",
            "Describe this result without jargon.",
        ],
        "held": [
            "Say this in plain English for me.",
            "How would you explain this result to a new team member?",
        ],
    },
    "next_step": {
        "applies": "any",
        "t": ["What should I do next with this application?", "What is the next step for this application?"],
        "l": [
            "What happens next?",
            "Do I need to do anything with this application?",
            "What would you suggest as a next step?",
            "How should I follow up on this result?",
            "What is the recommended follow-up here?",
            "Where do I go from here?",
        ],
        "held": [
            "Having seen this result, what should I do now?",
            "What action does this result call for?",
        ],
    },
}


def _pick_q(intent: dict, rng: random.Random, kinds: tuple[str, ...]) -> tuple[str, str]:
    """Return (question text, source) drawn from the requested question pools."""
    pools = []
    for k in kinds:
        src = "template" if k == "t" else "llm:claude-sonnet"
        pools += [(q, src) for q in intent[k]]
    return rng.choice(pools)


def choose_question(intent_name: str, rng: random.Random, split: str) -> tuple[str, str, bool]:
    """(question template, source, held_out) for a split. Test mixes held-out and seen phrasings."""
    intent = INTENTS[intent_name]
    if split == "test" and rng.random() < 0.5:
        q, src = _pick_q(intent, rng, ("held",))
        return q, src, True
    q, src = _pick_q(intent, rng, ("t", "l", "l"))  # paraphrases are more varied, so draw them more often
    return q, src, False


# ------------------------------------------------------------------ answers
def build_answer(intent_name: str, c: dict, rng: random.Random, factor: dict | None = None) -> str:
    """c: score_s, thr_s, flagged, items (factor dicts with name/shown/text/direction)."""
    s, t, items = c["score_s"], c["thr_s"], c["items"]
    flagged = c["flagged"]
    fs = factors_sentence(items, rng, flagged)
    state = "flagged for review" if flagged else "not flagged"
    rel = "at or above" if flagged else "below"
    top = items[0]
    top_dir = "raises" if top["direction"] == "raises" else "lowers"

    if intent_name == "why_flagged":
        return rng.choice([
            f"The model score is {s}, which is at or above the review threshold of {t}, so the application is flagged for review. {fs} A flag is a screening signal, not proof of fraud; the fraud analyst makes the decision.",
            f"It was flagged because the model score of {s} reaches the review threshold of {t}. {fs} The flag asks for a manual review and is not a verdict.",
            f"Flagged for review: the model score {s} is at or above the threshold {t}. {fs}",
        ])
    if intent_name == "why_not_flagged":
        return rng.choice([
            f"The model score is {s}, which is below the review threshold of {t}, so the application is not flagged. {fs} A score below the threshold does not guarantee the application is genuine; it only means the model did not rank it for review.",
            f"It was not flagged because the model score of {s} stays below the review threshold of {t}. {fs}",
            f"Not flagged: the model score {s} is below the threshold {t}. {fs} The fraud analyst can still look at the case if there is another reason to.",
        ])
    if intent_name == "score_meaning":
        end = ("so this application is flagged for review." if flagged else "so this application is not flagged.")
        return rng.choice([
            f"The model score is {s} and the review threshold is {t}. The threshold was set for a 5% false-positive rate, and applications at or above it are flagged for review. This score is {rel} the threshold, {end} The score ranks applications for review; it does not measure how likely fraud is.",
            f"A model score of {s} is {rel} the review threshold of {t}, {end} The threshold was set for a 5% false-positive rate. The score is a screening signal and not a verdict.",
        ])
    if intent_name == "above_threshold":
        if flagged:
            return rng.choice([
                f"Yes. The model score {s} is at or above the review threshold {t}, so the application is flagged for review.",
                f"Yes, the score of {s} reaches the threshold of {t}. The application is flagged for review.",
            ])
        return rng.choice([
            f"No. The model score {s} is below the review threshold {t}, so the application is not flagged.",
            f"No, the score of {s} stays under the threshold of {t}. The application is not flagged.",
        ])
    if intent_name == "top_factor":
        others = [i["text"] for i in items[1:]]
        tail = f" The other main factors are {join_list(others)}." if others else ""
        return rng.choice([
            f"The strongest factor is {top['text']}, which {top_dir} the model score {REL}.{tail}",
            f"{cap(top['text'])} has the largest effect here and {top_dir} the model score {REL}.{tail}",
        ])
    if intent_name == "list_factors":
        parts = [f"{i['text']}, which {'raises' if i['direction'] == 'raises' else 'lowers'} the model score" for i in items]
        body = "; ".join(parts[:-1]) + "; and " + parts[-1] if len(parts) > 1 else parts[0]
        return rng.choice([
            f"The main factors are: {body}. Raises and lowers are measured {REL}.",
            f"These are the main factors for this application: {body}. The model score is {s}. Raises and lowers are measured {REL}.",
        ])
    if intent_name == "factor_detail":
        assert factor is not None
        d = "raises" if factor["direction"] == "raises" else "lowers"
        return rng.choice([
            f"{cap(factor['name'])} is {factor['shown']} for this application and it {d} the model score {REL}. It is one of the main factors, but no single factor proves anything about the application.",
            f"Here {factor['name']} is {factor['shown']}, and that {d} the model score of {s} {REL}. Read it together with the other main factors rather than on its own.",
        ])
    if intent_name == "summary":
        return rng.choice([
            f"The model score {s} is {rel} the review threshold {t}, so the application is {state}, mainly because of {top['text']}.",
            f"Result: {state} (score {s}, threshold {t}); the leading factor is {top['text']}.",
        ])
    if intent_name == "plain_language":
        if flagged:
            return (f"The model gave this application a score of {s}. Applications scoring {t} or higher are flagged for review, "
                    f"so this one is flagged. {fs} The flag means a person should take a closer look; it does not mean the application is bad.")
        return (f"The model gave this application a score of {s}. Applications scoring {t} or higher are flagged for review, "
                f"so this one is not flagged. {fs} A low score is not a guarantee, but the model does not ask for a review here.")
    if intent_name == "next_step":
        if flagged:
            return rng.choice([
                f"The application is flagged for review (model score {s}, threshold {t}), so a fraud analyst should review it manually and check the main factors listed above. The decision stays with the analyst; I only explain the model score.",
                f"Because the model score {s} is at or above the threshold {t}, the next step is a manual review by the fraud analyst, starting with the main factors above. I do not make the decision.",
            ])
        return rng.choice([
            f"The model did not flag this application (score {s}, threshold {t}), so it does not ask for a review. The fraud analyst can still check the case if other information gives a reason. The decision stays with the analyst.",
            f"No review is requested by the model here: the score {s} is below the threshold {t}. If you have other concerns, the fraud analyst can still look at it; I do not make the decision.",
        ])
    raise KeyError(intent_name)
