"""Small text helpers shared by the template modules (list joining, factor sentences).

Author: FraudLM v2 team (COS30049.2-1).
"""

from __future__ import annotations

import random

from utils.prompt import DIRECTION_BASIS, FEATURE_LABELS, format_value

REL = DIRECTION_BASIS


def cap(text: str) -> str:
    """Capitalise the first letter only (labels such as 'ZIP' keep their case)."""
    return text[:1].upper() + text[1:]


def join_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def factor_items(factors: list[dict]) -> list[dict]:
    """Add display strings to the raw factor dicts of ml_scores.csv: name, shown value, text 'name (value)'."""
    out = []
    for f in factors:
        name = FEATURE_LABELS[f["feature"]]
        val = format_value(f["feature"], f.get("value"))
        out.append({**f, "name": name, "shown": val, "text": f"{name} ({val})"})
    return out


def factors_sentence(items: list[dict], rng: random.Random, flagged: bool | None = None) -> str:
    """One sentence that states which factors raise and which lower the model score.

    Only words and values taken from the context are used, so every number stays exact. It always says that
    the direction is measured "compared with a typical application" (G3 review F1). When `flagged` is given
    and the factors all point away from the result (not flagged but every factor raises the score, or flagged
    but every factor lowers it), the sentence says that the factors are not enough to change the result.
    """
    raising = [i["text"] for i in items if i["direction"] == "raises"]
    lowering = [i["text"] for i in items if i["direction"] == "lowers"]
    verb_r = "raises" if len(raising) == 1 else "raise"
    verb_l = "lowers" if len(lowering) == 1 else "lower"
    if flagged is False and raising and not lowering:
        return rng.choice([
            f"{cap(join_list(raising))} {'pushes' if len(raising) == 1 else 'push'} the model score up {REL}, but not by enough to reach the review threshold.",
            f"The main factors are {join_list(raising)}; they push the model score up {REL}, which is not enough to reach the review threshold.",
        ])
    if flagged is True and lowering and not raising:
        return rng.choice([
            f"{cap(join_list(lowering))} {'pushes' if len(lowering) == 1 else 'push'} the model score down {REL}, yet the score still reaches the review threshold.",
            f"The main factors are {join_list(lowering)}; they push the model score down {REL}, yet the score still reaches the review threshold.",
        ])
    if raising and lowering:
        forms = [
            f"{cap(join_list(raising))} {verb_r} the model score {REL}, while {join_list(lowering)} {verb_l} it.",
            f"On the main factors, {join_list(raising)} {verb_r} the model score and {join_list(lowering)} {verb_l} it, {REL}.",
        ]
    elif raising:
        forms = [
            f"{cap(join_list(raising))} {verb_r} the model score {REL}.",
            f"The main factors are {join_list(raising)}; all of them raise the model score {REL}."
            if len(raising) > 1 else f"The main factor is {raising[0]}, which raises the model score {REL}.",
        ]
    else:
        forms = [
            f"{cap(join_list(lowering))} {verb_l} the model score {REL}.",
            f"The main factors are {join_list(lowering)}; all of them lower the model score {REL}."
            if len(lowering) > 1 else f"The main factor is {lowering[0]}, which lowers the model score {REL}.",
        ]
    return rng.choice(forms)
