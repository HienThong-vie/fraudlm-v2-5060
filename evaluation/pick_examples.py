"""Pick good and bad example answers from a results JSON, by fixed rules (no hand picking).

Good = grounded answer that passes every metric that applies to it (faithful, no invented number, correct refusal /
honesty, names the reference's factors, policy compliant, not a copy of the context) and passes at least one,
highest ROUGE-L first.
Bad  = answer that fails at least one of those, worst ROUGE-L first; banking_qa / general rows (no context to check)
are ranked by ROUGE-L only and shown separately.
Ties are broken by row id, so the same file always gives the same picks.

Usage (from llm_ver2/):
  python evaluation/pick_examples.py evaluation/results/<file>.json --n 3

Author: FraudLM v2 team (COS30049.2-1).

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHECKS = ["explain_faithful", "score_correct", "threshold_correct", "flag_consistent", "refusal_correct",
          "not_in_context_honest", "ref_numbers_all_present"]


def failures(r: dict) -> list[str]:
    out = [k for k in CHECKS if r.get(k) is False]
    if r.get("hallucinated_number"):
        out.append(f"invented numbers {r.get('ungrounded')}")
    if r.get("policy_compliant") is False:
        out.append(f"policy wording: {r.get('policy_violation_sentences')}")
    if r.get("context_echo"):
        out.append("copies the context")
    if r.get("factor_mentions", 0) != r.get("factor_mentions_in_context", 0):
        out.append("names a factor that is not in the context")
    if r.get("factor_recall_ref") is not None and r["factor_recall_ref"] < 1:
        out.append("misses a factor that the reference names")
    if r.get("answer_empty"):
        out.append("empty answer")
    return out


def passes_something(r: dict) -> bool:
    """At least one check applied and passed (an answer with nothing to check is not called 'good')."""
    return any(r.get(k) is True for k in CHECKS) or r.get("factor_recall_ref") == 1


def pick(rows: list[dict], n: int):
    grounded = [r for r in rows if r["task"] in ("explain_score", "dataset_fact", "policy")]
    good = sorted([r for r in grounded if not failures(r) and passes_something(r)], key=lambda r: (-r["rougeL_f1"], r["id"]))[:n]
    bad = sorted([r for r in grounded if failures(r)], key=lambda r: (r["rougeL_f1"], r["id"]))[:n]
    free = sorted([r for r in rows if r["task"] in ("banking_qa", "general")], key=lambda r: (-r["rougeL_f1"], r["id"]))
    return good, bad, (free[:1] + free[-1:]) if free else []


def fmt(r: dict) -> str:
    one = lambda t: " ".join((t or "").split())  # noqa: E731  (one line for the notes)
    lines = [f"- `{r['id']}` ({r['task']}, ROUGE-L {r['rougeL_f1']:.2f})",
             f"  - Q: {one(r['instruction'])}",
             f"  - answer: {one(r['answer'])[:400] or '(empty)'}",
             f"  - reference: {one(r['reference'])[:300]}"]
    f = failures(r)
    if f:
        lines.append(f"  - fails: {'; '.join(f)}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args(argv)
    res = json.loads(Path(args.results).read_text(encoding="utf-8"))
    good, bad, free = pick(res["rows"], args.n)
    print(f"## {res['model']} on {res['split']} ({Path(args.results).name})\n\n### good\n")
    print("\n".join(fmt(r) for r in good) or "(none)")
    print("\n### bad\n")
    print("\n".join(fmt(r) for r in bad) or "(none)")
    print("\n### banking_qa / general: best and worst ROUGE-L\n")
    print("\n".join(fmt(r) for r in free) or "(none)")


if __name__ == "__main__":
    main()
