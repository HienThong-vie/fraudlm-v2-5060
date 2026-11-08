"""Prediction-grounded FraudLM chat for the eight ML projects.

Run:
  python chat_multisource.py --adapter checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt
  Add --debug to inspect unfiltered model output and the guard's exact matches.

Commands:
  datasets | list <dataset_id> | explain <dataset_id:case_id>
  ask <dataset_id:case_id> <question> | help | quit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat import Engine, _adapter_size  # noqa: E402
from utils.guard import FOOTER, check_answer, check_question  # noqa: E402
from utils.multisource_prompt import (  # noqa: E402
    format_prediction_context, guard_prediction_answer, load_prediction_store,
    prediction_guard_reasons,
    calculated_threshold_answer,
    source_facts,
)


HELP = """Commands
  datasets                         show available dataset ids
  facts <dataset_id>               show current registry model facts
  list <dataset_id>                show example case keys
  explain <dataset_id:case_id>     explain one stored prediction
  ask <dataset_id:case_id> <text>  ask a question grounded in that prediction
  debug on | debug off             show/hide raw model output and guard reasons
  help | quit
Score/threshold gap questions use exact arithmetic from the stored facts.
Free text asks a general fraud-screening question without a case in focus."""


class MultiSourceSession:
    """Command loop for chatting about stored classifier predictions.

    It holds no model itself: `generate_fn(question, context)` is injected (the real GPT-2 XL + LoRA
    engine in `main`, a stub in the tests), and `out` is the print function, so the session can be
    tested without a GPU.
    """

    def __init__(self, cases, generate_fn, out=print, debug=False, sources=None):
        """cases: {"dataset_id:case_id": stored prediction}; sources: registry entries keyed by dataset id."""
        self.cases = cases
        self.generate_fn = generate_fn
        self.out = out
        self.debug = debug
        self.sources = {s['id']: s for s in (sources or [])}

    def _answer(self, question, case=None):
        """Answer one question, optionally about one stored prediction (`case`).

        Order of checks: (1) the question guard, which refuses decision, protected-attribute and
        instruction-override requests before the model runs; (2) model-level questions, answered
        from the registry; (3) score/threshold gap questions, answered by exact arithmetic;
        (4) otherwise the adapter generates text, which the answer guard may replace.
        """
        # Question guard: refuse unsafe requests without calling the model.
        checked = check_question(question, None)
        if checked.kind != "ok":
            if self.debug:
                self.out(f"[debug] Question guard: {checked.kind}; model was not called.")
            self.out("FraudLM> " + checked.reply)
            return
        # Model-level questions with no case in focus read the current registry, never the adapter's memory.
        if case is None and self.sources and re.search(r'\b(?:classifier|threshold|metrics?|pr.?auc|recall|precision|model|dataset)\b', question, re.I):
            matches = [s for key, s in self.sources.items()
                       if re.search(r'(?<!\w)' + re.escape(key) + r'(?!\w)', question, re.I)]
            if len(matches) == 1:
                self.out('FraudLM> ' + source_facts(matches[0]).replace('\n', '\n         '))
            else:
                self.out('FraudLM> Use: facts <dataset_id>. Model settings are read from the current registry, not recalled from training.')
            self.out('         ' + FOOTER)
            return
        # Score/threshold gap questions: exact Decimal arithmetic on the stored numbers, no generation.
        calculated = calculated_threshold_answer(question, case) if case else None
        if calculated is not None:
            if self.debug:
                self.out("[debug] Stored-fact calculation; model was not called.")
            self.out("FraudLM> " + calculated)
            self.out("         " + FOOTER)
            self.out("  [calculated from stored facts]")
            return
        # Ground the prompt in the stored facts only (the target label is never included).
        context = format_prediction_context(case) if case else ""
        raw, info = self.generate_fn(question, context)
        if self.debug:
            self.out("[debug] Raw model answer (unfiltered; may be incorrect or unsafe):")
            self.out(raw if raw else "<empty>")
            self.out("[debug] End raw model answer")
            reasons = prediction_guard_reasons(raw, case) if case else []
            full = check_answer(raw, question=question, ctx=None, grounding_text=context if case else None)
            if full.status == 'guarded':
                reasons.extend(full.reasons)
            self.out("[debug] Answer guard: " + ("; ".join(reasons) if reasons else "no rule matched"))
        # Answer guard: unsafe or contradictory text is replaced by a fixed fact-only answer.
        raw, guarded = guard_prediction_answer(raw, case, question=question)
        if self.debug:
            self.out("[debug] Final answer (after guards):")
        self.out("FraudLM> " + raw.replace("\n", "\n         "))
        self.out("         " + FOOTER)
        suffix = " [guarded replacement]" if guarded else ""
        self.out(f"  [{info.get('prompt_tokens', '?')} prompt + {info.get('new_tokens', '?')} new tokens{suffix}]")

    def handle(self, line):
        """Run one input line; return False to end the session, True to keep going."""
        line = (line or "").strip()
        if not line:
            return True
        command, _, rest = line.partition(" ")
        low = command.lower()
        if low in {"quit", "exit", "q"}:
            return False
        if low in {"help", "?"}:
            self.out(HELP)
            return True
        if low == "debug":
            setting = rest.strip().lower()
            if setting not in {"on", "off"}:
                self.out("Usage: debug on | debug off")
            else:
                self.debug = setting == "on"
                self.out(f"Debug {'on: raw model output is unfiltered and may be unsafe' if self.debug else 'off'}.")
            return True
        if low == "datasets":
            ids = sorted({c["source_id"] for c in self.cases.values()})
            self.out("Dataset ids: " + ", ".join(ids))
            return True
        if low == 'facts':
            source = self.sources.get(rest.strip())
            self.out('FraudLM> ' + (source_facts(source) if source else 'Unknown dataset. Use: datasets').replace('\n', '\n         '))
            self.out('         ' + FOOTER)
            return True
        if low == "list":
            rows = [key for key, case in self.cases.items() if case["source_id"] == rest.strip()][:10]
            self.out("Case keys: " + (", ".join(rows) if rows else "none; type 'datasets'"))
            return True
        if low == "explain":
            # Case lookup: the key is "<dataset_id>:<case_id>", as printed by `list`.
            case = self.cases.get(rest.strip())
            if not case:
                self.out("Unknown case key. Use: list <dataset_id>")
            else:
                self._answer("Explain this classifier result without treating it as proof of fraud.", case)
            return True
        if low == "ask":
            key, _, question = rest.partition(" ")
            case = self.cases.get(key)
            if not case or not question:
                self.out("Usage: ask <dataset_id:case_id> <question>")
            else:
                self._answer(question, case)
            return True
        # Anything that is not a command is a general question with no case in focus.
        self._answer(line)
        return True


def parser():
    """Build the command-line parser; device and dtype defaults come from configs/device.json if present."""
    ap = argparse.ArgumentParser(description=__doc__)
    profile_file = ROOT/'configs/device.json'
    profile = json.loads(profile_file.read_text()) if profile_file.exists() else {}
    ap.add_argument("--adapter", required=True)
    ap.add_argument('--registry', default=str(ROOT / 'configs/prediction_sources.json'))
    ap.add_argument("--store", default=str(ROOT / "dataset_final_v2" / "prediction_store.jsonl"))
    ap.add_argument("--size", default=None, choices=["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    ap.add_argument("--device", default=profile.get('device', 'cuda'), choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default=profile.get('dtype', 'fp16'), choices=["auto", "fp16", "fp32", "bf16"])
    ap.add_argument("--cache-dir", default=str(ROOT / "checkpoint" / "pretrained"))
    ap.add_argument("--max-new-tokens", type=int, default=profile.get('max_new_tokens', 200))
    ap.add_argument("--once", action="append")
    ap.add_argument("--debug", action="store_true",
                    help="show unfiltered model output and guard trigger reasons; guards remain enabled")
    return ap


def main(argv=None):
    """Load the adapter and the prediction store, check them against the registry, then start the chat."""
    args = parser().parse_args(argv)
    if not Path(args.adapter).is_file():
        raise SystemExit(f"adapter not found: {args.adapter}")
    if not Path(args.store).is_file():
        raise SystemExit(f"prediction store not found: {args.store}; run preprocess_data/build_multisource_sft.py")
    size = args.size or _adapter_size(args.adapter) or "gpt2-xl"
    sources = json.loads(Path(args.registry).read_text(encoding='utf-8'))['sources']
    cases = load_prediction_store(args.store)
    source_map = {s['id']: s for s in sources}
    # Refuse to chat if the stored predictions no longer match the registry (model, threshold, checksum),
    # so the chat never mixes scores from one classifier with settings from another.
    for case in cases.values():
        source = source_map.get(case['source_id'])
        if source is None:
            raise SystemExit(f"Unknown source in prediction store: {case['source_id']}")
        if case['model'] != source['model'] or abs(float(case['threshold']) - source['threshold']) > 1e-12:
            raise SystemExit('Prediction store does not match registry; rebuild it before chatting.')
        if source.get('model_sha256') and case.get('model_sha256') != source['model_sha256']:
            raise SystemExit('Prediction store model checksum differs; rebuild it before chatting.')
        # Model-level metrics come from the registry at run time, not from what the adapter learned in training.
        case['model_metrics'] = {k: source[k] for k in ('pr_auc', 'recall', 'precision', 'f1', 'roc_auc', 'actual_test_fpr') if k in source}
    engine = Engine(size, args.adapter, args.device, args.dtype, args.cache_dir, None, None,
                    args.max_new_tokens)
    session = MultiSourceSession(cases, engine.generate, debug=args.debug, sources=sources)
    print(f"FraudLM multi-source | {size} | {len(session.cases):,} local prediction cases | type help")
    if args.once:
        for line in args.once:
            if not session.handle(line):
                break
        return 0
    while session.handle(input("You> ")):
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
