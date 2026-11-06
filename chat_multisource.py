"""Prediction-grounded FraudLM chat for the eight ML projects.

Run:
  python chat_multisource.py --adapter checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt

Commands:
  datasets | list <dataset_id> | explain <dataset_id:case_id>
  ask <dataset_id:case_id> <question> | help | quit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chat import Engine, _adapter_size  # noqa: E402
from utils.guard import FOOTER, check_question  # noqa: E402
from utils.multisource_prompt import (  # noqa: E402
    format_prediction_context, guard_prediction_answer, load_prediction_store,
)


HELP = """Commands
  datasets                         show available dataset ids
  list <dataset_id>                show example case keys
  explain <dataset_id:case_id>     explain one stored prediction
  ask <dataset_id:case_id> <text>  ask a question grounded in that prediction
  help | quit
Free text asks a general fraud-screening question without a case in focus."""


class MultiSourceSession:
    def __init__(self, cases, generate_fn, out=print):
        self.cases = cases
        self.generate_fn = generate_fn
        self.out = out

    def _answer(self, question, case=None):
        checked = check_question(question, None)
        if checked.kind != "ok":
            self.out("FraudLM> " + checked.reply)
            return
        context = format_prediction_context(case) if case else ""
        raw, info = self.generate_fn(question, context)
        guarded = False
        if case:
            raw, guarded = guard_prediction_answer(raw, case)
        self.out("FraudLM> " + raw.replace("\n", "\n         "))
        self.out("         " + FOOTER)
        suffix = " [guarded replacement]" if guarded else ""
        self.out(f"  [{info.get('prompt_tokens', '?')} prompt + {info.get('new_tokens', '?')} new tokens{suffix}]")

    def handle(self, line):
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
        if low == "datasets":
            ids = sorted({c["source_id"] for c in self.cases.values()})
            self.out("Dataset ids: " + ", ".join(ids))
            return True
        if low == "list":
            rows = [key for key, case in self.cases.items() if case["source_id"] == rest.strip()][:10]
            self.out("Case keys: " + (", ".join(rows) if rows else "none; type 'datasets'"))
            return True
        if low == "explain":
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
        self._answer(line)
        return True


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--store", default=str(ROOT / "dataset_multisource" / "prediction_store.jsonl"))
    ap.add_argument("--size", default=None, choices=["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl"])
    ap.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--dtype", default="fp16", choices=["auto", "fp16", "fp32", "bf16"])
    ap.add_argument("--cache-dir", default=str(ROOT / "checkpoint" / "pretrained"))
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--once", action="append")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if not Path(args.adapter).is_file():
        raise SystemExit(f"adapter not found: {args.adapter}")
    if not Path(args.store).is_file():
        raise SystemExit(f"prediction store not found: {args.store}; run preprocess_data/build_multisource_sft.py")
    size = args.size or _adapter_size(args.adapter) or "gpt2-xl"
    engine = Engine(size, args.adapter, args.device, args.dtype, args.cache_dir, None, None,
                    args.max_new_tokens)
    session = MultiSourceSession(load_prediction_store(args.store), engine.generate)
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
