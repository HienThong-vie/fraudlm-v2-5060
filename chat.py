"""
FraudLM v2 terminal chat demo (PLAN gate G5).

Purpose: a single-turn chatbot for a fraud analyst. It loads GPT-2 (+ the LoRA adapter, merged), looks up a
demo application (test months 6-7) in preprocessed/ml_scores_months6_7_demo.csv, builds the prompt ONLY
through utils/prompt.py (the same layout as in training), generates greedily and passes question and
answer through utils/guard.py. Decision support only: it never approves, rejects or blocks.

Run (from this folder):
    python chat.py --size gpt2-small                                   # base model, for testing
    python chat.py --size gpt2-xl --adapter checkpoint/xl_r16_v1/lora_adapter.pt --device cuda
    python chat.py --size gpt2-small --once "explain 795470" --once "quit"   # non-interactive

Commands: explain <app_id> | /app <app_id> | /none | list | help | quit  (free text = a question).

Author: FraudLM v2 team (COS30049.2-1).
Generation and weight loading come from utils/ (Raschka, S. (2024) Build a Large Language Model (From
Scratch), Manning; code repo Apache-2.0).

"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.chat_facts import FactIndex  # noqa: E402
from utils.guard import FOOTER, AppContext, check_answer, check_question, render  # noqa: E402
from utils.prompt import FEATURE_LABELS, format_prompt  # noqa: E402

DEMO_FILE = ROOT / "preprocessed" / "ml_scores_months6_7_demo.csv"
CACHE_DIR = ROOT / "checkpoint" / "pretrained"
MAX_NEW_TOKENS = 200       # lead's setting; the trained answers are 40-120 tokens
CONTEXT_LENGTH = 1024      # GPT-2 window (PLAN D4)
MAX_QUESTION_CHARS = 600   # a longer question is cut so prompt + answer always fit in the window
PARAM_NAMES = {"gpt2-small": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
RAM_FP32_GB = {"gpt2-small": 0.5, "gpt2-medium": 1.4, "gpt2-large": 3.1, "gpt2-xl": 6.2}

HELP_TEXT = """FraudLM chat (decision support for a fraud analyst)
  explain <app_id>    show a demo application and have the model explain its score
  /app <app_id>       put an application in focus (a question that names an id does this too)
  /none               clear the focus
  list                a few demo application ids (flagged and not flagged)
  help                this text
  quit                leave (also /quit, exit)
Single-turn: every question is answered on its own; the model does not remember earlier turns.
With an application in focus, questions about its score, flag or factors use its context.
Questions about the data or the classifier (fraud rate by feature level, by month, drift, single-feature recall,
PR-AUC, the review threshold ...) are answered from fact lines of the EDA tables and the model card; the lines
used are printed under the answer. Other questions are answered as general banking / fraud questions.
Demo applications are synthetic (BAF dataset, test months 6-7). FraudLM never approves, rejects or blocks."""

DEFAULT_EXPLAIN = {True: "Why was this application flagged for review?",
                   False: "Why was this application not flagged for review?"}

# questions that are about the application in focus (its score, flag, factors, next steps)
_APP_QUESTION = re.compile(
    r"\b(?:this|that|the)\s+(?:application|applicant|case|score|result|flag|factor|factors|one|model|number|threshold)\b"
    r"|\b(?:this|application|applicant|score|scored|threshold|flag|flagged|factor|factors|driver|drivers|signal|signals|"
    r"classifier|result|explain|interpret|summar\w+|tl;dr|follow[- ]?up|next\s+step|review)\b", re.I)
# the model starts a new template section instead of ending its answer (untuned or badly tuned model)
_SECTION_MARKER = re.compile(r"\n*###\s*(?:Response|Instruction|Input)\b")
_THIS_APP = re.compile(r"\b(?:this|that)\s+(?:application|applicant|case|score|result|flag|one)\b|\bthe\s+(?:application|applicant)\b", re.I)
_COMMAND = re.compile(r"^(?:/(app|explain)|(app|explain))\b\s*(.*)$", re.I)
_HAS_ID = re.compile(r"\d{3,9}")
_ID_AFTER_WORD = re.compile(r"\b(?:application|app|applicant|id|case|no\.?|number)\s*(?:id|number|no\.?)?\s*[:#]?\s*#?(\d{3,9})\b|#(\d{3,9})\b", re.I)
_BARE_NUMBER = re.compile(r"(?<![\d.,])(\d{3,9})(?![\d]|[.,]\d)")


# =====================================================================================
# 1. Demo data
# =====================================================================================

def load_demo_apps(path: Path) -> tuple[dict, int]:
    """Read the demo CSV into {app_id: AppContext}. Returns (apps, number of rows skipped).

    A row is skipped when its score prints equal to the threshold at 4 decimals or when `flagged` disagrees
    with score >= threshold (the same rule the dataset builder uses), so the screen and the prompt never
    show an ambiguous result. A factor that is not one of the 26 model inputs is also refused.
    """
    apps, skipped = {}, 0
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            score, thr = float(r["score"]), float(r["threshold"])
            factors = json.loads(r["top_factors"])
            ambiguous = f"{score:.4f}" == f"{thr:.4f}" or (score >= thr) != bool(int(float(r["flagged"])))
            if ambiguous or any(f["feature"] not in FEATURE_LABELS for f in factors):
                skipped += 1
                continue
            label = int(float(r["label"])) if r.get("label") not in (None, "") else None
            apps[int(r["app_id"])] = AppContext(int(r["app_id"]), int(r["month"]), score, thr, tuple(factors), label)
    return apps, skipped


# =====================================================================================
# 2. The model (loaded lazily so the guard and the session logic can be tested without torch)
# =====================================================================================

class Engine:
    """GPT-2 + merged LoRA adapter behind one call: generate(instruction, input_text) -> (text, info)."""

    def __init__(self, size, adapter, device, dtype, cache_dir, lora_rank, lora_alpha, max_new_tokens, log=print):
        import torch

        from utils.generate import EOS_ID, generate, get_tokenizer, ids_to_text_until_eos, text_to_ids
        from utils.load_gpt2_weights import load_pretrained_gpt2

        self.torch, self._generate, self.EOS_ID = torch, generate, EOS_ID
        self._text_to_ids, self._until_eos, self._tok = text_to_ids, ids_to_text_until_eos, get_tokenizer()
        self.size, self.max_new_tokens = size, max_new_tokens
        self.adapter_info = "none (base GPT-2 only; answers will be poor)"

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            log("note: --device cuda but no CUDA device found; using the CPU.")
            device = "cpu"
        if dtype == "auto":
            dtype = "fp16" if device == "cuda" else "fp32"
        if device == "cpu" and dtype == "fp16":
            log("note: fp16 on a CPU is very slow; using fp32.")
            dtype = "fp32"
        self.device, self.dtype_name = device, dtype
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        if device == "cpu" and size in ("gpt2-large", "gpt2-xl"):
            log(f"note: {size} on the CPU needs about {RAM_FP32_GB[size]:.0f}-{RAM_FP32_GB[size] + 2:.0f} GB of RAM in fp32 "
                f"({RAM_FP32_GB[size] / 2:.0f} GB with --dtype bf16) and each answer takes minutes. Use --device cuda for a live demo.")

        # PLAN dtype rule: load in fp32, apply + merge LoRA in fp32, THEN cast (merging into fp16 rounds the update away).
        model = load_pretrained_gpt2(size, cache_dir=str(cache_dir), device="cpu", dtype=torch.float32)
        if adapter is not None:
            self.adapter_info = self._merge_adapter(model, Path(adapter), lora_rank, lora_alpha)
        model.to(device=device, dtype=self.dtype)
        model.eval()
        self.model = model

    def _merge_adapter(self, model, path, rank, alpha) -> str:
        torch = self.torch
        from training_model.lora import apply_lora, load_lora, merge_lora

        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("lora"), dict):  # a train_state.pt also works
            state = state["lora"]
        run_cfg = {}
        cfg_path = path.parent / "run_config.json"
        if cfg_path.exists():
            run_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        a_keys = [k for k in state if k.endswith(".lora.A")]
        if not a_keys:
            raise ValueError(f"{path} holds no LoRA weights (expected keys ending in '.lora.A')")
        rank = rank or run_cfg.get("lora_rank") or int(state[a_keys[0]].shape[1])
        alpha_src = "--lora-alpha" if alpha else ("run_config.json" if run_cfg.get("lora_alpha") else "assumed 32")
        alpha = alpha or run_cfg.get("lora_alpha") or 32
        if run_cfg.get("lora_targets"):
            targets = tuple(t.strip() for t in str(run_cfg["lora_targets"]).split(",") if t.strip())
        else:
            targets = tuple(t for t, tag in (("attn", ".att."), ("ff", ".ff.")) if any(tag in k for k in state))
        apply_lora(model, rank=int(rank), alpha=float(alpha), targets=targets)
        load_lora(model, state)
        merge_lora(model)  # fp32 merge, no extra compute at chat time
        return f"{path.name} (rank {rank}, alpha {alpha} [{alpha_src}], targets {'+'.join(targets)}, merged in fp32)"

    @property
    def banner_model(self) -> str:
        return f"{self.size} ({PARAM_NAMES.get(self.size, '?')} params)"

    def generate(self, instruction: str, input_text: str = "") -> tuple[str, dict]:
        """Greedy answer to one prompt. Returns (text, info) with info = prompt_tokens, new_tokens, eos, seconds, prompt."""
        torch = self.torch
        prompt = format_prompt(instruction, input_text)  # the ONLY place the template is applied
        ids = self._text_to_ids(prompt)
        n_prompt = int(ids.shape[1])
        room = CONTEXT_LENGTH - n_prompt
        if room < 16:
            raise ValueError(f"the prompt is {n_prompt} tokens, too long for the {CONTEXT_LENGTH}-token window")
        max_new = min(self.max_new_tokens, room)
        t0 = time.perf_counter()
        # A fine-tuned model ends with EOS; an untuned one starts a new "### Response:" section instead, and there
        # is no point in running on to the token limit. generate() stops at EOS by itself; stop_fn (called after
        # every new token, with the cache making each step cheap) stops at a new section header.
        def stop_at_header(so_far) -> bool:
            return bool(_SECTION_MARKER.search(self._until_eos(so_far[0, n_prompt:].tolist())))

        with torch.inference_mode():
            out = self._generate(self.model, ids.to(self.device), max_new, context_size=CONTEXT_LENGTH,
                                 temperature=0.0, eos_id=self.EOS_ID, stop_fn=stop_at_header)
        new = out[0, n_prompt:].tolist()
        decoded = self._until_eos(new)
        if _SECTION_MARKER.search(decoded):
            stop = "marker"
        elif len(new) < max_new or self.EOS_ID in new:  # generate() drops the EOS it stops on
            stop = "eos"
        else:
            stop = "limit"
        text = _SECTION_MARKER.split(decoded)[0].strip()
        return text, {"prompt_tokens": n_prompt, "new_tokens": len(new), "stop": stop, "max_new": max_new,
                      "seconds": time.perf_counter() - t0, "prompt": prompt}


# =====================================================================================
# 3. The chat session (pure logic: it only needs a generate function, so it is unit-testable)
# =====================================================================================

class ChatSession:
    def __init__(self, apps: dict, generate_fn, out=print, show_label=False, show_prompt=False, show_raw=False,
                 known_numbers: tuple = (), facts=None):
        self.apps, self.generate_fn, self.out = apps, generate_fn, out
        self.facts = facts  # utils.chat_facts.FactIndex or None (dataset / model questions then get no facts)
        self.show_label, self.show_prompt, self.show_raw = show_label, show_prompt, show_raw
        self.known_numbers = known_numbers
        self.focus: AppContext | None = None

    # ---- output helpers ----
    def say(self, text: str, footer: bool = False) -> None:
        self.out("FraudLM> " + text.replace("\n", "\n         "))
        if footer:
            self.out("         " + FOOTER)

    def card(self, ctx: AppContext) -> str:
        lines = [f"Application {ctx.app_id} (demo, test month {ctx.month})", *("  " + l for l in ctx.card_text.splitlines())]
        if self.show_label:
            lines.append(f"  (demo only: actual label = {'fraud' if ctx.label == 1 else 'not fraud'}; the tool does not know this)")
        return "\n".join(lines)

    # ---- id handling ----
    def find_app(self, text: str):
        """(AppContext or None, unknown id or None) for the first application id the text mentions."""
        explicit = [int(next(g for g in m.groups() if g)) for m in _ID_AFTER_WORD.finditer(text)]
        bare = [int(m.group(1)) for m in _BARE_NUMBER.finditer(text)]
        for n in explicit + bare:
            if n in self.apps:
                return self.apps[n], None
        return None, (explicit[0] if explicit else None)

    def set_focus(self, ctx: AppContext) -> None:
        self.focus = ctx
        self.out(self.card(ctx))

    # ---- commands ----
    def handle(self, line: str) -> bool:
        """Process one input line. Returns False when the user quits."""
        line = (line or "").strip()
        if not line:
            return True
        low = line.lower()
        if low in ("quit", "/quit", "exit", "/exit", "q", ":q"):
            self.say("Bye.")
            return False
        if low in ("help", "/help", "?", "/?"):
            self.out(HELP_TEXT)
            return True
        if low in ("/none", "none", "/clear"):
            self.focus = None
            self.say("Focus cleared. Questions are now general banking / fraud questions.")
            return True
        if low in ("list", "/list") or low.startswith(("list ", "/list ")):
            self.cmd_list(low.split(None, 1)[1] if " " in low else "")
            return True
        if low in ("/show", "show"):
            self.out(self.card(self.focus) if self.focus else "No application in focus. Type 'list' or 'explain <app_id>'.")
            return True
        m = _COMMAND.match(line)
        if m:
            verb = (m.group(1) or m.group(2)).lower()
            arg = m.group(3).strip()
            # "explain how phishing works" is a question, not the explain command
            if not arg or _HAS_ID.search(arg) or line.startswith("/"):
                self.cmd_app(verb, arg)
                return True
        self.ask(line)
        return True

    def cmd_list(self, arg: str) -> None:
        want = None
        if arg.startswith(("not", "unflag", "un-")):
            want = False
        elif arg.startswith("flag"):
            want = True
        if not self.apps:
            self.say("No demo applications are loaded (check preprocessed/ml_scores_months6_7_demo.csv).")
            return
        rows = list(self.apps.values())
        parts = []
        for flagged, name in ((True, "flagged for review"), (False, "not flagged")):
            if want is not None and want != flagged:
                continue
            sel = [c for c in rows if c.flagged == flagged][:5]
            parts.append(f"{name}:\n" + "\n".join(f"  {c.app_id}   month {c.month}   model score {c.score_s}" for c in sel))
        self.say("Some demo applications (type 'explain <app_id>'):\n" + "\n".join(parts))

    def cmd_app(self, verb: str, arg: str) -> None:
        if not arg:
            if verb == "explain" and self.focus:
                self.explain(self.focus)
            elif verb == "app" and self.focus:
                self.out(self.card(self.focus))
            else:
                self.say(f"Usage: {verb} <app_id>. Type 'list' to see some demo ids.")
            return
        ctx, unknown = self.find_app(arg)
        if ctx is None:
            n = unknown if unknown is not None else arg
            self.say(f"Application {n} is not in the demo file ({len(self.apps)} test-month applications). Type 'list' to see some ids.")
            return
        self.set_focus(ctx)
        if verb == "explain":
            self.explain(ctx)

    def explain(self, ctx: AppContext) -> None:
        self.ask(DEFAULT_EXPLAIN[ctx.flagged], grounded=True, quiet_question=True)

    # ---- one question ----
    def ask(self, question: str, grounded: bool | None = None, quiet_question: bool = False) -> None:
        question = question.strip()[:MAX_QUESTION_CHARS]
        mentioned, unknown = self.find_app(question)
        if mentioned is not None:
            if self.focus is None or mentioned.app_id != self.focus.app_id:
                self.set_focus(mentioned)
            grounded = True
        elif unknown is not None:
            self.say(f"Application {unknown} is not in the demo file ({len(self.apps)} test-month applications). Type 'list' to see some ids.")
            return
        # question guard: decisions, protected attributes, injection, small talk never reach the model
        qc = check_question(question, self.focus)
        if qc.kind != "ok":
            self.say(qc.reply, footer=qc.kind in ("decision_request", "protected_reason", "injection"))
            return
        # routing: this application > dataset / model facts > the application in focus > general question
        sel = None
        if grounded is None:
            if self.focus is not None and _THIS_APP.search(question):
                grounded = True
            else:
                sel = self.facts.select(question, strict=self.focus is not None) if self.facts is not None else None
                grounded = sel is None and self.focus is not None and bool(_APP_QUESTION.search(question))
        ctx = self.focus if grounded else None
        if sel is not None:
            input_text, mode = sel.input_text, f"answering from {len(sel.lines)} dataset/model fact lines: {sel.note}"
        else:
            input_text = ctx.input_text if ctx else ""
            mode = f"answering from application {ctx.app_id}" if ctx else "general question, no application context"
        self.out(f"  [{mode}; single-turn; generating...]")
        try:
            raw, info = self.generate_fn(question, input_text)
        except KeyboardInterrupt:
            self.out("  (answer abandoned)")
            return
        except Exception as e:  # never crash the demo on a model error
            self.say(f"Sorry, I could not produce an answer ({type(e).__name__}: {e}). The chat is still running.")
            return
        chk = check_answer(raw, question, ctx, grounding_text=sel.input_text if sel else None,
                           known_numbers=self.known_numbers, fact_lines=sel.lines if sel else None)
        self.say(render(chk), footer=True)
        if sel is not None and chk.status != "guarded":  # a replaced answer already lists the lines
            self.out("  Facts used:\n" + "\n".join(f"  - {x}" for x in sel.lines))
        meta = f"{info.get('prompt_tokens', '?')} prompt + {info.get('new_tokens', '?')} new tokens in {info.get('seconds', 0):.1f}s"
        if info.get("stop") == "limit":
            meta += f"; cut at the {info.get('max_new', MAX_NEW_TOKENS)}-token limit"
        elif info.get("stop") == "marker":
            meta += "; stopped where the model started a new '### ...' section"
        self.out(f"  [{meta}]")
        if chk.status == "guarded":
            self.out("  [guard: the model's answer was replaced by a template built from the context: " + "; ".join(chk.reasons) + "]")
        elif chk.status == "rewritten":
            self.out("  [guard: wording adjusted: " + "; ".join(chk.reasons) + "]")
        if self.show_raw and chk.status == "guarded":
            self.out("  [raw model output, developer view only] " + (raw or "<empty>").replace("\n", " | "))
        if self.show_prompt:
            self.out("  [prompt sent to the model]\n" + "\n".join("    " + l for l in info.get("prompt", "").splitlines()))


# =====================================================================================
# 4. Command line
# =====================================================================================

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="FraudLM terminal chat (decision support for a fraud analyst).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--size", choices=sorted(PARAM_NAMES), default=None,
                    help="GPT-2 size; default = the size in the adapter's run_config.json, else gpt2-small")
    ap.add_argument("--adapter", default=None, help="checkpoint/<run_id>/lora_adapter.pt (omit to chat with the base model)")
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu", help="default cpu; auto picks cuda when available")
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto", help="auto = fp32 on cpu, fp16 on cuda")
    ap.add_argument("--demo-file", default=str(DEMO_FILE))
    ap.add_argument("--cache-dir", default=str(CACHE_DIR), help="where the OpenAI GPT-2 weights are cached")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--lora-rank", type=int, default=None, help="default: run_config.json, else read from the adapter")
    ap.add_argument("--lora-alpha", type=float, default=None, help="default: run_config.json, else 32")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--once", action="append", default=None, metavar="LINE",
                    help="run this input line and exit (repeat the flag for several lines, in order)")
    ap.add_argument("--no-facts", action="store_true", help="do not load the EDA / model-card fact lines (data questions get no facts)")
    ap.add_argument("--show-label", action="store_true", help="demo only: also print the actual label of an application")
    ap.add_argument("--show-prompt", action="store_true", help="developer: print the exact prompt sent to the model")
    ap.add_argument("--show-raw", action="store_true", help="developer: print the raw model output when the guard replaced it")
    return ap


def _utf8_streams() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _adapter_size(adapter: str | None) -> str | None:
    if not adapter:
        return None
    cfg = Path(adapter).parent / "run_config.json"
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("size")
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _utf8_streams()
    if args.adapter and not Path(args.adapter).is_file():
        print(f"ERROR: adapter file not found: {args.adapter}", file=sys.stderr)
        return 2
    trained_for = _adapter_size(args.adapter)
    if args.size and trained_for and args.size != trained_for:
        print(f"ERROR: the adapter was trained for {trained_for} but --size is {args.size}.", file=sys.stderr)
        return 2
    size = args.size or trained_for or "gpt2-small"

    apps, skipped = {}, 0
    try:
        apps, skipped = load_demo_apps(Path(args.demo_file))
    except (OSError, ValueError, KeyError) as e:
        print(f"warning: could not read the demo file {args.demo_file} ({e}); application lookups are disabled.")

    facts = None
    if not args.no_facts:
        try:
            facts = FactIndex.load()  # EDA tables + model card, read once (about 1 s)
        except Exception as e:  # missing ../reports/tables or preprocessed/_ml_repo: the chat still works
            print(f"warning: dataset facts unavailable ({type(e).__name__}: {e}); data and model questions get no facts.")

    if args.threads:
        import torch
        torch.set_num_threads(args.threads)
    print(f"Loading {size}" + (f" + adapter {Path(args.adapter).name}" if args.adapter else " (no adapter)") + " ...", flush=True)
    t0 = time.perf_counter()
    try:
        engine = Engine(size, args.adapter, args.device, args.dtype, args.cache_dir, args.lora_rank, args.lora_alpha,
                        args.max_new_tokens)
    except KeyboardInterrupt:
        print("interrupted while loading", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"ERROR: could not load the model ({type(e).__name__}: {e}).", file=sys.stderr)
        return 2
    thresholds = {c.threshold for c in apps.values()}
    known = (f"{next(iter(thresholds)):.4f}",) if len(thresholds) == 1 else ()
    print(f"FraudLM chat | base: {engine.banner_model} | adapter: {engine.adapter_info} | device: {engine.device} "
          f"{engine.dtype_name} | {len(apps)} demo applications | {len(facts.facts) if facts else 0} fact lines"
          + (f" ({skipped} skipped: ambiguous result)" if skipped else "")
          + f" | loaded in {time.perf_counter() - t0:.0f}s")
    print("Single-turn, greedy decoding, max "
          f"{args.max_new_tokens} new tokens. Type 'help' for commands, 'list' for demo ids, 'quit' to leave.")
    session = ChatSession(apps, engine.generate, show_label=args.show_label, show_prompt=args.show_prompt,
                          show_raw=args.show_raw, known_numbers=known, facts=facts)

    def run_line(line: str) -> bool:
        try:
            return session.handle(line)
        except KeyboardInterrupt:
            print("\n(interrupted)")
            return True
        except Exception as e:  # last line of defence: the demo must not crash
            print(f"Internal error ({type(e).__name__}: {e}). The chat is still running.")
            return True

    if args.once:
        for line in args.once:
            print(f"you> {line}")
            if not run_line(line):
                break
        return 0
    interactive = sys.stdin.isatty()
    while True:
        try:
            line = input("you> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            break
        if not interactive:
            print(line)  # piped input is not echoed by the terminal
        if not run_line(line):
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
