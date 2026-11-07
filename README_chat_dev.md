# FraudLM v2 chat demo: developer notes (facts only)

Developer reference for the original single-source chat. For current multi-source setup, see [README_RTX5060.md](README_RTX5060.md).
All commands run from `llm_ver2/` with the project venv (`.venv/Scripts/python`) or the conda env `fraudlm2`.

## Files

| File | What it is |
|---|---|
| `chat.py` | CLI + `Engine` (loads GPT-2, merges the LoRA adapter, greedy generation) + `ChatSession` (commands, focus, guard hooks) |
| `utils/guard.py` | Deterministic question guard + answer guard + fixed texts + `AppContext` (no torch, no model) |
| `utils/chat_facts.py` | `FactIndex`: picks the fact lines for a dataset / model question and lays out the `### Input:` exactly like the training `dataset_fact` rows (imports `load_facts()` from `preprocess_data/facts.py`, read-only) |
| `tests/test_guard.py` | 127 tests: refusals, protected attributes, invented numbers, contradictions, reference answers pass |
| `tests/test_chat.py` | 33 tests: session logic with a fake generator (incl. fact routing), scripted gpt2-small session, cached == uncached text, dummy adapter, error exits |
| `tests/test_chat_facts.py` | 42 tests: block layout, line selection per question type, coverage of the reference answers |
| `preprocessed/ml_scores_months6_7_demo.csv` | Only source of applications (600 rows, test months 6-7, built by data-builder) |

## Run

```
# base GPT-2 small, no adapter (testing; answers are poor, the guard replaces most of them)
.venv/Scripts/python chat.py --size gpt2-small

# fine-tuned model (adapter from the Kaggle run); size is read from run_config.json next to the adapter
.venv/Scripts/python chat.py --size gpt2-xl --adapter checkpoint/<run_id>/lora_adapter.pt --device cuda

# non-interactive: each --once is one input line, in order (also works with piped stdin)
.venv/Scripts/python chat.py --size gpt2-small --once "explain 795470" --once "/none" --once "quit"
```

| Flag | Default | Meaning |
|---|---|---|
| `--size` | adapter's `run_config.json` size, else `gpt2-small` | `gpt2-small/medium/large/xl`; a mismatch with the adapter's `run_config.json` size exits 2 |
| `--adapter` | none | `lora_adapter.pt` (LoRA keys only); a `train_state.pt` also works (reads its `lora` entry) |
| `--device` | `cpu` | `cpu`, `cuda`, `auto` (cuda if available); `cuda` without a GPU falls back to cpu with a note |
| `--dtype` | `auto` | `auto` = fp32 on cpu, fp16 on cuda; `bf16` on cpu halves RAM |
| `--max-new-tokens` | 200 | greedy; stops at EOS or at a new `### Response/Instruction/Input` header |
| `--lora-rank` / `--lora-alpha` | `run_config.json`, else rank from the weights / alpha 32 | only needed when there is no `run_config.json` |
| `--demo-file`, `--cache-dir`, `--threads` | repo paths | |
| `--no-facts` | off | do not load the EDA / model-card fact lines (data questions then get no facts) |
| `--once LINE` | | repeatable; run these lines then exit |
| `--show-label` | off | prints the demo label as "(demo only: actual label ...; the tool does not know this)" |
| `--show-prompt`, `--show-raw` | off | developer views: exact prompt; raw model output when the guard replaced it |

Git Bash on Windows rewrites an argument that starts with `/` (turns `/none` into a path). Use `export MSYS_NO_PATHCONV=1`, or type `none`.

## Commands in the chat

`explain <app_id>` (card + model explanation), `/app <app_id>`, `/none`, `list` (`list flagged`, `list not flagged`),
`/show`, `help`, `quit` (`/quit`, `exit`). Anything else is a question. A question naming an id
("why was application 796030 flagged?") sets the focus; an id not in the file gives "not in the demo file".
`explain how phishing works` is a question, not the command.

## How a question is handled

1. Id lookup in the demo file (regexp on "application/app/id/case/#<n>" and bare 3-9 digit numbers).
2. Question guard (`guard.check_question`), no model call for: empty, small talk, prompt injection, decision request
   (approve/reject/deny/decline/block/close/freeze/... this application/account/it, "Should I approve?", "Mark this as approved",
   "Make the decision"), protected-attribute questions (age/income/employment/demographic with a reason/fraud/score cue or "the applicant").
   First-person banking questions ("block my card", "close my account") are left to the model.
3. Routing, first match wins: (a) an id in the question or `explain` -> that application; (b) an application is in focus and the question says
   "this/that application|applicant|case|score|result|flag|one" -> the application; (c) a data / model question -> fact lines
   (`FactIndex.select`; with a focus only if it carries an explicit data cue such as fraud rate, month n, PSI, recall, PR-AUC, classifier);
   (d) a focus and a question about score/flag/factors/threshold/review -> the application; (e) otherwise a general question (empty `### Input:`, as for banking_qa/general in training).
4. Prompt = `utils.prompt.format_prompt(question, input_text)`; `input_text` = `format_score_context(...)` for an application, or the fact block
   `SYSTEM_RULES` + header (`Facts (BAF Base, development months 0-5):` / `Facts about the classifier:`) + `- <fact line>` rows, same layout as in training.
5. Greedy generation with the KV cache (`utils.generate.generate(..., stop_fn=...)`): stops at EOS or when the new text opens a `### Response/Instruction/Input` header.
6. Answer guard (`guard.check_answer`), then footer "Decision support only. Model output can be wrong."; for a fact answer the guard's number check runs against the
   fact block, and a rejected answer is replaced by "I could not produce a reliable answer from the facts" + the lines.

## Fact lines (`utils/chat_facts.py`)

- Source: `preprocess_data.facts.load_facts()` (154 lines: 43 level, 26 each recall/assoc/psi, 12 numeric, 8 pattern, 6 month, 4 model, 2 corr, 1 total) reads
  `../reports/tables/eda_*.csv` and `preprocessed/_ml_repo/artifacts/model_card.json`; loaded once at start-up (about 1 s). If either is missing the chat prints a warning and
  data questions get no facts (`--no-facts` does the same on purpose).
- Selection is keyword matching, deterministic (distractor lines drawn with a generator seeded by the question): a feature is found by its plain-language name
  (token overlap with `FEATURE_LABELS`) or by a level word (`windows`, `BA`, `TELEAPP`); the family by cue words (drift/PSI, correlat, Cramer/associat, median/KS,
  catches/alone, missing pattern, month n / which month, overall, model/classifier/threshold/PR-AUC).
- What is shown: a named level or feature = that line + 2 lines about other features (training had 2-4 distractors); "which level / month ..." = all levels / all 6 months;
  "which feature drifted/associated/catches the most" = the 5 strongest features, so the true maximum is always among them; model questions = the relevant 1-2 lines + 1 other;
  a feature with no fact for the question (e.g. fraud rate of a numeric feature) = 3 unrelated level lines (the policy "not in the facts" case). At most 8 lines (about 250 tokens).
- Measured on the 1,925 dataset_fact questions of train+val+test (excluding "which of these features" questions, whose reference answer is the maximum of a random
  subset): the chosen lines contain every number of the reference answer in 99.3% (1,912); 1 question is not recognised as a fact question. Ordinary questions are not
  routed to facts (tests); about 1% of banking_qa and 1.7% of general instructions are (concept questions that mention threshold/model/feature).

## Answer guard rules (all deterministic; any hit replaces the answer with a template built from the context and prints `[guarded]`)

| Rule | Detail |
|---|---|
| decision language | approve/reject/block/deny/decline/accept/turn down/close or freeze or cancel an account/application/card ... unless the assistant declines ("I never approve, reject or block", "can't approve, reject or block") or the analyst decides ("the fraud analyst decides whether to approve"). Without a context only applies to applications/applicants or "I would/can block ..." |
| protected attribute | age, income, salary, employment, job, young/old/elderly, demographic, gender, race ... as a reason; only the ruling-out forms pass ("Age, income and employment status are not inputs to the classifier, and I never cite them as a reason"). `date of birth` is not blocked (feature label). Without a context a reason/fraud/score cue is needed in the sentence |
| invented number | with a context: every number must be in the prompt input or the question (plus 0, 1, the number of factors, the app id, 2022); rounding (0.71 for 0.7137) counts as invented. Without a context: only 3+ decimals or fractional percentages are flagged |
| contradiction | says flagged when Result is not flagged (or the reverse); conditionals ("would be flagged if ...") pass |
| accusation | "this applicant is a fraudster", "this is definitely fraud" (hedged/negated passes) |
| nonsense | template header leaking (`### Response`), a sentence repeated 3 times, answer = the question, empty, with a context: no word about score/flag/factors/data |
| rewrite (not replaced) | "probability" -> "model score" (kept when negated: "not a probability of fraud") |

Measured on the dataset (`tests/test_guard.py`): the guard leaves 100% of the reference answers unchanged:
explain_score, policy, dataset_fact (grounded, val+test and also the 10,800 non-general train rows) and banking_qa.

## Dtype / merge (PLAN §3 rule, G1 review)

Base loaded in fp32 on the CPU -> `apply_lora` + `load_lora` + `merge_lora` in fp32 -> then `.to(device, dtype)`. Never `.half()` before the merge.
Dummy-adapter test: merged model equals the unmerged LoRA model (max logit diff < 1e-3, fp32 rounding).

## Measured / expected cost

- gpt2-small, CPU, this PC, with the KV cache (model-builder, 07:28): load 3-5 s; 18 new tokens after a 174-token prompt in 0.8 s, 20 after a 252-token prompt in 0.9 s
  (before the cache: about 0.23 s per token, 24 tokens in 5.6 s).
- Not measured (estimates scaled by parameter count): gpt2-xl on this CPU about 12x slower than small and about 6.2 GB RAM fp32 (+ loader overhead); the banner prints a note.
  gpt2-xl needs `checkpoint/pretrained/gpt2-xl/model.safetensors` (6.4 GB; downloaded from the HF hub on the first run, needs internet once).
- Adapter size: rank 16 on XL = 22,118,400 trainable params, about 88 MB in fp32.

## Tests

`.venv/Scripts/python -m pytest tests/test_guard.py tests/test_chat_facts.py tests/test_chat.py -q` (about 40 s; the chat tests load gpt2-small and skip if it is not cached;
the fact tests skip if the EDA tables / model card are not on the machine).

## Open issues / decisions for the lead

- The fact lines need `../reports/tables/eda_*.csv` and `preprocessed/_ml_repo/artifacts/model_card.json` at run time. For a demo machine without them the data-builder
  should expose a snapshot (for example `dataset/fraudlm_facts.json` with family, group, key, line, vals, raw per fact) that `FactIndex` can load instead of `load_facts()`.
- Fact questions are answered from at most 8 lines chosen by keywords; a question phrased far from the training wordings can get the default (unrelated) lines and then the
  model should say it is not in the facts. Concept questions that mention "threshold" or "model" (about 1% of banking_qa) are routed to the classifier facts.
- Only gpt2-small was run here; the XL / adapter path is exercised only with a dummy rank-4 adapter. `--device cuda` and bf16 on CPU are untested (no CUDA on this PC).
- `date_of_birth_distinct_emails_4w` (age proxy) can still be a factor on the card and in the drift / association fact lines; the guard does not block it (F10 decision pending).
- The guard stays conservative on purpose: for example "This does not mean the application is approved." is replaced by the template.
