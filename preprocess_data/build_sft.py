"""Build the FraudLM v2 instruction fine-tuning dataset (Alpaca JSON): gate G3.

Output (dataset/):
    fraudlm_sft_train.json, fraudlm_sft_val.json, fraudlm_sft_test.json
        JSON lists of {"id", "instruction", "input", "output", "task", "source"}
    fraudlm_demo_months67.json     explain_score examples on test months 6-7 applications: demo / evaluation ONLY,
                                   never used for training (PLAN.md Section 3)
    build_manifest.json, SOURCES.md, DATA_CARD_notes.md   facts about this build (counts, tokens, sources)

Five tasks (PLAN.md Section 4)
    explain_score : context from preprocessed/ml_scores.csv (development months 0-5), answers from templates
    dataset_fact  : context lines from the EDA tables and the model card, answers from templates
    policy        : refusals (decisions, protected reasons, verdicts), "not in the context", scope
    banking_qa    : Bitext retail-banking Q&A (CDLA-Sharing-1.0) + concept Q&A written by Claude (llm:claude-sonnet)
    general       : Alpaca-cleaned (CC-BY-4.0)

Splits: seed 42; applications and fact rows are split into disjoint pools; no (instruction, input) pair occurs
in two splits; the test split mixes seen and held-out question phrasings (held-out ids end in "-heldout").

Size is a CLI parameter:  --n-train 12000 --n-val 500 --n-test 500   (mix via --mix)
Run from llm_ver2/:  .venv/Scripts/python -m preprocess_data.build_sft

Author: FraudLM v2 team (COS30049.2-1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict

import pandas as pd

from preprocess_data import common as C
from preprocess_data import facts as F
from preprocess_data import public_sets as P
from preprocess_data import templates_explain as TE
from preprocess_data import templates_policy as TP
from preprocess_data.concepts import CONCEPTS
from preprocess_data.text_utils import factor_items
from utils.prompt import SYSTEM_RULES, format_example, format_score_context

TASKS = ("explain_score", "dataset_fact", "policy", "banking_qa", "general")
DEFAULT_MIX = {"explain_score": 0.38, "dataset_fact": 0.17, "policy": 0.13, "banking_qa": 0.22, "general": 0.10}
EXPLAIN_WEIGHTS = {"why_flagged": 3, "why_not_flagged": 3, "score_meaning": 1.5, "above_threshold": 1, "top_factor": 1.5,
                   "list_factors": 1.5, "factor_detail": 2, "summary": 1.5, "plain_language": 1.2, "next_step": 1.2}
POLICY_WEIGHTS = {"decision_request": 3, "protected_reason": 3, "verdict_request": 2, "missing_fact": 2,
                  "dataset_missing": 1, "injection": 1, "evade_request": 0.4, "scope": 0.5}


def split_of_hash(key: str, p_val: float = 0.08, p_test: float = 0.08) -> str:
    h = int(hashlib.sha256(f"apps-{C.SEED}:{key}".encode()).hexdigest()[:8], 16) / 2**32
    return "test" if h < p_test else "val" if h < p_test + p_val else "train"


# ------------------------------------------------------------------ application contexts
def context_block(c: dict) -> str:
    """The context without the rules line (what makes two applications look identical to the model)."""
    return format_score_context(c["score"], c["thr"], c["factors"], rules=False)


class Apps:
    """Scored applications (ml_scores csv) split into disjoint train / val / test pools."""

    def __init__(self, path, months_allowed: tuple[int, int]) -> None:
        df = pd.read_csv(path)
        lo, hi = months_allowed
        assert df["month"].between(lo, hi).all(), f"{path} holds months outside {months_allowed}"
        self.rows = []
        for r in df.itertuples():
            score, thr = float(r.score), float(r.threshold)
            s_s, t_s = f"{score:.4f}", f"{thr:.4f}"
            if s_s == t_s or (score >= thr) != bool(r.flagged):
                continue  # displayed numbers would be ambiguous: skip
            factors = json.loads(r.top_factors)
            items = factor_items(factors)
            self.rows.append({"app_id": int(r.app_id), "score": score, "thr": thr, "flagged": bool(r.flagged),
                              "score_s": s_s, "thr_s": t_s, "items": items, "factors": factors})
        self.pool = {"train": {True: [], False: []}, "val": {True: [], False: []}, "test": {True: [], False: []}}
        for r in self.rows:
            self.pool[split_of_hash(str(r["app_id"]))][r["flagged"]].append(r)
        # G3 review F5: a val / test application whose printed context block is byte-identical to one in
        # the train pool (same 4-decimal score and same factors) is dropped, so no context repeats across splits
        def blocks(split):
            return {context_block(r) for fl in (True, False) for r in self.pool[split][fl]}

        self.removed = {"val": 0, "test": 0}
        seen_blocks = blocks("train")
        for split in ("val", "test"):
            for fl in (True, False):
                kept = [r for r in self.pool[split][fl] if context_block(r) not in seen_blocks]
                self.removed[split] += len(self.pool[split][fl]) - len(kept)
                self.pool[split][fl] = kept
            seen_blocks |= blocks(split)

    def draw(self, rng: random.Random, split: str, flagged: bool | None = None) -> dict:
        if flagged is None:
            flagged = rng.random() < 0.5
        return rng.choice(self.pool[split][flagged])


def context_input(c: dict, extra: str = "") -> str:
    inp = format_score_context(c["score"], c["thr"], c["factors"])
    # format_score_context prints 4-decimal score / threshold exactly like score_s / thr_s
    assert f"Model score: {c['score_s']}" in inp and f"Review threshold: {c['thr_s']}" in inp
    return inp + ("\n" + extra if extra else "")


# ------------------------------------------------------------------ example builders
def make_explain(apps: Apps, rng: random.Random, split: str):
    c = apps.draw(rng, split)
    state = "flagged" if c["flagged"] else "not_flagged"
    names = [k for k, v in TE.INTENTS.items() if v["applies"] in (state, "any")]
    intent = rng.choices(names, weights=[EXPLAIN_WEIGHTS[n] for n in names])[0]
    q, src, held = TE.choose_question(intent, rng, split)
    factor = None
    if "{factor}" in q:
        factor = rng.choice(c["items"])
        q = q.replace("{factor}", factor["name"])
    ans = TE.build_answer(intent, c, rng, factor)
    return dict(instruction=q, input=context_input(c), output=ans, task="explain_score", source=src, held=held, intent=intent)


def make_policy(apps: Apps, bank: F.FactBank, rng: random.Random, split: str):
    names = list(POLICY_WEIGHTS)
    cat = rng.choices(names, weights=[POLICY_WEIGHTS[n] for n in names])[0]
    if cat == "dataset_missing":
        q, inp, ans, src, held = bank.missing_example(rng, split)
        return dict(instruction=q, input=inp, output=ans, task="policy", source=src, held=held, intent=cat)
    if cat == "injection":
        c = apps.draw(rng, split)
        intent = "why_flagged" if c["flagged"] else "why_not_flagged"
        q, src, held = TE.choose_question(intent, rng, split)
        inp = context_input(c, rng.choice(TP.INJECTIONS))
        return dict(instruction=q, input=inp, output=TP.build_injection_answer(c, rng), task="policy",
                    source=src, held=held, intent=cat)
    needs = TP.CATEGORIES[cat]["needs_ctx"]
    with_ctx = needs is True or (needs is None and rng.random() < 0.75)
    q, src, held = TP.choose_question(cat, rng, split)
    c = apps.draw(rng, split) if with_ctx else None
    if "{missing}" in q:
        q = q.replace("{missing}", TP.pick_missing_label({f["feature"] for f in c["factors"]}, rng))
    inp = context_input(c) if c else SYSTEM_RULES
    ans = TP.build_answer(cat, q, c, rng)
    return dict(instruction=q, input=inp, output=ans, task="policy", source=src, held=held, intent=cat)


def make_fact(bank: F.FactBank, rng: random.Random, split: str):
    pool = "eval" if split in ("val", "test") else "train"
    q, inp, ans, src, held = bank.make(rng, pool, split)
    return dict(instruction=q, input=inp, output=ans, task="dataset_fact", source=src, held=held, intent="fact")


# ------------------------------------------------------------------ banking_qa and general
def concept_rows() -> dict[str, list[dict]]:
    """Concept Q&A by split. Template + 2 paraphrases -> train; third paraphrase -> val for 20% of entries
    (hash), train for the rest; the held-out question -> test only."""
    out = {"train": [], "val": [], "test": []}
    src = "llm:claude-sonnet"
    for tq, paras, held_q, ans in CONCEPTS:
        key = tq
        third_to_val = int(hashlib.sha256(f"concept-{key}".encode()).hexdigest()[:4], 16) % 5 == 0
        for q, s in [(tq, "template")] + [(p, src) for p in paras[:2]]:
            out["train"].append(dict(instruction=q, input="", output=ans, task="banking_qa", source=s if s != "template" else src, held=False, intent="concept"))
        row3 = dict(instruction=paras[2], input="", output=ans, task="banking_qa", source=src, held=False, intent="concept")
        out["val" if third_to_val else "train"].append(row3)
        out["test"].append(dict(instruction=held_q, input="", output=ans, task="banking_qa", source=src, held=True, intent="concept"))
    return out


def public_rows(task: str, n_total: int) -> list[dict]:
    rows = P.load_bitext(n_total) if task == "banking_qa" else P.load_alpaca(n_total)
    for r in rows:
        r.update(held=False, intent=r.pop("tag"))
    return rows


# ------------------------------------------------------------------ assembly
def counts_for(n: int, mix: dict[str, float]) -> dict[str, int]:
    raw = {t: n * mix[t] / sum(mix.values()) for t in TASKS}
    out = {t: int(raw[t]) for t in TASKS}
    for t in sorted(TASKS, key=lambda t: raw[t] - out[t], reverse=True)[: n - sum(out.values())]:
        out[t] += 1
    return out


def finish(ex: dict) -> dict | None:
    """Token check; returns the clean 6-key row or None when the example is too long."""
    prompt, resp = format_example(ex)
    ntok = C.token_count(prompt + resp)
    if ntok > C.TARGET_TOKENS:
        return None
    return {"instruction": ex["instruction"].strip(), "input": ex["input"].strip(), "output": ex["output"].strip(),
            "task": ex["task"], "source": ex["source"], "_held": ex.get("held", False), "_ntok": ntok}


def build(args) -> dict:
    mix = dict(DEFAULT_MIX)
    if args.mix:
        mix = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in args.mix.split(",")}
    sizes = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    per_task = {s: counts_for(n, mix) for s, n in sizes.items()}

    apps = Apps(C.ML_SCORES, (0, C.DEV_MAX_MONTH))
    bank = F.FactBank()
    concepts = concept_rows()
    # Bitext rows needed = banking_qa quota minus the concept Q&A rows; small buffer for duplicate drops
    n_bitext = sum(max(0, per_task[s]["banking_qa"] - len(concepts[s])) for s in sizes) + 30
    pub = {"banking_qa": public_rows("banking_qa", n_bitext),
           "general": public_rows("general", sum(per_task[s]["general"] for s in sizes) + 200)}
    rng_pub = random.Random(C.SEED)
    for t in pub:
        rng_pub.shuffle(pub[t])

    seen: set[str] = set()
    data: dict[str, list[dict]] = {}
    shortfall: dict[str, dict] = defaultdict(dict)
    dropped_long = Counter()
    pub_cursor = {"banking_qa": 0, "general": 0}

    def accept(split: str, ex: dict, bucket: list) -> bool:
        key = C.example_key(ex["instruction"], ex["input"])
        if key in seen:
            return False
        row = finish(ex)
        if row is None:
            dropped_long[ex["task"]] += 1
            return False
        seen.add(key)
        bucket.append(row)
        return True

    for split_idx, split in enumerate(("test", "val", "train")):  # test and val claim their keys first
        rng = random.Random(C.SEED * 100 + split_idx)
        rows: list[dict] = []
        for task in TASKS:
            target = per_task[split][task]
            got = 0
            bucket: list[dict] = []
            if task in ("banking_qa", "general"):
                if task == "banking_qa":
                    cr = concepts[split][:]
                    rng.shuffle(cr)
                    for ex in cr[: target]:
                        got += accept(split, ex, bucket)
                p = pub[task]
                while got < target and pub_cursor[task] < len(p):
                    ex = p[pub_cursor[task]]
                    pub_cursor[task] += 1
                    got += accept(split, ex, bucket)
            else:
                attempts = 0
                while got < target and attempts < target * 40:
                    attempts += 1
                    if task == "explain_score":
                        ex = make_explain(apps, rng, split)
                    elif task == "dataset_fact":
                        ex = make_fact(bank, rng, split)
                    else:
                        ex = make_policy(apps, bank, rng, split)
                    if split != "test" and ex["held"]:
                        continue
                    got += accept(split, ex, bucket)
            if got < target:
                shortfall[split][task] = {"wanted": target, "got": got}
            rows += bucket
        random.Random(C.SEED + split_idx).shuffle(rows)
        data[split] = rows

    # ids and file output
    out = {}
    for split, rows in data.items():
        counter = Counter()
        final = []
        for r in rows:
            counter[r["task"]] += 1
            rid = f"{r['task']}-{split}-{counter[r['task']]:05d}" + ("-heldout" if r["_held"] else "")
            final.append({"id": rid, "instruction": r["instruction"], "input": r["input"], "output": r["output"],
                          "task": r["task"], "source": r["source"], "_ntok": r["_ntok"]})
        out[split] = final
    return {"data": out, "shortfall": dict(shortfall), "dropped_long": dict(dropped_long), "per_task_target": per_task,
            "apps_removed_for_context_overlap": apps.removed, "bitext_filter": P.LAST_STATS.get("bitext", {})}


def build_demo(n: int) -> list[dict]:
    """explain_score examples on test months 6-7 applications. Demo / evaluation only."""
    apps = Apps(C.ML_SCORES_DEMO, (C.DEV_MAX_MONTH + 1, 7))
    # a months 6-7 context that prints exactly like a development-month context would be ambiguous in the
    # tests (same score to 4 decimals and same factors), so such coincidences are skipped
    dev_blocks = {context_input(c).split(chr(10), 1)[1] for c in Apps(C.ML_SCORES, (0, C.DEV_MAX_MONTH)).rows}
    rng = random.Random(C.SEED + 7)
    allrows = apps.rows[:]
    rng.shuffle(allrows)
    out, seen = [], set()
    for i, c in enumerate(allrows):
        if len(out) >= n:
            break
        if context_input(c).split(chr(10), 1)[1] in dev_blocks:
            continue
        state = "flagged" if c["flagged"] else "not_flagged"
        names = [k for k, v in TE.INTENTS.items() if v["applies"] in (state, "any")]
        intent = names[i % len(names)]
        q, src, _ = TE.choose_question(intent, rng, "val")
        factor = None
        if "{factor}" in q:
            factor = rng.choice(c["items"])
            q = q.replace("{factor}", factor["name"])
        ex = dict(instruction=q, input=context_input(c), output=TE.build_answer(intent, c, rng, factor), task="explain_score", source=src)
        row = finish(ex)
        key = C.example_key(ex["instruction"], ex["input"])
        if row is None or key in seen:
            continue
        seen.add(key)
        out.append({"id": f"demo67-{len(out) + 1:05d}-app{c['app_id']}", "instruction": row["instruction"], "input": row["input"],
                    "output": row["output"], "task": "explain_score", "source": src, "_ntok": row["_ntok"]})
    return out


def token_stats(rows: list[dict]) -> dict:
    t = sorted(r["_ntok"] for r in rows)
    if not t:
        return {}
    return {"n": len(t), "min": t[0], "mean": round(statistics.mean(t), 1), "median": statistics.median(t),
            "p95": t[int(0.95 * (len(t) - 1))], "max": t[-1]}


def write_json(path, rows):
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-demo", type=int, default=300, help="explain_score examples on test months 6-7 (demo/eval only)")
    ap.add_argument("--mix", default="", help="e.g. explain_score=0.4,dataset_fact=0.15,policy=0.13,banking_qa=0.22,general=0.10")
    args = ap.parse_args()

    res = build(args)
    demo = build_demo(args.n_demo)
    C.DATASET.mkdir(exist_ok=True)

    manifest = {"args": vars(args), "seed": C.SEED, "tokenizer": "tiktoken gpt2", "max_tokens_kept": C.TARGET_TOKENS,
                "ml_scores_meta": json.loads(C.ML_SCORES_META.read_text(encoding="utf-8")), "splits": {},
                "shortfall": res["shortfall"], "dropped_for_length": res["dropped_long"], "files": {}}
    for split, rows in res["data"].items():
        path = C.DATASET / f"fraudlm_sft_{split}.json"
        write_json(path, rows)
        manifest["splits"][split] = {
            "n": len(rows),
            "per_task": dict(Counter(r["task"] for r in rows)),
            "per_source": dict(Counter(r["source"] for r in rows)),
            "per_task_source": {f"{t}|{src}": n for (t, src), n in Counter((r["task"], r["source"]) for r in rows).items()},
            "heldout_phrasing": sum(r["id"].endswith("-heldout") for r in rows),
            "tokens": token_stats(rows),
            "tokens_by_task": {t: token_stats([r for r in rows if r["task"] == t]) for t in TASKS},
        }
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": C.sha256_file(path)}
    dpath = C.DATASET / "fraudlm_demo_months67.json"
    write_json(dpath, demo)
    manifest["demo_months67"] = {"n": len(demo), "tokens": token_stats(demo), "note": "test months 6-7 applications; demo / evaluation only, never training"}
    manifest["files"][dpath.name] = {"bytes": dpath.stat().st_size, "sha256": C.sha256_file(dpath)}
    (C.DATASET / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest.update(extra_stats(res))
    (C.DATASET / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_notes(manifest)

    print(json.dumps({s: {"n": v["n"], "per_task": v["per_task"], "tokens": v["tokens"], "heldout": v["heldout_phrasing"]}
                      for s, v in manifest["splits"].items()}, indent=1))
    print("shortfall:", manifest["shortfall"], "| dropped for length:", manifest["dropped_for_length"])
    print("demo months 6-7:", manifest["demo_months67"])


def extra_stats(res: dict) -> dict:
    """Facts for DATA_CARD_notes.md: repeats between splits, context diversity, sampling (G3 review F8)."""
    data = res["data"]
    out: dict = {"bitext_filter": res["bitext_filter"], "apps_removed_for_context_overlap": res["apps_removed_for_context_overlap"]}
    train_out = Counter(r["output"] for r in data["train"])
    out["test_outputs_equal_to_a_train_output"] = {
        s: dict(Counter(r["task"] for r in data[s] if r["output"] in train_out)) for s in ("val", "test")}

    def blocks(rows):
        return [r["input"].split("Main factors:\n", 1)[1] for r in rows if "Main factors:" in r["input"]]

    tr_blocks = blocks([r for r in data["train"] if r["task"] in ("explain_score", "policy")])
    names = Counter()
    for b in tr_blocks:
        for ln in b.split("\n"):
            names[ln[2:].split(":")[0]] += 1
    explain_tr = [r for r in data["train"] if r["task"] == "explain_score"]
    flagged_tr = sum("Result: flagged for review" in r["input"] for r in explain_tr)
    all_raise_not_flagged = sum(
        "Result: not flagged" in r["input"] and "(lowers" not in r["input"] for r in explain_tr)
    out["context_diversity_train"] = {
        "context_rows": len(tr_blocks), "distinct_factor_blocks": len(set(tr_blocks)),
        "most_common_factors": names.most_common(8),
        "explain_score_flagged_share": round(flagged_tr / len(explain_tr), 3),
        "explain_score_not_flagged_all_factors_raise": all_raise_not_flagged,
    }
    out["test_factor_blocks_also_in_train"] = {
        "test_context_rows": len(blocks(data["test"])),
        "also_in_train": sum(b in set(tr_blocks) for b in blocks(data["test"]))}
    return out


def write_notes(m: dict) -> None:
    """SOURCES.md and DATA_CARD_notes.md: facts only (the team writes the prose for the report/README)."""
    src, ts = Counter(), Counter()
    for s in m["splits"].values():
        src.update(s["per_source"])
        ts.update(s["per_task_source"])
    bit, alp = C.RAW / "bitext_retail_banking" / "bitext.csv", C.RAW / "alpaca_cleaned" / "alpaca_data_cleaned.json"
    bsha = C.sha256_file(bit) if bit.is_file() else "n/a"
    asha = C.sha256_file(alp) if alp.is_file() else "n/a"
    bf = m["bitext_filter"]
    lines = [
        "# SOURCES (facts only, for the bibliography)", "",
        "Licences were read from the dataset cards on 2026-11-06. Each row of the JSON splits carries its own `source` field "
        "(public dataset name with licence, `template`, or `llm:claude-sonnet`), which says which licence applies to that row.", "",
        "| Name | URL | Authors / publisher | Licence | Used for | Rows in dataset/ |", "|---|---|---|---|---|---|",
        f"| Bitext Retail Banking LLM Chatbot Training Dataset | https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset | Bitext Innovations International (2024) | CDLA-Sharing-1.0 | banking_qa | {src.get(P.BITEXT_SOURCE, 0)} |",
        f"| Alpaca-Cleaned | https://huggingface.co/datasets/yahma/alpaca-cleaned | cleaning project gururise/AlpacaDataCleaned (https://github.com/gururise/AlpacaDataCleaned), published on the Hugging Face hub by yahma | CC-BY-4.0 on the dataset card; upstream Stanford Alpaca is CC BY-NC 4.0 | general | {src.get(P.ALPACA_SOURCE, 0)} |",
        "| Stanford Alpaca (upstream of Alpaca-Cleaned) | https://github.com/tatsu-lab/stanford_alpaca | Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P. and Hashimoto, T.B. (2023) | data CC BY-NC 4.0 | none directly | 0 |",
        f"| Concept Q&A (banking and fraud-screening definitions) | written for this project, `preprocess_data/concepts.py` | project work | project work | banking_qa | {ts.get('banking_qa|llm:claude-sonnet', 0)} |",
        "| Bank Account Fraud (BAF) suite, Base | https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 | Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R.P., Gama, J. and Bizarro, P. (2022), NeurIPS Datasets and Benchmarks | CC BY-NC-SA 4.0 | explain_score contexts (scored by the team LR), dataset_fact numbers (EDA tables) | see build_manifest.json |",
        "| Team classifier and model card | https://github.com/KennyDao1324/ai4cyber-fraud (commit " + str(m["ml_scores_meta"]["team_repo_commit"]) + ") | the team | team repo | model scores, threshold, model facts | n/a |",
        "| Template text (questions, answers, policy) | `preprocess_data/templates_*.py`, `facts.py` | project work | project work | explain_score, dataset_fact, policy | n/a |", "",
        "## Retrieval", "",
        f"- Bitext: Hugging Face `main` revision 3e3621092fc6baaf7f53ceb6f091c60ae99acb67 (last modified 2024-07-15), downloaded 2026-11-06; raw CSV sha256 {bsha}; {bf.get('raw', 0):,} rows.",
        f"- Alpaca-Cleaned: Hugging Face `main` revision 12567cabf869d7c92e573c7c783905fc160e9639 (last modified 2023-04-10), downloaded 2026-11-06; raw JSON sha256 {asha}; 51,760 rows.",
        "- Author names above come from the dataset cards; Huy to confirm them against the sources before the bibliography is final.", "",
        "## Licence facts", "",
        "- Stanford Alpaca data (Taori et al. 2023) is released under CC BY-NC 4.0 (non-commercial). Its outputs were generated with OpenAI text-davinci-003, and OpenAI's terms restrict using model output to develop competing models. Alpaca-Cleaned relabels the cleaned data as CC-BY-4.0. Use here: non-commercial coursework, research only.",
        "- Bitext (CDLA-Sharing-1.0) and BAF (CC BY-NC-SA 4.0) are share-alike: data derived from them that is shared must keep the same terms. The JSON splits therefore mix rows under several licences; the per-row `source` field shows which one applies. BAF is also non-commercial.",
        "- The Bitext and Alpaca rows are filtered copies (see Filters below); the BAF-derived numbers are statistics and model scores, not BAF rows.", "",
        "## Filters (Bitext)", "",
        f"- Raw {bf.get('raw', 0):,} rows; without offensive-language tag W {bf.get('after_tag_W', 0):,}; short, placeholder-free {bf.get('after_length_placeholders', 0):,}; "
        f"without protected / decision words, profanity, URLs or code {bf.get('after_unsafe_words', 0):,}; without answers that ask for account numbers, passwords, PINs, names or other personal data {bf.get('after_personal_data_requests', 0):,}; after de-duplication {bf.get('after_dedupe', 0):,}.",
        f"- Rows taken: {src.get(P.BITEXT_SOURCE, 0):,} across train, val and test (tier 1 intents close to account opening, security and fraud handling first, then fees, branches and ATMs, then loans and mortgages).", "",
        "## Evaluated and not used", "",
        "- Josephgflowers/Finance-Instruct-500k (Apache-2.0): about 518k rows; fraud-related questions mostly refer to a missing text or contain extraction noise.",
        "- gbharti/finance-alpaca (MIT label): informal forum answers, upstream licence unclear.",
        "- databricks/databricks-dolly-15k (CC-BY-SA-3.0, human-written): not chosen because Alpaca-Cleaned was available under a CC-BY label; this is a preference, not a licence ruling (Bitext and BAF are share-alike as well).",
        "- PolyAI/banking77 (CC-BY-4.0): intent labels only, no answers.",
    ]
    (C.DATASET / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    t = m["splits"]
    cd, dup, tb = m["context_diversity_train"], m["test_outputs_equal_to_a_train_output"], m["test_factor_blocks_also_in_train"]
    meta = m["ml_scores_meta"]
    notes = ["# DATA_CARD_notes (facts only)", "",
             f"- Seed {m['seed']}; tokenizer {m['tokenizer']}; examples above {m['max_tokens_kept']} tokens are dropped.",
             f"- ml_scores source: {meta['source']} (model {meta['model']}, threshold {meta['threshold']:.4f}, artefact sha256 {meta['artifact_sha256'][:16]}...).",
             "- explain_score and policy contexts use development months 0-5 applications only; months 6-7 appear only in fraudlm_demo_months67.json.",
             "- dataset_fact: for single-fact questions the target fact row comes from a train pool (80%) in train and from an eval pool (20%) in val and test (hash split); distractor lines and ranking questions use all rows.",
             "- Application pools (hash of app_id): train 84%, val 8%, test 8%; no application is shared across pools. "
             f"Val / test applications whose printed context equals a train context were dropped first (val {m['apps_removed_for_context_overlap']['val']}, test {m['apps_removed_for_context_overlap']['test']} applications).",
             "- Test ids ending in -heldout use question wordings that never occur in train or val.",
             "- Fields: id, instruction, input, output, task, source. source is `template`, `llm:claude-sonnet` (question wording or concept answers) or a public dataset name with licence.",
             "- Answers of explain_score, dataset_fact and policy are template-built; every number is copied from the input.",
             "- Factor direction (raises / lowers the model score) is measured against a typical application, the average applicant of months 0-4; the factor lines and the answers say so.",
             "", "## Repeats and sampling (read before reporting test metrics)", "",
             f"- Test outputs identical to a train output: {sum(dup['test'].values())} of {t['test']['n']} (by task: {dup['test']}); val: {sum(dup['val'].values())} of {t['val']['n']} (by task: {dup['val']}). "
             "These are by design (concept answers, month ranking, model facts); report metrics per task and for -heldout rows separately.",
             f"- Sampling: explain_score contexts are drawn about 50/50 flagged / not flagged (train share flagged {cd['explain_score_flagged_share']}); the real flag rate of the frozen model on months 0-5 is {meta.get('dev_flag_rate_months0_5', 'n/a')} "
             f"(5% false-positive budget on validation month 5). ml_scores.csv holds {meta['dev_rows']} rows, {meta['dev_flagged']} flagged.",
             f"- Not-flagged train explain_score contexts whose three factors all raise the score: {cd['explain_score_not_flagged_all_factors_raise']}; their answers say the factors are not enough to reach the threshold.",
             "", "## Context diversity", "",
             f"- Train explain_score + policy contexts: {cd['context_rows']}; distinct factor blocks: {cd['distinct_factor_blocks']}.",
             f"- Most frequent factors (name, count): {cd['most_common_factors']}.",
             f"- Test context rows whose factor block also occurs in train (with a different score): {tb['also_in_train']} of {tb['test_context_rows']}.",
             "", "| Split | Rows | " + " | ".join(TASKS) + " | held-out wording | tokens mean / p95 / max |", "|---|---|" + "---|" * (len(TASKS) + 2)]
    for s in ("train", "val", "test"):
        v = t[s]
        notes.append(f"| {s} | {v['n']} | " + " | ".join(str(v["per_task"].get(k, 0)) for k in TASKS)
                     + f" | {v['heldout_phrasing']} | {v['tokens']['mean']} / {v['tokens']['p95']} / {v['tokens']['max']} |")
    (C.DATASET / "DATA_CARD_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
