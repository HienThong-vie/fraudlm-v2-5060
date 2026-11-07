"""Evaluate the continued adapter on frozen, non-training instruction examples.

Raw and guard-assisted metrics are separate. Private row outputs are never published.
"""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.multisource_prompt import guard_prediction_answer, load_prediction_store


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def metrics(answer, case):
    numbers = re.findall(r'(?<![\w.])\d+\.\d+(?!\w)', answer)
    def repeated(value):
        return any(abs(Decimal(n) - Decimal(f'{value:.6f}')) <= Decimal('.0000005') for n in numbers)
    unflagged = bool(re.search(r'\bnot flagged\b', answer, re.I))
    positive_text = re.sub(r'\bnot flagged(?: for review)?\b', '', answer, flags=re.I)
    flagged = bool(re.search(r'\bflagged for review\b', positive_text, re.I))
    correct = (flagged and not unflagged) if case['score'] >= case['threshold'] else (unflagged and not flagged)
    from utils.multisource_prompt import prediction_guard_reasons
    return dict(score_repeated=repeated(case['score']), threshold_repeated=repeated(case['threshold']),
                flag_correct=bool(correct), guard_rule_violation=bool(prediction_guard_reasons(answer, case)))


def select_rows(test, train, store, per_source):
    train_ids = {r['id'] for r in train}
    train_inputs = {(r['instruction'].strip(), r.get('input', '').strip()) for r in train}
    train_contexts = {r.get('input', '').strip() for r in train if r.get('task') == 'multisource_prediction'}
    pools = {}
    cases_by_id = {'multi_' + case['case_key'].replace(':', '_'): case for case in store.values()}
    for row in test:
        if row.get('task') != 'multisource_prediction':
            continue
        assert row['id'] not in train_ids
        assert (row['instruction'].strip(), row['input'].strip()) not in train_inputs
        assert row['input'].strip() not in train_contexts
        case = cases_by_id[row['id']]
        pools.setdefault(case['source_id'], []).append((row, case))
    rng = random.Random(42)
    selected = []
    for source in sorted(pools):
        pool = sorted(pools[source], key=lambda pair: pair[0]['id'])
        flagged = [p for p in pool if p[1]['score'] >= p[1]['threshold']]
        unflagged = [p for p in pool if p[1]['score'] < p[1]['threshold']]
        part = rng.sample(flagged, min(len(flagged), per_source // 2)) + rng.sample(unflagged, min(len(unflagged), per_source // 2))
        remaining = [p for p in pool if p[0]['id'] not in {x[0]['id'] for x in part}]
        part += rng.sample(remaining, min(len(remaining), per_source - len(part)))
        selected += part
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', default='checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt')
    parser.add_argument('--per-source', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--max-seconds', type=int, default=3600)
    parser.add_argument('--rescore-only', action='store_true')
    args = parser.parse_args()
    train_path = ROOT / 'dataset_multisource/fraudlm_sft_train.json'
    val_path = ROOT / 'dataset_multisource/fraudlm_sft_val.json'
    test_path = ROOT / 'dataset_multisource/fraudlm_sft_test.json'
    adapter = ROOT / args.adapter
    config = json.loads((adapter.parent / 'run_config.json').read_text())
    assert sha(train_path)[:16] + '-' + sha(val_path)[:16] == config['data_sha256'], 'Training data differ from recorded run'
    store = load_prediction_store(ROOT / 'dataset_multisource/prediction_store.jsonl')
    selected = select_rows(json.loads(test_path.read_text()), json.loads(train_path.read_text()), store, args.per_source)
    public = ROOT / 'evaluation/continued_adapter'
    private = ROOT / '.private/evaluation/continued_adapter'
    public.mkdir(parents=True, exist_ok=True); private.mkdir(parents=True, exist_ok=True)
    selection = dict(adapter_sha256=sha(adapter), train_sha256=sha(train_path), test_sha256=sha(test_path),
        selected_ids=[r['id'] for r, _ in selected], seed=42, per_source=args.per_source,
        max_new_tokens=args.max_new_tokens, device='cpu', dtype='fp32',
        scope='Held-out instruction/case contexts from the recorded run2 data. No identical training ID/input/context. Not proof of unseen underlying entities across original mixed datasets. Old-model grounding; separate from refreshed-model context probes.')
    manifest_path = public / 'selection.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == selection, 'Evaluation selection changed'
    else:
        manifest_path.write_text(json.dumps(selection, indent=2))
    partial = private / 'rows.jsonl'
    done = {r['id']: r for r in map(json.loads, partial.read_text().splitlines())} if partial.exists() else {}
    if args.rescore_only:
        for row, case in selected:
            if row['id'] in done:
                done[row['id']]['raw_metrics'] = metrics(done[row['id']]['raw'], case)
                done[row['id']]['system_metrics'] = metrics(done[row['id']]['guarded'], case)
        write_summary(public, selection, done)
        return
    import torch
    torch.set_num_threads(4)
    from chat import Engine
    engine = Engine('gpt2-xl', adapter, 'cpu', 'fp32', ROOT / 'checkpoint/pretrained', None, None, args.max_new_tokens)
    started = time.monotonic()
    for row, case in selected:
        if row['id'] in done:
            continue
        if time.monotonic() - started > args.max_seconds:
            break
        raw, info = engine.generate(row['instruction'], row['input'])
        guarded, replaced = guard_prediction_answer(raw, case)
        result = dict(id=row['id'], source=case['source_id'], raw=raw, guarded=guarded,
            raw_metrics=metrics(raw, case), system_metrics=metrics(guarded, case), replaced=replaced,
            generation_seconds=info['seconds'], prompt_tokens=info['prompt_tokens'], new_tokens=info['new_tokens'], stop=info['stop'])
        with partial.open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        done[row['id']] = result
        print('EVALUATED', len(done), '/', len(selected), row['id'], round(info['seconds'], 1), 'seconds', flush=True)
        write_summary(public, selection, done)
    for row, case in selected:
        if row['id'] in done:
            done[row['id']]['raw_metrics'] = metrics(done[row['id']]['raw'], case)
            done[row['id']]['system_metrics'] = metrics(done[row['id']]['guarded'], case)
    write_summary(public, selection, done)


def write_summary(folder, selection, done):
    rows = list(done.values())
    summary = dict(status='complete' if len(rows) == len(selection['selected_ids']) else 'partial',
        metric_definition='Six-decimal numeric repetition tolerance5e-7; explicit comparison status; separate existing guard-rule violations. Sentence-final punctuation permitted; negative review phrases are not positive flags.',
        evaluated_rows=len(rows), requested_rows=len(selection['selected_ids']), adapter_sha256=selection['adapter_sha256'],
        scope=selection['scope'], model='GPT-2 XL + completed RTX5060 continued adapter',
        guard_replacements=sum(r['replaced'] for r in rows),
        raw_metrics={k: sum(r['raw_metrics'][k] for r in rows) / len(rows) for k in rows[0]['raw_metrics']} if rows else {},
        guard_assisted_metrics={k: sum(r['system_metrics'][k] for r in rows) / len(rows) for k in rows[0]['system_metrics']} if rows else {},
        generation_seconds=sum(r['generation_seconds'] for r in rows),
        limitation='Small fixed outcome-balanced per-source sample; not population-weighted, not comparable directly to run1 full-test rates. Automatic string metrics do not establish semantic advice quality. No retraining or test-driven checkpoint selection.')
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
