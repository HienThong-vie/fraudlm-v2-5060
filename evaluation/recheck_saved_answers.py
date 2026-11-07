"""Replay saved raw generations through current guards without regenerating or changing weights."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.multisource_prompt import guard_prediction_answer, load_prediction_store
from evaluation.evaluate_continued import metrics


def main():
    groups = {}
    for name, data_dir in (('continued_adapter', 'dataset_multisource'), ('refreshed_context_probe', 'dataset_final')):
        store = load_prediction_store(ROOT/data_dir/'prediction_store.jsonl')
        by_id = {'multi_'+c['case_key'].replace(':', '_'):c for c in store.values()}
        rows = [json.loads(line) for line in (ROOT/'.private/evaluation'/name/'rows.jsonl').read_text().splitlines()]
        checked = []
        for row in rows:
            case = store[row['case_key']] if 'case_key' in row else by_id[row['id']]
            answer, replaced = guard_prediction_answer(row['raw'], case)
            checked.append(dict(replaced=replaced, metrics=metrics(answer, case)))
        groups[name] = dict(n=len(rows), replacements=sum(r['replaced'] for r in checked),
            guarded_numeric_metrics={k:sum(r['metrics'][k] for r in checked)/len(checked) for k in checked[0]['metrics']})
    report = dict(protocol='Replay existing private raw generations through current guards; original generation protocols and raw metrics are unchanged.',
                  limitation='Post-review regression checks on known answers, not a new blind evaluation. Template fallbacks can guarantee stored facts but do not prove learned reasoning or semantic advice quality.', groups=groups)
    (ROOT/'evaluation/guard_recheck.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
