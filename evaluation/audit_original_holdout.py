"""Export a conservatively overlap-screened original test split without publishing rows."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.compare_original_validation import screen


def main():
    train_path = ROOT/'dataset_multisource/fraudlm_sft_train.json'
    test_path = ROOT/'dataset/fraudlm_sft_test.json'
    training = json.loads(train_path.read_text())
    rows = json.loads(test_path.read_text())
    kept, excluded = screen(rows, training)
    private = ROOT/'.private/evaluation/clean_original'
    private.mkdir(parents=True, exist_ok=True)
    (private/'fraudlm_sft_test.json').write_text(json.dumps(kept, indent=2))
    (private/'excluded_ids.json').write_text(json.dumps(excluded))
    summary = dict(original_rows=len(rows), retained_rows=len(kept), excluded_rows=len(excluded),
                   original_test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
                   continued_train_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
                   method='Match displayed score and factor lines to all compatible legacy BAF application IDs; exclude IDs present in continued training.',
                   limitation='Conservative context/application screening, not proof of globally unseen entities. Original non-case tasks may retain shared answer patterns. No new model-quality evaluation performed.')
    (ROOT/'evaluation/original_holdout_audit.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
