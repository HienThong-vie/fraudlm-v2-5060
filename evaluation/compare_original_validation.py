"""Matched adapter loss on original validation rows, with BAF contamination screening."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def signature(text):
    lines = text.splitlines()
    score = next((line for line in lines if line.startswith('Model score:')), None)
    if score is None or 'Main factors:' not in lines:
        return None
    start = lines.index('Main factors:') + 1
    factors = []
    for line in lines[start:]:
        if not line.startswith('- '):
            break
        factors.append(line)
    return (score, tuple(factors)) if factors else None


def screen(rows, training):
    """Map original displayed score/factors back to all matching legacy BAF IDs."""
    import pandas as pd
    from utils.prompt import format_score_context
    lookup = {}
    for row in pd.read_csv(ROOT / 'preprocessed/ml_scores.csv').itertuples():
        text = format_score_context(float(row.score), float(row.threshold), json.loads(row.top_factors), rules=False)
        lookup.setdefault(signature(text), set()).add(str(int(row.app_id)))
    exposed = {str(r['id']).removeprefix('multi_baf_base_') for r in training
               if str(r.get('id', '')).startswith('multi_baf_base_')}
    keep, excluded = [], []
    for row in rows:
        key = signature(row.get('input', ''))
        ids = lookup.get(key, set()) if key else set()
        if ids & exposed:
            excluded.append(row['id'])
        else:
            keep.append(row)
    return keep, excluded


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--original-adapter', type=Path, default=ROOT/'checkpoint/xl_r16_v1/lora_adapter.pt')
    ap.add_argument('--continued-adapter', type=Path, default=ROOT/'checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt')
    ap.add_argument('--output', type=Path, default=ROOT / 'evaluation/original_validation_comparison.json')
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    from chat import Engine
    from training_model.sft_data import SFTDataset, collate, IGNORE_INDEX
    from training_model.train_sft import batch_loss
    path = ROOT / 'dataset/fraudlm_sft_val.json'
    rows = json.loads(path.read_text())
    training = json.loads((ROOT / 'dataset_multisource/fraudlm_sft_train.json').read_text())
    clean, excluded = screen(rows, training)
    clean_ids = {r['id'] for r in clean}
    dataset = SFTDataset(rows, max_len=1024, name='original-val')
    assert len(dataset) == len(rows) == 500 and not dataset.n_truncated
    clean_items = [item for row, item in zip(dataset.rows, dataset.items) if row['id'] in clean_ids]
    result = dict(rows=500, screened_rows=len(clean_items), excluded_overlap_rows=len(excluded),
                  validation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  protocol='Same original 500 validation rows; response-token-weighted CE including EOS, max_len1024, micro_batch1, fp32 merged adapters, no generation.',
                  limitation='Diagnostic validation loss, not an independent test or semantic skill evaluation. BAF screening matches displayed scores and factors to all matching original application IDs; other latent entity overlap is not ruled out.',
                  device=args.device, models={})
    private = ROOT / '.private/evaluation'
    private.mkdir(parents=True, exist_ok=True)
    (private / 'original_validation_excluded.json').write_text(json.dumps(excluded))
    for name in ('xl_r16_v1', 'xl_r16_multidata_5060_v1'):
        adapter = args.original_adapter if name == 'xl_r16_v1' else args.continued_adapter
        if name == 'xl_r16_v1' and not adapter.is_file():
            adapter = ROOT/'checkpoint/base_adapter/xl_r16_v1/lora_adapter.pt'
        engine = Engine('gpt2-xl', adapter, args.device, 'fp32', ROOT/'checkpoint/pretrained', 16, 32, 1)
        total, tokens, clean_total, clean_tokens = 0., 0, 0., 0
        order = sorted(range(len(dataset)), key=lambda i:len(dataset.items[i]['input_ids']))
        with torch.no_grad():
            for number, i in enumerate(order, 1):
                x, y = collate([dataset.items[i]], torch.device(args.device))
                count = int((y != IGNORE_INDEX).sum())
                value = batch_loss(engine.model, x, y, None).item() * count
                total += value
                tokens += count
                if dataset.rows[i]['id'] in clean_ids:
                    clean_total += value
                    clean_tokens += count
                if number % 50 == 0:
                    print(name, number, '/500 validation rows', flush=True)
        loss = total / tokens
        screened = clean_total / clean_tokens
        result['models'][name] = dict(full_500_loss=loss, overlap_screened_loss=screened,
            adapter_sha256=hashlib.sha256(adapter.read_bytes()).hexdigest())
        result['status'] = 'complete' if len(result['models']) == 2 else 'partial'
        args.output.write_text(json.dumps(result, indent=2))
        print(name, result['models'][name], flush=True)
        del engine
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
