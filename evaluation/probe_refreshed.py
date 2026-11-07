"""Small frozen-context smoke probe, not a new independent model-performance estimate."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.evaluate_continued import metrics,sha
from utils.multisource_prompt import load_prediction_store,format_prediction_context,guard_prediction_answer


def main():
    store=load_prediction_store(ROOT/'dataset_final/prediction_store.jsonl')
    sources=sorted({c['source_id'] for c in store.values()})
    selected=[]
    for i,source in enumerate(sources):
        pool=sorted([c for c in store.values() if c['source_id']==source],key=lambda c:c['case_key'])
        desired=i%2==0
        eligible=[c for c in pool if (c['score']>=c['threshold'])==desired]
        selected.append((eligible or pool)[0])
    folder=ROOT/'evaluation/refreshed_context_probe';folder.mkdir(parents=True,exist_ok=True)
    private=ROOT/'.private/evaluation/refreshed_context_probe';private.mkdir(parents=True,exist_ok=True)
    protocol=dict(case_keys=[c['case_key'] for c in selected],n=8,max_new_tokens=160,
        adapter_sha256=sha(ROOT/'checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt'),
        registry_sha256=sha(ROOT/'configs/prediction_sources.json'),
        selection='First lexicographic case per source, alternating flagged/not-flagged before generation.',
        scope='Refreshed-context smoke probe, not a guaranteed unseen-entity holdout or population-quality estimate.')
    (folder/'selection.json').write_text(json.dumps(protocol,indent=2))
    import torch
    torch.set_num_threads(4)
    from chat import Engine
    engine=Engine('gpt2-xl',ROOT/'checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt','cpu','fp32',ROOT/'checkpoint/pretrained',None,None,160)
    rows=[]
    for case in selected:
        raw,info=engine.generate('Explain this classifier result without treating it as proof of fraud.',format_prediction_context(case))
        guarded,replaced=guard_prediction_answer(raw,case)
        row=dict(source=case['source_id'],case_key=case['case_key'],raw=raw,guarded=guarded,replaced=replaced,
                 raw_metrics=metrics(raw,case),system_metrics=metrics(guarded,case),seconds=info['seconds'])
        rows.append(row)
        with (private/'rows.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print('PROBED',len(rows),'/8',case['source_id'],round(info['seconds'],1),'seconds',flush=True)
    summary=dict(status='complete',n=len(rows),protocol=protocol,guard_replacements=sum(r['replaced'] for r in rows),
        raw_metrics={k:sum(r['raw_metrics'][k] for r in rows)/len(rows) for k in rows[0]['raw_metrics']},
        guard_assisted_metrics={k:sum(r['system_metrics'][k] for r in rows)/len(rows) for k in rows[0]['system_metrics']},
        generation_seconds=sum(r['seconds'] for r in rows))
    (folder/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
