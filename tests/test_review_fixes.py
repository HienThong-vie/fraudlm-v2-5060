"""Regression tests for multi-source grounding, answer checks and continued-run resume."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from chat_multisource import MultiSourceSession
from preprocess_data.build_multisource_sft import read_source
from utils.multisource_prompt import guard_prediction_answer, prediction_answer

CASE = dict(source_id='demo', case_key='demo:7', case_id='7', dataset_name='Demo',
            dataset_kind='synthetic', model='test', score=.2, threshold=.6, signals={})


@pytest.mark.parametrize('answer', [
    'The applicant is 60, which raises the score.',
    'The score is 0.99 and it is not flagged.',
    'It is not at risk.',
    'Demo case 899 is not flagged.',
    'This person is a criminal.',
])
def test_unsafe_grounded_answers_are_replaced(answer):
    safe, replaced = guard_prediction_answer(answer, CASE)
    assert replaced and safe == prediction_answer(CASE, 'explain')


def test_correct_negative_review_phrase_is_not_a_contradiction():
    answer = 'The score is 0.2, below the threshold of 0.6. It is not flagged for review.'
    assert guard_prediction_answer(answer, CASE) == (answer, False)


def test_old_classifier_name_cannot_override_current_case_grounding():
    case = dict(CASE, model='catboost')
    answer = 'Logistic regression gave score 0.2 against threshold 0.6. The case is not flagged.'
    safe, replaced = guard_prediction_answer(answer, case)
    assert replaced and 'Logistic regression' not in safe


def test_general_answer_is_checked_too():
    output = []
    session = MultiSourceSession({}, lambda q, c: ('I would approve the application.', {}), output.append)
    session.handle('What should a fraud analyst consider?')
    assert any('[guarded replacement]' in line for line in output)
    assert not any('FraudLM> I would approve' in line for line in output)


def test_model_facts_come_from_runtime_registry_without_generation():
    calls, output = [], []
    source = dict(id='demo', name='Demo', model='new model', threshold=.3, pr_auc=.4)
    session = MultiSourceSession({}, lambda *a: calls.append(a), output.append, sources=[source])
    session.handle('facts demo')
    session.handle('Which classifier does demo use?')
    assert not calls and sum('new model' in line for line in output) == 2


def test_signals_are_not_fabricated_attributions():
    plain = dict(CASE, signals={'amount':100})
    assert 'No model-attribution factors' in prediction_answer(plain, 'limitations')
    attributed = dict(CASE, signals={'main model factors':'housing status=BA (raises score)'})
    assert 'housing status=BA' in prediction_answer(attributed, 'explain')
    assert 'not causal' in prediction_answer(attributed, 'limitations')


def test_sampled_row_ids_preserve_csv_positions(tmp_path):
    frame = pd.DataFrame({'score': [i/100 for i in range(100)]})
    frame.to_csv(tmp_path/'pred.csv', index=False)
    source = dict(id='demo', name='Demo', kind='synthetic', model='test', prediction_file='pred.csv',
                  score_column='score', threshold=.5, id_column='__row__', extra_columns=[])
    rows = read_source(source, tmp_path, 10, 42)
    assert len(rows) == 10
    for row in rows:
        assert float(frame.iloc[int(row['case_id'])]['score']) == row['score']
        assert row['prediction_row_index'] == int(row['case_id'])


def test_initialised_run_resumes_without_force_or_initial_adapter(tmp_path, monkeypatch):
    from training_model import train_sft as trainer
    from training_model.gpt_model import GPTModel
    from training_model.lora import apply_lora, lora_state_dict
    tiny = dict(vocab_size=50257, context_length=128, emb_dim=16, n_heads=2, n_layers=1,
                drop_rate=0., qkv_bias=True)
    def model(*args, **kwargs):
        torch.manual_seed(12)
        return GPTModel(tiny)
    monkeypatch.setattr(trainer, 'load_pretrained_gpt2', model)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda:False)
    initial = model()
    apply_lora(initial, rank=2, alpha=4)
    adapter = tmp_path/'initial.pt'
    torch.save(lora_state_dict(initial), adapter)
    data = tmp_path/'data'
    data.mkdir()
    rows = [dict(id=f'row{i}', instruction='Explain this score.', input='Model score: 0.2',
                 output='A score is not proof of fraud.', task='explain_score') for i in range(4)]
    for split in ('train', 'val'):
        (data/f'fraudlm_sft_{split}.json').write_text(json.dumps(rows))
    args = ['--size','gpt2-small','--lora_rank','2','--lora_alpha','4','--data_dir',str(data),
            '--out_dir',str(tmp_path),'--run_id','continued','--resume','','--max_len','128',
            '--max_steps','2','--micro_batch','1','--grad_accum','1','--eval_at_start','off',
            '--eval_every','2','--save_every','1','--sample_tokens','1','--log_every','1']
    with pytest.raises(trainer.SimulatedKill):
        trainer.main(args+['--init_adapter',str(adapter),'--debug_crash_at_step','2'])
    assert trainer.main(args+['--resume','auto']) == 'done'
    config = json.loads((tmp_path/'continued/run_config.json').read_text())
    assert config['init_adapter_sha256'] == hashlib.sha256(adapter.read_bytes()).hexdigest()
    assert config['init_adapter'] == str(adapter)
