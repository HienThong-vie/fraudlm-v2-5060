import json
from pathlib import Path

from evaluation.evaluate_continued import metrics, select_rows
from utils.multisource_prompt import format_prediction_context

ROOT = Path(__file__).resolve().parents[1]


def test_current_grounding_uses_one_frozen_champion_per_source():
    registry = json.loads((ROOT / 'configs/prediction_sources.json').read_text())
    sources = registry['sources']
    assert len(sources) == 8 and len({s['id'] for s in sources}) == 8
    expected = {'baf_base':'catboost','ieee_cis':'xgboost','paysim':'random_forest',
                'fraud_handbook':'random_forest','sparkov':'catboost','banksim':'catboost',
                'ulb_creditcard':'xgboost','elliptic_bitcoin':'xgboost'}
    for source in sources:
        assert source['model'] == expected[source['id']]
        assert source['fpr_budget'] == .05 and len(source['model_sha256']) == 64
        assert source['id_column'] == 'case_id'
        assert 'test' not in source['extra_columns'] and 'target' not in source['extra_columns']


def test_offline_and_noncausal_input_notes_are_in_context():
    case = dict(dataset_name='Elliptic',dataset_kind='real',case_id='final-test-1',model='xgboost',
                score=.4,threshold=.5,experiment='feature_policy_v4',fpr_budget=.05,offline_only=True,signals={})
    context = format_prediction_context(case)
    assert 'not validated for live' in context and 'not feature-attribution' in context
    assert '5.0%' in context


def test_raw_evaluation_does_not_credit_missing_or_wrong_numbers():
    case = dict(score=.123456, threshold=.3)
    correct = metrics('Score 0.123456, threshold 0.300000. It is not flagged for review.', case)
    assert correct['score_repeated'] and correct['threshold_repeated'] and correct['flag_correct']
    wrong = metrics('Score 0.900000. It is flagged for review.', case)
    assert not wrong['score_repeated'] and not wrong['threshold_repeated'] and not wrong['flag_correct']


def test_private_final_rows_never_overlap_their_training_contexts():
    import pytest
    folder = ROOT / 'dataset_final_v2'
    if not (folder / 'fraudlm_sft_train.json').exists():
        pytest.skip('Private generated data not installed')
    train = json.loads((folder / 'fraudlm_sft_train.json').read_text())
    test = json.loads((folder / 'fraudlm_sft_test.json').read_text())
    inputs = {(r['instruction'],r['input']) for r in train}
    assert not inputs & {(r['instruction'],r['input']) for r in test}
    assert not any(r['task']=='multisource_model_fact' for r in train)
    manifest = json.loads((folder/'build_manifest.json').read_text())
    assert manifest['max_rows_per_source'] == 500
    assert manifest['includes_original_fraudlm_data'] and len(manifest['sources']) == 8
