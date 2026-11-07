"""Device launchers share code while keeping CPU and CUDA runtime settings separate."""
import json
from pathlib import Path
import zipfile

import pytest
import torch

import run_device
from utils import package_pc_inputs as packaging


@pytest.mark.parametrize('device,dtype,config', [('cpu','fp32','configs/mac_cpu.json'),
                                               ('cuda','fp16','configs/rtx5060_final_models.json')])
def test_device_launcher_arguments(tmp_path, monkeypatch, device, dtype, config):
    (tmp_path/'configs').mkdir()
    profile = dict(name=device, device=device, dtype=dtype, training_config=config, max_new_tokens=160)
    (tmp_path/'configs/device.json').write_text(json.dumps(profile))
    monkeypatch.setattr(run_device, 'ROOT', tmp_path)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda:True)
    commands = []
    monkeypatch.setattr(run_device.subprocess, 'call', lambda command, **kw: commands.append(command) or 0)
    assert run_device.main(['chat']) == 0
    assert commands[-1][commands[-1].index('--device')+1] == device
    assert commands[-1][commands[-1].index('--dtype')+1] == dtype
    assert run_device.main(['train']) == 0 and config in commands[-1]
    assert run_device.main(['compare']) == 0 and commands[-1][-1] == device


def test_cpu_and_cuda_training_configs_have_separate_run_outputs():
    root = Path(__file__).resolve().parents[1]
    mac = json.loads((root/'configs/mac_cpu.json').read_text())
    pc = json.loads((root/'configs/rtx5060_final_models.json').read_text())
    assert mac['run_id'] != pc['run_id']
    assert mac['base_dtype']=='fp32' and not mac['amp']
    assert pc['base_dtype']=='fp16' and pc['amp']
    assert mac['data_dir'] == pc['data_dir'] == 'dataset_final_v2'


def test_private_package_carries_optional_comparison_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(packaging, 'ROOT', tmp_path)
    files = ['dataset_final_v2/fraudlm_sft_train.json', 'dataset_final_v2/fraudlm_sft_val.json',
             'dataset_final_v2/fraudlm_sft_test.json', 'dataset_final_v2/prediction_store.jsonl',
             'dataset_final_v2/build_manifest.json', 'checkpoint/completed/lora_adapter.pt',
             'checkpoint/completed/run_config.json', 'dataset_multisource/fraudlm_sft_train.json',
             'checkpoint/xl_r16_v1/lora_adapter.pt', 'checkpoint/xl_r16_v1/run_config.json']
    for name in files:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    archive = packaging.package(tmp_path/'checkpoint/completed/lora_adapter.pt', tmp_path/'inputs.zip',
                                tmp_path/'dataset_final_v2', evaluation_inputs=True)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None and set(files).issubset(zipped.namelist())
    with pytest.raises(FileExistsError):
        packaging.package(tmp_path/'checkpoint/completed/lora_adapter.pt', archive, tmp_path/'dataset_final_v2')
