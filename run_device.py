"""Run the current branch's device profile using this Python interpreter."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'chat', 'train', 'compare'])
    args, extra = parser.parse_known_args(argv)
    profile = json.loads((ROOT/'configs/device.json').read_text())
    if args.action == 'check':
        import torch
        print('Profile:', profile['name'])
        print('Python:', sys.executable)
        print('PyTorch:', torch.__version__)
        print('Configured device:', profile['device'])
        if profile['device'] == 'cuda':
            if not torch.cuda.is_available():
                raise SystemExit('CUDA unavailable; install a compatible PyTorch CUDA build before using this branch.')
            print('GPU:', torch.cuda.get_device_name(0))
        x = torch.ones((8, 8), device=profile['device'])
        assert float((x@x).mean()) == 8.
        print('Device calculation: OK')
        return 0
    commands = {
        'chat': ['chat_multisource.py', '--adapter', 'checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt',
                 '--device', profile['device'], '--dtype', profile['dtype'], '--max-new-tokens', str(profile['max_new_tokens'])],
        'train': ['training_model/train_sft.py', '--config', profile['training_config']],
        'compare': ['evaluation/compare_original_validation.py', '--device', profile['device']],
    }
    if profile['device'] == 'cuda':
        import torch
        if not torch.cuda.is_available():
            raise SystemExit('CUDA unavailable. Use the mac branch for the CPU profile, or fix the CUDA installation.')
    environment = os.environ.copy()
    if profile['device'] == 'cpu':
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            environment.setdefault(name, '4')
    return subprocess.call([sys.executable, *commands[args.action], *extra], cwd=ROOT, env=environment)


if __name__ == '__main__':
    raise SystemExit(main())
