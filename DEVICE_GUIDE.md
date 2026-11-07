# RTX 5060 PC

This branch uses CUDA for chat, FP16 for generation, and the RTX 5060 training configuration. The shared safety, grounding and resume fixes are also present in the `mac` branch. Do not use this branch's device launcher without working CUDA.

## Open in PyCharm

```powershell
git clone --branch pc-rtx5060 --single-branch https://github.com/KennyDao1324/fraudlm-v2-5060.git fraudlm-pc
cd fraudlm-pc
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the CUDA-enabled PyTorch build from the [official installer](https://pytorch.org/get-started/locally/), then:

```powershell
python -m pip install -r requirements-rtx5060.txt
python run_device.py check
```

If your existing environment already trains successfully on the RTX 5060, reuse that interpreter rather than reinstalling it.

## Local inputs

Extract the privately supplied `fraudlm-device-inputs.zip` into the repository root. It supplies:

- `dataset_final_v2/`: refreshed SFT data and prediction store.
- `checkpoint/xl_r16_multidata_5060_v1/`: completed adapter and configuration.
- `checkpoint/xl_r16_v1/`: original adapter for comparison.
- `dataset_multisource/fraudlm_sft_train.json`: historical overlap-audit input.

The pretrained GPT-2 XL cache is downloaded on first load if absent; allow several GB of disk space. The archive is private, not a GitHub Release. Existing copies may be reused after checksum verification. Keep the entire input directory layout, not just the `.pt` file.

## Run

```powershell
python run_device.py chat
```

Try `datasets`, `facts baf_base`, then `list ieee_cis`. Copy a listed key into `explain <key>` or `ask <key> <question>`.

To start a **new** refreshed-data training run:

```powershell
python run_device.py train
```

Output: `checkpoint/xl_r16_final_models_5060_v1/`. Resume an interruption:

```powershell
python run_device.py train --init_adapter= --resume auto
```

After that new run finishes, explicitly select its adapter:

```powershell
python run_device.py chat --adapter checkpoint/xl_r16_final_models_5060_v1/lora_adapter.pt
```

For matched diagnostic validation loss:

```powershell
python run_device.py compare
```

This requires both adapters and historical audit inputs. It uses FP32 for a matched comparison; close other GPU applications. If 8 GB VRAM is insufficient, use the CPU comparison on the Mac rather than silently comparing different precisions.

In PyCharm choose `.venv` as interpreter and use **Device chat** or **Device training**. Training is not required merely to test the existing completed adapter. Guard improvements apply immediately, but changed training data do not alter saved weights until a new run is trained.

See [grounding/evaluation status](FINAL_GROUNDING.md) and [distribution restrictions](DISTRIBUTION.md). The model Release remains a draft; do not publish the private input ZIP.
