# Mac

This branch uses CPU/FP32 for the M1 Max. It does not assume CUDA or claim a validated MPS training path. Shared safety, grounding and resume fixes are also present in `pc-rtx5060`.

## Open in PyCharm

The existing local project is already on `mac`; you do not need to clone it again. For a fresh checkout:

```bash
git clone --branch mac --single-branch https://github.com/KennyDao1324/fraudlm-v2-5060.git fraudlm-mac
cd fraudlm-mac
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_device.py check
```

On an existing checkout, use its working interpreter instead of reinstalling packages. The verified local interpreter is `/opt/anaconda3/bin/python`; select it in PyCharm if using that environment. A fresh installation must be checked independently.

## Local inputs

The current Mac already has the private inputs. A fresh clone needs the privately supplied `fraudlm-device-inputs.zip` extracted into its root. It contains refreshed `dataset_final_v2/`, both original and completed adapters/configurations under `checkpoint/`, and the historical continued-training JSON used for overlap screening. Keep that ZIP off GitHub.

GPT-2 XL base weights are downloaded on first load if absent. The existing `checkpoint/pretrained/` cache can be reused. The base weights are not included in the private transfer ZIP.

## Chat

```bash
python run_device.py chat
```

Using the existing interpreter directly:

```bash
/opt/anaconda3/bin/python run_device.py chat
```

Try `datasets`, `facts baf_base` and `list ieee_cis`. Copy a listed key into `explain <key>` or `ask <key> <question>`. CPU generation is slower than the RTX 5060; the default answer cap is 160 tokens within the model's 1,024-token context window.

In PyCharm select the working interpreter and choose **Device chat**. You do not need to train again to use the current adapter or the updated guards.

## Validation comparison

```bash
python run_device.py compare
```

This runs both adapters on the same original 500 validation rows and separately reports an overlap-screened subset. It can take a long time on CPU. A comparison was already started locally during verification; do not start a duplicate while it is still running. Only a JSON containing both model results is a completed comparison.

## Optional CPU training

The RTX 5060 is preferable for the refreshed training run. CPU training on the Mac is available but slow:

```bash
python run_device.py train
```

It uses FP32, no CUDA AMP, gradient checkpointing and a separate output folder: `checkpoint/xl_r16_final_models_mac_v1/`. The eight-hour session limit saves state rather than promising completion. Resume with:

```bash
python run_device.py train --init_adapter= --resume auto
```

After completion, select that adapter explicitly:

```bash
python run_device.py chat --adapter checkpoint/xl_r16_final_models_mac_v1/lora_adapter.pt
```

In PyCharm use **Device training**, not the historical RTX run configuration. The launcher limits CPU thread defaults to four; close other memory-intensive applications before loading GPT-2 XL.

See [evaluation status](FINAL_GROUNDING.md) and [distribution restrictions](DISTRIBUTION.md). Saved weights are unchanged until a new training run is completed.
