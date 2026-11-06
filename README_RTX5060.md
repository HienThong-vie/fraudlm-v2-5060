# FraudLM v2 on an RTX 5060 8 GB

This is the same project model as `fraudlm-v2`: the team's own GPT-2 architecture, OpenAI GPT-2 XL
weights, and a rank-16 LoRA adapter. It does **continued instruction fine-tuning** on predictions from
the eight ML projects; it does not replace the model with Qwen and does not retrain GPT-2 XL from zero.

## What it learns

The local builder reads prediction outputs from:

| ID | Project | Data type |
|---|---|---|
| `baf_base` | BAF Base | synthetic |
| `ieee_cis` | IEEE-CIS Fraud Detection | real, anonymized |
| `paysim` | PaySim | synthetic |
| `fraud_handbook` | Fraud Detection Handbook | synthetic |
| `sparkov` | Sparkov | synthetic |
| `banksim` | BankSim | synthetic |
| `ulb_creditcard` | ULB Credit Card Fraud | real, anonymized |
| `elliptic_bitcoin` | Elliptic Bitcoin AML | real, anonymized |

That is five synthetic and three real-anonymized sources. The model sees the dataset, classifier,
score, frozen review threshold, result, approved signals (when available), and held-out model metrics.
It never sees the target label in the case prompt. A score is taught as a review signal, not proof.

## 1. Prepare the data on the Mac

Keep this repository beside the eight ML project folders in `Playground`, then run:

```bash
cd fraudlm-v2-5060
python preprocess_data/build_multisource_sft.py
```

The default selects up to 5,000 cases from each source, balanced between flagged and not flagged,
then mixes in the original FraudLM data to reduce catastrophic forgetting. Outputs are written to
`dataset_multisource/`. That folder is intentionally ignored by Git because the source licences differ.
Copy it privately to the same location in the Windows clone; do not publish it.

To deliberately include every available prediction instead, run
`python preprocess_data/build_multisource_sft.py --max-rows-per-source 0`. This can create hundreds of
thousands of examples and take much longer to fine-tune; the balanced default covers all eight projects
without letting the largest CSV dominate the model.

## 2. Set up PyCharm on the RTX 5060 PC

Use Python 3.11 and create a project virtual environment. In PyCharm's Terminal (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the current Windows CUDA build of PyTorch using the command shown by the official PyTorch
installer, then install the rest:

```powershell
pip install -r requirements-rtx5060.txt
python utils/check_cuda.py
```

The check must print `CUDA available: True` and your RTX 5060. If it prints False, the CPU-only
PyTorch wheel is installed.

## 3. Supply the existing FraudLM adapter

Copy both files from the original run into:

```text
checkpoint/base_adapter/xl_r16_v1/lora_adapter.pt
checkpoint/base_adapter/xl_r16_v1/run_config.json
```

They are not committed because the adapter is about 89 MB. You can instead download the original
`xl_r16_v1` GitHub Release and place the two files there. The rank must stay 16 when continuing this
adapter.

## 4. Train

In PyCharm, run the included **Train FraudLM RTX 5060** configuration, or use:

```powershell
python training_model/train_sft.py --config configs/rtx5060_8gb.json
```

The 8 GB settings are: FP16 frozen base, AMP, gradient checkpointing, micro-batch 1, accumulation 8,
and 384 input tokens. Output goes to `checkpoint/xl_r16_multidata_5060_v1/`.

To continue an interrupted run, use `--resume` and remove the init adapter for that command:

```powershell
python training_model/train_sft.py --config configs/rtx5060_8gb.json --init_adapter "" --resume
```

If CUDA reports out of memory, close other GPU applications and retry with `--max_len 256`. Do not
lower the LoRA rank when starting from the rank-16 adapter. A new run may use rank 8, but it cannot load
the existing rank-16 weights.

## 5. Chat with the trained model

```powershell
python chat_multisource.py `
  --adapter checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt `
  --max-new-tokens 300
```

Try:

```text
datasets
list ieee_cis
explain ieee_cis:3663555
ask ieee_cis:3663555 What does this score mean and what should I check next?
```

The prediction store is a lookup source at chat time. Continued fine-tuning teaches response style and
fraud concepts, but it is not a database: adding every raw prediction to weights would not guarantee
exact recall. The chat therefore supplies the selected case facts in the prompt and applies a deterministic
safety guard to contradictory results, fraud accusations, and automated customer decisions.

## Files to open in PyCharm

- `training_model/train_sft.py` — continued LoRA training
- `configs/rtx5060_8gb.json` — 8 GB GPU settings
- `preprocess_data/build_multisource_sft.py` — builds the eight-source instruction dataset
- `chat_multisource.py` — prediction-grounded terminal chat
- `utils/check_cuda.py` — confirms PyTorch can use the RTX GPU

Run all tests with `python -m pytest tests -q`.
