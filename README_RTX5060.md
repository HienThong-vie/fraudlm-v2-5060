# FraudLM v2 on RTX 5060

This project uses a custom GPT-2 XL architecture, pretrained GPT-2 weights and rank-16 LoRA. It supports continued instruction fine-tuning on eight classifiers' outputs, not pretraining from scratch.

## Windows setup

Clone the repository and open the folder in PyCharm:

```powershell
git clone https://github.com/KennyDao1324/fraudlm-v2-5060.git
cd fraudlm-v2-5060
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a CUDA-compatible PyTorch build using the [official installer](https://pytorch.org/get-started/locally/), then:

```powershell
python -m pip install -r requirements-rtx5060.txt
python utils/check_cuda.py
```

The check must report CUDA and the RTX 5060. Select `.venv` as the PyCharm interpreter. Weights, source prediction files and private generated datasets are not included in a clone.

## Prepare refreshed data

On the Mac with the source prediction files installed:

```bash
python preprocess_data/build_multisource_sft.py
```

The default selects at most 500 cases per source, preserves the original instruction training data, omits model-fact targets and writes `dataset_final_v2/`. Original validation/test contexts exposed during the completed run are screened out using its private training file (`--prior-training-file`); the builder stops if this required audit input is absent. Source paths, prediction-file hashes and original CSV data-row indices are recorded. Earlier data directories remain unchanged for reproducibility.

Sources are BAF, IEEE-CIS, PaySim, Fraud Detection Handbook, Sparkov, BankSim, ULB and Elliptic: five synthetic and three real/anonymized sources. Elliptic is an offline cryptocurrency AML benchmark. Observed inputs are not feature attributions. Target labels are excluded from case prompts.

Some classifier test predictions are used for LLM training. This does not change classifier weights, but those cases are not independent LLM test cases. Original application contexts must be screened for overlap before making held-out claims.

Transfer refreshed data and the completed adapter privately:

```bash
python utils/package_pc_inputs.py --adapter checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt
```

Extract the private ZIP into the Windows repository root. Keep it off GitHub. Redistribution permission remains a separate issue; see [distribution status](DISTRIBUTION.md).

## Train a separate refreshed run

Install the completed adapter and its `run_config.json` in `checkpoint/xl_r16_multidata_5060_v1/`, then:

```powershell
python training_model/train_sft.py --config configs/rtx5060_final_models.json
```

Or choose **Train final models RTX 5060** in PyCharm. This writes `xl_r16_final_models_5060_v1`, leaving the completed run untouched. Settings use FP16 base weights, AMP, gradient checkpointing, batch 1 and accumulation 8. Inspect saved data truncation counters before accepting a run.

Resume with:

```powershell
python training_model/train_sft.py --config configs/rtx5060_final_models.json --init_adapter= --resume auto
```

Initial-adapter provenance is restored automatically. For a new reproducibility experiment, add `--deterministic on` and set `$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"` before starting Python. Strict CUDA repeatability remains unverified; no test is silently skipped to claim success.

The historical **Train FraudLM RTX 5060** configuration and `configs/rtx5060_8gb.json` remain for reproducing the completed run, not training refreshed data.

## Chat

```powershell
python chat_multisource.py --adapter checkpoint/xl_r16_multidata_5060_v1/lora_adapter.pt --max-new-tokens 200
```

The default store is `dataset_final_v2/prediction_store.jsonl`. The completed adapter has not been retrained on it. Try:

```text
datasets
facts baf_base
list ieee_cis
```

Copy a real key from `list`, then use `explain <key>` or `ask <key> <question>`. Versioned prepared-data IDs are not raw bank transaction IDs. `facts` reads the registry directly. Store/model/threshold mismatches stop startup.

Every generated answer passes the shared answer guard; case answers also receive contradiction checks. Score-gap calculations use exact decimal arithmetic. Guards are heuristic, not a guarantee of correct or safe advice.

## Evaluate

```bash
python evaluation/compare_original_validation.py --device cpu
python -m pytest tests/test_review_fixes.py tests/test_multisource.py tests/test_multisource_debug.py -q
```

The comparison uses the same original 500 validation rows for both adapters and separately screens original BAF contexts exposed during continued training. It is diagnostic loss, not an independent semantic-quality test. See [grounding status](FINAL_GROUNDING.md) for generation checks and remaining limitations.
