# FraudLM v2

GPT-2 XL with rank-16 LoRA for explaining fraud-classifier outputs. The model supports analyst review; it does not predict transactions independently, make customer decisions or prove fraud.

## Start here

- [This branch's device setup](DEVICE_GUIDE.md)
- [RTX 5060 setup and commands](README_RTX5060.md)
- [Current grounding and evaluation status](FINAL_GROUNDING.md)
- [Data and model distribution status](DISTRIBUTION.md)
- [Original adapter evaluation](evaluation/EVAL_NOTES.md)

Open the repository folder in PyCharm. Install Python and a CUDA-compatible PyTorch build on the PC, then install `requirements-rtx5060.txt`. Run `python utils/check_cuda.py` before training.

## Project layout

| Path | Contents |
|---|---|
| `training_model/` | GPT-2 architecture, LoRA, supervised fine-tuning |
| `preprocess_data/` | Data preparation and multi-source SFT builder |
| `configs/` | Current prediction registry and training configurations |
| `utils/` | Prompt formatting, safety checks and private transfer packaging |
| `evaluation/` | Evaluation scripts, protocols and aggregate results |
| `chat_multisource.py` | Prediction lookup and guarded chat |
| `.run/` | Shared PyCharm run configurations |
| `checkpoint/` | Local weights and training state; not bundled in Git |
| `dataset_final_v2/` | Local refreshed training data and prediction store; Git-ignored |

## Current model

The completed RTX run is `xl_r16_multidata_5060_v1`: 6,016 steps and final validation loss 0.0589 on a mostly templated set. That loss must not be compared directly with the original adapter's loss on a different dataset. Small context checks do not establish improved generalisation or retention of all original skills.

The current registry uses one frozen classifier per dataset. Model facts are read at runtime, not recalled from adapter weights. The completed adapter still contains older training facts; it has not been retrained on the refreshed data. The separate `rtx5060_final_models.json` configuration is for a future run.

## Sources and limitations

Architecture and training references include Sebastian Raschka's *Build a Large Language Model (From Scratch)* and the [companion repository](https://github.com/rasbt/LLMs-from-scratch). GPT-2 pretrained weights originate from OpenAI. Source, dataset and licence records remain in `dataset/SOURCES.md` and the existing project documentation.

The sources have different redistribution conditions. Private data transfer is not permission to redistribute. The continued model Release is temporarily unpublished pending review. Neither the adapter nor its rule-based guard is suitable for autonomous financial decisions.
