# FraudLM v2: RTX 5060 continued-training edition

For the RTX 5060 8 GB workflow, eight-project prediction data, PyCharm configurations, training and
multi-source chat, start with **[README_RTX5060.md](README_RTX5060.md)**.

This copy keeps the original team's custom GPT-2 XL + LoRA implementation. The original developer
quickstart remains below for reference.

Developer quickstart so team members can run the chat demo. This is **not** the submitted README.
GPT-2 (own PyTorch code) + OpenAI GPT-2 XL weights + LoRA r16 adapter, run `xl_r16_v1`.
Full chat notes: `README_chat_dev.md`. Numbers to quote: `evaluation/EVAL_NOTES.md` only.

## 1. Setup (Python 3.11)

```bash
git clone https://github.com/HienThong-vie/fraudlm-v2.git
cd fraudlm-v2

# conda
conda env create -f environment.yml
conda activate fraudlm2
# or venv + pip
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Download the trained adapter (89 MB, GitHub Release `xl_r16_v1`)

```bash
gh release download xl_r16_v1 -R HienThong-vie/fraudlm-v2 -p lora_adapter.pt -D checkpoint/xl_r16_v1
```

Without `gh`:
`curl -L -o checkpoint/xl_r16_v1/lora_adapter.pt https://github.com/HienThong-vie/fraudlm-v2/releases/download/xl_r16_v1/lora_adapter.pt`.
Or download `lora_adapter.pt` from the Releases page and put it in `checkpoint/xl_r16_v1/`
(next to `run_config.json`, which is already in the repo).

## 3. Team classifier repo (required for the fact lines)

The fact lines come from the EDA tables (in this repo) plus the team model card (in Kha's repo, frozen commit
d0fc662). Without this step the chat still starts, but it has no fact lines at all and some tests fail.

```bash
git clone https://github.com/KennyDao1324/ai4cyber-fraud preprocessed/_ml_repo
git -C preprocessed/_ml_repo checkout d0fc662
```

## 4. Run the chat

```bash
python chat.py --size gpt2-xl --adapter checkpoint/xl_r16_v1/lora_adapter.pt
```

- First run downloads the OpenAI GPT-2 XL weights (about 6.2 GB) into `checkpoint/pretrained/`.
- Needs about 7 GB free RAM; add `--dtype bf16` to halve it. About 15-22 s per answer on CPU.
  With an NVIDIA GPU and CUDA torch: `--device cuda`.
- Commands: `list flagged`, `explain <app_id>`, `help`, `quit`. Anything else is a question.
- The chat only knows the 600 demo applications (test months 6-7) in `preprocessed/ml_scores_months6_7_demo.csv`
  and the fact lines it shows. It cannot open files you paste in; it should say a question is not in its facts.
- Decision support only: it never approves, rejects or blocks an application.

## 5. Tests

```bash
python -m pytest tests -q
```

## Data licences

`dataset/SOURCES.md`: BAF (CC BY-NC-SA 4.0), Bitext (CDLA-Sharing-1.0), Alpaca-Cleaned (CC BY 4.0, upstream
CC BY-NC 4.0). Non-commercial coursework use only.
