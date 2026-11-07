"""Download the public instruction sets used for the `banking_qa` and `general` tasks.

Files go to preprocessed/raw/<name>/ (gitignored). Plain HTTPS downloads from the public Hugging Face hub,
no account or token needed (and none is read). Licences were checked on the dataset cards on 2026-11-06.

    bitext_retail_banking : Bitext-retail-banking-llm-chatbot-training-dataset, CDLA-Sharing-1.0, 25,545 rows
    alpaca_cleaned        : yahma/alpaca-cleaned, CC-BY-4.0, 51,760 rows

Considered and not used (see dataset/SOURCES.md): Josephgflowers/Finance-Instruct-500k (Apache-2.0; most
fraud-related questions refer to a missing text, so too noisy), gbharti/finance-alpaca (forum answers,
licence provenance unclear), databricks-dolly-15k (CC-BY-SA-3.0, ShareAlike).

Run from llm_ver2/:  .venv/Scripts/python -m preprocess_data.fetch_public

Author: FraudLM v2 team (COS30049.2-1).
"""

from __future__ import annotations

import urllib.request

from preprocess_data import common as C

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
SETS = {
    "bitext_retail_banking": ("bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
                              "bitext-retail-banking-llm-chatbot-training-dataset.csv", "bitext.csv"),
    "alpaca_cleaned": ("yahma/alpaca-cleaned", "alpaca_data_cleaned.json", "alpaca_data_cleaned.json"),
}


def main() -> None:
    for name, (repo, remote, local) in SETS.items():
        dest = C.RAW / name / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"{name}: already downloaded ({dest.stat().st_size:,} bytes, sha256 {C.sha256_file(dest)[:16]}...)")
            continue
        url = HF.format(repo=repo, file=remote)
        print(f"{name}: downloading {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"{name}: {dest.stat().st_size:,} bytes, sha256 {C.sha256_file(dest)}")


if __name__ == "__main__":
    main()
