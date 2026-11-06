import json

import pandas as pd
import pytest
import torch

from preprocess_data.build_multisource_sft import balanced_sample, make_example, read_source
from chat_multisource import MultiSourceSession
from training_model.gpt_model import GPTModel
from training_model.lora import apply_lora, lora_state_dict
from training_model.train_sft import load_initial_adapter, parse_config
from utils.multisource_prompt import format_prediction_context, guard_prediction_answer

TINY = {"vocab_size": 100, "context_length": 16, "emb_dim": 16, "n_heads": 4,
        "n_layers": 1, "drop_rate": 0.0, "qkv_bias": True}


def case(score=0.8):
    return {"case_key": "demo:7", "source_id": "demo", "case_id": "7", "dataset_name": "Demo",
            "dataset_kind": "synthetic", "model": "test model", "score": score, "threshold": 0.6,
            "signals": {}}


def test_context_has_facts_but_no_target_label():
    text = format_prediction_context(case())
    assert "Model score: 0.800000" in text and "flagged for review" in text
    assert "target" not in text.lower() and "ground truth" not in text.lower()
    example = make_example(case())
    assert example["source"] == "demo" and example["task"] == "multisource_prediction"


def test_guard_replaces_contradiction_and_automatic_decision():
    safe, changed = guard_prediction_answer("This case is not flagged.", case())
    assert changed and "flagged for review" in safe
    safe, changed = guard_prediction_answer("Reject the application.", case())
    assert changed and "not proof" in safe
    raw = "The case is flagged for review, so an analyst should investigate it."
    assert guard_prediction_answer(raw, case()) == (raw, False)


def test_balanced_sample_is_reproducible():
    frame = pd.DataFrame({"_score": [0.1] * 20 + [0.9] * 20})
    a = balanced_sample(frame, 0.5, 10, 42)
    b = balanced_sample(frame, 0.5, 10, 42)
    assert a.index.tolist() == b.index.tolist()
    assert (a["_score"] >= 0.5).value_counts().to_dict() == {False: 5, True: 5}


def test_reader_ignores_label_column(tmp_path):
    pd.DataFrame({"target": [1, 0], "fraud_probability": [0.8, 0.2]}).to_csv(tmp_path / "p.csv", index=False)
    source = {"id": "x", "name": "X", "kind": "synthetic", "prediction_file": "p.csv",
              "id_column": "__row__", "score_column": "fraud_probability", "threshold": 0.5,
              "model": "m", "extra_columns": []}
    rows = read_source(source, tmp_path, 0, 42)
    assert len(rows) == 2 and all("target" not in json.dumps(row).lower() for row in rows)


def test_resume_and_init_adapter_are_mutually_exclusive():
    with pytest.raises(SystemExit, match="cannot be used together"):
        parse_config(["--resume", "auto", "--init_adapter", "adapter.pt"])


def test_initial_adapter_round_trip(tmp_path):
    first = GPTModel(TINY)
    apply_lora(first, 2, 4)
    with torch.no_grad():
        for name, parameter in first.named_parameters():
            if name.endswith(".lora.B"):
                parameter.fill_(0.25)
    path = tmp_path / "lora_adapter.pt"
    torch.save(lora_state_dict(first), path)
    (tmp_path / "run_config.json").write_text(json.dumps({
        "size": "gpt2-small", "lora_rank": 2, "lora_alpha": 4, "lora_targets": "attn,ff"}))
    fresh = GPTModel(TINY)
    apply_lora(fresh, 2, 4)
    cfg = {"size": "gpt2-small", "lora_rank": 2, "lora_alpha": 4, "lora_targets": "attn,ff"}
    digest = load_initial_adapter(fresh, path, cfg)
    assert len(digest) == 64
    assert all(torch.equal(lora_state_dict(first)[k], lora_state_dict(fresh)[k])
               for k in lora_state_dict(first))


def test_multisource_session_commands_and_grounding():
    output = []
    calls = []

    def generate(question, context):
        calls.append((question, context))
        return "This case is flagged for review and needs analyst investigation.", {"prompt_tokens": 10, "new_tokens": 8}

    session = MultiSourceSession({"demo:7": case()}, generate, output.append)
    assert session.handle("datasets")
    assert session.handle("list demo")
    assert session.handle("explain demo:7")
    assert calls and "Model score: 0.800000" in calls[0][1]
    assert any("Decision support only" in line for line in output)
    assert not session.handle("quit")
