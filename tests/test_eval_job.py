"""
Tests for kaggle/eval_job.py (the G5 Kaggle evaluation orchestrator) on CPU with gpt2-small.

A tiny run (3 rows per split) writes B0 / FT / B1 results and figures; a run stopped by the time guard in
the middle of a split keeps the rows done so far, and the next run resumes them instead of starting again.

Author: FraudLM v2 team (COS30049.2-1).

"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PRETRAINED = ROOT / "checkpoint" / "pretrained" / "gpt2-small" / "model.safetensors"
pytestmark = pytest.mark.skipif(not PRETRAINED.exists(), reason="gpt2-small weights not downloaded")


@pytest.fixture(scope="module")
def eval_job():
    spec = importlib.util.spec_from_file_location("eval_job", ROOT / "kaggle" / "eval_job.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def adapter(tmp_path_factory):
    """Dummy gpt2-small adapter (rank 4) with run_config.json and a 3-row train_log.csv."""
    from training_model.lora import apply_lora, lora_state_dict
    from training_model.gpt_model import GPT_CONFIGS, GPTModel

    d = tmp_path_factory.mktemp("ck") / "dummy_r4_v1"
    d.mkdir()
    torch.manual_seed(42)
    m = GPTModel(GPT_CONFIGS["gpt2-small"])
    apply_lora(m, 4, 8)
    torch.save(lora_state_dict(m), d / "lora_adapter.pt")
    (d / "run_config.json").write_text(json.dumps({"size": "gpt2-small", "lora_rank": 4, "lora_alpha": 8,
                                                   "lora_targets": "attn,ff", "base_dtype": "fp16"}))
    (d / "train_log.csv").write_text("step,epoch,train_loss,val_loss,lr,tokens_seen,elapsed_s\n"
                                     "0,0,,2.6,0,0,1\n5,0.5,2.4,2.3,1e-4,100,2\n")
    return d / "lora_adapter.pt"


def _args(adapter, out, *extra):
    return ["--adapter", str(adapter), "--limit", "3", "--out_root", str(out), "--data_dir", str(ROOT / "dataset"),
            "--cache_dir", str(ROOT / "checkpoint" / "pretrained"), "--device", "cpu", "--max_new_tokens", "8",
            *extra]


def test_tiny_run_writes_all_results_and_figures(eval_job, adapter, tmp_path):
    assert eval_job.main(_args(adapter, tmp_path)) == "done"
    res = tmp_path / "results"
    for label in ("B0", "FT-gpt2-small", "B1-gpt2-small"):
        for split in ("test-n3", "demo67-n3"):
            r = json.loads((res / f"dummy_r4_v1_{label}_{split}.json").read_text(encoding="utf-8"))
            assert r["n_rows"] == 3
    ft = json.loads((res / "dummy_r4_v1_FT-gpt2-small_test-n3.json").read_text(encoding="utf-8"))
    assert ft["model_info"]["base_dtype"] == "float32"  # auto on CPU; fp16 on CUDA
    assert ft["model_info"]["lora_rank"] == 4
    assert (tmp_path / "figures" / "loss_curve_dummy_r4_v1.png").exists()
    assert (tmp_path / "figures" / "metrics_bars_dummy_r4_v1_test-n3_all.png").exists()
    assert json.loads((tmp_path / "eval_status.json").read_text())["status"] == "done"


def test_time_limit_mid_split_keeps_rows_and_resumes(eval_job, adapter, tmp_path, monkeypatch, capsys):
    run_eval = eval_job.run_eval
    real = getattr(run_eval, "_unguarded_generate_answer", run_eval.generate_answer)
    calls = {"n": 0}

    def stops_after_one(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise eval_job.TimeLimit()
        return real(*a, **kw)

    monkeypatch.setattr(run_eval, "_unguarded_generate_answer", stops_after_one)
    assert eval_job.main(_args(adapter, tmp_path, "--models", "ft", "--splits", "test")) == "time_limit"
    partial = tmp_path / "results" / "dummy_r4_v1_FT-gpt2-small_test-n3.partial.jsonl"
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 1  # the one finished row is on disk

    monkeypatch.setattr(run_eval, "_unguarded_generate_answer", real)
    capsys.readouterr()
    assert eval_job.main(_args(adapter, tmp_path, "--models", "ft", "--splits", "test")) == "done"
    assert "resume: 1 rows already generated" in capsys.readouterr().out
    assert not partial.exists()
    assert (tmp_path / "results" / "dummy_r4_v1_FT-gpt2-small_test-n3.json").exists()


def test_committed_eval_bundle_is_current():
    spec = importlib.util.spec_from_file_location("make_bundle", ROOT / "kaggle" / "make_bundle.py")
    mb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mb)
    import hashlib
    job, out, mods = mb.BUNDLES["eval"]
    text = (ROOT / out).read_text(encoding="utf-8")
    for rel in mods:
        sha = hashlib.sha256((ROOT / rel).read_text(encoding="utf-8").encode("utf-8")).hexdigest()[:12]
        assert f"(sha256 {sha})" in text, f"{out} is stale for {rel}; re-run kaggle/make_bundle.py"
    assert text.endswith(mb.prepare_job(ROOT / job)), f"{out} is stale for {job}; re-run kaggle/make_bundle.py"
    assert '"evaluation")]' in text  # re-runs in one notebook session forget old evaluation modules
