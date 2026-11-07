"""
Smoke tests for the trainer's code (CPU only, a few seconds).

G2 part: the Kaggle bundle builds into valid Python that carries the current repo modules, and the
feasibility measurement loop runs a real LoRA step on a tiny model (finite loss, only LoRA params
change, base weights untouched).
G4 part: train_sft.py on gpt2-small with 32 real training rows for 10 steps: val loss falls, only the
adapter is saved, a run killed after step 7 resumes from the step-5 checkpoint and gives the same
loss trajectory and weights as an uninterrupted run, and the --max_hours guard saves and resumes.

Author: FraudLM v2 team (COS30049.2-1).

"""

import hashlib
import json
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    """Import a script from kaggle/ (that folder is not a package) by file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bundle_builds_and_carries_current_modules(tmp_path):
    make_bundle = _load(ROOT / "kaggle" / "make_bundle.py", "make_bundle")
    out = tmp_path / "bundle.py"
    make_bundle.build(ROOT / "kaggle" / "feasibility.py", out)
    text = out.read_text(encoding="utf-8")
    compile(text, str(out), "exec")
    assert "REPO_ONLY" not in text  # the repo sys.path block must be stripped
    for rel in make_bundle.MODULES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
        assert f"# ---- {rel}  (sha256 {sha}) ----" in text, f"{rel} missing or stale in the bundle"


def test_committed_bundle_is_current():
    """Fails when a bundled module changed after the bundle was made: run kaggle/make_bundle.py again."""
    make_bundle = _load(ROOT / "kaggle" / "make_bundle.py", "make_bundle")
    text = (ROOT / "kaggle" / "feasibility_bundle.py").read_text(encoding="utf-8")
    for rel in make_bundle.MODULES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
        assert f"(sha256 {sha})" in text, f"kaggle/feasibility_bundle.py is stale for {rel}; re-run make_bundle.py"
    job = make_bundle.prepare_job(ROOT / "kaggle" / "feasibility.py")
    assert text.endswith(job), "kaggle/feasibility_bundle.py is stale for feasibility.py; re-run make_bundle.py"


def test_feasibility_measure_runs_one_lora_step_on_tiny_model():
    feas = _load(ROOT / "kaggle" / "feasibility.py", "feasibility")
    from training_model.gpt_model import GPTModel
    from training_model.lora import apply_lora, lora_state_dict

    torch.manual_seed(42)
    cfg = {"vocab_size": 50257, "context_length": 64, "drop_rate": 0.0, "qkv_bias": True,
           "emb_dim": 32, "n_layers": 2, "n_heads": 2}
    model = GPTModel(cfg)
    base_before = {k: v.clone() for k, v in model.state_dict().items()}
    apply_lora(model, rank=4, alpha=8)
    init = lora_state_dict(model)
    tokens = feas.get_tokens()

    res = feas.measure(model, init, tokens, torch.device("cpu"), None, ckpt=True,
                       micro_batch=2, seq_len=32, steps=2, warmup=1)
    assert res["finite"] and res["tokens_per_s"] > 0

    after = model.state_dict()
    # Base weights are frozen: unchanged after the optimiser steps (keys are renamed by the wrapper).
    for k, v in base_before.items():
        k2 = k if k in after else k.replace(".weight", ".linear.weight").replace(".bias", ".linear.bias")
        assert torch.equal(after[k2], v), f"base weight {k} changed"
    # LoRA B started at zero and must have moved.
    assert any(after[k].abs().sum() > 0 for k in init if k.endswith(".B"))


# ---------------------------------------------------------------------------------------------
# G4: training loop smoke tests (gpt2-small on CPU, 32 real training examples)
# ---------------------------------------------------------------------------------------------

import csv  # noqa: E402

import pytest  # noqa: E402

PRETRAINED = ROOT / "checkpoint" / "pretrained" / "gpt2-small" / "model.safetensors"
needs_small = pytest.mark.skipif(not PRETRAINED.exists(), reason="gpt2-small weights not downloaded")


def _args(out_dir, run_id, *extra):
    return ["--size", "gpt2-small", "--cache_dir", str(ROOT / "checkpoint" / "pretrained"),
            "--out_dir", str(out_dir), "--run_id", run_id, "--resume", "",
            "--max_train_rows", "32", "--max_val_rows", "16", "--eval_rows", "8", "--max_steps", "10",
            "--micro_batch", "2", "--grad_accum", "1", "--lr", "3e-4",
            "--log_every", "1", "--eval_every", "5", "--save_every", "5", "--sample_tokens", "5", *extra]


def _log(run_dir):
    with open(run_dir / "train_log.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def train_sft():
    from training_model import train_sft as mod
    return mod


@pytest.fixture(scope="module")
def uninterrupted(tmp_path_factory, train_sft):
    out = tmp_path_factory.mktemp("full")
    assert train_sft.main(_args(out, "full")) == "done"
    return out / "full"


@needs_small
def test_training_runs_and_val_loss_falls(uninterrupted):
    rows = _log(uninterrupted)
    assert [r["step"] for r in rows] == [str(i) for i in range(11)]
    # PLAN 3 contract columns first, then the two GradScaler columns (G4 review R7).
    assert list(rows[0]) == ["step", "epoch", "train_loss", "val_loss", "lr", "tokens_seen", "elapsed_s",
                             "grad_scale", "amp_skipped"]
    val0, val10 = float(rows[0]["val_loss"]), float(rows[10]["val_loss"])
    print(f"\n[G4 smoke] val subset loss step 0 {val0:.4f} -> step 10 {val10:.4f}; "
          f"train loss first 3 {[r['train_loss'] for r in rows[1:4]]} last 3 {[r['train_loss'] for r in rows[8:]]}")
    assert val10 < val0
    for f in ("lora_adapter.pt", "lora_adapter_best.pt", "train_state.pt", "run_config.json",
              "samples.txt", "run_summary.json"):
        assert (uninterrupted / f).exists(), f
    # Only the adapter is saved, never the 124M base.
    sd = torch.load(uninterrupted / "lora_adapter.pt")
    assert all(".lora." in k for k in sd) and sum(v.numel() for v in sd.values()) == 2_654_208
    assert (uninterrupted / "samples.txt").read_text(encoding="utf-8").count("[explain_score]") == 3


@needs_small
def test_resume_after_kill_continues_identically(tmp_path, train_sft, uninterrupted):
    with pytest.raises(train_sft.SimulatedKill):
        train_sft.main(_args(tmp_path, "r", "--debug_crash_at_step", "7"))  # last save was at step 5
    assert train_sft.main(_args(tmp_path, "r", "--resume")) == "done"
    a, b = _log(uninterrupted), _log(tmp_path / "r")
    assert [r["step"] for r in b] == [r["step"] for r in a]  # no duplicate or missing rows
    for ra, rb in zip(a, b):
        for col in ("train_loss", "val_loss", "lr", "tokens_seen"):
            if ra[col]:
                assert abs(float(ra[col]) - float(rb[col])) <= 2e-4, (ra["step"], col, ra[col], rb[col])
    wa = torch.load(uninterrupted / "lora_adapter.pt")
    wb = torch.load(tmp_path / "r" / "lora_adapter.pt")
    assert max((wa[k] - wb[k]).abs().max().item() for k in wa) < 1e-5


@needs_small
def test_time_guard_saves_and_resumes(tmp_path, train_sft):
    args = _args(tmp_path, "t", "--max_steps", "3", "--eval_at_start", "off")
    assert train_sft.main(args + ["--max_hours", "1e-9"]) == "time_limit"
    assert (tmp_path / "t" / "train_state.pt").exists()
    assert json.loads((tmp_path / "t" / "run_summary.json").read_text())["status"] == "time_limit"
    with pytest.raises(SystemExit):  # a fresh start must not overwrite an existing checkpoint
        train_sft.main(args)
    assert train_sft.main(args + ["--resume"]) == "done"
    assert [r["step"] for r in _log(tmp_path / "t")] == ["1", "2", "3"]


def test_committed_train_bundle_is_current():
    make_bundle = _load(ROOT / "kaggle" / "make_bundle.py", "make_bundle")
    job, out, mods = make_bundle.BUNDLES["train"]
    text = (ROOT / out).read_text(encoding="utf-8")
    for rel in mods:
        src = (ROOT / rel).read_text(encoding="utf-8")
        sha = hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
        assert f"(sha256 {sha})" in text, f"{out} is stale for {rel}; re-run kaggle/make_bundle.py"
    assert text.endswith(make_bundle.prepare_job(ROOT / job)), \
        f"{out} is stale for {job}; re-run kaggle/make_bundle.py"


# ---------------------------------------------------------------------------------------------
# G4 code review fixes R1-R4, R5
# ---------------------------------------------------------------------------------------------

@needs_small
def test_train_state_loads_on_cpu_with_uint8_rng_and_holds_the_adapter(uninterrupted, train_sft):
    """R1: RNG states must come back as CPU ByteTensors (set_rng_state rejects anything else, so
    map_location="cuda" broke every GPU resume). R5: the adapter lives inside train_state.pt."""
    state = train_sft.load_train_state(uninterrupted / "train_state.pt")
    rng = state["rng_cpu"]
    assert rng.device.type == "cpu" and rng.dtype == torch.uint8
    torch.set_rng_state(rng)  # must not raise
    adapter = torch.load(uninterrupted / "lora_adapter.pt")
    assert set(state["lora"]) == set(adapter)
    assert all(torch.equal(state["lora"][k], adapter[k]) for k in adapter)


def test_load_train_state_moves_rng_to_cpu(tmp_path, train_sft):
    """R1 without a GPU: a state whose RNG tensor was saved as another dtype/device comes back CPU uint8."""
    path = tmp_path / "train_state.pt"
    torch.save({"rng_cpu": torch.get_rng_state().to(torch.int16), "rng_cuda": None}, path)
    state = train_sft.load_train_state(path)
    assert state["rng_cpu"].device.type == "cpu" and state["rng_cpu"].dtype == torch.uint8
    torch.set_rng_state(state["rng_cpu"])


def test_start_val_loss_guard(train_sft):
    """R3: a broken base (NaN or a loss far above pretrained GPT-2's ~2-3) stops before training."""
    train_sft.check_start_val_loss(2.6)
    for bad in (float("nan"), float("inf"), 6.0):
        with pytest.raises(SystemExit):
            train_sft.check_start_val_loss(bad)


def test_run_id_follows_size(train_sft):
    """R4: if G2 picks Large, the default run id (and the checkpoint folder) changes with it."""
    assert train_sft.parse_config([])["run_id"] == "xl_r16_v1"
    assert train_sft.parse_config(["--size", "gpt2-large"])["run_id"] == "large_r16_v1"
    assert train_sft.parse_config(["--run_id", "mine"])["run_id"] == "mine"


def test_data_hash_mismatch_stops_the_run(tmp_path, train_sft, monkeypatch):
    """R2: a dataset that differs from the bundled hash stops before any model is loaded."""
    monkeypatch.setattr(train_sft, "EXPECTED_DATA_SHA256", "0000000000000000-0000000000000000")
    with pytest.raises(SystemExit, match="differs from the one this bundle was built with"):
        train_sft.main(_args(tmp_path, "m"))


def test_bundle_bakes_the_current_data_hash(tmp_path):
    """R2: the training bundle carries the sha of today's dataset/ files (built fresh into tmp)."""
    make_bundle = _load(ROOT / "kaggle" / "make_bundle.py", "make_bundle")
    job, _, mods = make_bundle.BUNDLES["train"]
    out = tmp_path / "train_bundle.py"
    make_bundle.build(ROOT / job, out, mods)
    text = out.read_text(encoding="utf-8")
    assert f'EXPECTED_DATA_SHA256 = "{make_bundle.repo_data_sha256()}"' in text
    assert "BUNDLE_FILLS_DATA_SHA" not in text


def test_committed_train_bundle_has_current_data_hash():
    """R2: the committed kaggle/train_bundle.py matches dataset/ (fails after a data rebuild until the
    lead re-runs make_bundle.py; then the Kaggle dataset needs a New Version as well)."""
    make_bundle = _load(ROOT / "kaggle" / "make_bundle.py", "make_bundle")
    text = (ROOT / "kaggle" / "train_bundle.py").read_text(encoding="utf-8")
    assert f'EXPECTED_DATA_SHA256 = "{make_bundle.repo_data_sha256()}"' in text
