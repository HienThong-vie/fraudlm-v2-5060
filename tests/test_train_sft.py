"""
Tests for how training_model/train_sft.py ends a run (CPU only, no model needed).

On Kaggle an early stop (time limit, NaN abort) must exit 0, because a script kernel that exits
non-zero is marked as a failed version and its output may not reach the next (resume) version.
Locally the exit codes 3 (time limit) and 2 (NaN abort) stay so shell scripts can tell them apart.
The last log line must say why the run stopped and what run_summary.json / progress.json record.

Author: FraudLM v2 team (COS30049.2-1).

"""

import json
from pathlib import Path

import pytest

from training_model import train_sft


def _run_dir(tmp_path, status, step=1200):
    (tmp_path / "progress.json").write_text(json.dumps({"step": step, "total_steps": 3000}))
    (tmp_path / "run_summary.json").write_text(json.dumps({"status": status, "steps": step}))
    return tmp_path


@pytest.fixture
def local(monkeypatch):
    """Pretend we are not on Kaggle, whatever machine runs the tests."""
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists",
                        lambda self: False if self.as_posix() == "/kaggle/working" else real_exists(self))


@pytest.mark.parametrize("result,code,name", [("done", 0, "done"), ("time_limit", 3, "time_limit"),
                                              ("nan", 2, "nan_abort")])
def test_local_exit_codes_unchanged(tmp_path, capsys, local, result, code, name):
    assert train_sft.finish(result, _run_dir(tmp_path, name)) == code
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert last.startswith(f"[stop] reason={name}") and f"exit {code}" in last


@pytest.mark.parametrize("result,name", [("time_limit", "time_limit"), ("nan", "nan_abort"), ("done", "done")])
def test_kaggle_always_exits_0_and_says_why(tmp_path, capsys, monkeypatch, result, name):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert train_sft.finish(result, _run_dir(tmp_path, name)) == 0
    last = capsys.readouterr().out.strip().splitlines()[-1]
    assert f"reason={name}" in last
    assert f"run_summary.json status={name}" in last
    assert "last saved step 1200/3000" in last
    assert "exit 0" in last and "Kaggle" in last


def test_missing_status_files_do_not_crash(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert train_sft.finish("nan", tmp_path) == 0
    assert "run_summary.json status=missing" in capsys.readouterr().out


def test_kaggle_detection_uses_env_var(monkeypatch, local):
    assert not train_sft.in_kaggle_kernel()
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert train_sft.in_kaggle_kernel()
