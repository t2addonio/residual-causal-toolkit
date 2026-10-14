import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from rct_paths import artifacts_dir  # noqa: E402


def test_cli_flag_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("RCT_ARTIFACTS", str(tmp_path / "env"))
    out = artifacts_dir(tmp_path / "cli")
    assert out == tmp_path / "cli" and out.is_dir()


def test_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("RCT_ARTIFACTS", str(tmp_path / "env"))
    out = artifacts_dir()
    assert out == tmp_path / "env" and out.is_dir()


def test_default_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("RCT_ARTIFACTS", raising=False)
    monkeypatch.chdir(tmp_path)
    out = artifacts_dir()
    assert out.resolve() == (tmp_path / "artifacts").resolve() and out.is_dir()
