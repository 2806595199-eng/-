from pathlib import Path
import types

from core.tabpfn_runtime import configure_tabpfn_cache, resolve_tabpfn_device


def test_configure_tabpfn_cache_sets_project_local_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)

    cache_dir = configure_tabpfn_cache(tmp_path / "tabpfn_cache")

    assert cache_dir == (tmp_path / "tabpfn_cache").resolve()
    assert Path(cache_dir).exists()
    assert Path(cache_dir).name == "tabpfn_cache"


def test_resolve_tabpfn_device_auto_prefers_cuda(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert resolve_tabpfn_device("auto") == "cuda"


def test_resolve_tabpfn_device_auto_falls_back_to_cpu(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    assert resolve_tabpfn_device("auto") == "cpu"
