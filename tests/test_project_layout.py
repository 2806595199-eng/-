import importlib
from pathlib import Path


def test_new_package_layout_imports():
    modules = [
        "core.config",
        "core.data_types",
        "core.feature_engineer",
        "training.data_loader",
        "training.data_quality",
        "training.model_exporter",
        "training.gen_sim_data",
        "training.model_registry",
        "training.model_trainer",
        "training.model_update",
        "training.update_scheduler",
        "training.train",
        "serving.cost_calculator",
        "serving.inference_engine",
        "serving.online_history",
        "serving.optimizer",
        "serving.pump_converter",
        "serving.serve",
    ]

    for module in modules:
        assert importlib.import_module(module)


def test_only_cli_compatibility_entrypoints_remain_at_root():
    keep = {"train.py", "serve.py", "gen_sim_data.py"}
    removed = {
        "config.py",
        "data_types.py",
        "feature_engineer.py",
        "data_loader.py",
        "data_quality.py",
        "model_exporter.py",
        "model_registry.py",
        "model_trainer.py",
        "model_update.py",
        "cost_calculator.py",
        "inference_engine.py",
        "optimizer.py",
        "pump_converter.py",
    }

    for filename in keep:
        assert Path(filename).exists()
    for filename in removed:
        assert not Path(filename).exists()


def test_cli_entrypoint_imports_still_work():
    for module in ["train", "serve", "gen_sim_data"]:
        assert importlib.import_module(module)
