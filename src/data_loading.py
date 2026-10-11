from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "heartcycle_rq1_config.yaml"
KEYS = ["dataset_id", "subject_id", "recording_id", "beat_id"]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["sources"] = {
        key: os.path.expandvars(value) if isinstance(value, str) else value
        for key, value in config.get("sources", {}).items()
    }
    unresolved = [value for value in config.get("sources", {}).values() if isinstance(value, str) and "${" in value]
    if unresolved:
        raise RuntimeError("Set P1_DATA_ROOT or provide a resolved local configuration before accessing external data")
    return config


def output_path(*parts: str) -> Path:
    path = ROOT / "outputs" / Path(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_heartcycle_csv(path: str | Path, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    table = pd.read_csv(path, low_memory=False)
    if "dataset_id" in table:
        table = table.loc[table["dataset_id"].eq(config["project"]["dataset_id"])].copy()
    return table


def load_external_module(name: str, path: str | Path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_tables(config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or load_config()
    paths = config["sources"]
    phase3 = read_heartcycle_csv(paths["phase3_beats"], config)
    phase4 = read_heartcycle_csv(paths["phase4_candidates"], config)
    phase8 = read_heartcycle_csv(paths["phase8_features"], config)
    return phase3, phase4, phase8

