from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_loading import load_config, load_external_module


def build_reference_table(config: dict | None = None) -> pd.DataFrame:
    """Reuse the inspected original builder while preserving every alignment field."""
    config = config or load_config()
    import src as active_src
    legacy_src = str(Path(config["sources"]["variability_root"]) / "src")
    if legacy_src not in active_src.__path__:
        active_src.__path__.append(legacy_src)
    builder = load_external_module("heartcycle_reference_builder", config["sources"]["reference_builder"])
    before = builder.source_snapshot()
    inventory, phase4, beats = builder.load_base_beats()
    beats, b_names, x_names = builder.add_candidates(beats, phase4)
    beats = builder.add_echo_references(beats)
    checks = builder.validate(beats, inventory, b_names, x_names, before)
    failed = [name for name, (passed, _) in checks.items() if not passed]
    if failed:
        raise RuntimeError("Reference validation failed: " + "; ".join(failed))
    valid = config["analysis"]["reference_alignment_status"]
    beats["b_reference_eligibility"] = beats["echo_avo_alignment_status"].eq(valid)
    beats["x_reference_eligibility"] = beats["echo_avc_alignment_status"].eq(valid)
    beats["reference_validation_checks"] = ";".join(checks)
    return beats

