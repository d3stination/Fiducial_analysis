"""Generate reproducible Phase 1 inventory and loading-QC tables."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import platform
import sys

import h5py
import numpy as np
import scipy

from . import __version__
from .loaders import load_recording
from .manifest import MANIFEST_VERSION, dataset_manifest
from .schema import Recording


INVENTORY_COLUMNS = (
    "dataset_id", "dataset_name", "cohort", "condition", "paired_group",
    "subject_id", "source_subject_id", "recording_id", "source_file",
    "file_format", "loader", "load_status", "warning_codes", "has_ecg",
    "has_icg", "icg_kind", "sample_count", "duration_s",
    "sampling_rate_hz", "sampling_rate_source", "timestamp_source",
    "timestamp_sampling_rate_hz", "annotation_available", "reference_names",
    "reference_sources",
)

LOADING_COLUMNS = (
    "dataset_id", "cohort", "condition", "source_file", "load_status",
    "warning_codes", "error_codes", "error_type", "error_message",
    "subject_id", "recording_id",
)

SUMMARY_COLUMNS = (
    "dataset_id", "dataset_name", "cohort", "condition", "paired_group",
    "expected_files", "discovered_files", "loaded_files", "warning_files",
    "failed_files", "expected_subjects", "discovered_subjects",
    "manifest_status", "issue_codes",
)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def inventory_row(recording: Recording, load_status: str) -> dict:
    """Create one machine-readable inventory row from a validated recording."""

    reference_sources = sorted({reference.source for reference in recording.references.values()})
    return {
        "dataset_id": recording.dataset_id,
        "dataset_name": recording.dataset_name,
        "cohort": recording.cohort,
        "condition": recording.condition,
        "paired_group": recording.paired_group,
        "subject_id": recording.subject_id,
        "source_subject_id": recording.source_subject_id,
        "recording_id": recording.recording_id,
        "source_file": str(recording.source_path),
        "file_format": recording.source_path.suffix.lower().lstrip("."),
        "loader": recording.metadata.get("loader", recording.dataset_name),
        "load_status": load_status,
        "warning_codes": ";".join(recording.warnings),
        "has_ecg": recording.ecg.size == recording.sample_count,
        "has_icg": recording.icg.size == recording.sample_count,
        "icg_kind": recording.icg_kind,
        "sample_count": recording.sample_count,
        "duration_s": f"{recording.duration_s:.9f}",
        "sampling_rate_hz": f"{recording.sampling_rate_hz:.9f}",
        "sampling_rate_source": recording.sampling_rate_source,
        "timestamp_source": recording.timestamp_source,
        "timestamp_sampling_rate_hz": f"{recording.timestamp_sampling_rate_hz:.9f}",
        "annotation_available": any(
            reference.source == "rebeat_expert_annotation"
            for reference in recording.references.values()
        ),
        "reference_names": ";".join(sorted(recording.references)),
        "reference_sources": ";".join(reference_sources),
    }


def run_inventory(project_root: Path, output_dir: Path) -> dict[str, int]:
    """Load every manifest file and write Phase 1 inventory/QC outputs."""

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    inventory_rows: list[dict] = []
    loading_rows: list[dict] = []
    summary_rows: list[dict] = []

    for spec in dataset_manifest(project_root):
        paths = spec.discover(project_root)
        loaded = 0
        warning_files = 0
        failures = 0
        subjects: set[str] = set()
        for path in paths:
            try:
                recording = load_recording(spec, path, project_root)
                recording.metadata["loader"] = spec.loader
                errors = recording.validation_errors()
                if errors:
                    failures += 1
                    loading_rows.append({
                        "dataset_id": spec.dataset_id,
                        "cohort": spec.cohort,
                        "condition": spec.condition,
                        "source_file": str(path),
                        "load_status": "FAILED",
                        "warning_codes": ";".join(recording.warnings),
                        "error_codes": ";".join(errors),
                        "error_type": "ValidationError",
                        "error_message": "; ".join(errors),
                        "subject_id": recording.subject_id,
                        "recording_id": recording.recording_id,
                    })
                    continue
                status = "WARNING" if recording.warnings else "PASSED"
                loaded += 1
                warning_files += bool(recording.warnings)
                subjects.add(recording.subject_id)
                inventory_rows.append(inventory_row(recording, status))
                loading_rows.append({
                    "dataset_id": spec.dataset_id,
                    "cohort": spec.cohort,
                    "condition": spec.condition,
                    "source_file": str(path),
                    "load_status": status,
                    "warning_codes": ";".join(recording.warnings),
                    "error_codes": "",
                    "error_type": "",
                    "error_message": "",
                    "subject_id": recording.subject_id,
                    "recording_id": recording.recording_id,
                })
            except Exception as error:  # Per-file failures must remain visible.
                failures += 1
                loading_rows.append({
                    "dataset_id": spec.dataset_id,
                    "cohort": spec.cohort,
                    "condition": spec.condition,
                    "source_file": str(path),
                    "load_status": "FAILED",
                    "warning_codes": "",
                    "error_codes": "loader_exception",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "subject_id": "",
                    "recording_id": "",
                })
        issues = []
        if not spec.directory(project_root).is_dir():
            issues.append("source_directory_missing")
        if len(paths) != spec.expected_files:
            issues.append("file_count_mismatch")
        if len(subjects) != spec.expected_subjects:
            issues.append("subject_count_mismatch")
        if failures:
            issues.append("loading_failures_present")
        summary_rows.append({
            "dataset_id": spec.dataset_id,
            "dataset_name": spec.dataset_name,
            "cohort": spec.cohort,
            "condition": spec.condition,
            "paired_group": spec.paired_group,
            "expected_files": spec.expected_files,
            "discovered_files": len(paths),
            "loaded_files": loaded,
            "warning_files": warning_files,
            "failed_files": failures,
            "expected_subjects": spec.expected_subjects,
            "discovered_subjects": len(subjects),
            "manifest_status": "PASSED" if not issues else "FAILED",
            "issue_codes": ";".join(issues),
        })

    _write_csv(output_dir / "recording_inventory.csv", INVENTORY_COLUMNS, inventory_rows)
    _write_csv(output_dir / "loading_qc.csv", LOADING_COLUMNS, loading_rows)
    _write_csv(output_dir / "dataset_inventory_qc.csv", SUMMARY_COLUMNS, summary_rows)
    aggregate = {
        "manifest_groups": len(summary_rows),
        "discovered_files": len(loading_rows),
        "loaded_files": len(inventory_rows),
        "warning_files": sum(row["load_status"] == "WARNING" for row in loading_rows),
        "failed_files": sum(row["load_status"] == "FAILED" for row in loading_rows),
        "passed_manifest_groups": sum(row["manifest_status"] == "PASSED" for row in summary_rows),
    }
    metadata = {
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "project_root": str(project_root),
        "output_directory": str(output_dir),
        "manifest_version": MANIFEST_VERSION,
        "pipeline_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "h5py": h5py.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "aggregate": aggregate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase1_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )
    return aggregate

