"""Read-only loaders for HeartCycle, ReBeatICG, and IITB recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np
from scipy.io import loadmat

from .manifest import DatasetSpec
from .schema import Recording, ReferenceSeries


@dataclass
class _HeartCycleChannel:
    index: int
    label: str
    data: np.ndarray
    time: np.ndarray | None
    rate_text: str


def _octave_raw(node: h5py.Group | h5py.Dataset) -> np.ndarray:
    """Extract the payload from an Octave-style HDF5 field."""

    if isinstance(node, h5py.Group):
        if "value" not in node:
            raise ValueError("Octave field has no value payload")
        node = node["value"]
    return np.asarray(node[()])


def _octave_text(node: h5py.Group | h5py.Dataset) -> str:
    """Decode an Octave string field while rejecting non-text payloads."""

    raw = _octave_raw(node).reshape(-1)
    if raw.dtype.kind in {"S", "U"}:
        return "".join(
            item.decode("utf-8", errors="replace") if isinstance(item, bytes)
            else str(item) for item in raw
        ).strip("\x00")
    codes = [int(value) for value in raw]
    if any(value < 0 or value > 0x10FFFF for value in codes):
        return ""
    return "".join(chr(value) for value in codes).strip("\x00")


def _heartcycle_channels(handle: h5py.File) -> list[_HeartCycleChannel]:
    root = handle.get("measure/value")
    if not isinstance(root, h5py.Group):
        raise ValueError("HeartCycle file lacks measure/value")
    channels: list[_HeartCycleChannel] = []
    for key, node in root.items():
        if not re.fullmatch(r"_\d+", key) or not isinstance(node, h5py.Group):
            continue
        value = node.get("value")
        if not isinstance(value, h5py.Group) or "label" not in value or "data" not in value:
            continue
        label = _octave_text(value["label"])
        data = np.asarray(_octave_raw(value["data"])).squeeze()
        time = None
        if "time" in value:
            time = np.asarray(_octave_raw(value["time"]), dtype=float).squeeze()
        rate_text = _octave_text(value["Rate"]) if "Rate" in value else ""
        channels.append(_HeartCycleChannel(int(key[1:]), label, data, time, rate_text))
    return sorted(channels, key=lambda channel: channel.index)


def _adjacent_channel(
    by_index: dict[int, _HeartCycleChannel], anchor: int, offset: int, label: str,
) -> _HeartCycleChannel | None:
    channel = by_index.get(anchor + offset)
    return channel if channel is not None and channel.label.upper() == label else None


def _add_heartcycle_reference(
    references: dict[str, ReferenceSeries], name: str,
    channel: _HeartCycleChannel | None, path: Path, source: str,
    coordinate_system: str, unit: str, note: str = "",
) -> None:
    if channel is None:
        return
    references[name] = ReferenceSeries(
        np.asarray(channel.data).reshape(-1), unit, coordinate_system,
        f"{source}:channel_{channel.index:03d}", path, note,
    )


def load_heartcycle(spec: DatasetSpec, path: Path, project_root: Path) -> Recording:
    """Load one HeartCycle HDF5 file using its native ECG/IMP timestamps."""

    del project_root
    warnings: list[str] = []
    with h5py.File(path, "r") as handle:
        channels = _heartcycle_channels(handle)
    by_index = {channel.index: channel for channel in channels}
    impedance = [channel for channel in channels if channel.label.upper() == "IMP"]
    if len(impedance) != 1:
        raise ValueError(f"expected one IMP channel, found {len(impedance)}")
    imp = impedance[0]
    imp_data = np.asarray(imp.data, dtype=float).reshape(-1)
    imp_time = np.asarray(imp.time, dtype=float).reshape(-1) if imp.time is not None else np.array([])
    ecg_candidates = [
        channel for channel in channels
        if channel.label.upper() == "ECG"
        and np.asarray(channel.data).size == imp_data.size
        and channel.time is not None
        and np.asarray(channel.time).size == imp_time.size
        and np.allclose(np.asarray(channel.time).reshape(-1), imp_time, rtol=0, atol=1e-9)
    ]
    if len(ecg_candidates) != 1:
        raise ValueError(f"expected one ECG aligned to IMP, found {len(ecg_candidates)}")
    ecg = np.asarray(ecg_candidates[0].data, dtype=float).reshape(-1)
    if imp_time.size < 2 or not np.all(np.diff(imp_time) > 0):
        raise ValueError("HeartCycle IMP timestamps are unavailable or invalid")
    sampling_rate = float(1.0 / np.median(np.diff(imp_time)))
    nominal_rate = float(imp.rate_text) if imp.rate_text else float("nan")
    if np.isfinite(nominal_rate) and abs(sampling_rate - nominal_rate) / nominal_rate > 0.01:
        warnings.append("heartcycle_nominal_timestamp_rate_difference_over_1pct")

    references: dict[str, ReferenceSeries] = {}
    primary_labels = (
        (1, "RPEAKS", "r_time_s", "time_s", "s"),
        (2, "AVO", "avo_time_s", "time_s", "s"),
        (3, "PEP", "pep_labelled_ms", "measurement", "ms"),
        (4, "AVC", "avc_time_s", "time_s", "s"),
        (5, "LVET", "lvet_labelled_ms", "measurement", "ms"),
    )
    for offset, label, suffix, coordinates, unit in primary_labels:
        channel = _adjacent_channel(by_index, imp.index, offset, label)
        _add_heartcycle_reference(
            references, f"heartcycle_ecg_imp_{suffix}", channel, path,
            "heartcycle_ecg_imp", coordinates, unit,
            "Original labelled field; retained as reference/QC only.",
        )
        if channel is None:
            warnings.append(f"heartcycle_ecg_imp_{label.lower()}_missing")

    echo_channels = [channel for channel in channels if channel.label.upper() == "ECHO"]
    if len(echo_channels) != 1:
        warnings.append(f"heartcycle_echo_block_count_{len(echo_channels)}")
    else:
        echo = echo_channels[0]
        echo_labels = (
            (1, "RPEAKS", "r_time_s", "time_s", "s"),
            (2, "AVO", "avo_time_s", "time_s", "s"),
            (3, "PEP", "pep_labelled_ms", "measurement", "ms"),
            (4, "AVC", "avc_time_s", "time_s", "s"),
            (5, "LVET", "lvet_labelled_ms", "measurement", "ms"),
        )
        for offset, label, suffix, coordinates, unit in echo_labels:
            channel = _adjacent_channel(by_index, echo.index, offset, label)
            _add_heartcycle_reference(
                references, f"heartcycle_echo_{suffix}", channel, path,
                "heartcycle_echo", coordinates, unit,
                "Separate ECHO time base; timestamp matching is required before comparison.",
            )
            if channel is None:
                warnings.append(f"heartcycle_echo_{label.lower()}_missing")

    match = re.match(r"(CH\d+)_", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("HeartCycle filename does not contain a CH subject identifier")
    source_subject = match.group(1).upper()
    return Recording(
        spec.dataset_id, spec.dataset_name, spec.cohort, spec.condition,
        spec.paired_group, f"heartcycle:{source_subject}", source_subject,
        path.stem, path, imp_time, ecg, imp_data, sampling_rate,
        "heartcycle_imp_timestamp_median", "source_timestamps", "impedance",
        references=references,
        metadata={
            "nominal_sampling_rate_hz": nominal_rate,
            "ecg_channel_index": ecg_candidates[0].index,
            "imp_channel_index": imp.index,
            "signal_unit": "unverified_device_unit",
        },
        warnings=tuple(warnings),
    )


def load_rebeat(spec: DatasetSpec, path: Path, project_root: Path) -> Recording:
    """Load one raw ReBeat recording and retain optional annotations unchanged."""

    source = loadmat(path, squeeze_me=True)
    missing = [name for name in ("ECG", "ICG", "samplFreq") if name not in source]
    if missing:
        raise ValueError(f"ReBeat MAT file missing: {', '.join(missing)}")
    ecg = np.asarray(source["ECG"], dtype=float).reshape(-1)
    icg = np.asarray(source["ICG"], dtype=float).reshape(-1)
    sampling_rate = float(np.asarray(source["samplFreq"]).item())
    if ecg.size != icg.size:
        raise ValueError("ReBeat ECG and ICG lengths differ")
    match = re.search(r"Subject_(\d+)", path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("ReBeat filename lacks Subject_<number>")
    subject_number = int(match.group(1))
    source_subject = f"Subject_{subject_number}"
    references: dict[str, ReferenceSeries] = {}
    warnings: list[str] = []
    annotation_path = None
    if spec.annotation_directory is not None:
        annotation_path = (
            Path(project_root) / spec.annotation_directory
            / f"Annotat_Subject_{subject_number}_task_{spec.task_code}.mat"
        )
    if annotation_path is not None and annotation_path.is_file():
        annotation = loadmat(annotation_path, squeeze_me=True)
        points = np.asarray(annotation.get("annotPoints"))
        if points.ndim != 2 or points.shape[1] < 4:
            warnings.append("rebeat_annotation_invalid_shape")
        else:
            points = np.asarray(points[:, :4], dtype=float)
            if not np.all(np.isfinite(points)):
                warnings.append("rebeat_annotation_contains_nan_or_inf")
            if np.any(points < 0) or np.any(points > ecg.size):
                warnings.append("rebeat_annotation_outside_possible_index_bounds")
            ordered = (points[:, 0] < points[:, 1]) & (points[:, 1] < points[:, 2])
            if not np.all(ordered):
                warnings.append("rebeat_annotation_b_c_x_order_violation")
            for column, name in enumerate(("expert_B", "expert_C", "expert_X", "expert_O")):
                references[name] = ReferenceSeries(
                    points[:, column], "sample",
                    "matlab_sample_index_unresolved", "rebeat_expert_annotation",
                    annotation_path,
                    "Original index retained; 0-based versus 1-based origin is unresolved.",
                )
    else:
        warnings.append("rebeat_annotation_missing")
    time_s = np.arange(ecg.size, dtype=float) / sampling_rate
    return Recording(
        spec.dataset_id, spec.dataset_name, spec.cohort, spec.condition,
        spec.paired_group, f"rebeat:Subject_{subject_number:02d}", source_subject,
        path.stem, path, time_s, ecg, icg, sampling_rate,
        "mat_file:samplFreq", "generated_from_native_rate", "dzdt",
        references=references,
        metadata={
            "annotation_path": str(annotation_path) if annotation_path else "",
            "annotation_index_origin": "unresolved",
            "signal_unit": "unverified",
        },
        warnings=tuple(warnings),
    )


def load_iitb(spec: DatasetSpec, path: Path, project_root: Path) -> Recording:
    """Load one IITB matrix using the July 2025 database documentation."""

    del project_root
    source = loadmat(path, squeeze_me=True)
    if "data" not in source:
        raise ValueError("IITB MAT file lacks data matrix")
    data = np.asarray(source["data"], dtype=float)
    if data.ndim != 2 or data.shape[1] < 5:
        raise ValueError(f"IITB data must be N x >=5, found {data.shape}")
    source_subject = path.stem.split("_", 1)[0].upper()
    namespace = "iitb_cvd" if spec.dataset_id == "iitb_cvd" else "iitb"
    sampling_rate = 500.0
    warnings = (
        "iitb_digitized_signal_units_unverified",
        "iitb_column_5_undocumented",
        "iitb_doppler_timebase_not_sample_synchronized",
        "iitb_legacy_c_max_applied_to_documented_negative_dzdt",
    )
    return Recording(
        spec.dataset_id, spec.dataset_name, spec.cohort, spec.condition,
        spec.paired_group, f"{namespace}:{source_subject}", source_subject,
        path.stem, path, np.arange(data.shape[0], dtype=float) / sampling_rate,
        data[:, 3], data[:, 2], sampling_rate,
        "dataset_documentation:IITB-ICG-Database-Description_2025jul08.pdf",
        "generated_from_documented_500_hz_rate",
        "dzdt",
        extra_signals={
            "iitb_basal_impedance_z0": data[:, 0],
            "iitb_negative_impedance_variation": data[:, 1],
            "iitb_column_5_undocumented": data[:, 4],
        },
        metadata={
            "documented_channel_mapping": (
                "column_1=Z0;column_2=-z(t);column_3=-dz/dt;"
                "column_4=ECG;column_5=undocumented"
            ),
            "icg_signal_label": "-dz/dt",
            "icg_sign_convention": "documented_negative_preserved_not_inverted",
            "source_column_count": int(data.shape[1]),
            "digitized_signal_units": "unverified",
            "expert_bx_annotations": "unavailable",
            "doppler_reference_alignment": "independent_timebase_visual_ecg_rr_alignment",
        },
        warnings=warnings,
    )


def load_recording(spec: DatasetSpec, path: Path, project_root: Path) -> Recording:
    """Dispatch a manifest entry to its format-specific read-only loader."""

    loaders = {
        "heartcycle": load_heartcycle,
        "rebeat": load_rebeat,
        "iitb": load_iitb,
    }
    try:
        loader = loaders[spec.loader]
    except KeyError as error:
        raise ValueError(f"unsupported loader: {spec.loader}") from error
    return loader(spec, Path(path), Path(project_root))

