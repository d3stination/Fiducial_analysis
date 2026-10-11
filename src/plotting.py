from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .data_loading import ROOT, output_path
from .morphology_analysis import _legacy_api


FACTORS = [
    ("sharpness", "Sharpness"),
    ("local_slope", "Local slope"),
    ("amplitude", "Signal amplitude"),
    ("snr", "Signal-to-noise ratio"),
    ("competing_features", "Competing features"),
    ("beat_to_beat_variability", "Beat-to-beat morphology"),
]


def load_figure_style() -> dict:
    with (ROOT / "configs" / "figure_style.yaml").open(encoding="utf-8") as handle:
        return json.load(handle)


def set_style(style: dict) -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = style["font_family"] if style["font_family"] in available else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": family, "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": style["dpi"],
        "savefig.dpi": style["dpi"], "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(output_path("figures", filename), bbox_inches="tight", dpi=300)
    plt.close(fig)


def _processed_signal(beat: pd.Series, config: dict):
    root, spec, load_recording, preprocess_recording, central_difference = _legacy_api(config)
    recording = load_recording(spec, Path(beat["source_file"]), root)
    processed = preprocess_recording(recording)
    return recording, central_difference(recording.icg, recording.sampling_rate_hz), processed.analysis_dzdt


def _participant_handles(style: dict) -> list[Line2D]:
    return [
        Line2D([0], [0], marker=style["participant_markers"][participant], linestyle="none",
               markerfacecolor=colour, markeredgecolor="#30363A", markeredgewidth=.35,
               markersize=5, label=participant.split(":")[-1])
        for participant, colour in style["participant_colours"].items()
    ]


def figure1(master: pd.DataFrame, phase3: pd.DataFrame, phase4: pd.DataFrame,
            registry: pd.DataFrame, disagreement: pd.DataFrame,
            config: dict, style: dict) -> pd.DataFrame:
    keys = ["subject_id", "recording_id", "beat_id"]
    b = disagreement.loc[disagreement.candidate_type.eq("B"), keys + ["within_beat_range_ms", "all_algorithms_valid"]].rename(
        columns={"within_beat_range_ms": "b_range_ms", "all_algorithms_valid": "b_complete"})
    x = disagreement.loc[disagreement.candidate_type.eq("X"), keys + ["within_beat_range_ms", "all_algorithms_valid"]].rename(
        columns={"within_beat_range_ms": "x_range_ms", "all_algorithms_valid": "x_complete"})
    candidates = master.merge(b, on=keys).merge(x, on=keys)
    complete = candidates.loc[candidates.b_complete & candidates.x_complete]
    major_b = complete.sort_values(["b_range_ms", "x_range_ms"], ascending=False).iloc[0]
    remaining = complete.drop(index=major_b.name)
    major_x = remaining.sort_values(["x_range_ms", "b_range_ms"], ascending=False).iloc[0]

    status = phase4.assign(valid=lambda d: d.detection_status.eq("DETECTED") & d.qc_status.isin(["PASS", "FLAGGED"]))
    failures = status.assign(failed=lambda d: ~d.valid).groupby(keys).failed.sum().reset_index()
    used = pd.DataFrame([major_b[keys].to_dict(), major_x[keys].to_dict()]).assign(used=True)
    fail_key = failures.merge(used, on=keys, how="left").loc[lambda d: d.used.isna()].sort_values("failed", ascending=False).iloc[0]
    failed = candidates.merge(pd.DataFrame([fail_key[keys].to_dict()]), on=keys).iloc[0]

    selected = [
        ("A", "Major B disagreement", major_b, "maximum all-method B range among complete B/X beats"),
        ("B", "Major X disagreement", major_x, "maximum remaining all-method X range among complete B/X beats"),
        ("C", "Failed or ambiguous", failed, "maximum number of missed or invalid candidates"),
    ]
    selection_rows = [{"panel_group": "Clean agreement", "selection_status": "NOT AVAILABLE",
                       "selection_reason": "No prespecified defensible clean-agreement criterion exists; no beat is labelled clean."}]

    set_style(style)
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 9.1))
    fig.subplots_adjust(top=.92, bottom=.18, left=.08, right=.98, hspace=.48, wspace=.16)
    signal = style["signal_styles"]
    for row_index, (panel, category, row, reason) in enumerate(selected):
        beat = phase3.loc[(phase3.subject_id.eq(row.subject_id)) &
                          (phase3.recording_id.eq(row.recording_id)) &
                          (phase3.beat_id.eq(row.beat_id))].iloc[0]
        recording, raw, filtered = _processed_signal(beat, config)
        start = int(beat.beat_start_r_sample_inclusive)
        stop = int(beat.beat_end_next_r_sample_exclusive)
        time_ms = 1000 * (recording.time_s[start:stop] - beat.beat_start_timestamp_s)
        beat_rows = phase4.loc[(phase4.subject_id.eq(row.subject_id)) &
                               (phase4.recording_id.eq(row.recording_id)) &
                               (phase4.beat_id.eq(row.beat_id))]
        for column_index, kind in enumerate(("B", "X")):
            axis = axes[row_index, column_index]
            axis.plot(time_ms, raw[start:stop], color=signal["raw"]["colour"],
                      lw=signal["raw"]["linewidth"], label="Raw-derived dZ/dt")
            axis.plot(time_ms, filtered[start:stop], color=signal["filtered"]["colour"],
                      lw=signal["filtered"]["linewidth"], label="Filtered dZ/dt")
            current = beat_rows.loc[beat_rows.candidate_type.eq(kind)]
            invalid_display = []
            for candidate in current.itertuples(index=False):
                valid = candidate.detection_status == "DETECTED" and candidate.qc_status in {"PASS", "FLAGGED"}
                if not valid or not np.isfinite(candidate.candidate_timestamp_s):
                    invalid_display.append(style["algorithm_display_names"][candidate.candidate_name])
                    continue
                relative_ms = 1000 * (candidate.candidate_timestamp_s - beat.beat_start_timestamp_s)
                y = np.interp(candidate.candidate_timestamp_s, recording.time_s, filtered)
                axis.scatter(relative_ms, y, s=34,
                             marker=style["algorithm_markers"][candidate.candidate_name],
                             color=style["algorithm_colours"][candidate.candidate_name],
                             edgecolor="white", linewidth=.45, zorder=4)
            event = "echo_avo_time_s" if kind == "B" else "echo_avc_time_s"
            eligible = "b_reference_eligibility" if kind == "B" else "x_reference_eligibility"
            reference = "ECHO AVO" if kind == "B" else "ECHO AVC"
            if bool(row[eligible]):
                reference_ms = 1000 * (row[event] - row.r_peak_time_s)
                ref_style = style["reference_styles"][reference]
                axis.axvline(reference_ms, color=ref_style["colour"], ls=ref_style["linestyle"], lw=1.25)
            if row_index == 0:
                local_handles = [
                    Line2D([0], [0], color=signal["raw"]["colour"], lw=signal["raw"]["linewidth"], label="Raw-derived dZ/dt"),
                    Line2D([0], [0], color=signal["filtered"]["colour"], lw=signal["filtered"]["linewidth"], label="Filtered dZ/dt"),
                    Line2D([0], [0], color=style["reference_styles"][reference]["colour"], ls=style["reference_styles"][reference]["linestyle"], label=reference),
                ]
                axis.legend(handles=local_handles, loc="upper right", frameon=False, fontsize=6)
            if category == "Failed or ambiguous" and invalid_display:
                message = "Not detected/invalid: " + ", ".join(invalid_display)
                axis.text(.99, .03, textwrap.fill(message, 34), transform=axis.transAxes,
                          ha="right", va="bottom", fontsize=7, color="#5A3434",
                          bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "edgecolor": "#C9B7B3"})
            axis.set_title(f"{panel}{column_index + 1}  {category} — {kind}-point", loc="left", fontweight="semibold")
            axis.set_xlabel("Time from detected R (ms)")
            axis.set_ylabel("dZ/dt (unverified unit/s)")
            axis.grid(False)
        selection_rows.append({
            "panel_group": category, "selection_status": "SELECTED",
            "subject_id": row.subject_id, "recording_id": row.recording_id,
            "beat_id": row.beat_id, "selection_reason": reason,
            "b_range_ms": row.b_range_ms, "x_range_ms": row.x_range_ms,
            "failed_or_invalid_candidates": int(fail_key.failed) if category == "Failed or ambiguous" else 0,
            "echo_avo_shown": bool(row.b_reference_eligibility),
            "echo_avc_shown": bool(row.x_reference_eligibility),
        })
    algorithm_handles = [
        Line2D([0], [0], marker=style["algorithm_markers"][row.candidate_name], linestyle="none",
               markerfacecolor=style["algorithm_colours"][row.candidate_name], markeredgecolor="white",
               markersize=5, label=style["algorithm_display_names"][row.candidate_name])
        for row in registry.itertuples(index=False)
    ]
    fig.legend(handles=algorithm_handles, loc="lower center", bbox_to_anchor=(.5, .01),
               ncol=8, frameon=False, title="Algorithm display names")
    fig.suptitle("Same-beat B- and X-point localisation", fontsize=12, fontweight="semibold", y=.985)
    save_figure(fig, "figure1_same_beat_overlays.png")
    return pd.DataFrame(selection_rows)


def figure2(participant_accuracy: pd.DataFrame, kind: str, style: dict) -> None:
    set_style(style)
    data = participant_accuracy.loc[participant_accuracy.candidate_type.eq(kind)].copy()
    methods = data.candidate_name.drop_duplicates().tolist()
    fig, axis = plt.subplots(figsize=(10.4, 5.2))
    fig.subplots_adjust(top=.90, bottom=.24, left=.09, right=.98)
    for position, method in enumerate(methods, 1):
        values = data.loc[data.candidate_name.eq(method), "median_absolute_error_ms"].dropna()
        box = axis.boxplot(values, positions=[position], widths=.56, patch_artist=True,
                           showfliers=False, medianprops={"color": "#30363A", "linewidth": 1.1},
                           whiskerprops={"color": "#606A70"}, capprops={"color": "#606A70"})
        box["boxes"][0].set(facecolor=style["algorithm_colours"][method], alpha=.28,
                             edgecolor=style["algorithm_colours"][method], linewidth=1.0)
        rows = data.loc[data.candidate_name.eq(method)].sort_values("subject_id")
        for offset_index, row in enumerate(rows.itertuples(index=False)):
            offset = ((offset_index % 5) - 2) * .025
            axis.scatter(position + offset, row.median_absolute_error_ms, s=25,
                         marker=style["participant_markers"][row.subject_id],
                         color=style["participant_colours"][row.subject_id],
                         edgecolor="#30363A", linewidth=.35, zorder=4)
    axis.set_xticks(range(1, len(methods) + 1), [style["algorithm_display_names"][method] for method in methods])
    axis.set_xlabel(f"{kind}-point algorithm")
    axis.set_ylabel("Participant median absolute timing error (ms)")
    reference = "ECHO AVO" if kind == "B" else "ECHO AVC"
    axis.set_title(f"{kind}-point reference accuracy ({reference})", fontweight="semibold")
    axis.grid(False)
    fig.legend(handles=_participant_handles(style), loc="lower center", bbox_to_anchor=(.5, .01),
               ncol=9, frameon=False, title="Participant")
    save_figure(fig, f"figure2_reference_accuracy_{kind}.png")


def figure3(beat_accuracy: pd.DataFrame, kind: str, style: dict, bins_count: int) -> None:
    set_style(style)
    data = beat_accuracy.loc[beat_accuracy.candidate_type.eq(kind)].copy()
    methods = data.candidate_name.drop_duplicates().tolist()
    values = data.signed_error_ms.to_numpy(dtype=float)
    edges = np.linspace(np.nanmin(values), np.nanmax(values), bins_count + 1)
    ncols = 2 if kind == "B" else 3
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 2.0 * nrows), sharex=True, sharey=True)
    fig.subplots_adjust(top=.91, bottom=.10, left=.075, right=.985, hspace=.34, wspace=.08)
    axes = np.atleast_1d(axes).ravel()
    all_counts = [np.histogram(data.loc[data.candidate_name.eq(method), "signed_error_ms"], bins=edges)[0] for method in methods]
    ymax = max(int(counts.max()) for counts in all_counts)
    centres = (edges[:-1] + edges[1:]) / 2
    colours = np.where(centres < 0, style["signed_error_colours"]["negative"], style["signed_error_colours"]["positive"])
    for axis, method, counts in zip(axes, methods, all_counts, strict=False):
        axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color=colours,
                 edgecolor="white", linewidth=.25)
        axis.axvline(0, color=style["signed_error_colours"]["zero"], lw=1.0)
        axis.set_title(style["algorithm_display_names"][method], fontweight="semibold")
        axis.set_ylim(0, ymax * 1.06)
        axis.grid(False)
    for axis in axes[len(methods):]:
        axis.set_visible(False)
    fig.supxlabel(
        f"Signed {kind}-point error: detected time − {'ECHO AVO' if kind == 'B' else 'ECHO AVC'} (ms)",
        y=.035,
    )
    fig.supylabel("Beat count")
    fig.suptitle(f"{kind}-point signed-error distributions", fontsize=12, fontweight="semibold", y=.992)
    handles = [
        Patch(facecolor=style["signed_error_colours"]["negative"], label="Negative: detected earlier"),
        Patch(facecolor=style["signed_error_colours"]["positive"], label="Positive: detected later"),
        Line2D([0], [0], color=style["signed_error_colours"]["zero"], label="Zero error"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .967), ncol=3, frameon=False)
    fig.text(.5, .006, "Tolerance exceedance is unavailable because no approved threshold is specified.", ha="center", fontsize=8)
    save_figure(fig, f"figure3_signed_error_{kind}.png")


def figure5(participant_disagreement: pd.DataFrame, kind: str, style: dict) -> None:
    set_style(style)
    data = participant_disagreement.loc[participant_disagreement.candidate_type.eq(kind)].sort_values("subject_id")
    measures = [("median_range_ms", "Participant median"), ("p95_range_ms", "Participant 95th percentile")]
    fig, axis = plt.subplots(figsize=(6.8, 5.0))
    fig.subplots_adjust(top=.90, bottom=.29, left=.13, right=.97)
    for position, (column, _) in enumerate(measures, 1):
        values = data[column].dropna()
        box = axis.boxplot(values, positions=[position], widths=.5, patch_artist=True, showfliers=False,
                           medianprops={"color": "#30363A", "linewidth": 1.2},
                           whiskerprops={"color": "#606A70"}, capprops={"color": "#606A70"})
        box["boxes"][0].set(facecolor=style["box_style"]["facecolour"],
                             edgecolor=style["box_style"]["edgecolour"], linewidth=1.0)
        for index, row in enumerate(data.itertuples(index=False)):
            offset = ((index % 5) - 2) * .025
            axis.scatter(position + offset, getattr(row, column), s=28,
                         marker=style["participant_markers"][row.subject_id],
                         color=style["participant_colours"][row.subject_id],
                         edgecolor="#30363A", linewidth=.35, zorder=4)
    axis.set_xticks([1, 2], [label for _, label in measures])
    axis.set_xlabel("Participant summary measure")
    axis.set_ylabel(f"Within-beat {kind}-algorithm range (ms)")
    axis.set_title(f"{kind}-point cross-algorithm disagreement", fontweight="semibold")
    axis.grid(False)
    fig.legend(handles=_participant_handles(style), loc="lower center", bbox_to_anchor=(.5, .01),
               ncol=6, frameon=False, title="Each marker represents one participant")
    save_figure(fig, f"figure5_within_beat_disagreement_{kind}.png")


def _horizontal_category_boxes(axis: plt.Axes, panel: pd.DataFrame, methods: list[str],
                               display: dict, style: dict) -> None:
    for method_index, method in enumerate(methods):
        method_rows = panel.loc[panel.candidate_name.eq(method)]
        categories = [category for category in ("low", "consistent", "high", "variable") if category in set(method_rows.category)]
        for category_index, category in enumerate(categories):
            values = method_rows.loc[method_rows.category.eq(category), "participant_median_outcome"].dropna()
            if values.empty:
                continue
            offset = -.18 if category_index == 0 else .18
            box = axis.boxplot(values, positions=[method_index + offset], widths=.28,
                               vert=False, patch_artist=True, showfliers=False,
                               medianprops={"color": "#30363A", "linewidth": .9},
                               whiskerprops={"color": "#687177", "linewidth": .7},
                               capprops={"color": "#687177", "linewidth": .7})
            box["boxes"][0].set(facecolor=style["category_colours"][category], alpha=.45,
                                 edgecolor=style["category_colours"][category], linewidth=.8)
    axis.set_yticks(range(len(methods)), [display[method] for method in methods])
    axis.invert_yaxis()
    axis.grid(False)


def figure6(morphology_results: pd.DataFrame, registry: pd.DataFrame, kind: str, style: dict) -> None:
    set_style(style)
    data = morphology_results.loc[morphology_results.candidate_type.eq(kind) &
                                  morphology_results.outcome.eq("absolute_error_ms")]
    methods = registry.loc[registry.candidate_type.eq(kind), "candidate_name"].tolist()
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.0))
    fig.subplots_adjust(top=.90, bottom=.07, left=.08, right=.98, hspace=.34, wspace=.12)
    for panel_label, axis, (factor, label) in zip("ABCDEF", axes.ravel(), FACTORS, strict=True):
        panel = data.loc[data.factor.eq(factor)]
        _horizontal_category_boxes(axis, panel, methods, style["algorithm_display_names"], style)
        axis.set_title(f"{panel_label}  {label}", loc="left", fontweight="semibold")
        axis.set_xlabel("Participant median absolute timing error (ms)")
        axis.set_ylabel(f"{kind}-point algorithm")
    fig.suptitle(f"{kind}-point morphology and ECHO-reference error", fontsize=12, fontweight="semibold", y=.992)
    handles = [
        Patch(facecolor=style["category_colours"]["low"], edgecolor=style["category_colours"]["low"],
              alpha=.45, label="Lower / consistent category"),
        Patch(facecolor=style["category_colours"]["high"], edgecolor=style["category_colours"]["high"],
              alpha=.45, label="Higher / variable category"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .962), ncol=2, frameon=False,
               title="Outcome-blind within-participant strata")
    save_figure(fig, f"figure6_morphology_{kind}.png")


def figure6_waveforms(master: pd.DataFrame, phase3: pd.DataFrame, categories: pd.DataFrame,
                      registry: pd.DataFrame, kind: str, config: dict, style: dict) -> pd.DataFrame:
    set_style(style)
    selector = registry.loc[registry.candidate_type.eq(kind), "candidate_name"].iloc[0]
    available = categories.loc[categories.candidate_name.eq(selector) & categories.category.ne("unavailable")]
    event = "echo_avo_time_s" if kind == "B" else "echo_avc_time_s"
    eligible = "b_reference_eligibility" if kind == "B" else "x_reference_eligibility"
    keys = ["subject_id", "recording_id", "beat_id"]
    valid_reference_beats = master.loc[master[eligible].fillna(False), keys].drop_duplicates()
    available = available.merge(valid_reference_beats, on=keys, how="inner")
    fig, axes = plt.subplots(6, 2, figsize=(9.4, 12.4), sharex=False, sharey=True)
    fig.subplots_adjust(top=.94, bottom=.055, left=.085, right=.985, hspace=.52, wspace=.10)
    selection_rows = []
    for row_index, (factor, label) in enumerate(FACTORS):
        panel = available.loc[available.factor.eq(factor)]
        categories_order = [category for category in ("low", "consistent", "high", "variable") if category in set(panel.category)]
        if len(categories_order) != 2:
            categories_order = panel.groupby("category").factor_value.median().sort_values().index.tolist()[:2]
        row_limits = []
        for column_index, category in enumerate(categories_order):
            group = panel.loc[panel.category.eq(category)].copy()
            target = group.factor_value.median()
            chosen = group.loc[(group.factor_value - target).abs().idxmin()]
            beat = phase3.loc[(phase3.subject_id.eq(chosen.subject_id)) &
                              (phase3.recording_id.eq(chosen.recording_id)) &
                              (phase3.beat_id.eq(chosen.beat_id))].iloc[0]
            recording, _, filtered = _processed_signal(beat, config)
            start = int(beat.beat_start_r_sample_inclusive)
            stop = int(beat.beat_end_next_r_sample_exclusive)
            time_ms = 1000 * (recording.time_s[start:stop] - beat.beat_start_timestamp_s)
            waveform = filtered[start:stop]
            scale = waveform.std()
            waveform = (waveform - waveform.mean()) / scale if scale > 0 else waveform - waveform.mean()
            axis = axes[row_index, column_index]
            axis.plot(time_ms, waveform, color=style["signal_styles"]["filtered"]["colour"], lw=1.1)
            master_row = master.loc[(master.subject_id.eq(chosen.subject_id)) &
                                    (master.recording_id.eq(chosen.recording_id)) &
                                    (master.beat_id.eq(chosen.beat_id))].iloc[0]
            reference = "ECHO AVO" if kind == "B" else "ECHO AVC"
            if bool(master_row[eligible]):
                ref_style = style["reference_styles"][reference]
                axis.axvline(1000 * (master_row[event] - master_row.r_peak_time_s),
                             color=ref_style["colour"], ls=ref_style["linestyle"], lw=1.0)
            category_label = category.replace("_", " ").title()
            axis.set_title(f"{label}: {category_label}", fontweight="semibold")
            row_limits.append(float(time_ms.max()))
            axis.grid(False)
            selection_rows.append({"candidate_type": kind, "factor": factor, "category": category,
                                   "subject_id": chosen.subject_id, "recording_id": chosen.recording_id,
                                   "beat_id": chosen.beat_id, "feature_value": chosen.factor_value,
                                   "selection_rule": "reference-eligible beat closest to category median; outcome not used"})
        if row_limits:
            for axis in axes[row_index]:
                axis.set_xlim(0, max(row_limits))
    reference_label = "ECHO AVO" if kind == "B" else "ECHO AVC"
    line_word = "dashed" if kind == "B" else "dotted"
    fig.suptitle(f"{kind}-point morphology/quality examples ({line_word}: {reference_label})", fontsize=12, fontweight="semibold", y=.987)
    fig.supxlabel("Time from detected R (ms)", y=.018)
    fig.supylabel("Standardized filtered dZ/dt (a.u.)", x=.018)
    save_figure(fig, f"figure6_waveforms_{kind}.png")
    return pd.DataFrame(selection_rows)
