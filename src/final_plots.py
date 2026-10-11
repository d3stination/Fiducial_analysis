from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .final_analysis import KEYS, participant_balanced_limits, selected_pair
from .morphology_analysis import _legacy_api
from .plotting import load_figure_style, set_style


FACTORS = [
    ("sharpness", "Sharpness"), ("slope", "Slope"), ("amplitude", "Amplitude"),
    ("snr", "SNR"), ("competing_features", "Competing"),
    ("beat_to_beat_variability", "Beat variability"),
]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _participant_handles(style: dict) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
               markerfacecolor=colour, markeredgecolor="#30363A", markeredgewidth=.35,
               label=participant.split(":")[-1])
        for participant, colour in style["participant_colours"].items()
    ]


def _recording_signals(row: pd.Series, config: dict):
    root, spec, load_recording, preprocess_recording, central_difference = _legacy_api(config)
    recording = load_recording(spec, Path(row.source_file), root)
    processed = preprocess_recording(recording)
    raw_dzdt = central_difference(recording.icg, recording.sampling_rate_hz)
    start, stop = int(row.r_peak_sample_index), int(row.next_r_peak_sample_index)
    time_ms = 1000 * (recording.time_s[start:stop] - row.r_peak_time_s)
    return recording, processed, raw_dzdt, start, stop, time_ms


def select_waveform_examples(master: pd.DataFrame, b_status: pd.DataFrame,
                             x_status: pd.DataFrame) -> pd.DataFrame:
    def spread(status: pd.DataFrame, label: str) -> pd.DataFrame:
        valid = status.loc[status.valid].pivot(index=KEYS, columns="algorithm_id", values="candidate_time_from_r_ms")
        expected = status.algorithm_id.nunique()
        output = pd.DataFrame({f"{label}_valid": valid.notna().sum(axis=1)})
        output[f"{label}_range_ms"] = valid.max(axis=1) - valid.min(axis=1)
        output[f"{label}_complete"] = output[f"{label}_valid"].eq(expected)
        return output.reset_index()

    b = spread(b_status, "B")
    x = spread(x_status, "X")
    summary = master[KEYS].merge(b, on=KEYS).merge(x, on=KEYS)
    complete = summary.loc[summary.B_complete & summary.X_complete].copy()
    complete["agreement_score"] = complete.B_range_ms.rank(pct=True) + complete.X_range_ms.rank(pct=True)
    typical = complete.sort_values(["agreement_score", "B_range_ms", "X_range_ms"]).iloc[0]
    remaining = complete.loc[~complete.beat_id.eq(typical.beat_id)]
    major_b = remaining.sort_values(["B_range_ms", "X_range_ms"], ascending=False).iloc[0]
    remaining = remaining.loc[~remaining.beat_id.eq(major_b.beat_id)]
    major_x = remaining.sort_values(["X_range_ms", "B_range_ms"], ascending=False).iloc[0]
    failures = pd.concat([
        b_status.assign(failed=~b_status.valid & b_status.attempted),
        x_status.assign(failed=~x_status.valid & x_status.attempted),
    ]).groupby(KEYS, observed=True).failed.sum().reset_index()
    used = {typical.beat_id, major_b.beat_id, major_x.beat_id}
    failure = failures.loc[~failures.beat_id.isin(used)].sort_values("failed", ascending=False).iloc[0]
    failure = summary.merge(pd.DataFrame([failure[KEYS].to_dict()]), on=KEYS).iloc[0]
    rows = []
    for category, row, rule in (
        ("Typical low disagreement", typical, "minimum combined percentile rank of complete B and X ranges"),
        ("Major B disagreement", major_b, "maximum complete-beat B range after excluding the typical example"),
        ("Major X disagreement", major_x, "maximum complete-beat X range after excluding prior examples"),
        ("Failed or ambiguous", failure, "maximum number of attempted missed or invalid detections after excluding prior examples"),
    ):
        values = row[KEYS + ["B_range_ms", "X_range_ms"]].to_dict()
        values.update({"category": category, "selection_rule": rule})
        rows.append(values)
    return pd.DataFrame(rows)


def figure_f01(kind: str, master: pd.DataFrame, status: pd.DataFrame, registry: pd.DataFrame,
               selection: pd.DataFrame, config: dict, output: Path) -> dict:
    style = load_figure_style()
    set_style(style)
    reference = "ECHO AVO" if kind == "B" else "ECHO AVC"
    event = "echo_avo_time_s" if kind == "B" else "echo_avc_time_s"
    eligible = "b_reference_eligibility" if kind == "B" else "x_reference_eligibility"
    fig, axes = plt.subplots(4, 2, figsize=(11.2, 10.4))
    fig.subplots_adjust(top=.90, bottom=.13, hspace=.58, wspace=.18)
    for row_index, selected in selection.iterrows():
        row = master.merge(pd.DataFrame([selected[KEYS].to_dict()]), on=KEYS).iloc[0]
        recording, processed, raw_dzdt, start, stop, time_ms = _recording_signals(row, config)
        ecg = recording.ecg[start:stop]
        filtered_ecg = processed.filtered_ecg[start:stop]
        axes[row_index, 0].plot(time_ms, ecg, color="#B8BDC1", lw=.65, label="Raw ECG")
        axes[row_index, 0].plot(time_ms, filtered_ecg, color="#344F60", lw=1.0, label="Filtered ECG")
        axes[row_index, 0].set_ylabel("ECG amplitude\n(device units)")
        axes[row_index, 0].set_title(f"{chr(65 + row_index)}1  {selected.category} — ECG", loc="left", fontweight="semibold")
        axes[row_index, 1].plot(time_ms, raw_dzdt[start:stop], color="#B8BDC1", lw=.65, label="Raw-derived dZ/dt")
        axes[row_index, 1].plot(time_ms, processed.analysis_dzdt[start:stop], color="#263F4F", lw=1.1, label="Filtered dZ/dt")
        beat_status = status.loc[(status.subject_id.eq(row.subject_id)) &
                                 (status.recording_id.eq(row.recording_id)) &
                                 (status.beat_id.eq(row.beat_id))]
        for candidate in beat_status.loc[beat_status.valid].itertuples(index=False):
            y = np.interp(candidate.candidate_timestamp_s, recording.time_s, processed.analysis_dzdt)
            axes[row_index, 1].scatter(candidate.candidate_time_from_r_ms, y, s=28,
                                       marker=style["algorithm_markers"][candidate.candidate_name],
                                       color=style["algorithm_colours"][candidate.candidate_name],
                                       edgecolor="white", linewidth=.35, zorder=4)
        if bool(row[eligible]):
            ref = style["reference_styles"][reference]
            axes[row_index, 1].axvline(1000 * (row[event] - row.r_peak_time_s), color=ref["colour"],
                                       linestyle=ref["linestyle"], lw=1.15)
        axes[row_index, 1].set_ylabel("dZ/dt\n(device units/s)")
        axes[row_index, 1].set_title(f"{chr(65 + row_index)}2  {selected.category} — {kind}-point", loc="left", fontweight="semibold")
        for axis in axes[row_index]:
            axis.set_xlabel("Time from ECG R (ms)")
            axis.grid(False)
    axes[0, 0].legend(frameon=False, loc="upper right")
    axes[0, 1].legend(handles=[
        Line2D([0], [0], color="#B8BDC1", lw=.65, label="Raw-derived dZ/dt"),
        Line2D([0], [0], color="#263F4F", lw=1.1, label="Filtered dZ/dt"),
        Line2D([0], [0], color=style["reference_styles"][reference]["colour"],
               linestyle=style["reference_styles"][reference]["linestyle"], label=reference),
    ], frameon=False, loc="upper right")
    algorithm_handles = [
        Line2D([0], [0], marker=style["algorithm_markers"][row.candidate_name], linestyle="none",
               markerfacecolor=style["algorithm_colours"][row.candidate_name], markeredgecolor="white",
               markersize=5, label=row.algorithm_id)
        for row in registry.loc[registry.candidate_type.eq(kind)].itertuples(index=False)
    ]
    fig.legend(handles=algorithm_handles, ncol=len(algorithm_handles), loc="lower center",
               bbox_to_anchor=(.5, .025), frameon=False, title=f"{kind}-point algorithms")
    fig.suptitle(f"{kind}-point same-beat ECG–ICG localisation", fontsize=12, fontweight="semibold", y=.985)
    _save(fig, output)
    return {"selection": selection.copy()}


def figure_f02(kind: str, participant: pd.DataFrame, registry: pd.DataFrame, output: Path) -> dict:
    style = load_figure_style()
    set_style(style)
    methods = registry.loc[registry.candidate_type.eq(kind), "algorithm_id"].tolist()
    fig, axis = plt.subplots(figsize=(10.4, 5.2))
    fig.subplots_adjust(top=.90, bottom=.25, left=.09, right=.98)
    for position, method in enumerate(methods, 1):
        rows = participant.loc[participant.algorithm_id.eq(method)].sort_values("subject_id")
        box = axis.boxplot(rows.median_absolute_error_ms.dropna(), positions=[position], widths=.55,
                           patch_artist=True, showfliers=False,
                           medianprops={"color": "#30363A", "linewidth": 1.1},
                           whiskerprops={"color": "#606A70"}, capprops={"color": "#606A70"})
        box["boxes"][0].set(facecolor="#E6EAED", edgecolor="#4D5961")
        for index, row in enumerate(rows.itertuples(index=False)):
            axis.scatter(position + ((index % 5) - 2) * .024, row.median_absolute_error_ms,
                         marker="o", s=25, color=style["participant_colours"][row.subject_id],
                         edgecolor="#30363A", linewidth=.35, zorder=4)
    axis.set_xticks(range(1, len(methods) + 1), methods)
    axis.set_xlabel(f"{kind}-point algorithm")
    axis.set_ylabel("Participant median absolute timing error (ms)")
    axis.set_title(f"{kind}-point accuracy relative to {'ECHO AVO' if kind == 'B' else 'ECHO AVC'}", fontweight="semibold")
    axis.grid(False)
    fig.legend(handles=_participant_handles(style), ncol=9, frameon=False, loc="lower center",
               bbox_to_anchor=(.5, .01), title="Participant (circular markers)")
    _save(fig, output)
    summary = participant.groupby("algorithm_id").median_absolute_error_ms.agg(
        median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75)
    ).reset_index()
    return {"summary": summary}


def figure_f03(kind: str, errors: pd.DataFrame, registry: pd.DataFrame, output: Path) -> dict:
    style = load_figure_style()
    set_style(style)
    methods = registry.loc[registry.candidate_type.eq(kind), "algorithm_id"].tolist()
    values = errors.signed_error_ms.to_numpy(float)
    edges = np.linspace(np.nanmin(values), np.nanmax(values), 31)
    ncols = 2 if kind == "B" else 3
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 2.0 * nrows), sharex=True, sharey=True)
    fig.subplots_adjust(top=.91, bottom=.08, left=.075, right=.985, hspace=.34, wspace=.08)
    axes = np.atleast_1d(axes).ravel()
    counts = [np.histogram(errors.loc[errors.algorithm_id.eq(method), "signed_error_ms"], bins=edges)[0] for method in methods]
    ymax = max(int(count.max()) for count in counts)
    centres = (edges[:-1] + edges[1:]) / 2
    colours = np.where(centres < 0, style["signed_error_colours"]["negative"], style["signed_error_colours"]["positive"])
    for axis, method, count in zip(axes, methods, counts, strict=False):
        axis.bar(edges[:-1], count, width=np.diff(edges), align="edge", color=colours, edgecolor="white", linewidth=.25)
        axis.axvline(0, color=style["signed_error_colours"]["zero"], lw=1.0)
        axis.set_title(method, fontweight="semibold")
        axis.set_ylim(0, ymax * 1.06)
        axis.grid(False)
    for axis in axes[len(methods):]:
        axis.set_visible(False)
    fig.supxlabel(f"Signed timing error: detected {kind} − {'ECHO AVO' if kind == 'B' else 'ECHO AVC'} (ms)")
    fig.supylabel("Beat count")
    fig.suptitle(f"{kind}-point signed timing-error distributions", fontsize=12, fontweight="semibold", y=.992)
    fig.legend(handles=[
        Patch(facecolor=style["signed_error_colours"]["negative"], label="Negative: detected earlier"),
        Patch(facecolor=style["signed_error_colours"]["positive"], label="Positive: detected later"),
        Line2D([0], [0], color=style["signed_error_colours"]["zero"], label="Zero error"),
    ], loc="upper center", bbox_to_anchor=(.5, .967), ncol=3, frameon=False)
    _save(fig, output)
    summary = errors.groupby("algorithm_id").agg(
        beats=("beat_id", "size"), participants=("subject_id", "nunique"),
        median_signed_error_ms=("signed_error_ms", "median"),
        signed_error_iqr_ms=("signed_error_ms", lambda x: x.quantile(.75) - x.quantile(.25)),
        p90_absolute_error_ms=("absolute_error_ms", lambda x: x.quantile(.90)),
        p95_absolute_error_ms=("absolute_error_ms", lambda x: x.quantile(.95)),
    ).reset_index()
    return {"summary": summary}


def figure_f04(kind: str, status: pd.DataFrame, registry: pd.DataFrame, output: Path) -> dict:
    style = load_figure_style()
    set_style(style)
    data = status.loc[status.attempted].groupby("algorithm_id", observed=True).agg(
        denominator=("beat_id", "size"), valid=("valid", "sum"), missed=("missed", "sum"), invalid=("invalid", "sum")
    ).reindex(registry.loc[registry.candidate_type.eq(kind), "algorithm_id"]).reset_index()
    for column in ("valid", "missed", "invalid"):
        data[f"{column}_pct"] = 100 * data[column] / data.denominator
    fig, axis = plt.subplots(figsize=(10.2, 5.0))
    fig.subplots_adjust(top=.84)
    positions = np.arange(len(data))
    axis.bar(positions, data.valid_pct, color="#557A95", label="Valid")
    axis.bar(positions, data.missed_pct, bottom=data.valid_pct, color="#C2A14A", label="Missed")
    axis.bar(positions, data.invalid_pct, bottom=data.valid_pct + data.missed_pct, color="#A65A4E", label="Invalid")
    axis.set_xticks(positions, data.algorithm_id)
    axis.set_xlabel(f"{kind}-point algorithm")
    axis.set_ylabel("Detection rate among attempted beats (%)")
    axis.set_ylim(0, 100)
    fig.suptitle(f"{kind}-point detection performance", fontweight="semibold", y=.98)
    fig.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(.5, .935))
    axis.grid(False)
    _save(fig, output)
    return {"summary": data}


def figure_f05(kind: str, status: pd.DataFrame, participant: pd.DataFrame,
               registry: pd.DataFrame, output: Path) -> dict:
    style = load_figure_style()
    set_style(style)
    participant_once = participant.drop_duplicates("subject_id").sort_values("subject_id")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    fig.subplots_adjust(top=.88, bottom=.24, wspace=.28)
    measures = [("participant_median_range_ms", "Median"), ("participant_p95_range_ms", "95th percentile")]
    for position, (column, _) in enumerate(measures, 1):
        box = axes[0].boxplot(participant_once[column], positions=[position], widths=.5, patch_artist=True,
                              showfliers=False, medianprops={"color": "#30363A", "linewidth": 1.1})
        box["boxes"][0].set(facecolor="#E6EAED", edgecolor="#4D5961")
        for index, row in enumerate(participant_once.itertuples(index=False)):
            axes[0].scatter(position + ((index % 5) - 2) * .025, getattr(row, column), marker="o", s=26,
                            color=style["participant_colours"][row.subject_id], edgecolor="#30363A", linewidth=.35)
    axes[0].set_xticks([1, 2], [label for _, label in measures])
    axes[0].set_xlabel("Participant summary of within-beat range")
    axes[0].set_ylabel(f"Across-algorithm {kind}-point range (ms)")
    axes[0].set_title("A  All-method disagreement", loc="left", fontweight="semibold")
    pair_a, pair_b = selected_pair(kind)
    complete = status.loc[status.valid].pivot(index=KEYS, columns="algorithm_id", values="candidate_time_from_r_ms")
    complete = complete.dropna()
    pair = complete[[pair_a, pair_b]].copy().reset_index()
    pair["mean_timing_ms"] = pair[[pair_a, pair_b]].mean(axis=1)
    pair["difference_ms"] = pair[pair_a] - pair[pair_b]
    for subject, rows in pair.groupby("subject_id"):
        axes[1].scatter(rows.mean_timing_ms, rows.difference_ms, marker="o", s=12,
                        color=style["participant_colours"][subject], alpha=.62, edgecolor="none")
    bias, low, high = participant_balanced_limits(pair)
    axes[1].axhline(bias, color="#30363A", lw=1.1)
    axes[1].axhline(low, color="#777777", lw=1.0, ls="--")
    axes[1].axhline(high, color="#777777", lw=1.0, ls="--")
    axes[1].set_xlabel(f"Mean {pair_a}/{pair_b} timing from ECG R (ms)")
    axes[1].set_ylabel(f"{pair_a} − {pair_b} timing (ms)")
    axes[1].set_title("B  Selected-pair agreement", loc="left", fontweight="semibold")
    for axis in axes:
        axis.grid(False)
    fig.legend(handles=_participant_handles(style), ncol=9, frameon=False, loc="lower center",
               bbox_to_anchor=(.5, .01), title="Participant (circular markers)")
    fig.suptitle(f"{kind}-point within-beat disagreement and agreement", fontsize=12, fontweight="semibold")
    _save(fig, output)
    return {"pair": pair, "bias": bias, "loa_low": low, "loa_high": high, "participant": participant_once}


def _effect_matrix(morphology: pd.DataFrame, outcome: str) -> pd.DataFrame:
    methods = sorted(morphology.algorithm_id.drop_duplicates(), key=lambda value: int(value[1:]))
    matrix = pd.DataFrame(index=methods, columns=[label for _, label in FACTORS], dtype=float)
    for factor, label in FACTORS:
        category = f"{factor}_category"
        rows = morphology.loc[morphology[category].ne("unavailable") & morphology.analysis_status.ne("INELIGIBLE")].copy()
        if outcome == "error":
            rows = rows.loc[rows.strict_common_reference & rows.valid].dropna(subset=["absolute_error_ms"])
            part = rows.groupby(["subject_id", "algorithm_id", category], observed=True).absolute_error_ms.median().reset_index(name="value")
        else:
            rows["failed"] = rows.missed | rows.invalid
            part = rows.groupby(["subject_id", "algorithm_id", category], observed=True).failed.mean().mul(100).reset_index(name="value")
        levels = [level for level in ("low", "consistent", "high", "variable") if level in set(part[category])]
        if len(levels) != 2:
            continue
        med = part.groupby(["algorithm_id", category], observed=True).value.median().unstack(category)
        effect = med[levels[1]] - med[levels[0]]
        matrix.loc[effect.index, label] = effect
    return matrix


def _morphology_examples(morphology: pd.DataFrame) -> pd.DataFrame:
    unique = morphology.loc[morphology.reference_eligible].drop_duplicates(KEYS).copy()
    feature_columns = [
        "reference_local_sharpness_abs_d2_device_units_s3", "signal_to_noise_ratio_db",
        "reference_region_competing_features_count", "beat_to_beat_shape_nrmse",
    ]
    standardized = unique.groupby("subject_id")[feature_columns].transform(
        lambda x: (x - x.median()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1)
    )
    unique["clear_score"] = standardized.iloc[:, 0] + standardized.iloc[:, 1] - standardized.iloc[:, 2] - standardized.iloc[:, 3]
    clear = unique.loc[unique.clear_score.idxmax()]
    ambiguous = unique.loc[unique.clear_score.idxmin()]
    return pd.DataFrame([
        {**clear[KEYS].to_dict(), "example": "Clear reference-region morphology", "selection_rule": "maximum outcome-blind sharpness/SNR minus competing/variability score"},
        {**ambiguous[KEYS].to_dict(), "example": "Ambiguous or noisy reference-region morphology", "selection_rule": "minimum outcome-blind sharpness/SNR minus competing/variability score"},
    ])


def figure_f06(kind: str, master: pd.DataFrame, status: pd.DataFrame, morphology: pd.DataFrame,
               registry: pd.DataFrame, config: dict, output: Path) -> dict:
    style = load_figure_style()
    set_style(style)
    examples = _morphology_examples(morphology)
    reference = "ECHO AVO" if kind == "B" else "ECHO AVC"
    event = "echo_avo_time_s" if kind == "B" else "echo_avc_time_s"
    fig, axes = plt.subplots(3, 2, figsize=(11.4, 9.2), gridspec_kw={"height_ratios": [1, 1.25, 1.25]})
    fig.subplots_adjust(top=.93, bottom=.08, hspace=.48, wspace=.22)
    for column, example in examples.iterrows():
        row = master.merge(pd.DataFrame([example[KEYS].to_dict()]), on=KEYS).iloc[0]
        recording, processed, _, start, stop, time_ms = _recording_signals(row, config)
        ecg = processed.filtered_ecg[start:stop]
        ecg_scale = ecg.std()
        ecg = (ecg - ecg.mean()) / ecg_scale if ecg_scale > 0 else ecg - ecg.mean()
        signal = processed.analysis_dzdt[start:stop]
        scale = signal.std()
        signal = (signal - signal.mean()) / scale if scale > 0 else signal - signal.mean()
        axes[0, column].plot(time_ms, ecg, color="#526D7A", lw=1.0)
        axes[0, column].set_title(f"{chr(65 + column)}  {example.example} — ECG", loc="left", fontweight="semibold")
        axes[0, column].set_xlabel("Time from ECG R (ms)")
        axes[0, column].set_ylabel("Standardized ECG (a.u.)")
        axes[1, column].plot(time_ms, signal, color="#263F4F", lw=1.1)
        ref_ms = 1000 * (row[event] - row.r_peak_time_s)
        axes[1, column].axvspan(ref_ms - 25, ref_ms + 25, color="#D9D9D9", alpha=.45)
        ref = style["reference_styles"][reference]
        axes[1, column].axvline(ref_ms, color=ref["colour"], linestyle=ref["linestyle"], lw=1.15)
        beat_status = status.loc[(status.subject_id.eq(row.subject_id)) &
                                 (status.recording_id.eq(row.recording_id)) &
                                 (status.beat_id.eq(row.beat_id)) & status.valid]
        for candidate in beat_status.itertuples(index=False):
            y = np.interp(candidate.candidate_time_from_r_ms, time_ms, signal)
            axes[1, column].scatter(candidate.candidate_time_from_r_ms, y, s=27,
                                    marker=style["algorithm_markers"][candidate.candidate_name],
                                    color=style["algorithm_colours"][candidate.candidate_name],
                                    edgecolor="white", linewidth=.35, zorder=4)
        axes[1, column].set_title(f"{chr(67 + column)}  Reference region and {kind} candidates", loc="left", fontweight="semibold")
        axes[1, column].set_xlabel("Time from ECG R (ms)")
        axes[1, column].set_ylabel("Standardized filtered dZ/dt (a.u.)")
    error = _effect_matrix(morphology, "error")
    failure = _effect_matrix(morphology, "failure")
    for axis, matrix, title, cbar_label, cmap in (
        (axes[2, 0], error, "E  Higher-category minus lower-category error", "Difference in participant median error (ms)", "RdBu_r"),
        (axes[2, 1], failure, "F  Higher-category minus lower-category failure", "Difference in participant median failure (percentage points)", "RdBu_r"),
    ):
        limit = np.nanmax(np.abs(matrix.to_numpy(float)))
        image = axis.imshow(matrix.to_numpy(float), aspect="auto", cmap=cmap, vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_xlabel("Outcome-blind within-participant morphology category")
        axis.set_ylabel(f"{kind}-point algorithm")
        axis.set_title(title, loc="left", fontweight="semibold")
        fig.colorbar(image, ax=axis, fraction=.046, pad=.04, label=cbar_label)
    for axis in axes.ravel():
        axis.grid(False)
    fig.suptitle(f"{kind}-point morphology and localisation reliability", fontsize=12, fontweight="semibold")
    _save(fig, output)
    return {"examples": examples, "error_effects": error, "failure_effects": failure}
