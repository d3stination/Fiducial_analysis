"""Generate the final, audited RQ1 figure families in the mandated order."""

from __future__ import annotations

import json
import os
import pickle
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .data_loading import ROOT, load_config
from .final_rq1_results import DATASETS, FEATURES, KEYS, PAIR, load_master, load_status, methods, table_path
from .morphology_analysis import _legacy_api


STYLE = json.loads((ROOT / "configs" / "figure_style.yaml").read_text())
OUTPUT_ROOT = Path(os.environ.get("RQ1_OUTPUT_ROOT", ROOT / "results"))
PALETTE = STYLE["participant_palette"]
CONDITION_LABELS = {"overall": "Overall", "normal": "Rest", "post_exercise": "Post-exercise", "baseline": "Baseline", "cognitive_workload": "Cognitive workload"}
DATASET_SLUG = {"HeartCycle": "HeartCycle", "IITB": "IITB", "ReBeat": "ReBeat"}
MORPH_SELECTIONS = {
    ("HeartCycle", "B"): ("sharpness", "median_algorithm_absolute_reference_error_ms"),
    ("HeartCycle", "X"): ("competing", "median_algorithm_absolute_reference_error_ms"),
    ("IITB", "B"): ("amplitude", "timing_range_ms"),
    ("IITB", "X"): ("slope", "timing_range_ms"),
    ("ReBeat", "B"): ("sharpness", "median_algorithm_absolute_reference_error_ms"),
    ("ReBeat", "X"): ("competing", "timing_range_ms"),
}
MORPH_XLABELS = {
    "sharpness": r"Peak/turning-point sharpness (s$^{-2}$)",
    "slope": r"Local waveform slope (s$^{-1}$)",
    "amplitude": "Region amplitude (device units)",
    "snr": "Signal-quality proxy (dB)",
    "competing": "Competing extrema (count)",
    "variability": "Adjacent-beat morphology NRMSE (a.u.)",
    "rr_interval_ms": "RR interval (ms)",
}
MORPH_YLABELS = {
    "median_algorithm_absolute_reference_error_ms": "Across-algorithm median absolute reference error (ms)",
    "timing_range_ms": "Within-beat algorithm timing range (ms)",
}
METHOD_SHORT = {
    "B1": "Last local minimum", "B2": "d3 downward zero", "B3": "d3 maximum",
    "B4": "45% upstroke", "B5": "25% upstroke", "B6": "d2 maximum",
    "B7": "d2 upward zero", "B8": "55% R–C", "B9": "Adaptive 10% upstroke",
    "B10": "15% upstroke", "X1": "First local minimum", "X2": "Global minimum",
    "X3": "d3 upward zero", "X4": "d3 minimum", "X5": "45% downstroke",
    "X6": "25% downstroke", "X7": "d2 upward zero", "X8": "d2 after minimum",
    "X9": "d2 minimum", "X10": "Weissler HR", "X11": "Adaptive half descent",
    "X12": "RR-adaptive minimum", "X13": "15% downstroke",
}
MORPH_FORMULAS = {
    "sharpness": "95th percentile of the absolute second derivative of the within-beat-standardised waveform in the B/X region",
    "slope": "95th percentile of the absolute first derivative of the within-beat-standardised waveform in the B/X region",
    "amplitude": "peak-to-peak filtered dZ/dt amplitude within the B/X region",
    "snr": "20 log10 of the 51-ms Savitzky–Golay smooth-signal SD divided by residual RMS",
    "competing": "count of local maxima plus local minima within the B/X region",
    "variability": "NRMSE between the current and preceding 200-point standardised beat shapes",
}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": STYLE["dpi"], "savefig.dpi": STYLE["dpi"],
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def natural_key(value: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def participant_colours(dataset: str, subjects: list[str]) -> dict[str, str]:
    subjects = sorted(set(subjects), key=natural_key)
    if dataset == "HeartCycle":
        return {subject: STYLE["participant_colours"].get(subject, PALETTE[index % len(PALETTE)]) for index, subject in enumerate(subjects)}
    return {subject: PALETTE[index % len(PALETTE)] for index, subject in enumerate(subjects)}


def output_dirs(dataset: str, kind: str) -> tuple[Path, Path]:
    suffix = Path(kind) if dataset == "HeartCycle" else Path(dataset) / kind
    figure_dir = OUTPUT_ROOT / "figures" / suffix
    report_dir = OUTPUT_ROOT / "reports" / suffix
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir, report_dir


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=STYLE["dpi"], facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def report(path: Path, question: str, sample: str, calculation: str, axes: str, result: str, interpretation: str, limitation: str, contribution: str) -> None:
    data_path = f"../../figure_data/{path.parent.name}/{path.stem}.csv"
    if path.parent.parent.name in {"IITB", "ReBeat"}:
        data_path = f"../../../figure_data/{path.parent.parent.name}/{path.parent.name}/{path.stem}.csv"
    text = (
        f"**Question:** {question}\n\n"
        f"**Dataset/sample:** {sample}\n\n"
        f"**Source data:** [{path.stem}.csv]({data_path}) and the canonical tables identified in the calculation.\n\n"
        f"**Calculation:** {calculation}\n\n"
        f"**Axes:** {axes}\n\n"
        f"**Result:** {result}\n\n"
        f"**Interpretation:** {interpretation}\n\n"
        f"**Limitation:** {limitation}\n\n"
        f"**RQ1 contribution:** {contribution}\n\n"
        f"**Three-sentence reader guide:** What is shown: {question} "
        f"What the evidence supports: {result} Why it matters for RQ1: {contribution}\n\n"
        f"**Manuscript-ready caption:** {question} {calculation} {result} {limitation}\n\n"
        f"**Reproducibility:** Generated by `python -m src.final_rq1_figures`; matching PNG, vector PDF and plotting-data CSV share this stem.\n"
    )
    path.write_text(text, encoding="utf-8")


def condition_handles(conditions: list[str]) -> list[Line2D]:
    return [Line2D([0], [0], marker=STYLE["condition_markers"][condition], linestyle="none",
                   markerfacecolor=STYLE["condition_colours"][condition], markeredgecolor="white",
                   markersize=6, label=CONDITION_LABELS[condition]) for condition in conditions]


def method_label(algorithm: str) -> str:
    return f"{algorithm} · {METHOD_SHORT[algorithm]}"


def bootstrap_median(values: np.ndarray, seed: int = 20261009) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    estimate = float(np.median(values))
    if len(values) < 3:
        return estimate, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = np.median(rng.choice(values, (4000, len(values)), replace=True), axis=1)
    return estimate, float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def _figure4_failure_legacy(dataset: str, kind: str) -> None:
    status = load_status(dataset, kind)
    conditions = DATASETS[dataset]["conditions"]
    summary = status.loc[status.attempted].groupby(["subject_id", "condition", "algorithm_id"], as_index=False).agg(
        attempted=("beat_id", "size"), missed=("missed", "sum"), invalid=("invalid", "sum")
    )
    summary["failure_pct"] = 100 * (summary.missed + summary.invalid) / summary.attempted
    plotted = []
    for (condition, algorithm), rows in summary.groupby(["condition", "algorithm_id"]):
        estimate, low, high = bootstrap_median(rows.failure_pct.to_numpy())
        plotted.append({"condition": condition, "algorithm_id": algorithm, "estimate": estimate, "low": low, "high": high})
    plotted = pd.DataFrame(plotted)
    totals = status.loc[status.attempted].groupby(["condition", "algorithm_id"]).agg(
        attempted=("beat_id", "size"), missed=("missed", "sum"), invalid=("invalid", "sum")
    ).reset_index()
    totals["pooled_pct"] = 100 * (totals.missed + totals.invalid) / totals.attempted
    plotted = plotted.merge(totals, on=["condition", "algorithm_id"], validate="one_to_one")
    order = methods(kind)
    y = np.arange(len(order))
    finite = np.concatenate([plotted.estimate.to_numpy(float), plotted.high.to_numpy(float), plotted.pooled_pct.to_numpy(float)])
    full_max = max(1.0, float(np.nanmax(finite)))
    centre = np.concatenate([plotted.estimate.to_numpy(float), plotted.pooled_pct.to_numpy(float)])
    detail_limit = max(1.0, float(np.ceil(np.nanquantile(centre, .80) * 2.7) / 2))
    show_detail = full_max > max(3.0, 2.5 * detail_limit)
    if show_detail:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.9 if kind == "B" else 5.8),
                                 sharey=True, gridspec_kw={"width_ratios": [2.4, 1]}, constrained_layout=True)
    else:
        fig, axis = plt.subplots(figsize=(8.0, 4.9 if kind == "B" else 5.8), constrained_layout=True)
        axes = np.asarray([axis])
    offsets = np.linspace(-.16, .16, len(conditions)) if len(conditions) > 1 else [0]
    for axis in axes:
        for offset, condition in zip(offsets, conditions):
            rows = plotted.loc[plotted.condition.eq(condition)].set_index("algorithm_id").reindex(order)
            low_error = np.maximum(0, rows.estimate.to_numpy(float) - rows.low.to_numpy(float))
            high_error = np.maximum(0, rows.high.to_numpy(float) - rows.estimate.to_numpy(float))
            axis.errorbar(rows.estimate, y + offset, xerr=[low_error, high_error],
                          fmt=STYLE["condition_markers"][condition], color=STYLE["condition_colours"][condition],
                          markeredgecolor="white", markeredgewidth=.45, markersize=5, capsize=2,
                          linewidth=1, zorder=3)
            axis.scatter(rows.pooled_pct, y + offset, facecolors="none", edgecolors=STYLE["condition_colours"][condition],
                         marker="D", s=26, linewidth=.9, zorder=4)
        axis.axvline(0, color="#777777", linewidth=.7)
        axis.set_xlabel("Missed + invalid detection rate (%)")
        axis.set_xlim(left=0)
    axes[0].set_xlim(0, full_max * 1.07)
    axes[0].set_yticks(y, [method_label(item) for item in order])
    axes[0].invert_yaxis()
    axes[0].set_ylabel(f"{kind}-point algorithm definition")
    axes[0].set_title("Full range")
    if show_detail:
        axes[1].set_xlim(0, detail_limit)
        axes[1].set_title("Low-rate detail (truncated view)")
        axes[1].tick_params(labelleft=False)
    condition_legend = condition_handles(conditions)
    summary_legend = [
        Line2D([0], [0], marker="o", color="#555555", linewidth=1, markersize=5,
               markerfacecolor="#555555", label="Participant median + bootstrap 95% CI"),
        Line2D([0], [0], marker="D", color="#555555", linestyle="none", markersize=5,
               markerfacecolor="none", label="Pooled beat-weighted rate"),
    ]
    axes[0].legend(handles=condition_legend + summary_legend, frameon=False, loc="best")
    fig.suptitle(f"{dataset}: {kind}-point detection failure\n"
                 f"n={status.subject_id.nunique()} participants; {status.beat_id.nunique()} beats", fontsize=11)
    figure_dir, report_dir = output_dirs(dataset, kind)
    stem = f"F04_{DATASET_SLUG[dataset]}_{kind}_detection_failure"
    save(fig, figure_dir / f"{stem}.png")
    worst = totals.sort_values("pooled_pct", ascending=False).iloc[0]
    sample = f"{status.subject_id.nunique()} participants, {status.recording_id.nunique()} recordings, {status.beat_id.nunique()} beats; {int(status.attempted.sum())} eligible algorithm-beat attempts."
    result = f"The largest pooled failure was {worst.algorithm_id} in {CONDITION_LABELS[worst.condition]}: {int(worst.missed)} missed + {int(worst.invalid)} invalid of {int(worst.attempted)} attempts ({worst.pooled_pct:.2f}%). Complete numerators and denominators are in RQ1_dataset_algorithm_results.csv."
    report(report_dir / f"{stem}.md", "Which definitions fail, and does failure differ by condition?", sample,
           "Participant failure rate = 100 × (missed + invalid) / eligible attempted beats. Filled condition markers are participant medians with participant-bootstrap 95% CIs; open diamonds are pooled beat-weighted rates. When shown, the right panel repeats the low-rate region while the left panel retains every interval and extreme.",
           f"Y: {kind}-point algorithm definition. X: missed + invalid detection rate (%).",
           result, "Failure is definition- and condition-specific; missed and invalid states remain separate in the source table.",
           "A failure rate is not timing accuracy. Sparse events yield imprecise intervals, and upstream-ineligible beats are excluded from the denominator.",
           "Quantifies whether a localisation definition produces an event reliably before its timing is interpreted.")


def figure4_failure(dataset: str, kind: str) -> None:
    status = load_status(dataset, kind)
    conditions = DATASETS[dataset]["conditions"]
    attempted = status.loc[status.attempted]
    participant = attempted.groupby(["subject_id", "condition", "algorithm_id"], as_index=False).agg(
        attempted_n=("beat_id", "size"), missed_n=("missed", "sum"), invalid_n=("invalid", "sum")
    )
    participant["failure_pct"] = 100 * (participant.missed_n + participant.invalid_n) / participant.attempted_n
    primary = []
    for (condition, algorithm), rows in participant.groupby(["condition", "algorithm_id"]):
        estimate, low, high = bootstrap_median(rows.failure_pct.to_numpy())
        primary.append({"condition": condition, "algorithm_id": algorithm, "estimate": estimate, "low": low, "high": high})
    primary = pd.DataFrame(primary)
    pooled = status.groupby(["condition", "algorithm_id"], as_index=False).agg(
        all_record_n=("beat_id", "size"), attempted_n=("attempted", "sum"), missed_n=("missed", "sum"),
        invalid_n=("invalid", "sum"), upstream_n=("attempted", lambda values: (~values).sum())
    )
    pooled["pooled_total_pct"] = 100 * (pooled.missed_n + pooled.invalid_n) / pooled.attempted_n
    pooled["missed_pct"] = 100 * pooled.missed_n / pooled.attempted_n
    pooled["invalid_pct"] = 100 * pooled.invalid_n / pooled.attempted_n
    pooled["upstream_pct"] = 100 * pooled.upstream_n / pooled.all_record_n
    primary = primary.merge(pooled, on=["condition", "algorithm_id"], validate="one_to_one")

    order = methods(kind)
    y = np.arange(len(order))
    offsets = np.linspace(-.16, .16, len(conditions)) if len(conditions) > 1 else [0]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.1 if kind == "B" else 6.0), sharey=True,
                             gridspec_kw={"width_ratios": [1.45, 1]}, constrained_layout=True)
    for offset, condition in zip(offsets, conditions):
        rows = primary.loc[primary.condition.eq(condition)].set_index("algorithm_id").reindex(order)
        low_error = np.maximum(0, rows.estimate.to_numpy(float) - rows.low.to_numpy(float))
        high_error = np.maximum(0, rows.high.to_numpy(float) - rows.estimate.to_numpy(float))
        axes[0].errorbar(rows.estimate, y + offset, xerr=[low_error, high_error],
                         fmt=STYLE["condition_markers"][condition], color=STYLE["condition_colours"][condition],
                         markeredgecolor="white", markeredgewidth=.45, markersize=5, capsize=2, linewidth=1, zorder=3)
        axes[0].scatter(rows.pooled_total_pct, y + offset, facecolors="none",
                        edgecolors=STYLE["condition_colours"][condition], marker="D", s=27, linewidth=.9, zorder=4)
        for shift, column, marker in [(-.055, "missed_pct", "o"), (0, "invalid_pct", "s"), (.055, "upstream_pct", "x")]:
            axes[1].scatter(rows[column], y + offset + shift, color=STYLE["condition_colours"][condition],
                            marker=marker, s=22, alpha=.9, linewidth=.8)
    axes[0].axvline(0, color="#777777", linewidth=.7)
    axes[0].set_xlim(left=0)
    axes[0].set_xlabel("Participant missed + invalid rate (%)")
    axes[0].set_title("Primary participant summary\nmedian + participant-bootstrap 95% CI")
    axes[1].set_xscale("symlog", linthresh=.25, linscale=1.1)
    maximum_component = max(1.0, float(pooled[["missed_pct", "invalid_pct", "upstream_pct"]].max().max()))
    axes[1].set_xlim(0, maximum_component * 1.12)
    axes[1].set_xlabel("Pooled status-specific rate (%)")
    axes[1].set_title("Status composition\ncommon nonlinear scale; linear 0–0.25%")
    axes[0].set_yticks(y, [method_label(item) for item in order])
    axes[0].invert_yaxis()
    axes[0].set_ylabel(f"{kind}-point algorithm definition")
    axes[0].legend(handles=condition_handles(conditions) + [
        Line2D([0], [0], marker="o", color="#555555", linewidth=1, markersize=5, label="Participant median + 95% CI"),
        Line2D([0], [0], marker="D", color="#555555", markerfacecolor="none", linestyle="none", markersize=5, label="Pooled missed + invalid"),
    ], frameon=False, loc="best")
    axes[1].legend(handles=[
        Line2D([0], [0], marker="o", color="#555555", linestyle="none", label="Missed / attempted"),
        Line2D([0], [0], marker="s", color="#555555", linestyle="none", label="Invalid / attempted"),
        Line2D([0], [0], marker="x", color="#555555", linestyle="none", label="Upstream-ineligible / all records"),
    ], frameon=False, loc="best")
    fig.suptitle(f"{dataset}: {kind}-point detection failure\n"
                 f"n={status.subject_id.nunique()} participants; {status.beat_id.nunique()} beats; {int(attempted.shape[0])} attempted method-beat records", fontsize=11)
    figure_dir, report_dir = output_dirs(dataset, kind)
    stem = f"F04_{DATASET_SLUG[dataset]}_{kind}_detection_failure"
    save(fig, figure_dir / f"{stem}.png")
    worst = pooled.sort_values("pooled_total_pct", ascending=False).iloc[0]
    report(report_dir / f"{stem}.md", "Which definitions fail, and are failures missed, invalid or upstream-ineligible?",
           f"{status.subject_id.nunique()} participants, {status.recording_id.nunique()} recordings and {status.beat_id.nunique()} beats; {int(attempted.shape[0])} eligible attempted method-beat records.",
           "The primary estimate is each participant's missed + invalid percentage, summarised by its median and participant-bootstrap 95% CI. The open diamond is the secondary pooled beat-weighted total. The right panel separates pooled missed/attempted, invalid/attempted and upstream-ineligible/all-record rates on one explicitly labelled symmetric-log scale; the left panel retains every uncertainty interval on an unclipped linear scale.",
           f"Y: {kind}-point algorithm definition. Left X: participant missed + invalid rate (%). Right X: pooled status-specific rate (%), nonlinear only outside ±0.25%.",
           f"The largest pooled attempted-failure rate was {worst.algorithm_id} in {CONDITION_LABELS[worst.condition]}: {int(worst.missed_n)} missed + {int(worst.invalid_n)} invalid of {int(worst.attempted_n)} attempts ({worst.pooled_total_pct:.2f}%); {int(worst.upstream_n)} additional records were upstream-ineligible and excluded from that denominator.",
           "The participant estimate is primary; pooled totals and status composition show whether a similar percentage arises from different execution outcomes.",
           "Failure is not timing accuracy. The right-panel nonlinear scale expands rare statuses but does not change their values; upstream ineligibility is contextual and remains distinct from attempted algorithm failure.",
           "Quantifies detection reliability and verified status composition before timing estimates are interpreted.")


def figure5_subject(dataset: str, kind: str) -> None:
    values = pd.read_csv(table_path(dataset, kind, "range_participant_summary"))
    conditions = DATASETS[dataset]["conditions"]
    subjects = sorted(values.subject_id.unique(), key=natural_key)
    colours = participant_colours(dataset, subjects)
    y_map = {subject: index for index, subject in enumerate(subjects)}
    fig, axis = plt.subplots(figsize=(7.2, max(4.6, .26 * len(subjects) + 1.8)), constrained_layout=True)
    if len(conditions) == 2:
        wide = values.pivot(index="subject_id", columns="condition", values="participant_median_range_ms").reindex(columns=conditions).dropna()
        for subject, row in wide.iterrows():
            y = y_map[subject]
            axis.plot([row[conditions[0]], row[conditions[1]]], [y, y], color=colours[subject], alpha=.65, linewidth=.9)
            for condition in conditions:
                axis.scatter(row[condition], y, color=STYLE["condition_colours"][condition], marker=STYLE["condition_markers"][condition], edgecolor="white", linewidth=.4, s=27, zorder=3)
        change = wide[conditions[1]] - wide[conditions[0]]
        estimate, low, high = bootstrap_median(change.to_numpy())
        result = f"The median paired change was {estimate:.2f} ms (participant-bootstrap 95% CI {low:.2f} to {high:.2f}); {int((change > 0).sum())}/{len(change)} participants increased and {int((change < 0).sum())}/{len(change)} decreased."
    else:
        for row in values.itertuples():
            axis.scatter(row.participant_median_range_ms, y_map[row.subject_id], color=colours[row.subject_id], marker="o", edgecolor="white", linewidth=.4, s=28)
        estimate, low, high = bootstrap_median(values.participant_median_range_ms.to_numpy())
        result = f"The participant median disagreement was {estimate:.2f} ms (participant-bootstrap 95% CI {low:.2f} to {high:.2f})."
    summary_handles = []
    for condition in conditions:
        condition_values = values.loc[values.condition.eq(condition), "participant_median_range_ms"]
        median = condition_values.median(); p95 = condition_values.quantile(.95)
        colour = STYLE["condition_colours"][condition]
        axis.axvline(median, color=colour, linewidth=1.0, alpha=.85)
        axis.axvline(p95, color=colour, linewidth=.9, linestyle=":", alpha=.85)
        summary_handles += [
            Line2D([0], [0], color=colour, linewidth=1.0, label=f"{CONDITION_LABELS[condition]} participant median"),
            Line2D([0], [0], color=colour, linewidth=.9, linestyle=":", label=f"{CONDITION_LABELS[condition]} participant P95"),
        ]
    axis.set_yticks(range(len(subjects)), [subject.split(":", 1)[-1] for subject in subjects])
    axis.invert_yaxis()
    axis.set_xlabel(f"Participant median within-beat {kind}-point timing range (ms)")
    axis.set_ylabel("Participant ID")
    axis.set_title(f"{dataset}: participant {kind}-point disagreement\n"
                   f"n={values.subject_id.nunique()} participants; {int(values.common_valid_beats.sum())} participant-summed common-valid beats")
    if len(conditions) == 2:
        axis.legend(handles=condition_handles(conditions) + summary_handles, frameon=False, loc="best")
    else:
        axis.legend(handles=summary_handles, frameon=False, loc="best")
    figure_dir, report_dir = output_dirs(dataset, kind)
    stem = f"F05A_{DATASET_SLUG[dataset]}_{kind}_subject_disagreement"
    save(fig, figure_dir / f"{stem}.png")
    report(report_dir / f"{stem}.md", "Which participants have greater across-method timing disagreement?",
           f"{values.subject_id.nunique()} participants; {int(values.common_valid_beats.sum())} participant-summed common-valid beats across displayed conditions.",
           f"For each common-valid beat, {kind} range = latest minus earliest of all {len(methods(kind))} methods; the displayed point is each participant-condition median. Vertical solid/dotted lines are the across-participant median/P95, not within-person error bars.",
           f"Y: participant ID. X: participant median within-beat {kind}-point timing range (ms).", result,
           "Participant trajectories show whether a condition effect is consistent or driven by a subset of people.",
           "Across-method range measures consensus, not physiological correctness; it is defined only on all-method valid beats.",
           "Separates subject heterogeneity from a pooled change in algorithm disagreement.")


def figure5_definition(dataset: str, kind: str) -> None:
    values = pd.read_csv(table_path(dataset, kind, "definition_participant_summary"))
    conditions = DATASETS[dataset]["conditions"]
    order = methods(kind)
    y = np.arange(len(order))
    fig, axis = plt.subplots(figsize=(8.2, 5.2 if kind == "B" else 6.3), constrained_layout=True)
    offsets = np.linspace(-.19, .19, len(conditions)) if len(conditions) > 1 else [0]
    for offset, condition in zip(offsets, conditions):
        condition_rows = values.loc[values.condition.eq(condition)]
        for index, algorithm in enumerate(order):
            rows = condition_rows.loc[condition_rows.algorithm_id.eq(algorithm)].sort_values(
                ["participant_median_peer_difference_ms", "subject_id"]
            )
            spread = np.linspace(-.075, .075, len(rows)) if len(rows) > 1 else np.zeros(len(rows))
            axis.scatter(rows.participant_median_peer_difference_ms, index + offset + spread,
                         color="#98A2A8", marker=STYLE["condition_markers"][condition],
                         s=15, alpha=.52, edgecolor="white", linewidth=.25, rasterized=True)
            if len(rows):
                q1, median, q3 = rows.participant_median_peer_difference_ms.quantile([.25, .5, .75])
                axis.plot([q1, q3], [index + offset] * 2, color=STYLE["condition_colours"][condition], linewidth=2.2, zorder=3)
                axis.scatter(median, index + offset, color=STYLE["condition_colours"][condition], marker="D", s=28, edgecolor="white", linewidth=.5, zorder=4)
    axis.set_yticks(y, [method_label(item) for item in order])
    axis.invert_yaxis()
    axis.set_xlabel("Participant median absolute difference from other-method median (ms)")
    axis.set_ylabel(f"{kind}-point algorithm definition")
    common_n = int(values.common_valid_beats.sum() / len(order))
    axis.set_title(f"{dataset}: definition-wise {kind}-point disagreement\n"
                   f"n={values.subject_id.nunique()} participants; {common_n} participant-summed common-valid beats")
    axis.legend(handles=condition_handles(conditions) + [Line2D([0], [0], marker="D", color="#555555", linewidth=2, markersize=5, label="Median and IQR")], frameon=False, loc="best")
    figure_dir, report_dir = output_dirs(dataset, kind)
    stem = f"F05B_{DATASET_SLUG[dataset]}_{kind}_definition_disagreement"
    save(fig, figure_dir / f"{stem}.png")
    group = values.groupby(["condition", "algorithm_id"]).participant_median_peer_difference_ms.median().reset_index()
    worst = group.sort_values("participant_median_peer_difference_ms", ascending=False).iloc[0]
    report(report_dir / f"{stem}.md", "Which definitions select positions most different from the other methods?",
           f"{values.subject_id.nunique()} participants and {common_n} participant-summed common-valid beats across conditions; each participant estimate uses identical all-method common-valid beats.",
           "For every common-valid beat, absolute difference = |target method − median(other methods)|; each neutral small point is one participant, vertically spread deterministically within its condition lane. Condition-specific diamonds and horizontal bars are across-participant medians and IQRs.",
           f"Y: {kind}-point algorithm definition. X: participant median absolute peer difference (ms).",
           f"The largest across-participant median was {worst.algorithm_id} in {CONDITION_LABELS[worst.condition]} ({worst.participant_median_peer_difference_ms:.2f} ms).",
           "Large values identify definitions that operationalise a different waveform location.",
           "Divergence is not inaccuracy, and common-valid restriction can omit difficult beats.",
           "Identifies which mathematical definitions contribute most to localisation variability.")


def figure7_changes(dataset: str, kind: str) -> None:
    if dataset == "HeartCycle":
        return
    changes = pd.read_csv(ROOT / "results" / "tables" / "RQ1_paired_condition_changes.csv")
    changes = changes.loc[(changes.dataset.eq(dataset)) & changes.candidate_type.eq(kind)]
    failure = changes.loc[changes.outcome.eq("failure_rate")].copy()
    method_summary = failure.groupby("algorithm_id").stress_minus_normal.median().abs().sort_values(ascending=False)
    selected = method_summary.index[0]
    failure_rows = failure.loc[failure.algorithm_id.eq(selected)]
    panels = [("within_beat_range", changes.loc[changes.outcome.eq("within_beat_range")],
               "Median timing-range change (ms)", "all-method range")]
    if not (dataset == "ReBeat" and kind == "B"):
        panels.insert(0, ("failure_rate", failure_rows, "Failure-rate change (percentage points)", f"selected method {selected}"))
    figure_dir, report_dir = output_dirs(dataset, kind)
    for outcome, rows, xlabel, detail in panels:
        subjects = sorted(rows.subject_id, key=natural_key)
        colours = participant_colours(dataset, subjects)
        rows = rows.set_index("subject_id").reindex(subjects).reset_index()
        fig, axis = plt.subplots(figsize=(7.0, max(4.6, .26 * len(rows) + 1.8)), constrained_layout=True)
        axis.axvline(0, color=STYLE["signed_error_colours"]["zero"], linewidth=.9)
        axis.scatter(rows.stress_minus_normal, np.arange(len(rows)), c=[colours[subject] for subject in rows.subject_id], marker="o", s=28, edgecolor="white", linewidth=.4)
        estimate, low, high = bootstrap_median(rows.stress_minus_normal.to_numpy())
        group_y = len(rows) + .55
        axis.errorbar(estimate, group_y, xerr=[[estimate - low], [high - estimate]], fmt="D",
                      color="#222222", markeredgecolor="white", markersize=6, capsize=3, linewidth=1.2,
                      label="Group median + participant-bootstrap 95% CI")
        axis.set_yticks([*np.arange(len(rows)), group_y], [*[subject.split(":", 1)[-1] for subject in rows.subject_id], "Group median"])
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Participant ID")
        axis.set_title(f"{dataset}: paired {kind}-point {outcome.replace('_', ' ')}\n"
                       f"n={len(rows)} matched participants; {detail}")
        axis.legend(frameon=False, loc="best")
        stem = f"F07_{DATASET_SLUG[dataset]}_{kind}_{outcome}_change"
        save(fig, figure_dir / f"{stem}.png")
        limitation = (f"The failure panel represents {selected}, not all definitions; timing-range change remains disagreement rather than accuracy."
                      if outcome == "failure_rate" else
                      "Timing-range change is across-method disagreement rather than independent-reference accuracy.")
        report(report_dir / f"{stem}.md", "How much does the same participant change from normal to stress?",
               f"{len(rows)} matched participants with both conditions; {detail}.",
               "Signed participant change = stress condition minus normal condition, using paired participant summaries only.",
               f"Y: participant ID. X: {xlabel}; zero indicates no paired change.",
               f"Median paired change was {estimate:.2f} ({rows.unit.iloc[0]}; participant-bootstrap 95% CI {low:.2f} to {high:.2f}); {int((rows.stress_minus_normal > 0).sum())}/{len(rows)} participants increased.",
               "The signed display makes direction and participant consistency explicit beyond the two-value Figure 5A view.",
               limitation,
               "Quantifies within-person condition sensitivity without mixing unmatched participants.")
    if dataset == "ReBeat" and kind == "B":
        stem = "F07_ReBeat_B_failure_rate_change"
        estimate, low, high = bootstrap_median(failure_rows.stress_minus_normal.to_numpy())
        report(report_dir / f"{stem}.md", "How much does the same participant's B-point failure rate change from baseline to cognitive workload?",
               f"{len(failure_rows)} matched participants with both conditions; selected method {selected}.",
               "Signed participant change = cognitive workload minus baseline failure rate for the selected definition. The former PNG was removed after backup because it duplicated Figure 4 without adding a material group shift.",
               "No active figure. The retained numerical outcome is expressed in percentage points.",
               f"Median paired change was {estimate:.2f} percentage points (participant-bootstrap 95% CI {low:.2f} to {high:.2f}); {int((failure_rows.stress_minus_normal > 0).sum())}/{len(failure_rows)} participants increased.",
               "The participant-level changes are already represented by the condition-specific Figure 4 distributions; a second dedicated panel did not add a distinct scientific conclusion.",
               f"This result applies to selected method {selected}, not all B definitions; a zero median does not imply no participant heterogeneity.",
               "Retains the paired numerical result while removing a redundant scientific figure.")


_SIGNALS = None
_HEARTCYCLE_CACHE = {}


def signal_bundle():
    global _SIGNALS
    if _SIGNALS is None:
        _legacy_api(load_config())
        with (ROOT / "work" / "phase2" / "verified_replay.pkl").open("rb") as handle:
            _SIGNALS = pickle.load(handle)[4]
    return _SIGNALS


def recording_signal(dataset: str, recording_id: str, source_file: str):
    if dataset != "HeartCycle":
        return signal_bundle()[recording_id]
    if recording_id not in _HEARTCYCLE_CACHE:
        root, _, loader, preprocess, _ = _legacy_api(load_config())
        from src.variability_analysis.manifest import dataset_manifest
        spec = next(item for item in dataset_manifest(root) if item.dataset_id == "heartcycle_healthy")
        recording = loader(spec, Path(source_file), root)
        _HEARTCYCLE_CACHE[recording_id] = (recording, preprocess(recording))
    return _HEARTCYCLE_CACHE[recording_id]


def beat_samples(dataset: str, row: pd.Series) -> tuple[int, int, int]:
    if dataset == "HeartCycle":
        return int(row.r_peak_sample_index), int(row.c_sample_index), int(row.next_r_peak_sample_index)
    return int(row.beat_start_r_sample_inclusive), int(row.c_sample), int(row.beat_end_next_r_sample_exclusive)


def select_waveform_examples(dataset: str) -> list[dict]:
    b_range = pd.read_csv(table_path(dataset, "B", "common_beat_ranges"))
    x_range = pd.read_csv(table_path(dataset, "X", "common_beat_ranges"))
    b_range = b_range.merge(load_master(dataset, "B")[["beat_id", "rr_interval_ms"]], on="beat_id", validate="one_to_one")
    x_range = x_range.merge(load_master(dataset, "X")[["beat_id", "rr_interval_ms"]], on="beat_id", validate="one_to_one")
    combined = b_range[[*KEYS, "condition", "timing_range_ms"]].rename(columns={"timing_range_ms": "b_range"}).merge(
        x_range[[*KEYS, "condition", "timing_range_ms"]].rename(columns={"timing_range_ms": "x_range"}), on=[*KEYS, "condition"], how="inner")
    low_b, low_x = combined.b_range.quantile(.10), combined.x_range.quantile(.10)
    clean = combined.copy()
    if dataset in ["HeartCycle", "ReBeat"]:
        reference = pd.read_csv(ROOT / "results" / "tables" / "RQ1_reference_errors.csv")
        reference = set(reference.loc[reference.dataset.eq(dataset), "beat_id"])
        referenced = clean.loc[clean.beat_id.isin(reference)]
        if len(referenced):
            clean = referenced
    clean["b_percentile_rank"] = clean.b_range.rank(method="average", pct=True)
    clean["x_percentile_rank"] = clean.x_range.rank(method="average", pct=True)
    clean["joint_rank"] = clean.b_percentile_rank + clean.x_percentile_rank
    clean_row = clean.sort_values(["joint_rank", "b_range", "x_range", "beat_id"]).iloc[0]
    b_rr = b_range.rr_interval_ms.quantile([.05, .95])
    x_rr = x_range.rr_interval_ms.quantile([.05, .95])
    major_b = b_range.loc[b_range.rr_interval_ms.between(b_rr.iloc[0], b_rr.iloc[1])].sort_values(["timing_range_ms", "beat_id"], ascending=[False, True]).iloc[0]
    major_x = x_range.loc[x_range.rr_interval_ms.between(x_rr.iloc[0], x_rr.iloc[1])].sort_values(["timing_range_ms", "beat_id"], ascending=[False, True]).iloc[0]
    x_status = load_status(dataset, "X")
    failure = x_status.groupby([*KEYS, "condition"], as_index=False).agg(failed=("valid", lambda values: (~values).sum()))
    failure_row = failure.sort_values(["failed", "beat_id"], ascending=[False, True]).iloc[0]
    return [
        {"label": "Clear B agreement", "kind": "B", "row": clean_row, "criterion": f"All 23 methods valid; lowest joint empirical B/X disagreement rank{' among beats with a valid reference' if dataset in ['HeartCycle', 'ReBeat'] else ''}. Its B range is {clean_row.b_range:.2f} ms and X range is {clean_row.x_range:.2f} ms; dataset 10th-percentile cut-points are {low_b:.2f} and {low_x:.2f} ms."},
        {"label": "Major B disagreement", "kind": "B", "row": major_b, "criterion": "Largest all-method common-valid B timing range among beats within the dataset 5th–95th percentile RR interval, avoiding an unrepresentative RR outlier."},
        {"label": "Major X disagreement", "kind": "X", "row": major_x, "criterion": "Largest all-method common-valid X timing range among beats within the dataset 5th–95th percentile RR interval, avoiding an unrepresentative RR outlier."},
        {"label": "Failed/ambiguous X", "kind": "X", "row": failure_row, "criterion": "Largest number of missed or invalid X definitions on an upstream-eligible beat."},
    ]


def reference_relative_ms(dataset: str, kind: str, beat_id: str, r_time: float) -> tuple[float | None, str | None]:
    errors = pd.read_csv(ROOT / "results" / "tables" / "RQ1_reference_errors.csv")
    rows = errors.loc[(errors.dataset.eq(dataset)) & errors.candidate_type.eq(kind) & errors.beat_id.eq(beat_id)]
    if not len(rows):
        return None, None
    row = rows.iloc[0]
    return 1000 * (row.reference_event_time_s - r_time), row.reference_type


def plot_waveform(dataset: str, kind: str, selected: pd.Series, title: str, stem: str, criterion: str) -> None:
    master = load_master(dataset, kind)
    beat = master.loc[master.beat_id.eq(selected.beat_id)].iloc[0]
    status = load_status(dataset, kind)
    detections = status.loc[status.beat_id.eq(selected.beat_id)].copy()
    recording, processed = recording_signal(dataset, beat.recording_id, beat.source_file)
    r, c, nr = beat_samples(dataset, beat)
    signal = np.asarray(processed.analysis_dzdt, float)
    raw_derivative = _legacy_api(load_config())[-1](recording.icg, recording.sampling_rate_hz)
    scale = np.std(signal[r:nr])
    raw_scale = np.std(raw_derivative[r:nr])
    standardized = (signal - np.mean(signal[r:nr])) / scale if scale > 0 else signal * np.nan
    raw_standardized = ((raw_derivative - np.mean(raw_derivative[r:nr])) / raw_scale
                        if raw_scale > 0 else raw_derivative * np.nan)
    if kind == "B":
        start, stop = max(r, r - int(.02 * recording.sampling_rate_hz)), min(nr, c + int(.06 * recording.sampling_rate_hz))
    else:
        start, stop = max(r, c - int(.06 * recording.sampling_rate_hz)), min(nr, c + int(.55 * (nr - c)))
    relative_time = 1000 * (recording.time_s[start:stop] - recording.time_s[r])
    waveform = standardized[start:stop]
    raw_waveform = raw_standardized[start:stop]
    fig, axis = plt.subplots(figsize=(10.2, 6.0))
    fig.subplots_adjust(left=.10, right=.98, top=.87, bottom=.30)
    axis.plot(relative_time, raw_waveform, color=STYLE["signal_styles"]["raw"]["colour"],
              linewidth=STYLE["signal_styles"]["raw"]["linewidth"], alpha=.78, label="Raw-derived dZ/dt")
    axis.plot(relative_time, waveform, color=STYLE["signal_styles"]["filtered"]["colour"], linewidth=1.25, label="Filtered dZ/dt")
    valid = detections.loc[detections.valid & detections.candidate_time_from_r_ms.between(relative_time.min(), relative_time.max())]
    search_start = detections.search_start_sample.dropna()
    search_stop = detections.search_stop_sample_exclusive.dropna()
    if len(search_start) and len(search_stop):
        left = 1000 * (recording.time_s[int(search_start.min())] - recording.time_s[r])
        right_index = min(int(search_stop.max()) - 1, len(recording.time_s) - 1)
        right = 1000 * (recording.time_s[right_index] - recording.time_s[r])
        axis.axvspan(max(left, relative_time.min()), min(right, relative_time.max()), color="#E6EAED", alpha=.5, zorder=0, label="Union of method search intervals")
        axis.axvline(max(left, relative_time.min()), color="#AAB2B8", linewidth=.7, linestyle=":")
        axis.axvline(min(right, relative_time.max()), color="#AAB2B8", linewidth=.7, linestyle=":")
    for candidate in valid.itertuples():
        x = candidate.candidate_time_from_r_ms
        y = np.interp(x, relative_time, waveform)
        colour = STYLE["algorithm_colours"][candidate.candidate_name]
        axis.scatter(x, y, marker="^" if kind == "B" else "v", s=34, color=colour, edgecolor="white", linewidth=.45, zorder=4)
    rejected = detections.loc[detections.invalid & detections.candidate_time_from_r_ms.between(relative_time.min(), relative_time.max())]
    for candidate in rejected.itertuples():
        x = candidate.candidate_time_from_r_ms
        y = np.interp(x, relative_time, waveform)
        colour = STYLE["algorithm_colours"][candidate.candidate_name]
        axis.scatter(x, y, marker="^" if kind == "B" else "v", s=48, facecolors="none",
                     edgecolors=colour, linewidth=1.15, zorder=5)
    y_min, y_max = float(np.nanmin(waveform)), float(np.nanmax(waveform))
    y_span = max(y_max - y_min, 1e-6)
    reference, reference_name = reference_relative_ms(dataset, kind, beat.beat_id, recording.time_s[r])
    if reference is not None and relative_time.min() <= reference <= relative_time.max():
        y = np.interp(reference, relative_time, waveform)
        style = STYLE["reference_styles"].get(reference_name, {"colour": "#2F6B4F", "marker": "D", "linestyle": "--"})
        axis.scatter(reference, y, marker=style["marker"], color=style["colour"], s=42, edgecolor="white", linewidth=.5, zorder=5, label=reference_name)
        axis.axvline(reference, color=style["colour"], linestyle=style["linestyle"], linewidth=.8, alpha=.75)
    missed = detections.loc[detections.missed, "algorithm_id"].tolist()
    invalid = detections.loc[detections.invalid, "algorithm_id"].tolist()
    reason_records = pd.read_csv(ROOT / "work" / "p1_completion" / "phase4_failure_reason_records.csv", low_memory=False)
    reasons = reason_records.loc[
        reason_records.dataset.eq(dataset) & reason_records.candidate_type.eq(kind) & reason_records.beat_id.eq(beat.beat_id),
        ["algorithm_id", "primary_reason", "primary_reason_detail"],
    ]
    reason_text = "; ".join(
        f"{row.algorithm_id}: {row.primary_reason}" + (f" ({row.primary_reason_detail})" if pd.notna(row.primary_reason_detail) and row.primary_reason_detail else "")
        for row in reasons.itertuples()
    )
    reason_plot_text = "; ".join(
        f"{reason}: {','.join(group.algorithm_id)}" for reason, group in reasons.groupby("primary_reason", sort=False)
    )
    axis.set_xlabel("Time relative to detected ECG R (ms)")
    axis.set_ylabel("Within-beat standardised dZ/dt (z score)")
    axis.set_ylim(y_min - .08 * y_span, y_max + .16 * y_span)
    beat_number = int(beat.beat_number) if "beat_number" in beat.index and pd.notna(beat.beat_number) else beat.beat_id.rsplit("b", 1)[-1]
    axis.set_title(
        f"{dataset} · {title}\n"
        f"Participant {beat.subject_id.split(':')[-1]} | recording {beat.recording_id} | "
        f"{CONDITION_LABELS.get(beat.condition, beat.condition)} | beat {beat_number}"
    )
    if missed or invalid:
        axis.text(.0, -.16,
                  f"Missed: {', '.join(missed) if missed else 'none'}   |   Invalid: {', '.join(invalid) if invalid else 'none'}",
                  transform=axis.transAxes, ha="left", va="top", fontsize=7.2, color="#444444")
    if reason_plot_text:
        axis.text(.0, -.22, f"Verified primary reasons — {reason_plot_text}", transform=axis.transAxes,
                  ha="left", va="top", fontsize=6.2, color="#555555")
    axis.text(.0, -.10, "Selection: " + textwrap.fill(criterion, 145), transform=axis.transAxes,
              ha="left", va="top", fontsize=6.6, color="#333333")
    signal_handles, signal_labels = axis.get_legend_handles_labels()
    axis.legend(signal_handles, signal_labels, frameon=False, loc="upper right", ncol=2)
    algorithm_handles = [
        Line2D([0], [0], marker="^" if kind == "B" else "v", linestyle="none", markersize=5,
               markerfacecolor=STYLE["algorithm_colours"][row.candidate_name], markeredgecolor="white",
               label=row.algorithm_id)
        for row in sorted(detections.itertuples(), key=lambda item: natural_key(item.algorithm_id))
    ]
    fig.legend(handles=algorithm_handles, title=f"Frozen {kind}-point definitions", frameon=False,
               loc="lower center", bbox_to_anchor=(.52, .015), ncol=7 if kind == "X" else 5,
               columnspacing=1.2, handletextpad=.3)
    figure_dir, report_dir = output_dirs(dataset, kind)
    save(fig, figure_dir / f"{stem}.png")
    range_value = selected.get("timing_range_ms", selected.get(f"{kind.lower()}_range", np.nan))
    status_text = f"missed: {', '.join(missed) if missed else 'none'}; invalid: {', '.join(invalid) if invalid else 'none'}"
    result = (f"Selected beat {beat.beat_id}; displayed valid-method range was {range_value:.2f} ms; {status_text}."
              if np.isfinite(range_value) else f"Selected beat {beat.beat_id}; {status_text}.")
    report(report_dir / f"{stem}.md", "What waveform morphology is present where methods agree, disagree, or fail?",
           f"{dataset}, {beat.subject_id}, recording {beat.recording_id}, condition {beat.condition}, one pre-specified representative beat.",
           f"Selection criterion: {criterion} Markers use actual candidate timestamps. Raw-derived and filtered dZ/dt are each standardised within the same R–R beat for morphology display; detection calculations remain unchanged.",
           "X: time relative to detected ECG R (ms). Y: within-beat standardised dZ/dt (z score).",
           result + (f" Verified execution reasons: {reason_text}." if reason_text else ""), "The displayed waveform makes the selected landmark ambiguity and method clustering visible without treating consensus as truth.",
           "A representative beat illustrates mechanism but does not estimate population frequency; a reference marker appears only on a validated matched beat.",
           "Links numerical disagreement/failure to the waveform actually processed by the frozen definitions.")


def figure1_waveforms(dataset: str) -> None:
    for index, item in enumerate(select_waveform_examples(dataset), start=1):
        kind = item["kind"]
        label_code = ["A", "B", "C", "D"][index - 1]
        stem = f"F01{label_code}_{DATASET_SLUG[dataset]}_{kind}_{item['label'].lower().replace('/', '_').replace(' ', '_')}"
        plot_waveform(dataset, kind, item["row"], item["label"], stem, item["criterion"])


def figure2_accuracy(dataset: str, kind: str) -> None:
    participant = pd.read_csv(ROOT / "results" / "tables" / "RQ1_reference_participant_accuracy.csv")
    participant = participant.loc[(participant.dataset.eq(dataset)) & participant.candidate_type.eq(kind)]
    if participant.empty:
        return
    conditions = list(participant.condition.drop_duplicates())
    order = methods(kind)
    fig, axes = plt.subplots(1, len(conditions), figsize=(7.7 * len(conditions), 5.1 if kind == "B" else 6.1), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(20261009)
    if dataset == "ReBeat":
        q95 = participant.median_absolute_error_ms.quantile(.95)
        max_q3 = participant.groupby(["condition", "algorithm_id"]).median_absolute_error_ms.quantile(.75).max()
        main_limit = float(np.ceil(max(q95, 1.2 * max_q3) / 10) * 10)
        full_limit = float(np.ceil(participant.median_absolute_error_ms.max() / 10) * 10)
    else:
        main_limit = full_limit = np.nan
    for axis, condition in zip(axes, conditions):
        data = participant.loc[participant.condition.eq(condition)]
        for index, algorithm in enumerate(order):
            rows = data.loc[data.algorithm_id.eq(algorithm)]
            jitter = rng.uniform(-.11, .11, len(rows))
            display_x = rows.median_absolute_error_ms.to_numpy(float)
            marker = np.full(len(rows), "o", dtype=object)
            if dataset == "ReBeat":
                outside = display_x > main_limit
                inside = ~outside
                axis.scatter(display_x[inside], index + jitter[inside],
                             color="#87939B", marker="o", s=17, alpha=.58, edgecolor="white", linewidth=.3)
                axis.scatter(np.full(outside.sum(), main_limit * .985), index + jitter[outside],
                             color="#59666F", marker=">", s=23, alpha=.85, edgecolor="white", linewidth=.3, clip_on=False)
            else:
                axis.scatter(display_x, index + jitter, color="#87939B",
                             marker="o", s=17, alpha=.58, edgecolor="white", linewidth=.3)
            if len(rows):
                q1, median, q3 = rows.median_absolute_error_ms.quantile([.25, .5, .75])
                axis.plot([q1, q3], [index, index], color="#222222", linewidth=2.2)
                axis.scatter(median, index, marker="D", color="#222222", edgecolor="white", linewidth=.4, s=30, zorder=4)
        axis.set_title(CONDITION_LABELS.get(condition, condition))
        if dataset == "ReBeat":
            axis.set_xlim(0, main_limit)
            inset = axis.inset_axes([.58, .53, .39, .39])
            for index, algorithm in enumerate(order):
                rows = data.loc[data.algorithm_id.eq(algorithm)]
                inset.scatter(rows.median_absolute_error_ms, np.full(len(rows), index),
                              color="#87939B", s=5, alpha=.45, linewidth=0, rasterized=True)
                if len(rows):
                    q1, median, q3 = rows.median_absolute_error_ms.quantile([.25, .5, .75])
                    inset.plot([q1, q3], [index, index], color="#222222", linewidth=.7)
                    inset.scatter(median, index, marker="D", color="#222222", s=6, linewidth=0)
            inset.set_xlim(0, full_limit)
            inset.set_ylim(len(order) - .5, -.5)
            inset.set_facecolor("white")
            inset.patch.set_alpha(.94)
            inset.set_yticks([])
            inset.set_xlabel("Error (ms)", fontsize=6.5, labelpad=1)
            inset.tick_params(axis="x", labelsize=6.2, length=2)
            inset.text(.98, .96, "Full range\n(all participants)", transform=inset.transAxes,
                       ha="right", va="top", fontsize=6.5)
        axis.set_xlabel("Participant median absolute reference timing error (ms)")
        axis.set_yticks(np.arange(len(order)), [method_label(item) for item in order])
    axes[0].invert_yaxis()
    axes[0].set_ylabel(f"{kind}-point algorithm definition")
    fig.suptitle(f"{dataset}: {kind}-point reference accuracy\n"
                 f"n={participant.subject_id.nunique()} participants; {int(participant.common_reference_beats.sum() / len(order))} participant-summed reference beats per method", fontsize=11)
    figure_dir, report_dir = output_dirs(dataset, kind)
    stem = f"F02_{DATASET_SLUG[dataset]}_{kind}_reference_accuracy"
    save(fig, figure_dir / f"{stem}.png")
    medians = participant.groupby(["condition", "algorithm_id"]).median_absolute_error_ms.median().reset_index()
    best = medians.sort_values("median_absolute_error_ms").iloc[0]
    counts = participant.groupby("condition").agg(participants=("subject_id", "nunique"), participant_method_rows=("algorithm_id", "size"))
    report(report_dir / f"{stem}.md", "Which definition is closer to the available reference across participants?",
           "; ".join(f"{CONDITION_LABELS.get(c, c)}: {int(row.participants)} participants" for c, row in counts.iterrows()) + "; strict all-method common-reference beats only.",
           "Each restrained grey circle is one participant median absolute detected-minus-reference timing error; the black diamond and horizontal bar show the across-participant median and IQR. A horizontal counterpart to the requested algorithm-on-X layout is used because 10/13 long algorithm definitions remain readable without rotation. In ReBeat, right-pointing boundary markers identify observations beyond the main linear range and the inset displays every participant on the full linear range.",
           f"Y: {kind}-point algorithm definition. X: participant median absolute reference timing error (ms){f'; ReBeat main linear range 0–{main_limit:.0f} ms with full-range inset to {full_limit:.0f} ms' if dataset == 'ReBeat' else ''}.",
           f"The lowest displayed across-participant median was {best.algorithm_id} in {CONDITION_LABELS.get(best.condition, best.condition)} ({best.median_absolute_error_ms:.2f} ms). Coverage and excluded reference beats are reported separately.",
           "Lower error supports closer correspondence to this specific reference, while the participant dots show heterogeneity.",
           "HeartCycle ECHO and ReBeat expert annotations are different reference types; expert ICG annotations are not independent ECHO. Median error alone does not establish a universal winner.",
           "Provides reference-supported evidence while preventing success-only selection from being hidden inside the plot.")


def figure3_signed_errors(dataset: str, kind: str) -> None:
    errors = pd.read_csv(ROOT / "results" / "tables" / "RQ1_reference_errors.csv")
    errors = errors.loc[(errors.dataset.eq(dataset)) & errors.candidate_type.eq(kind)]
    for condition, data in errors.groupby("condition"):
        order = methods(kind)
        absolute = data.signed_error_ms.abs()
        linear_limit = max(20.0, float(np.ceil(absolute.quantile(.80) / 10) * 10))
        full_limit = max(linear_limit, float(np.ceil(absolute.max() / 20) * 20))
        if full_limit > linear_limit:
            negative = -np.geomspace(full_limit, linear_limit, 10)
            positive = np.geomspace(linear_limit, full_limit, 10)
            edges = np.unique(np.concatenate([negative[:-1], np.linspace(-linear_limit, linear_limit, 21), positive[1:]]))
        else:
            edges = np.linspace(-linear_limit, linear_limit, 31)
        columns = 2
        rows = int(np.ceil(len(order) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(11.2, 1.85 * rows + 1.3), sharex=True, sharey=True, constrained_layout=True)
        axes = np.asarray(axes).ravel()
        max_count = 0
        counts = {}
        for algorithm in order:
            values = data.loc[data.algorithm_id.eq(algorithm), "signed_error_ms"]
            count, _ = np.histogram(values, bins=edges)
            counts[algorithm] = count
            max_count = max(max_count, int(count.max()))
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)
        colours = [STYLE["signed_error_colours"]["negative"] if center < 0 else STYLE["signed_error_colours"]["positive"] for center in centers]
        for axis, algorithm in zip(axes, order):
            axis.bar(centers, counts[algorithm], width=widths, color=colours, edgecolor="white", linewidth=.25)
            axis.axvline(0, color=STYLE["signed_error_colours"]["zero"], linewidth=1.2)
            axis.set_title(method_label(algorithm), fontsize=8.1)
            axis.set_xscale("symlog", linthresh=linear_limit, linscale=1.2)
            axis.set_xlim(-full_limit, full_limit)
            axis.set_ylim(0, max_count * 1.06)
        for axis in axes[len(order):]:
            axis.set_visible(False)
        for axis in axes[-columns:]:
            if axis.get_visible(): axis.set_xlabel("Earlier ← signed detected − reference error (ms) → Later")
        for index in range(0, len(order), columns):
            axes[index].set_ylabel("Heartbeats (count)")
        fig.suptitle(f"{dataset}: {kind}-point signed reference errors · {CONDITION_LABELS.get(condition, condition)}\n"
                     f"n={data.subject_id.nunique()} participants; {data.beat_id.nunique()} strict common-reference beats · complete range; linear centre ±{linear_limit:.0f} ms", fontsize=11)
        fig.legend(handles=[
            Patch(facecolor=STYLE["signed_error_colours"]["negative"], label="Negative (early)"),
            Patch(facecolor=STYLE["signed_error_colours"]["positive"], label="Positive (late)"),
            Line2D([0], [0], color=STYLE["signed_error_colours"]["zero"], linewidth=1, label="Zero error"),
        ], frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(.995, .995))
        figure_dir, report_dir = output_dirs(dataset, kind)
        suffix = "" if len(errors.condition.unique()) == 1 else f"_{condition}"
        stem = f"F03_{DATASET_SLUG[dataset]}_{kind}_signed_reference_error{suffix}"
        save(fig, figure_dir / f"{stem}.png")
        stats = data.groupby("algorithm_id").agg(median=("signed_error_ms", "median"), p95_abs=("absolute_error_ms", lambda values: values.quantile(.95)), n=("beat_id", "size")).reset_index()
        early = stats.sort_values("median").iloc[0]; late = stats.sort_values("median", ascending=False).iloc[0]; tail = stats.sort_values("p95_abs", ascending=False).iloc[0]
        report(report_dir / f"{stem}.md", "Are definitions systematically early or late, and do large tails occur?",
               f"{data.subject_id.nunique()} participants, {data.beat_id.nunique()} strict common-reference beats, {CONDITION_LABELS.get(condition, condition)}.",
               f"Signed error = detected time − reference time. Every algorithm uses identical hybrid bin edges and the same count scale over the complete symmetric range (±{full_limit:.0f} ms). The common symmetric-log x-axis is linear from −{linear_limit:.0f} to +{linear_limit:.0f} ms and logarithmically compressed outside that interval, so central structure and every tail observation remain visible without miniature insets.",
               "X: signed detected-minus-reference error (ms), common symmetric-log scale with a labelled linear centre. Y: heartbeats (count). Blue indicates earlier-than-reference and muted red later-than-reference; the dark vertical line is zero error.",
               f"Most negative median: {early.algorithm_id} {early['median']:.2f} ms; most positive median: {late.algorithm_id} {late['median']:.2f} ms; largest P95 absolute error: {tail.algorithm_id} {tail.p95_abs:.2f} ms (numerical statistics are kept in this report, not the panels).",
               "The sign separates systematic early/late localisation from the magnitude of tail errors.",
               "Repeated beats are clustered within participants and histograms are descriptive. The nonlinear display scale changes visual spacing but not bin counts; common edges/scales preserve algorithm comparability. No tolerance-exceedance percentage is calculated because no approved threshold exists.",
               "Reveals systematic timing bias and occasional large errors hidden by an unsigned median alone.")


def figure9_bland_altman(dataset: str, kind: str) -> None:
    values = pd.read_csv(ROOT / "results" / "tables" / "RQ1_bland_altman_values.csv")
    summary = pd.read_csv(ROOT / "results" / "tables" / "RQ1_bland_altman_summary.csv")
    values = values.loc[(values.dataset.eq(dataset)) & values.candidate_type.eq(kind)]
    summary = summary.loc[(summary.dataset.eq(dataset)) & summary.candidate_type.eq(kind)]
    a, b = PAIR[kind]
    for condition, data in values.groupby("condition"):
        stat = summary.loc[summary.condition.eq(condition)].iloc[0]
        colours = participant_colours(dataset, data.subject_id.tolist())
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
        for axis in axes:
            axis.scatter(data.mean_time_from_r_ms, data.signed_difference_a_minus_b_ms,
                         c=[colours[subject] for subject in data.subject_id], s=7, alpha=.20, linewidth=0, rasterized=True)
            axis.axhline(stat.participant_balanced_bias_ms, color="#222222", linewidth=1.2, label="Participant-balanced bias")
            axis.axhline(stat.participant_balanced_loa_low_ms, color="#777777", linestyle="--", linewidth=1, label="95% limits of agreement")
            axis.axhline(stat.participant_balanced_loa_high_ms, color="#777777", linestyle="--", linewidth=1)
            axis.axhline(0, color="#B0B0B0", linewidth=.7)
            axis.set_xlabel(f"Mean of {a} and {b} timing relative to ECG R (ms)")
            axis.set_ylabel(f"Signed difference {a} − {b} (ms)")
        axes[0].set_title("Complete range")
        x_low, x_high = data.mean_time_from_r_ms.quantile([.01, .99])
        y_low, y_high = data.signed_difference_a_minus_b_ms.quantile([.01, .99])
        x_pad = max(1.0, .04 * (x_high - x_low))
        y_pad = max(1.0, .05 * (y_high - y_low))
        axes[1].set_xlim(x_low - x_pad, x_high + x_pad)
        axes[1].set_ylim(y_low - y_pad, y_high + y_pad)
        axes[1].set_title("Central 98% view")
        axes[0].legend(frameon=False, loc="best")
        fig.suptitle(f"{dataset}: {kind}-point method agreement · {CONDITION_LABELS.get(condition, condition)}\n"
                     f"n={int(stat.participants)} participants; {int(stat.paired_beats)} paired beats", fontsize=11)
        figure_dir, report_dir = output_dirs(dataset, kind)
        suffix = "" if len(values.condition.unique()) == 1 else f"_{condition}"
        stem = f"F09_{DATASET_SLUG[dataset]}_{kind}_{a}_{b}_bland_altman{suffix}"
        save(fig, figure_dir / f"{stem}.png")
        report(report_dir / f"{stem}.md", "How closely do two related operational definitions agree on identical beats?",
               f"{int(stat.participants)} participants and {int(stat.paired_beats)} paired valid beats, {CONDITION_LABELS.get(condition, condition)}; selected pair {a}/{b} shares the same waveform-threshold family.",
               "Difference = algorithm A minus B; bias and empirical limits use equal participant weights, with clustered participant bootstrap uncertainty in RQ1_bland_altman_summary.csv. The left panel preserves all observations; the right panel repeats the same data over pooled 1st–99th-percentile x/y limits.",
               f"Both panels — X: mean {a}/{b} timing relative to ECG R (ms). Y: signed {a} − {b} difference (ms).",
               f"Participant-balanced bias was {stat.participant_balanced_bias_ms:.2f} ms (95% CI {stat.bias_ci_low:.2f} to {stat.bias_ci_high:.2f}); limits of agreement were {stat.participant_balanced_loa_low_ms:.2f} to {stat.participant_balanced_loa_high_ms:.2f} ms.",
               "The selected threshold definitions can be systematically offset even when both return valid events.",
               "Bland–Altman agreement is not independent-reference accuracy; repeated-beat limits depend on the participant-balanced empirical model.",
               "Quantifies method interchangeability for a justified pair without exhaustively screening arbitrary pairs.")


def waveform_only(axis: plt.Axes, dataset: str, kind: str, beat_id: str, title: str, show_legend: bool = False,
                  highlight_algorithm: str | None = None) -> None:
    master = load_master(dataset, kind)
    beat = master.loc[master.beat_id.eq(beat_id)].iloc[0]
    detections = load_status(dataset, kind)
    detections = detections.loc[detections.beat_id.eq(beat_id)].copy()
    recording, processed = recording_signal(dataset, beat.recording_id, beat.source_file)
    r, c, nr = beat_samples(dataset, beat)
    signal = np.asarray(processed.analysis_dzdt, float)
    raw_derivative = _legacy_api(load_config())[-1](recording.icg, recording.sampling_rate_hz)
    scale = np.std(signal[r:nr])
    raw_scale = np.std(raw_derivative[r:nr])
    z = (signal - np.mean(signal[r:nr])) / scale if scale > 0 else signal * np.nan
    raw_z = ((raw_derivative - np.mean(raw_derivative[r:nr])) / raw_scale
             if raw_scale > 0 else raw_derivative * np.nan)
    start, stop = (r, min(nr, c + int(.08 * recording.sampling_rate_hz))) if kind == "B" else (max(r, c - int(.05 * recording.sampling_rate_hz)), min(nr, c + int(.55 * (nr - c))))
    x = 1000 * (recording.time_s[start:stop] - recording.time_s[r])
    filtered = z[start:stop]
    axis.plot(x, raw_z[start:stop], color=STYLE["signal_styles"]["raw"]["colour"],
              linewidth=STYLE["signal_styles"]["raw"]["linewidth"], alpha=.75, label="Raw-derived dZ/dt")
    axis.plot(x, filtered, color=STYLE["signal_styles"]["filtered"]["colour"], linewidth=1.1, label="Filtered dZ/dt")
    valid = detections.loc[detections.valid & detections.candidate_time_from_r_ms.between(x.min(), x.max())]
    for candidate in valid.itertuples():
        location = candidate.candidate_time_from_r_ms
        highlighted = highlight_algorithm is None or candidate.algorithm_id == highlight_algorithm
        axis.scatter(location, np.interp(location, x, filtered),
                     color=STYLE["algorithm_colours"][candidate.candidate_name] if highlighted else "#B5BDC2",
                     marker="^" if kind == "B" else "v", s=38 if highlighted else 16,
                     alpha=1 if highlighted else .55, edgecolor="white", linewidth=.35, zorder=5 if highlighted else 3)
    rejected = detections.loc[detections.invalid & detections.candidate_time_from_r_ms.between(x.min(), x.max())]
    for candidate in rejected.itertuples():
        location = candidate.candidate_time_from_r_ms
        highlighted = highlight_algorithm is None or candidate.algorithm_id == highlight_algorithm
        axis.scatter(location, np.interp(location, x, filtered), facecolors="none",
                     edgecolors=STYLE["algorithm_colours"][candidate.candidate_name] if highlighted else "#AEB6BB",
                     marker="^" if kind == "B" else "v", s=48 if highlighted else 22,
                     linewidth=1.1 if highlighted else .7, zorder=6 if highlighted else 3)
    if highlight_algorithm is not None:
        selected = detections.loc[detections.algorithm_id.eq(highlight_algorithm)].iloc[0]
        if pd.notna(selected.search_start_sample) and pd.notna(selected.search_stop_sample_exclusive):
            left = 1000 * (recording.time_s[int(selected.search_start_sample)] - recording.time_s[r])
            right_index = min(int(selected.search_stop_sample_exclusive) - 1, len(recording.time_s) - 1)
            right = 1000 * (recording.time_s[right_index] - recording.time_s[r])
            if x.min() <= left <= x.max(): axis.axvline(left, color="#8D979E", linestyle=":", linewidth=.8)
            if x.min() <= right <= x.max(): axis.axvline(right, color="#8D979E", linestyle=":", linewidth=.8)
        if selected.analysis_status in ["MISSED", "INVALID", "INELIGIBLE"]:
            reasons = pd.read_csv(ROOT / "work" / "p1_completion" / "phase4_failure_reason_records.csv", low_memory=False)
            reason = reasons.loc[reasons.dataset.eq(dataset) & reasons.candidate_type.eq(kind)
                                 & reasons.beat_id.eq(beat_id) & reasons.algorithm_id.eq(highlight_algorithm)]
            if len(reason):
                row = reason.iloc[0]
                axis.text(.02, .04, f"{highlight_algorithm} {selected.analysis_status.lower()}: {row.primary_reason}",
                          transform=axis.transAxes, fontsize=6.4, color="#4E5960",
                          bbox={"facecolor": "white", "edgecolor": "#C7CDD1", "alpha": .88, "pad": 2})
    combined = np.concatenate([raw_z[start:stop], filtered])
    y_min, y_max = float(np.nanmin(combined)), float(np.nanmax(combined))
    y_span = max(y_max - y_min, 1e-6)
    reference, reference_name = reference_relative_ms(dataset, kind, beat.beat_id, recording.time_s[r])
    if reference is not None and x.min() <= reference <= x.max():
        style = STYLE["reference_styles"].get(reference_name, {"colour": "#2F6B4F", "marker": "D", "linestyle": "--"})
        axis.scatter(reference, np.interp(reference, x, filtered), marker=style["marker"], color=style["colour"],
                     s=31, edgecolor="white", linewidth=.4, zorder=5, label=reference_name)
        axis.axvline(reference, color=style["colour"], linestyle=style["linestyle"], linewidth=.65, alpha=.75)
    beat_number = int(beat.beat_number) if "beat_number" in beat.index and pd.notna(beat.beat_number) else beat.beat_id.rsplit("b", 1)[-1]
    axis.set_title(f"{title}\n{beat.subject_id.split(':')[-1]} | {CONDITION_LABELS.get(beat.condition, beat.condition)} | beat {beat_number}", fontsize=8.2)
    axis.set_xlabel("Time from ECG R (ms)")
    axis.set_ylabel("Standardised dZ/dt (z score)")
    axis.set_ylim(y_min - .06 * y_span, y_max + .12 * y_span)
    if show_legend:
        handles, labels = axis.get_legend_handles_labels()
        handles.append(Line2D([0], [0], marker="^" if kind == "B" else "v", color="#B5BDC2",
                              linestyle="none", markersize=4, label="Other frozen definitions"))
        if highlight_algorithm is not None:
            chosen = detections.loc[detections.algorithm_id.eq(highlight_algorithm)].iloc[0]
            handles.append(Line2D([0], [0], marker="^" if kind == "B" else "v",
                                  color=STYLE["algorithm_colours"][chosen.candidate_name], linestyle="none",
                                  markersize=5, label=f"Selected {highlight_algorithm}"))
        axis.legend(handles=handles, frameon=False, fontsize=6.0, loc="best", ncol=2)


def figure6_method_specific(dataset: str, kind: str) -> None:
    from work.p1_completion.phase5_morphology_analysis import load_features

    selected_ids = {"B": ["B1", "B7", "B8"], "X": ["X4", "X8", "X12"]}[kind]
    evidence = pd.read_csv(ROOT / "work" / "p1_completion" / "phase5_all_method_morphology_evidence.csv")
    candidates = evidence.loc[
        evidence.dataset.eq(dataset) & evidence.candidate_type.eq(kind) & evidence.algorithm_id.isin(selected_ids)
        & evidence.participant_effect_n.ge(5)
    ].copy()
    supported = candidates.loc[candidates.exploratory_supported_after_bh]
    chosen = (supported if len(supported) else candidates).sort_values(
        ["bh_q_value", "participant_effect_n"], ascending=[True, False], na_position="last"
    ).iloc[0]
    feature, algorithm, condition, outcome = chosen.feature, chosen.algorithm_id, chosen.condition, chosen.outcome
    feature_units = {
        "sharpness": "dataset-specific curvature units", "slope": "dataset-specific slope units",
        "amplitude": "dataset-specific signal units", "snr": "dB", "competing_features": "count",
        "beat_to_beat_variability": "normalised error",
    }[feature]
    beat_features = load_features()
    beat_features = beat_features.loc[beat_features.dataset.eq(dataset) & beat_features.candidate_type.eq(kind)]
    if dataset != "HeartCycle":
        beat_features = beat_features.loc[beat_features.condition.eq(condition)]
    status = load_status(dataset, kind)
    status = status.loc[status.algorithm_id.eq(algorithm) & status.condition.eq(condition)]
    base = status.merge(beat_features[KEYS + [feature]], on=KEYS, how="left", validate="many_to_one")
    if outcome == "attempted detection failure":
        base = base.loc[base.attempted & base[feature].notna()].copy()
        base["outcome_value"] = base.analysis_status.isin(["MISSED", "INVALID"]).astype(int)
        base["outcome_label"] = np.where(base.outcome_value.eq(1), "Failed", "Valid")
        participant_plot = base.groupby(["subject_id", "outcome_label"], as_index=False)[feature].median()
        usable_subjects = participant_plot.groupby("subject_id").outcome_label.nunique()
        participant_plot = participant_plot.loc[participant_plot.subject_id.isin(usable_subjects[usable_subjects.eq(2)].index)]
        plot_rows = participant_plot
        export_rows = base
        outcome_label = "Detection status"
        y_label = f"{chosen.feature_label} ({feature_units})"
        valid_candidates = base.loc[base.outcome_label.eq("Valid")]
        failed_candidates = base.loc[base.outcome_label.eq("Failed")]
        example_low = valid_candidates.loc[(valid_candidates[feature] - valid_candidates[feature].median()).abs().idxmin()]
        example_high = failed_candidates.loc[(failed_candidates[feature] - failed_candidates[feature].median()).abs().idxmin()]
    else:
        if outcome.startswith("strict common-reference"):
            values = pd.read_csv(ROOT / "results" / "tables" / "RQ1_reference_errors.csv", low_memory=False)
            values = values.loc[values.dataset.eq(dataset) & values.candidate_type.eq(kind) & values.algorithm_id.eq(algorithm) & values.condition.eq(condition)]
            outcome_column = "absolute_error_ms"
            outcome_label = "Absolute reference timing error (ms)"
        else:
            values = pd.read_csv(table_path(dataset, kind, "definition_disagreement"))
            values = values.loc[values.algorithm_id.eq(algorithm) & values.condition.eq(condition)]
            outcome_column = "absolute_difference_from_other_method_median_ms"
            outcome_label = "Absolute difference from other-method median (ms)"
        plot_rows = values.merge(beat_features[KEYS + [feature]], on=KEYS, how="left", validate="many_to_one").dropna(subset=[feature, outcome_column])
        plot_rows["outcome_value"] = plot_rows[outcome_column]
        export_rows = plot_rows
        y_label = outcome_label
        example_low = plot_rows.sort_values(feature).iloc[max(0, int(.10 * (len(plot_rows) - 1)))]
        example_high = plot_rows.sort_values(feature).iloc[min(len(plot_rows) - 1, int(.90 * (len(plot_rows) - 1)))]

    participant_effects = pd.read_csv(ROOT / "work" / "p1_completion" / "phase5_participant_effects.csv")
    effects = participant_effects.loc[
        participant_effects.dataset.eq(dataset) & participant_effects.candidate_type.eq(kind)
        & participant_effects.algorithm_id.eq(algorithm) & participant_effects.condition.eq(condition)
        & participant_effects.outcome.eq(outcome) & participant_effects.feature.eq(feature)
    ].copy()
    effects["_natural_order"] = effects.subject_id.map(lambda value: tuple(natural_key(value)))
    effects = effects.sort_values("_natural_order")
    fig = plt.figure(figsize=(10.8, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
    left, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    quantitative = fig.add_subplot(grid[1, :])
    if outcome == "attempted detection failure":
        waveform_only(left, dataset, kind, example_low.beat_id, f"Typical valid beat · {algorithm}",
                      show_legend=True, highlight_algorithm=algorithm)
        waveform_only(right, dataset, kind, example_high.beat_id, f"Typical failed beat · {algorithm}",
                      highlight_algorithm=algorithm)
        effect_xlabel = f"Failed − valid participant median {chosen.feature_label.lower()} ({feature_units})"
    else:
        waveform_only(left, dataset, kind, example_low.beat_id, f"Lower {chosen.feature_label.lower()} · {algorithm}",
                      show_legend=True, highlight_algorithm=algorithm)
        waveform_only(right, dataset, kind, example_high.beat_id, f"Higher {chosen.feature_label.lower()} · {algorithm}",
                      highlight_algorithm=algorithm)
        effect_xlabel = f"Within-participant Spearman rho: {chosen.feature_label.lower()} vs {outcome_label.lower()}"
    effect_colours = participant_colours(dataset, effects.subject_id.tolist())
    y_positions = np.arange(len(effects))
    quantitative.axvline(0, color="#858E94", linewidth=.8)
    quantitative.scatter(effects.participant_effect, y_positions,
                         c=[effect_colours[subject] for subject in effects.subject_id], s=30,
                         edgecolor="white", linewidth=.4, zorder=3)
    group_y = len(effects) + .55
    median_effect = float(chosen.median_participant_effect)
    low_effect = float(chosen.bootstrap_95ci_low); high_effect = float(chosen.bootstrap_95ci_high)
    quantitative.errorbar(median_effect, group_y,
                          xerr=[[median_effect - low_effect], [high_effect - median_effect]], fmt="D",
                          color="#222222", markeredgecolor="white", markersize=6, capsize=3, linewidth=1.2)
    quantitative.set_yticks([*y_positions, group_y],
                            [*[subject.split(":", 1)[-1] for subject in effects.subject_id], "Median [95% CI]"])
    quantitative.invert_yaxis()
    quantitative.set_xlabel(effect_xlabel)
    quantitative.set_ylabel("Participant ID")
    condition_label = CONDITION_LABELS.get(condition, condition)
    support_label = "supported exploratory association" if chosen.exploratory_supported_after_bh else "unsupported after multiplicity control"
    quantitative.set_title(f"Participant-aware effect: {chosen.feature_label} → {outcome}\n"
                           f"{algorithm} · {condition_label} · n={int(chosen.participant_effect_n)} · {support_label}")
    fig.suptitle(f"{dataset}: {algorithm} morphology and {kind}-point {outcome}", fontsize=11)
    figure_dir, report_dir = output_dirs(dataset, kind)
    stem = f"F06_{DATASET_SLUG[dataset]}_{kind}_{algorithm}_{feature}_mechanism"
    save(fig, figure_dir / f"{stem}.png")
    beat_export = export_rows.assign(row_type="method_beat",
                                     selected_low_example=export_rows.beat_id.eq(example_low.beat_id),
                                     selected_high_or_failure_example=export_rows.beat_id.eq(example_high.beat_id))
    effect_export = effects.drop(columns="_natural_order").assign(row_type="participant_effect")
    pd.concat([beat_export, effect_export], ignore_index=True, sort=False).to_csv(
        figure_data_dir(dataset, kind) / f"{stem}.csv", index=False)
    result = (f"Median participant effect {chosen.median_participant_effect:.3g}; 95% participant-bootstrap CI "
              f"{chosen.bootstrap_95ci_low:.3g} to {chosen.bootstrap_95ci_high:.3g}; BH q="
              f"{chosen.bh_q_value:.3g}; {int(chosen.participant_effect_n)} participants. Evidence status: {support_label}.")
    report(report_dir / f"{stem}.md", "Which measured waveform characteristic is associated with method-specific localisation error, failure or disagreement?",
           f"{dataset}, {condition_label}, {kind}-point method {algorithm}; {int(chosen.participant_effect_n)} estimable participants and {int(chosen.available_observation_n)} available method-beat observations.",
           f"The complete six-feature/all-method audit selected this display without changing the frozen method. Outcome: {outcome}. Feature: {chosen.feature_label}. Participant effect: {chosen.effect_metric}; minimum data rule: {chosen.minimum_participant_data}. The two waveforms are actual low/high-feature or valid/failed examples from the plotted sample.",
           f"Quantitative panel — X: {effect_xlabel}; Y: participant ID plus the participant-median effect and bootstrap 95% CI. Waveform panels use time from ECG R (ms) and standardised dZ/dt, emphasise {algorithm}, show its search boundaries, and display its verified rejection reason when failed.",
           result,
           "This links a reproducible feature to a method-specific outcome while showing the waveform actually processed by the frozen detector.",
           "The analysis is exploratory and participant-aware but not causal. HeartCycle reference-local features are availability-limited; failure examples establish the recorded rejection path, not that morphology caused it.",
           "Provides a verified mechanism-oriented explanation after the complete all-method selection step; all negative and unsupported tests remain in the Phase 5 evidence table.")


def figure6_morphology(dataset: str, kind: str) -> None:
    feature, outcome = MORPH_SELECTIONS[(dataset, kind)]
    values = pd.read_csv(ROOT / "results" / "tables" / "RQ1_morphology_plot_values.csv", low_memory=False)
    data = values.loc[(values.dataset.eq(dataset)) & values.candidate_type.eq(kind), ["subject_id", "beat_id", "condition", feature, outcome]].dropna()
    if data.empty:
        return
    colours = participant_colours(dataset, data.subject_id.tolist())
    if len(data) > 3000:
        data_plot = pd.concat(
            [frame.sample(min(len(frame), 120), random_state=20261009) for _, frame in data.groupby("subject_id")],
            ignore_index=True,
        )
    else:
        data_plot = data
    representative_subject = data.groupby("subject_id").size().sort_values(ascending=False).index[0]
    subject_rows = data.loc[data.subject_id.eq(representative_subject)].sort_values(feature)
    low_row = subject_rows.iloc[max(0, int(.10 * (len(subject_rows) - 1)))]
    high_row = subject_rows.iloc[min(len(subject_rows) - 1, int(.90 * (len(subject_rows) - 1)))]
    fig = plt.figure(figsize=(11.0, 5.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1])
    scatter = fig.add_subplot(grid[:, 0])
    low_axis = fig.add_subplot(grid[0, 1]); high_axis = fig.add_subplot(grid[1, 1])
    for condition, rows in data_plot.groupby("condition"):
        scatter.scatter(rows[feature], rows[outcome], c=[colours[subject] for subject in rows.subject_id],
                        marker=STYLE["condition_markers"][condition], s=10, alpha=.28, linewidth=0, rasterized=True)
    feature_xlabel = MORPH_XLABELS[feature]
    if dataset == "HeartCycle" and feature == "sharpness":
        feature_xlabel = r"Reference-local sharpness (device units/s$^3$)"
    scatter.set_xlabel(feature_xlabel)
    scatter.set_ylabel(MORPH_YLABELS[outcome])
    scope = ("all-definition median reference error" if outcome.startswith("median_algorithm")
             else "all-definition within-beat range")
    scatter.set_title(f"{DATASET_SLUG[dataset]} · {kind}-point · {FEATURES[feature]}\nOutcome: {scope}")
    if feature in ["sharpness", "slope"] and (data[feature] > 0).all():
        scatter.set_xscale("log")
    if outcome != "failure" and (data[outcome] > 0).all() and data[outcome].max() / data[outcome].min() > 100:
        scatter.set_yscale("log")
    if len(DATASETS[dataset]["conditions"]) > 1:
        scatter.legend(handles=condition_handles(DATASETS[dataset]["conditions"]), frameon=False, loc="best")
    waveform_only(low_axis, dataset, kind, low_row.beat_id, "Lower feature example", show_legend=True)
    waveform_only(high_axis, dataset, kind, high_row.beat_id, "Higher feature example")
    figure_dir, report_dir = output_dirs(dataset, kind)
    stem = f"F06_{DATASET_SLUG[dataset]}_{kind}_{feature}_association"
    save(fig, figure_dir / f"{stem}.png")
    evidence = pd.read_csv(ROOT / "results" / "tables" / "RQ1_morphology_evidence.csv")
    outcome_name = "reference_absolute_error" if outcome.startswith("median_algorithm") else "across_method_disagreement"
    rows = evidence.loc[(evidence.dataset.eq(dataset)) & evidence.candidate_type.eq(kind) & evidence.feature.eq(feature) & evidence.outcome.eq(outcome_name)]
    result_parts = []
    for row in rows.itertuples():
        result_parts.append(f"{CONDITION_LABELS.get(row.condition, row.condition)}: median within-participant Spearman rho {row.median_within_participant_spearman_rho:.2f} (95% bootstrap CI {row.bootstrap_ci_low:.2f} to {row.bootstrap_ci_high:.2f}; n={int(row.participant_effect_n)} participants)")
    if dataset == "HeartCycle" and feature == "sharpness":
        feature_formula = "absolute second derivative of filtered dZ/dt at the validated reference event (device units/s³)"
    elif dataset == "HeartCycle" and feature == "competing":
        feature_formula = "turning points plus first-derivative sign changes within the reference-associated B/X region"
    else:
        feature_formula = MORPH_FORMULAS[feature]
    outcome_formula = ("median across all frozen definitions of |detected − reference| on each strict common-reference beat"
                       if outcome.startswith("median_algorithm") else
                       "latest minus earliest detection across all frozen definitions on each all-method common-valid beat")
    examples = (f"Lower-feature example: {low_row.beat_id}, {feature}={low_row[feature]:.4g}, outcome={low_row[outcome]:.2f} ms; "
                f"higher-feature example: {high_row.beat_id}, {feature}={high_row[feature]:.4g}, outcome={high_row[outcome]:.2f} ms.")
    report(report_dir / f"{stem}.md", "Is a reproducibly measured waveform characteristic associated with localisation error or disagreement?",
           f"{data.subject_id.nunique()} participants and {data.beat_id.nunique()} covered beats; scatter display capped at 120 deterministic observations per participant when needed; statistics use all covered beats.",
           f"Feature formula: {feature_formula}. Outcome formula: {outcome_formula}. Effect size is the median within-participant Spearman correlation with participant-bootstrap 95% CI. Waveforms are lower/higher-feature examples from the same high-coverage participant; coloured {kind} markers are actual timestamps from every valid frozen definition and the reference marker is shown only when matched. {examples}",
           f"X: {feature_xlabel}. Y: {MORPH_YLABELS[outcome]}; waveform panels use time from ECG R (ms) and within-beat z score.",
           "; ".join(result_parts) + ".",
           "The association is participant-aware and the adjacent waveforms connect the measured feature to the actual algorithm-localisation pattern; it does not establish causation.",
           "This was a focused selection after all six hypotheses were audited; selection is exploratory, nonlinear effects may be missed, and HeartCycle physical signal units remain unverified. The plotted outcome is an aggregate across all definitions, so this figure does not identify a specifically sensitive method.",
           "Links poorer localisation evidence to a concrete waveform property while retaining unsupported hypotheses in the complete evidence table.")


def figure_data_dir(dataset: str, kind: str) -> Path:
    suffix = Path(kind) if dataset == "HeartCycle" else Path(dataset) / kind
    directory = ROOT / "results" / "figure_data" / suffix
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _figure8_cross_dataset_legacy(kind: str) -> None:
    reference_condition = {"HeartCycle": "overall", "IITB": "normal", "ReBeat": "baseline"}
    rows = []
    for dataset in DATASETS:
        status = load_status(dataset, kind)
        condition = reference_condition[dataset]
        status = status.loc[status.condition.eq(condition) & status.attempted]
        participant = status.groupby(["subject_id", "algorithm_id", "candidate_name"], as_index=False).agg(
            attempted_n=("beat_id", "size"), missed_n=("missed", "sum"), invalid_n=("invalid", "sum")
        )
        participant["failure_rate_pct"] = 100 * (participant.missed_n + participant.invalid_n) / participant.attempted_n
        participant["dataset"] = dataset
        participant["condition"] = condition
        pooled = status.groupby(["algorithm_id"], as_index=False).agg(
            pooled_attempted_n=("beat_id", "size"), pooled_missed_n=("missed", "sum"), pooled_invalid_n=("invalid", "sum")
        )
        pooled["pooled_failure_rate_pct"] = 100 * (pooled.pooled_missed_n + pooled.pooled_invalid_n) / pooled.pooled_attempted_n
        rows.append(participant.merge(pooled, on="algorithm_id", validate="many_to_one"))
    data = pd.concat(rows, ignore_index=True)
    order = methods(kind)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0 if kind == "B" else 6.0), sharey=True, sharex=True, constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        subset = data.loc[data.dataset.eq(dataset)]
        for index, algorithm in enumerate(order):
            values = subset.loc[subset.algorithm_id.eq(algorithm), "failure_rate_pct"]
            q1, median, q3 = values.quantile([.25, .5, .75])
            pooled = subset.loc[subset.algorithm_id.eq(algorithm), "pooled_failure_rate_pct"].iloc[0]
            axis.plot([q1, q3], [index, index], color=STYLE["condition_colours"][reference_condition[dataset]], linewidth=2.2)
            axis.scatter(median, index, color=STYLE["condition_colours"][reference_condition[dataset]], s=29, marker="o", edgecolor="white", linewidth=.5, zorder=3)
            axis.scatter(pooled, index, facecolors="none", edgecolors="#333333", s=28, marker="D", linewidth=.8, zorder=4)
        condition_label = CONDITION_LABELS.get(reference_condition[dataset], reference_condition[dataset])
        axis.set_title(f"{dataset} · {condition_label}\nn={subset.subject_id.nunique()} participants")
        axis.set_xlabel("Detection failure rate (%)")
        axis.set_xlim(left=0)
    axes[0].set_yticks(y, [method_label(item) for item in order])
    axes[0].invert_yaxis()
    axes[0].set_ylabel(f"{kind}-point algorithm definition")
    axes[0].legend(handles=[
        Line2D([0], [0], marker="o", color="#555555", linewidth=2, markersize=5, label="Participant median and IQR"),
        Line2D([0], [0], marker="D", color="#333333", markerfacecolor="none", linestyle="none", markersize=5, label="Pooled rate"),
    ], frameon=False, loc="best")
    fig.suptitle(f"Cross-dataset {kind}-point detection failure patterns", fontsize=11)
    figure_dir, report_dir = output_dirs("HeartCycle", kind)
    stem = f"F08_cross_dataset_{kind}_detection_failure"
    save(fig, figure_dir / f"{stem}.png")
    data.to_csv(figure_data_dir("HeartCycle", kind) / f"{stem}.csv", index=False)
    maxima = data.groupby(["dataset", "algorithm_id"], as_index=False).pooled_failure_rate_pct.first().sort_values(["dataset", "pooled_failure_rate_pct"], ascending=[True, False]).groupby("dataset").head(1)
    result = "; ".join(f"{row.dataset}: {row.algorithm_id} {row.pooled_failure_rate_pct:.2f}%" for row in maxima.itertuples())
    report(report_dir / f"{stem}.md", "Do the same frozen definitions show similar failure patterns across recording environments?",
           "HeartCycle overall (17 participants), IITB rest (18), and ReBeat baseline (24); each panel retains its own participants and acquisition context.",
           "For each participant and method, failure = 100 × (missed + invalid) / eligible attempts. Circles and bars are participant medians/IQRs; open diamonds are pooled rates. Datasets are aligned visually but never pooled.",
           f"Y: {kind}-point definition. X: missed + invalid detection rate (%), with a common scale across dataset panels.",
           f"Largest pooled rates in each panel were {result}.",
           "Repeated high-failure patterns suggest definition-specific vulnerability, while differences in magnitude may depend on acquisition, cohort and protocol.",
           "HeartCycle, IITB and ReBeat differ in sampling, equipment, condition, preprocessing context and reference availability; this is a robustness comparison, not a pooled ranking or disease-effect analysis.",
           "Tests whether implementation-level failure patterns recur without conflating heterogeneous reference types.")


def write_dataset_comparability() -> None:
    table = pd.DataFrame([
        {"dataset": "HeartCycle", "participants": 17, "sampling_hz": "196.60–204.36", "equipment": "HeartCycle source acquisition; not harmonised with IITB/ReBeat", "displayed_condition": "Overall healthy cohort", "reference_availability": "ECHO AVO/AVC on eligible beats", "signal_quality_context": "Variable sampling; physical dZ/dt calibration unverified"},
        {"dataset": "IITB", "participants": 18, "sampling_hz": "500", "equipment": "IITB source acquisition; not harmonised with HeartCycle/ReBeat", "displayed_condition": "Rest (paired post-exercise available)", "reference_availability": "No independent B/X timing reference", "signal_quality_context": "Exercise-related physiology/motion contextual QC retained separately"},
        {"dataset": "ReBeat", "participants": 24, "sampling_hz": "250", "equipment": "ReBeat source acquisition; not harmonised with HeartCycle/IITB", "displayed_condition": "Baseline (paired cognitive workload available)", "reference_availability": "Expert ICG B/X annotation on validated subset", "signal_quality_context": "Two cognitive expert files absent; acquisition QC kept contextual"},
    ])
    table.to_csv(ROOT / "results" / "tables" / "RQ1_dataset_comparability.csv", index=False)


def figure8_cross_dataset(kind: str) -> None:
    reference_condition = {"HeartCycle": "overall", "IITB": "normal", "ReBeat": "baseline"}
    rows = []
    for dataset in DATASETS:
        status = load_status(dataset, kind)
        condition = reference_condition[dataset]
        status = status.loc[status.condition.eq(condition) & status.attempted]
        participant = status.groupby(["subject_id", "algorithm_id", "candidate_name"], as_index=False).agg(
            attempted_n=("beat_id", "size"), missed_n=("missed", "sum"), invalid_n=("invalid", "sum"))
        participant["failure_rate_pct"] = 100 * (participant.missed_n + participant.invalid_n) / participant.attempted_n
        participant["dataset"] = dataset; participant["condition"] = condition
        pooled = status.groupby("algorithm_id", as_index=False).agg(
            pooled_attempted_n=("beat_id", "size"), pooled_missed_n=("missed", "sum"), pooled_invalid_n=("invalid", "sum"))
        pooled["pooled_failure_rate_pct"] = 100 * (pooled.pooled_missed_n + pooled.pooled_invalid_n) / pooled.pooled_attempted_n
        rows.append(participant.merge(pooled, on="algorithm_id", validate="many_to_one"))
    data = pd.concat(rows, ignore_index=True)
    order = methods(kind); y = np.arange(len(order))
    full_max = max(5.0, float(data.pooled_failure_rate_pct.max()) * 1.08,
                   float(data.groupby(["dataset", "algorithm_id"]).failure_rate_pct.quantile(.75).max()) * 1.08)
    detail_limit = 5.0
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 8.0 if kind == "B" else 9.2), sharey=True,
                             gridspec_kw={"height_ratios": [1, 1]}, constrained_layout=True)
    for column, dataset in enumerate(DATASETS):
        subset = data.loc[data.dataset.eq(dataset)]
        colour = STYLE["condition_colours"][reference_condition[dataset]]
        for row_index, axis in enumerate(axes[:, column]):
            for index, algorithm in enumerate(order):
                values = subset.loc[subset.algorithm_id.eq(algorithm), "failure_rate_pct"]
                q1, median, q3 = values.quantile([.25, .5, .75])
                pooled_value = subset.loc[subset.algorithm_id.eq(algorithm), "pooled_failure_rate_pct"].iloc[0]
                axis.plot([q1, q3], [index, index], color=colour, linewidth=2.1)
                axis.scatter(median, index, color=colour, s=28, marker="o", edgecolor="white", linewidth=.5, zorder=3)
                axis.scatter(pooled_value, index, facecolors="none", edgecolors="#333333", s=27, marker="D", linewidth=.8, zorder=4)
            axis.set_xlim(0, full_max if row_index == 0 else detail_limit)
            axis.set_xlabel("Detection failure rate (%)")
            if row_index == 0:
                condition_label = CONDITION_LABELS.get(reference_condition[dataset], reference_condition[dataset])
                axis.set_title(f"{dataset} · {condition_label}\nn={subset.subject_id.nunique()} participants · complete 0–{full_max:.0f}% scale")
            else:
                axis.set_title(f"Magnified common 0–{detail_limit:.0f}% view\nvalues outside window remain in the complete view above", fontsize=8.5)
    axes[0, 0].set_yticks(y, [method_label(item) for item in order]); axes[0, 0].invert_yaxis()
    axes[0, 0].set_ylabel(f"{kind}-point algorithm definition"); axes[1, 0].set_ylabel(f"{kind}-point algorithm definition")
    axes[0, 0].legend(handles=[
        Line2D([0], [0], marker="o", color="#555555", linewidth=2, markersize=5, label="Participant median and IQR"),
        Line2D([0], [0], marker="D", color="#333333", markerfacecolor="none", linestyle="none", markersize=5, label="Pooled rate"),
    ], frameon=False, loc="best")
    fig.suptitle(f"Cross-dataset {kind}-point detection failure patterns", fontsize=11)
    figure_dir, report_dir = output_dirs("HeartCycle", kind)
    stem = f"F08_cross_dataset_{kind}_detection_failure"
    save(fig, figure_dir / f"{stem}.png")
    data.to_csv(figure_data_dir("HeartCycle", kind) / f"{stem}.csv", index=False)
    write_dataset_comparability()
    maxima = data.groupby(["dataset", "algorithm_id"], as_index=False).pooled_failure_rate_pct.first().sort_values(
        ["dataset", "pooled_failure_rate_pct"], ascending=[True, False]).groupby("dataset").head(1)
    result = "; ".join(f"{row.dataset}: {row.algorithm_id} {row.pooled_failure_rate_pct:.2f}%" for row in maxima.itertuples())
    report(report_dir / f"{stem}.md", "Do the same frozen definitions show similar failure patterns across recording environments?",
           "HeartCycle overall (17 participants), IITB rest (18), and ReBeat baseline (24); acquisition/reference differences are tabulated in RQ1_dataset_comparability.csv.",
           f"Failure = 100 × (missed + invalid) / eligible attempts. Circles/bars are participant medians/IQRs and open diamonds pooled rates. All upper panels share the honest 0–{full_max:.0f}% scale; every lower panel repeats the same values over the identical 0–{detail_limit:.0f}% magnified scale. No dataset-specific scale is used and no value is removed from the complete view.",
           f"Y: {kind}-point definition. X: missed + invalid detection rate (%); common complete and common magnified scales across all datasets.",
           f"Largest pooled rates in each dataset were {result}.",
           "Recurring high-failure definitions suggest implementation-level vulnerability, while the common magnified row makes verified low-rate differences visible.",
           "Sampling, equipment, protocols, conditions and references are not equivalent across datasets; this is a separated robustness comparison, not a pooled ranking or disease-effect analysis.",
           "Tests whether failure patterns recur while displaying rather than hiding acquisition non-comparability.")


def waveform_plot_data(dataset: str, kind: str, selected: pd.Series) -> pd.DataFrame:
    master = load_master(dataset, kind)
    beat = master.loc[master.beat_id.eq(selected.beat_id)].iloc[0]
    status = load_status(dataset, kind)
    detections = status.loc[status.beat_id.eq(selected.beat_id)].copy()
    recording, processed = recording_signal(dataset, beat.recording_id, beat.source_file)
    r, c, nr = beat_samples(dataset, beat)
    signal = np.asarray(processed.analysis_dzdt, float)
    raw = _legacy_api(load_config())[-1](recording.icg, recording.sampling_rate_hz)
    scale, raw_scale = np.std(signal[r:nr]), np.std(raw[r:nr])
    filtered_z = (signal - np.mean(signal[r:nr])) / scale
    raw_z = (raw - np.mean(raw[r:nr])) / raw_scale
    start, stop = ((max(r, r - int(.02 * recording.sampling_rate_hz)), min(nr, c + int(.06 * recording.sampling_rate_hz)))
                   if kind == "B" else (max(r, c - int(.06 * recording.sampling_rate_hz)), min(nr, c + int(.55 * (nr - c)))))
    x = 1000 * (recording.time_s[start:stop] - recording.time_s[r])
    waveform = pd.DataFrame({
        "row_type": "waveform", "dataset": dataset, "candidate_type": kind, "subject_id": beat.subject_id,
        "recording_id": beat.recording_id, "beat_id": beat.beat_id, "condition": beat.condition,
        "time_from_r_ms": x, "raw_derived_dzdt_z": raw_z[start:stop], "filtered_dzdt_z": filtered_z[start:stop],
    })
    reasons = pd.read_csv(ROOT / "work" / "p1_completion" / "phase4_failure_reason_records.csv", low_memory=False)
    reasons = reasons.loc[reasons.dataset.eq(dataset) & reasons.candidate_type.eq(kind) & reasons.beat_id.eq(beat.beat_id), ["algorithm_id", "primary_reason", "primary_reason_detail"]]
    detection_rows = detections.merge(reasons, on="algorithm_id", how="left")
    detection_rows = detection_rows.assign(
        row_type="algorithm_detection", dataset=dataset, condition=beat.condition,
        time_from_r_ms=detection_rows.candidate_time_from_r_ms,
    )
    reference, reference_name = reference_relative_ms(dataset, kind, beat.beat_id, recording.time_s[r])
    parts = [waveform, detection_rows]
    if reference is not None:
        parts.append(pd.DataFrame([{
            "row_type": "reference", "dataset": dataset, "candidate_type": kind, "subject_id": beat.subject_id,
            "recording_id": beat.recording_id, "beat_id": beat.beat_id, "condition": beat.condition,
            "time_from_r_ms": reference, "reference_type": reference_name,
        }]))
    return pd.concat(parts, ignore_index=True, sort=False)


def write_waveform_selection_audit() -> None:
    rows = []
    for dataset in DATASETS:
        for label_code, item in zip("ABCD", select_waveform_examples(dataset)):
            row = item["row"]
            stem = f"F01{label_code}_{DATASET_SLUG[dataset]}_{item['kind']}_{item['label'].lower().replace('/', '_').replace(' ', '_')}"
            data = waveform_plot_data(dataset, item["kind"], row)
            data.to_csv(figure_data_dir(dataset, item["kind"]) / f"{stem}.csv", index=False)
            failures = data.loc[data.row_type.eq("algorithm_detection") & data.analysis_status.isin(["INELIGIBLE", "MISSED", "INVALID"])]
            rows.append({
                "dataset": dataset, "panel": label_code, "category": item["label"], "candidate_type": item["kind"],
                "subject_id": row.subject_id, "recording_id": row.recording_id, "beat_id": row.beat_id,
                "condition": row.condition, "selection_criterion": item["criterion"],
                "timing_range_ms": row.get("timing_range_ms", row.get(f"{item['kind'].lower()}_range", np.nan)),
                "failed_or_ineligible_algorithm_n": len(failures),
                "verified_failure_reasons": "; ".join(f"{r.algorithm_id}: {r.primary_reason} ({r.primary_reason_detail})" for r in failures.itertuples()),
                "reference_present": bool(data.row_type.eq("reference").any()),
            })
    pd.DataFrame(rows).to_csv(OUT := ROOT / "work" / "p1_completion" / "phase7_waveform_case_selection.csv", index=False)


def export_plotting_data() -> None:
    participant_accuracy = pd.read_csv(ROOT / "results" / "tables" / "RQ1_reference_participant_accuracy.csv")
    reference_errors = pd.read_csv(ROOT / "results" / "tables" / "RQ1_reference_errors.csv", low_memory=False)
    paired = pd.read_csv(ROOT / "results" / "tables" / "RQ1_paired_condition_changes.csv")
    ba = pd.read_csv(ROOT / "results" / "tables" / "RQ1_bland_altman_values.csv", low_memory=False)
    for dataset in DATASETS:
        for kind in ["B", "X"]:
            directory = output_dirs(dataset, kind)[0]
            data_directory = figure_data_dir(dataset, kind)
            for image_path in directory.glob("*.png"):
                stem = image_path.stem
                target = data_directory / f"{stem}.csv"
                if stem.startswith("F01") or stem.startswith("F08_") or target.exists():
                    continue
                if stem.startswith("F02_"):
                    data = participant_accuracy.loc[(participant_accuracy.dataset.eq(dataset)) & participant_accuracy.candidate_type.eq(kind)]
                elif stem.startswith("F03_"):
                    data = reference_errors.loc[(reference_errors.dataset.eq(dataset)) & reference_errors.candidate_type.eq(kind)]
                    for condition in DATASETS[dataset]["conditions"]:
                        if stem.endswith(f"_{condition}"):
                            data = data.loc[data.condition.eq(condition)]
                elif stem.startswith("F04_"):
                    status = load_status(dataset, kind)
                    data = status.groupby(["subject_id", "condition", "algorithm_id", "candidate_name"], as_index=False).agg(
                        all_record_n=("beat_id", "size"), attempted_n=("attempted", "sum"), missed_n=("missed", "sum"),
                        invalid_n=("invalid", "sum"), upstream_ineligible_n=("attempted", lambda values: (~values).sum())
                    )
                    data["failure_rate_pct"] = 100 * (data.missed_n + data.invalid_n) / data.attempted_n
                    data["upstream_ineligible_pct"] = 100 * data.upstream_ineligible_n / data.all_record_n
                elif stem.startswith("F05A_"):
                    data = pd.read_csv(table_path(dataset, kind, "range_participant_summary"))
                elif stem.startswith("F05B_"):
                    data = pd.read_csv(table_path(dataset, kind, "definition_participant_summary"))
                elif stem.startswith("F06_"):
                    values = pd.read_csv(ROOT / "results" / "tables" / "RQ1_morphology_plot_values.csv", low_memory=False)
                    data = values.loc[(values.dataset.eq(dataset)) & values.candidate_type.eq(kind)]
                elif stem.startswith("F07_"):
                    outcome = "failure_rate" if "failure_rate" in stem else "within_beat_range"
                    data = paired.loc[(paired.dataset.eq(dataset)) & paired.candidate_type.eq(kind) & paired.outcome.eq(outcome)]
                elif stem.startswith("F09_"):
                    data = ba.loc[(ba.dataset.eq(dataset)) & ba.candidate_type.eq(kind)]
                    for condition in DATASETS[dataset]["conditions"]:
                        if stem.endswith(f"_{condition}"):
                            data = data.loc[data.condition.eq(condition)]
                else:
                    continue
                data.to_csv(target, index=False)


def write_legend_mapping() -> None:
    registry = pd.read_csv(ROOT / "configs" / "canonical_method_registry.csv")
    lines = ["# Final figure legend mapping", "", "## Algorithms", "", "| ID | Full definition name | Colour | Marker |", "|---|---|---|---|"]
    for row in registry.itertuples():
        lines.append(f"| {row.algorithm_id} | {row.candidate_name} | `{STYLE['algorithm_colours'][row.candidate_name]}` | `{STYLE['algorithm_markers'][row.candidate_name]}` |")
    lines += ["", "## Conditions", "", "| Condition | Colour | Marker |", "|---|---|---|"]
    for condition, colour in STYLE["condition_colours"].items():
        lines.append(f"| {condition} | `{colour}` | `{STYLE['condition_markers'][condition]}` |")
    lines += ["", "## Signed error", "", f"Negative/early: `{STYLE['signed_error_colours']['negative']}`; positive/late: `{STYLE['signed_error_colours']['positive']}`; zero: `{STYLE['signed_error_colours']['zero']}`.", "", "## Participants", "", "Participant colours are assigned deterministically within each dataset using the configured muted, colourblind-safe participant palette. The full rendered mapping is below.", "", "| Dataset | Participant | Colour |", "|---|---|---|"]
    for dataset in DATASETS:
        subjects = load_master(dataset, "B").subject_id.unique().tolist()
        for subject, colour in participant_colours(dataset, subjects).items():
            lines.append(f"| {dataset} | {subject} | `{colour}` |")
    report_root = OUTPUT_ROOT / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "FIGURE_LEGEND_MAPPING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_all_figures() -> None:
    # Mandated generation order: 4, 5, 7, 1, 2/3, 9, 6, 8.
    for dataset in DATASETS:
        for kind in ["B", "X"]:
            figure4_failure(dataset, kind)
    for dataset in DATASETS:
        for kind in ["B", "X"]:
            figure5_subject(dataset, kind); figure5_definition(dataset, kind)
    for dataset in ["IITB", "ReBeat"]:
        for kind in ["B", "X"]:
            figure7_changes(dataset, kind)
    for dataset in DATASETS:
        figure1_waveforms(dataset)
    for dataset in ["HeartCycle", "ReBeat"]:
        for kind in ["B", "X"]:
            figure2_accuracy(dataset, kind); figure3_signed_errors(dataset, kind)
    for dataset in DATASETS:
        for kind in ["B", "X"]:
            figure9_bland_altman(dataset, kind)
    for dataset in DATASETS:
        for kind in ["B", "X"]:
            figure6_method_specific(dataset, kind)
    for kind in ["B", "X"]:
        figure8_cross_dataset(kind)
    export_plotting_data()
    write_waveform_selection_audit()
    write_legend_mapping()
    print("Final RQ1 figures generated in mandated order.")


if __name__ == "__main__":
    build_all_figures()
