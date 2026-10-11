from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.data_loading import load_config
from src.final_analysis import KEYS, participant_balanced_limits, selected_pair
from src.final_plots import _recording_signals
from src.plotting import load_figure_style, set_style


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs"
RESULTS = ROOT / "results"
FACTORS = [
    ("sharpness", "Sharpness / curvature"),
    ("slope", "Local slope"),
    ("amplitude", "Amplitude / prominence"),
    ("snr", "Signal quality / SNR"),
    ("competing_features", "Competing features"),
    ("beat_to_beat_variability", "Beat-to-beat variability"),
]
DISPLAY_NAMES = {
    "B1": "Last local minimum", "B2": "3rd-deriv last ↓0", "B3": "3rd-deriv maximum",
    "B4": "45% upstroke", "B5": "25% upstroke", "B6": "2nd-deriv maximum",
    "B7": "2nd-deriv last ↑0", "B8": "55% R–C", "B9": "Adaptive 10%",
    "B10": "15% upstroke", "X1": "First local minimum", "X2": "Global minimum",
    "X3": "3rd-deriv first ↑0", "X4": "3rd-deriv minimum", "X5": "45% downstroke",
    "X6": "25% downstroke", "X7": "2nd-deriv first ↑0", "X8": "2nd-deriv after minimum",
    "X9": "2nd-deriv minimum", "X10": "Weissler HR", "X11": "Adaptive half descent",
    "X12": "RR-adaptive global min", "X13": "15% downstroke",
}


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def short_name(name: str) -> str:
    name = re.sub(r"^[BX]_", "", name)
    replacements = {
        "dzdt": "dZ/dt", "d2": "2nd derivative", "d3": "3rd derivative",
        "Cminus": "C−", "Cplus": "C+", "upstroke": "upstroke",
        "downstroke": "downstroke", "local": "local", "global": "global",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name.replace("_", " ")


def method_order(registry: pd.DataFrame, kind: str) -> list[str]:
    return registry.loc[registry.candidate_type.eq(kind), "algorithm_id"].tolist()


def method_labels(registry: pd.DataFrame, kind: str) -> dict[str, str]:
    rows = registry.loc[registry.candidate_type.eq(kind)]
    return {r.algorithm_id: f"{r.algorithm_id} · {DISPLAY_NAMES[r.algorithm_id]}" for r in rows.itertuples()}


def participant_colour(style: dict, subject: str) -> str:
    return style["participant_colours"].get(f"heartcycle_healthy:{subject}", "#66717A")


def display_id(value: str) -> str:
    return str(value).split(":")[-1].split("|")[-1]


def algorithm_colour(style: dict, row: pd.Series | object) -> str:
    return style["algorithm_colours"].get(getattr(row, "candidate_name", row["candidate_name"]), "#456B82")


def load_tables() -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame, dict, dict]:
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for kind in ("B", "X"):
        tables[kind] = {
            path.name: pd.read_csv(path, low_memory=False)
            for path in sorted((SOURCE / "tables" / kind).glob("*.csv"))
        }
    registry = pd.read_csv(ROOT / "configs" / "canonical_method_registry.csv")
    return tables, registry, load_config(), load_figure_style()


def copy_tables(tables: dict[str, dict[str, pd.DataFrame]]) -> None:
    for kind, collection in tables.items():
        target = RESULTS / "tables" / kind
        target.mkdir(parents=True, exist_ok=True)
        for name, table in collection.items():
            table.to_csv(target / name, index=False)


def joint_master(tables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    b = tables["B"]["B_master_beats.csv"].copy()
    x = tables["X"]["X_master_beats.csv"][KEYS + ["reference_event_time_s", "reference_eligible"]].copy()
    b = b.rename(columns={"reference_event_time_s": "echo_avo_time_s", "reference_eligible": "b_reference_eligibility"})
    x = x.rename(columns={"reference_event_time_s": "echo_avc_time_s", "reference_eligible": "x_reference_eligibility"})
    return b.merge(x, on=KEYS, validate="one_to_one")


def select_examples(tables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    master = joint_master(tables)
    spreads = []
    for kind in ("B", "X"):
        status = tables[kind][f"{kind}_detection_status.csv"]
        wide = status.loc[status.valid].pivot(index=KEYS, columns="algorithm_id", values="candidate_time_from_r_ms")
        spread = pd.DataFrame({f"{kind}_valid": wide.notna().sum(axis=1), f"{kind}_range_ms": wide.max(axis=1) - wide.min(axis=1)}).reset_index()
        spread[f"{kind}_complete"] = spread[f"{kind}_valid"].eq(status.algorithm_id.nunique())
        spreads.append(spread)
    summary = master.merge(spreads[0], on=KEYS).merge(spreads[1], on=KEYS)
    morphology = tables["B"]["B_morphology_features.csv"].drop_duplicates(KEYS)[KEYS + ["signal_to_noise_ratio_db"]]
    summary = summary.merge(morphology, on=KEYS, how="left")
    summary["participant_snr_median"] = summary.groupby("subject_id").signal_to_noise_ratio_db.transform("median")
    clean_pool = summary.loc[
        summary.B_complete & summary.X_complete & summary.b_reference_eligibility & summary.x_reference_eligibility &
        summary.signal_to_noise_ratio_db.ge(summary.participant_snr_median)
    ].copy()
    clean_pool["agreement_score"] = clean_pool.B_range_ms.rank(pct=True) + clean_pool.X_range_ms.rank(pct=True)
    clean = clean_pool.sort_values(["agreement_score", "B_range_ms", "X_range_ms"]).iloc[0]
    remaining = summary.loc[summary.B_complete & summary.X_complete & ~summary.beat_id.eq(clean.beat_id)]
    major_b = remaining.sort_values(["B_range_ms", "X_range_ms"], ascending=False).iloc[0]
    remaining = remaining.loc[~remaining.beat_id.eq(major_b.beat_id)]
    major_x = remaining.sort_values(["X_range_ms", "B_range_ms"], ascending=False).iloc[0]
    failures = pd.concat([
        tables[k][f"{k}_detection_status.csv"].assign(failed=lambda d: d.attempted & ~d.valid)
        for k in ("B", "X")
    ]).groupby(KEYS).failed.sum().reset_index()
    used = {clean.beat_id, major_b.beat_id, major_x.beat_id}
    failure_key = failures.loc[~failures.beat_id.isin(used)].sort_values("failed", ascending=False).iloc[0]
    failed = summary.merge(pd.DataFrame([failure_key[KEYS].to_dict()]), on=KEYS).iloc[0]
    rows = [
        ("F01A", "Clean agreement", clean, "all B/X methods valid; both references eligible; SNR at or above the participant median; minimum combined percentile rank of B and X ranges"),
        ("F01B", "Major B disagreement", major_b, "maximum B range among remaining beats with every B and X method valid"),
        ("F01C", "Major X disagreement", major_x, "maximum X range among remaining beats with every B and X method valid"),
        ("F01D", "Failed or ambiguous", failed, "maximum count of attempted missed or invalid B/X detections after excluding prior examples"),
    ]
    output = []
    for code, category, row, rule in rows:
        values = row.to_dict()
        values.update({"code": code, "category": category, "selection_rule": rule, "failed_methods": int(failures.merge(pd.DataFrame([row[KEYS].to_dict()]), on=KEYS).failed.iloc[0])})
        output.append(values)
    return pd.DataFrame(output)


def plot_f01(selection: pd.DataFrame, tables: dict[str, dict[str, pd.DataFrame]], registry: pd.DataFrame,
             config: dict, style: dict) -> list[dict]:
    master = joint_master(tables)
    outputs = []
    for selected in selection.itertuples(index=False):
        row = master.merge(pd.DataFrame([{k: getattr(selected, k) for k in KEYS}]), on=KEYS).iloc[0]
        recording, processed, _, start, stop, time_ms = _recording_signals(row, config)
        signal = processed.analysis_dzdt[start:stop]
        fig, ax = plt.subplots(figsize=(8.2, 4.1))
        ax.plot(time_ms, signal, color="#263F4F", lw=1.35)
        kinds = ("B", "X") if selected.code in ("F01A", "F01D") else (("B",) if selected.code == "F01B" else ("X",))
        handles = []
        for kind in kinds:
            status = tables[kind][f"{kind}_detection_status.csv"]
            beat = status.loc[np.logical_and.reduce([status[k].eq(getattr(selected, k)) for k in KEYS])]
            marker = "^" if kind == "B" else "v"
            for candidate in beat.loc[beat.valid].itertuples(index=False):
                y = np.interp(candidate.candidate_timestamp_s, recording.time_s, processed.analysis_dzdt)
                colour = style["algorithm_colours"][candidate.candidate_name]
                ax.scatter(candidate.candidate_time_from_r_ms, y, s=48, marker=marker, color=colour,
                           edgecolor="white", linewidth=.45, zorder=4)
            handles.append(Line2D([0], [0], marker=marker, linestyle="none", color="#536A78", label=f"{kind}-point methods"))
            event = row.echo_avo_time_s if kind == "B" else row.echo_avc_time_s
            eligible = row.b_reference_eligibility if kind == "B" else row.x_reference_eligibility
            if bool(eligible) and pd.notna(event):
                ref_name = "ECHO AVO" if kind == "B" else "ECHO AVC"
                ref = style["reference_styles"][ref_name]
                xref = 1000 * (event - row.r_peak_time_s)
                ax.axvline(xref, color=ref["colour"], linestyle=ref["linestyle"], lw=1.4, label=ref_name)
        ax.set(xlabel="Time relative to ECG R-peak (ms)", ylabel="Filtered dZ/dt (device units/s)")
        ax.set_title(
            f"{selected.code}  {selected.category} — {display_id(selected.subject_id)}\n"
            f"Recording {display_id(selected.recording_id)} · beat {display_id(selected.beat_id)}",
            loc="left", fontweight="semibold"
        )
        existing, labels = ax.get_legend_handles_labels()
        ax.legend(handles + existing, [h.get_label() for h in handles] + labels, frameon=False, ncol=2, loc="best")
        folder = "B" if selected.code in ("F01A", "F01B") else "X"
        filename = f"{selected.code}_{selected.category.lower().replace(' ', '_')}.png"
        path = RESULTS / "figures" / folder / filename
        save(fig, path)
        outputs.append({"code": selected.code, "kind": folder, "filename": filename, "selection": selected})
    return outputs


def plot_accuracy(kind: str, tables: dict[str, pd.DataFrame], registry: pd.DataFrame, style: dict) -> dict:
    participant = tables[f"{kind}_participant_summary.csv"]
    methods, labels = method_order(registry, kind), method_labels(registry, kind)
    fig, ax = plt.subplots(figsize=(max(9, len(methods) * .72), 5.2))
    values = [participant.loc[participant.algorithm_id.eq(m), "median_absolute_error_ms"].dropna() for m in methods]
    ax.boxplot(values, positions=np.arange(len(methods)), widths=.52, patch_artist=True, showfliers=False,
               boxprops={"facecolor": "#D8DEE2", "edgecolor": "#536A78"}, medianprops={"color": "#172B36", "linewidth": 1.2},
               whiskerprops={"color": "#536A78"}, capprops={"color": "#536A78"})
    for pos, method in enumerate(methods):
        rows = participant.loc[participant.algorithm_id.eq(method)].sort_values("subject_id")
        offsets = np.linspace(-.13, .13, len(rows))
        for offset, row in zip(offsets, rows.itertuples()):
            ax.scatter(pos + offset, row.median_absolute_error_ms, s=25, marker="o",
                       color=participant_colour(style, row.subject_id), edgecolor="white", linewidth=.35, zorder=3)
    ax.set_xticks(np.arange(len(methods)), [textwrap.fill(labels[m], 18) for m in methods], rotation=35, ha="right")
    common_beats = tables[f"{kind}_reference_errors.csv"].beat_id.nunique()
    ax.set(xlabel=f"{kind}-point algorithm definition", ylabel="Participant median absolute timing error (ms)",
           title=f"{kind}-point reference accuracy\n17 participants · {common_beats:,} strict common beats")
    fig.subplots_adjust(bottom=.30)
    path = RESULTS / "figures" / kind / f"F02_{kind}_reference_accuracy.png"
    save(fig, path)
    return {"filename": path.name, "values": values}


def plot_histograms(kind: str, tables: dict[str, pd.DataFrame], registry: pd.DataFrame) -> dict:
    errors = tables[f"{kind}_reference_errors.csv"]
    methods, labels = method_order(registry, kind), method_labels(registry, kind)
    limit = float(np.ceil(errors.signed_error_ms.abs().quantile(.995) / 25) * 25)
    bins = np.linspace(-limit, limit, 31)
    counts = [np.histogram(errors.loc[errors.algorithm_id.eq(m), "signed_error_ms"], bins=bins)[0] for m in methods]
    ymax = max(int(c.max()) for c in counts)
    cols, rows = 3, math.ceil(len(methods) / 3)
    fig, axes = plt.subplots(rows, cols, figsize=(10.8, rows * 2.35), sharex=True, sharey=True)
    for ax, method, count in zip(np.ravel(axes), methods, counts):
        centres, widths = (bins[:-1] + bins[1:]) / 2, np.diff(bins)
        ax.bar(centres[centres < 0], count[centres < 0], width=widths[centres < 0], color="#50789A")
        ax.bar(centres[centres >= 0], count[centres >= 0], width=widths[centres >= 0], color="#B26050")
        ax.axvline(0, color="#555B60", lw=1)
        ax.set_title(textwrap.fill(labels[method], 27), fontsize=8.5)
        ax.set_xlim(-limit, limit); ax.set_ylim(0, ymax * 1.04)
    for ax in np.ravel(axes)[len(methods):]: ax.set_visible(False)
    fig.supxlabel("Signed timing error: detected minus ECHO reference (ms)")
    fig.supylabel("Number of heartbeats")
    fig.suptitle(f"{kind}-point signed timing-error distributions", fontweight="semibold")
    fig.tight_layout(rect=(.03, .03, 1, .97))
    path = RESULTS / "figures" / kind / f"F03_{kind}_signed_error.png"
    save(fig, path)
    return {"filename": path.name, "limit": limit}


def bootstrap_ci(values: pd.Series, seed: int) -> tuple[float, float, float]:
    values = values.dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    med = float(np.median(values))
    boot = np.median(rng.choice(values, (4000, len(values)), replace=True), axis=1)
    return med, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def plot_failure(kind: str, tables: dict[str, pd.DataFrame], registry: pd.DataFrame) -> dict:
    status, participant = tables[f"{kind}_detection_status.csv"], tables[f"{kind}_participant_summary.csv"]
    methods, labels = method_order(registry, kind), method_labels(registry, kind)
    attempted = status.loc[status.attempted].groupby("algorithm_id").agg(denominator=("beat_id", "size"), missed=("missed", "sum"), invalid=("invalid", "sum")).reindex(methods)
    attempted[["missed_pct", "invalid_pct"]] = 100 * attempted[["missed", "invalid"]].div(attempted.denominator, axis=0)
    cis = [bootstrap_ci(participant.loc[participant.algorithm_id.eq(m), "total_failure_rate_pct"], 20261009 + i) for i, m in enumerate(methods)]
    x = np.arange(len(methods)); med = np.array([c[0] for c in cis]); low = np.array([c[1] for c in cis]); high = np.array([c[2] for c in cis])
    fig, ax = plt.subplots(figsize=(max(9, len(methods) * .72), 5.2))
    ax.bar(x, attempted.missed_pct, color="#C7A348", width=.65, label="Missed")
    ax.bar(x, attempted.invalid_pct, bottom=attempted.missed_pct, color="#A95B4C", width=.65, label="Invalid")
    ax.errorbar(x, med, yerr=[med-low, high-med], fmt="o", color="#263F4F", mfc="white", ms=4.5, lw=1,
                capsize=2.5, label="Participant median failure (95% bootstrap CI)")
    tick_labels = [textwrap.fill(labels[m], 18) + f"\nn={int(attempted.loc[m, 'denominator']):,}" for m in methods]
    ax.set_xticks(x, tick_labels, rotation=35, ha="right")
    ax.set(xlabel=f"{kind}-point algorithm definition", ylabel="Missed + invalid detection rate (%)", title=f"{kind}-point detection failure")
    ax.legend(frameon=False, ncol=3)
    fig.subplots_adjust(bottom=.30)
    path = RESULTS / "figures" / kind / f"F04_{kind}_detection_failure.png"
    save(fig, path)
    return {"filename": path.name, "summary": attempted.reset_index(), "ci": cis}


def complete_timing(status: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    wide = status.loc[status.valid].pivot(index=KEYS, columns="algorithm_id", values="candidate_time_from_r_ms")
    return wide.reindex(columns=methods).dropna()


def plot_subject_disagreement(kind: str, tables: dict[str, pd.DataFrame], registry: pd.DataFrame, style: dict) -> dict:
    methods = method_order(registry, kind)
    timing = complete_timing(tables[f"{kind}_detection_status.csv"], methods)
    spread = (timing.max(axis=1) - timing.min(axis=1)).rename("range_ms").reset_index()
    summary = spread.groupby("subject_id").range_ms.agg(median="median", p95=lambda x: x.quantile(.95), beats="size").reset_index().sort_values("subject_id")
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    for pos, row in zip(x, summary.itertuples()):
        colour = participant_colour(style, row.subject_id)
        ax.vlines(pos, row.median, row.p95, color=colour, lw=2)
        ax.scatter([pos, pos], [row.median, row.p95], s=[32, 42], marker="o", color=colour, edgecolor="white", linewidth=.4, zorder=3)
    ax.set_xticks(x, [display_id(v) for v in summary.subject_id], rotation=45, ha="right")
    ax.set(xlabel="Participant ID", ylabel="Within-beat algorithm timing spread (ms)", title=f"{kind}-point participant-wise disagreement")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="#536A78", label="Participant median"), Line2D([0], [0], marker="o", color="#536A78", markersize=7, label="Participant P95")], frameon=False)
    path = RESULTS / "figures" / kind / f"F05A_{kind}_subject_disagreement.png"
    save(fig, path)
    return {"filename": path.name, "timing": timing, "summary": summary}


def definition_disagreement(timing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in timing.columns:
        values = (timing[method] - timing.drop(columns=method).median(axis=1)).abs().rename("absolute_peer_difference_ms").reset_index()
        values["algorithm_id"] = method
        rows.append(values)
    beat = pd.concat(rows, ignore_index=True)
    return beat.groupby(["subject_id", "algorithm_id"]).absolute_peer_difference_ms.agg(median="median", beats="size").reset_index()


def plot_definition_disagreement(kind: str, timing: pd.DataFrame, registry: pd.DataFrame, style: dict) -> dict:
    participant = definition_disagreement(timing)
    methods, labels = method_order(registry, kind), method_labels(registry, kind)
    fig, ax = plt.subplots(figsize=(max(9, len(methods) * .72), 5.2))
    values = [participant.loc[participant.algorithm_id.eq(m), "median"] for m in methods]
    ax.boxplot(values, positions=np.arange(len(methods)), widths=.52, patch_artist=True, showfliers=False,
               boxprops={"facecolor": "#D8DEE2", "edgecolor": "#536A78"}, medianprops={"color": "#172B36"}, whiskerprops={"color": "#536A78"}, capprops={"color": "#536A78"})
    for pos, method in enumerate(methods):
        rows = participant.loc[participant.algorithm_id.eq(method)].sort_values("subject_id")
        for offset, row in zip(np.linspace(-.13, .13, len(rows)), rows.itertuples()):
            ax.scatter(pos+offset, row.median, s=24, marker="o", color=participant_colour(style, row.subject_id), edgecolor="white", linewidth=.3)
    ax.set_xticks(np.arange(len(methods)), [textwrap.fill(labels[m], 18) for m in methods], rotation=35, ha="right")
    ax.set(xlabel=f"{kind}-point algorithm definition", ylabel="Participant median absolute difference from peer median (ms)", title=f"{kind}-point definition-wise disagreement")
    fig.subplots_adjust(bottom=.30)
    path = RESULTS / "figures" / kind / f"F05B_{kind}_definition_disagreement.png"
    save(fig, path)
    return {"filename": path.name, "participant": participant}


def plot_bland_altman(kind: str, timing: pd.DataFrame, registry: pd.DataFrame, style: dict) -> dict:
    a, b = selected_pair(kind)
    pair = timing[[a, b]].reset_index()
    pair["mean_ms"] = pair[[a, b]].mean(axis=1)
    pair["difference_ms"] = pair[a] - pair[b]
    bias, low, high = participant_balanced_limits(pair)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for subject, rows in pair.groupby("subject_id"):
        ax.scatter(rows.mean_ms, rows.difference_ms, s=22, marker="o", alpha=.72, color=participant_colour(style, subject), edgecolor="white", linewidth=.25)
    for value, linestyle in ((bias, "-"), (low, "--"), (high, "--")):
        ax.axhline(value, color="#4E5960", linestyle=linestyle, lw=1.1)
    ax.set(xlabel=f"Mean {a}/{b} time relative to ECG R (ms)", ylabel=f"{a} minus {b} detection time (ms)", title=f"{kind}-point selected-pair agreement: {a} versus {b}")
    path = RESULTS / "figures" / kind / f"F05C_{kind}_selected_pair_bland_altman.png"
    save(fig, path)
    return {"filename": path.name, "pair": pair, "bias": bias, "low": low, "high": high, "a": a, "b": b}


def morphology_examples(kind: str, tables: dict[str, pd.DataFrame]) -> tuple[pd.Series, pd.Series]:
    morphology = tables[f"{kind}_morphology_features.csv"].drop_duplicates(KEYS).copy()
    morphology["clarity_rank"] = morphology.signal_to_noise_ratio_db.rank(pct=True) + morphology.reference_local_sharpness_abs_d2_device_units_s3.rank(pct=True) - morphology.reference_region_competing_features_count.rank(pct=True)
    eligible = morphology.loc[morphology.reference_eligible]
    clear = eligible.sort_values("clarity_rank", ascending=False).iloc[0]
    ambiguous = eligible.loc[~eligible.beat_id.eq(clear.beat_id)].sort_values("clarity_rank").iloc[0]
    return clear, ambiguous


def plot_morphology_waveform(kind: str, label: str, chosen: pd.Series, tables: dict[str, pd.DataFrame],
                             config: dict, style: dict) -> dict:
    master = tables[f"{kind}_master_beats.csv"]
    row = master.merge(pd.DataFrame([chosen[KEYS].to_dict()]), on=KEYS).iloc[0]
    recording, processed, _, start, stop, time_ms = _recording_signals(row, config)
    signal = processed.analysis_dzdt[start:stop]
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.plot(time_ms, signal, color="#263F4F", lw=1.35)
    status = tables[f"{kind}_detection_status.csv"]
    beat = status.loc[np.logical_and.reduce([status[k].eq(chosen[k]) for k in KEYS])]
    marker = "^" if kind == "B" else "v"
    for candidate in beat.loc[beat.valid].itertuples(index=False):
        y = np.interp(candidate.candidate_timestamp_s, recording.time_s, processed.analysis_dzdt)
        ax.scatter(candidate.candidate_time_from_r_ms, y, s=48, marker=marker, color=style["algorithm_colours"][candidate.candidate_name], edgecolor="white", linewidth=.4)
    ref_ms = 1000 * (row.reference_event_time_s - row.r_peak_time_s)
    ref_name = "ECHO AVO" if kind == "B" else "ECHO AVC"
    ref = style["reference_styles"][ref_name]
    ax.axvspan(ref_ms-20, ref_ms+20, color="#D8DEE2", alpha=.55, label="Reference neighbourhood (±20 ms display window)")
    ax.axvline(ref_ms, color=ref["colour"], linestyle=ref["linestyle"], lw=1.4, label=ref_name)
    descriptor = "high relative SNR/sharpness with fewer competing features" if label == "clear" else "low relative SNR/sharpness or more competing features"
    ax.set(
        xlabel="Time relative to ECG R-peak (ms)", ylabel="Filtered dZ/dt (device units/s)",
        title=f"{kind}-point {label} reference-region morphology — {display_id(row.subject_id)}\n"
              f"Recording {display_id(row.recording_id)} · beat {display_id(row.beat_id)}"
    )
    ax.text(.01, .02, descriptor, transform=ax.transAxes, fontsize=8, color="#4E5960")
    ax.legend(frameon=False)
    code = "F06A" if label == "clear" else "F06B"
    path = RESULTS / "figures" / kind / f"{code}_{kind}_{label}_morphology.png"
    save(fig, path)
    return {"filename": path.name, "chosen": chosen, "descriptor": descriptor, "code": code}


def plot_morphology_error(kind: str, tables: dict[str, pd.DataFrame], registry: pd.DataFrame, style: dict) -> dict:
    morphology = tables[f"{kind}_morphology_features.csv"]
    methods, labels = method_order(registry, kind), method_labels(registry, kind)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4), sharey=True)
    summaries, counts = [], []
    for ax, (factor, title) in zip(np.ravel(axes), FACTORS):
        category = f"{factor}_category"
        rows = morphology.loc[morphology.strict_common_reference & morphology[category].ne("unavailable")]
        participant = rows.groupby(["subject_id", "algorithm_id", category], observed=True).absolute_error_ms.median().reset_index(name="participant_median_error_ms")
        categories = list(dict.fromkeys(rows[category].tolist()))
        categories = sorted(categories, key=lambda x: (x in ("high", "variable"), x))
        for subject, subject_rows in participant.groupby("subject_id"):
            seed = int(hashlib.sha256(f"{kind}|{factor}|{subject}".encode()).hexdigest()[:8], 16)
            ax.scatter(subject_rows[category].map({c:i for i,c in enumerate(categories)}) + np.random.default_rng(seed).normal(0, .025, len(subject_rows)),
                       subject_rows.participant_median_error_ms, s=7, marker="o", color=participant_colour(style, subject), alpha=.10, linewidth=0)
        for method in methods:
            med = participant.loc[participant.algorithm_id.eq(method)].groupby(category).participant_median_error_ms.median().reindex(categories)
            colour = style["algorithm_colours"][registry.set_index("algorithm_id").at[method, "candidate_name"]]
            ax.plot(range(len(categories)), med, marker="o", ms=3.2, lw=.9, color=colour, alpha=.9)
            for cat, value in med.items(): summaries.append({"factor": factor, "category": cat, "algorithm_id": method, "median_participant_error_ms": value})
        count = rows.drop_duplicates(KEYS).groupby(category).agg(beats=("beat_id", "size"), participants=("subject_id", "nunique")).reset_index().rename(columns={category: "category"})
        count["factor"] = factor; counts.append(count)
        ax.set_xticks(range(len(categories)), [c.replace("_", " ").title() for c in categories])
        ax.set_title(title)
        ax.set_xlabel("Morphology / quality category")
    axes[0,0].set_ylabel("Participant median absolute reference error (ms)")
    axes[1,0].set_ylabel("Participant median absolute reference error (ms)")
    handles = [Line2D([0],[0], marker="o", color=style["algorithm_colours"][r.candidate_name], lw=1, ms=3.5, label=f"{r.algorithm_id} · {DISPLAY_NAMES[r.algorithm_id]}") for r in registry.loc[registry.candidate_type.eq(kind)].itertuples()]
    fig.legend(handles=handles, frameon=False, ncol=2, bbox_to_anchor=(1.01, .5), loc="center left", fontsize=7)
    fig.suptitle(f"{kind}-point morphology category versus reference error", fontweight="semibold")
    fig.tight_layout(rect=(0,0,.82,.96))
    path = RESULTS / "figures" / kind / f"F06C_{kind}_morphology_vs_error.png"
    save(fig, path)
    return {"filename": path.name, "summary": pd.DataFrame(summaries), "counts": pd.concat(counts, ignore_index=True)}


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, digits: int = 1) -> str:
    frame = frame[columns].copy() if columns else frame.copy()
    for column in frame.select_dtypes(include="number"):
        frame[column] = frame[column].round(digits)
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def report(path: Path, title: str, sections: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [f"# {title}"]
    for i, heading in enumerate(("Question", "Dataset", "Algorithms", "Axes and units", "N", "Key numerical result", "Interpretation", "Limitation", "Decision"), 1):
        text.extend(["", f"## {i}. {heading}", "", sections[heading]])
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def algorithm_text(registry: pd.DataFrame, kind: str) -> str:
    return "; ".join(f"{r.algorithm_id} = {r.candidate_name}" for r in registry.loc[registry.candidate_type.eq(kind)].itertuples())


def create_reports(tables: dict[str, dict[str, pd.DataFrame]], registry: pd.DataFrame, selection: pd.DataFrame,
                   metadata: dict[str, dict]) -> None:
    for row in metadata["F01"]:
        selected = row["selection"]
        kinds = "B and X" if selected.code in ("F01A", "F01D") else ("B" if selected.code == "F01B" else "X")
        relevant = []
        for kind in ("B", "X"):
            status = tables[kind][f"{kind}_detection_status.csv"]
            beat = status.loc[np.logical_and.reduce([status[k].eq(getattr(selected, k)) for k in KEYS])]
            failed = beat.loc[beat.attempted & ~beat.valid, "algorithm_id"].tolist()
            if failed: relevant.append(f"{kind} failed/invalid: {', '.join(failed)}")
        report(RESULTS / "reports" / row["kind"] / row["filename"].replace(".png", ".md"), f"{selected.code} {selected.category}", {
            "Question": "What does the filtered ICG waveform show when methods agree, disagree or fail?",
            "Dataset": f"HeartCycle; participant {selected.subject_id}, recording {selected.recording_id}, beat {selected.beat_id}. Figure: `{row['filename']}`. Source tables: B/X master and detection-status tables.",
            "Algorithms": f"Displayed point types: {kinds}. " + " ".join(algorithm_text(registry, k) for k in ("B","X") if k in kinds),
            "Axes and units": "x = time relative to detected ECG R (ms); y = filtered dZ/dt (device units/s; physical impedance calibration is unverified).",
            "N": "One reproducibly selected beat. Selection rule: " + selected.selection_rule,
            "Key numerical result": f"B range = {selected.B_range_ms:.1f} ms; X range = {selected.X_range_ms:.1f} ms; attempted failed/invalid methods = {selected.failed_methods}. " + ("; ".join(relevant) if relevant else "No attempted failures in this example."),
            "Interpretation": "Marker placement on the same waveform shows whether definitions converge on one landmark or select different local features.",
            "Limitation": "This is an illustrative deterministic example, not a population estimate. Filtered-only display intentionally simplifies the original raw/filtered request; filtering can suppress high-frequency morphology.",
            "Decision": "Use this example only to explain waveform-level behaviour; quantitative accuracy, failure, and disagreement are reported in F02–F05.",
        })
    for kind in ("B", "X"):
        t, ref = tables[kind], ("ECHO AVO" if kind == "B" else "ECHO AVC")
        participant, errors, status = t[f"{kind}_participant_summary.csv"], t[f"{kind}_reference_errors.csv"], t[f"{kind}_detection_status.csv"]
        algorithms = algorithm_text(registry, kind)
        ordered = method_order(registry, kind)
        acc = participant.groupby("algorithm_id").median_absolute_error_ms.agg(median="median", q1=lambda x:x.quantile(.25), q3=lambda x:x.quantile(.75)).reindex(ordered).reset_index()
        acc["IQR_ms"] = acc.q3-acc.q1
        winners = participant.loc[participant.groupby("subject_id").median_absolute_error_ms.idxmin(), ["subject_id","algorithm_id","median_absolute_error_ms"]].sort_values("subject_id")
        report(RESULTS/"reports"/kind/f"F02_{kind}_reference_accuracy.md", f"F02 {kind} reference accuracy", {
            "Question": f"Which {kind}-point method is closest to {ref}, and how consistent is performance across participants?",
            "Dataset": f"HeartCycle strict common reference set. Figure: `F02_{kind}_reference_accuracy.png`; sources: `{kind}_reference_errors.csv`, `{kind}_participant_summary.csv`.",
            "Algorithms": algorithms,
            "Axes and units": f"x = {kind}-point definition; y = participant median absolute detected-minus-{ref} error (ms). Circular dots are participants.",
            "N": f"{errors.subject_id.nunique()} participants, {errors.recording_id.nunique()} recordings represented, {errors.beat_id.nunique()} identical common beats per algorithm.",
            "Key numerical result": markdown_table(acc, ["algorithm_id","median","IQR_ms"]) + "\n\nParticipant-wise lowest median error:\n\n" + markdown_table(winners),
            "Interpretation": f"The lowest overall participant median is {acc.loc[acc['median'].idxmin(),'algorithm_id']}; the smallest between-participant IQR is {acc.loc[acc.IQR_ms.idxmin(),'algorithm_id']}. Participant winners show whether performance is broadly consistent or person-dependent.",
            "Limitation": "A narrow participant box is not evidence of low beat-to-beat variability. ECHO and ICG landmarks may have physiological offset, and ECHO uses a separately aligned time base.",
            "Decision": "Use strict-common reference accuracy together with signed bias and failure frequency; do not select a method from one summary alone.",
        })
        signed = errors.groupby("algorithm_id").agg(median_signed_error_ms=("signed_error_ms","median"), signed_error_IQR_ms=("signed_error_ms",lambda x:x.quantile(.75)-x.quantile(.25)), P90_absolute_error_ms=("absolute_error_ms",lambda x:x.quantile(.90)), P95_absolute_error_ms=("absolute_error_ms",lambda x:x.quantile(.95)), beats=("beat_id","size"), participants=("subject_id","nunique")).reindex(ordered).reset_index()
        report(RESULTS/"reports"/kind/f"F03_{kind}_signed_error.md", f"F03 {kind} signed timing error", {
            "Question": f"Does each {kind}-point definition detect systematically earlier or later than {ref}, and how large are its tails?",
            "Dataset": f"HeartCycle strict common reference set. Figure: `F03_{kind}_signed_error.png`; source: `{kind}_reference_errors.csv`.",
            "Algorithms": algorithms,
            "Axes and units": "x = detected minus ECHO reference timing error (ms); y = number of heartbeats. Negative is earlier, positive is later.",
            "N": f"{errors.subject_id.nunique()} participants and {errors.beat_id.nunique()} beats per algorithm.",
            "Key numerical result": markdown_table(signed),
            "Interpretation": "Signed medians establish bias direction; IQR and absolute-error P90/P95 distinguish broad distributions and tail errors.",
            "Limitation": "No clinically or analytically validated tolerance was supplied, so percentage-outside-tolerance is unavailable and was not fabricated.",
            "Decision": "Report directional bias and tails alongside F02 absolute error.",
        })
        failure = metadata[kind]["failure"]["summary"]
        ci = metadata[kind]["failure"]["ci"]
        failure["failure_pct"] = failure.missed_pct + failure.invalid_pct
        failure["participant_median_failure_pct"] = [v[0] for v in ci]
        failure["participant_95CI_low"] = [v[1] for v in ci]; failure["participant_95CI_high"] = [v[2] for v in ci]
        report(RESULTS/"reports"/kind/f"F04_{kind}_detection_failure.md", f"F04 {kind} detection failure", {
            "Question": f"Which {kind}-point definitions most often miss or return an invalid detection?",
            "Dataset": f"All attempted HeartCycle beats. Figure: `F04_{kind}_detection_failure.png`; sources: `{kind}_detection_status.csv`, `{kind}_participant_summary.csv`.",
            "Algorithms": algorithms,
            "Axes and units": f"x = {kind}-point definition; y = missed + invalid detections among eligible attempted beats (%). Bars partition missed and invalid; points/intervals are participant median total failure with 95% participant bootstrap CI.",
            "N": f"17 participants, 212 recordings, 1,706 master beats; algorithm-specific attempted denominators are shown below.\n\n" + markdown_table(failure, ["algorithm_id","denominator","missed_pct","invalid_pct","failure_pct","participant_median_failure_pct","participant_95CI_low","participant_95CI_high"]),
            "Key numerical result": f"Highest overall failure: {failure.loc[failure.failure_pct.idxmax(),'algorithm_id']} ({failure.failure_pct.max():.2f}%). Lowest overall failure: {failure.loc[failure.failure_pct.idxmin(),'algorithm_id']} ({failure.failure_pct.min():.2f}%).",
            "Interpretation": "The plot isolates algorithm reliability; missed and invalid categories are mutually exclusive and ECHO absence is not a detection failure.",
            "Limitation": "This is an overall HeartCycle comparison, not the supervisor's unavailable rest-versus-exercise comparison. Bootstrap intervals summarize participant heterogeneity rather than binomial beat independence.",
            "Decision": "Consider failure burden jointly with reference timing error.",
        })
        subject = metadata[kind]["subject"]
        report(RESULTS/"reports"/kind/f"F05A_{kind}_subject_disagreement.md", f"F05A {kind} participant-wise disagreement", {
            "Question": f"Which participants show greater same-beat disagreement across all {kind}-point definitions?",
            "Dataset": f"HeartCycle all-method-valid beats. Figure: `{subject['filename']}`; source: `{kind}_detection_status.csv`.",
            "Algorithms": algorithms,
            "Axes and units": "x = participant ID; y = within-beat maximum minus minimum algorithm timing (ms). Points show participant median and P95.",
            "N": f"17 participants and {len(subject['timing'])} complete identical beats. Participant counts:\n\n" + markdown_table(subject["summary"], ["subject_id","beats"]),
            "Key numerical result": markdown_table(subject["summary"]),
            "Interpretation": "Larger participant ranges mean definitions select more widely separated locations on the same beats.",
            "Limitation": "Across-method range is not ECHO reference error and can be driven by one outlying definition.",
            "Decision": "Use this result to localize participant heterogeneity in method sensitivity.",
        })
        definition = metadata[kind]["definition"]
        dsum = definition["participant"].groupby("algorithm_id")["median"].agg(
            overall_median_ms="median", IQR_ms=lambda x: x.quantile(.75)-x.quantile(.25)
        ).reindex(ordered).reset_index()
        report(RESULTS/"reports"/kind/f"F05B_{kind}_definition_disagreement.md", f"F05B {kind} definition-wise disagreement", {
            "Question": f"Which {kind}-point definitions tend to lie farther from the other definitions' median location?",
            "Dataset": f"HeartCycle all-method-valid beats. Figure: `{definition['filename']}`; source: `{kind}_detection_status.csv`.",
            "Algorithms": algorithms,
            "Axes and units": f"x = {kind}-point definition; y = participant median absolute target-time minus median-of-other-methods time (ms).",
            "N": f"17 participants and {len(subject['timing'])} identical complete beats per definition.",
            "Key numerical result": markdown_table(dsum),
            "Interpretation": f"{dsum.loc[dsum.overall_median_ms.idxmax(),'algorithm_id']} deviates most from the peer-method median by this disagreement metric; this does not establish inaccuracy.",
            "Limitation": "Peer consensus is not ground truth. This leave-one-target calculation defines disagreement only; it is not a sensitivity analysis and does not alter the method set.",
            "Decision": "Use it to identify definitions selecting systematically different morphological landmarks, then interpret against F02/F03 reference evidence.",
        })
        ba = metadata[kind]["ba"]
        report(RESULTS/"reports"/kind/f"F05C_{kind}_selected_pair_bland_altman.md", f"F05C {kind} selected-pair agreement", {
            "Question": f"Is the timing difference between related definitions {ba['a']} and {ba['b']} systematic?",
            "Dataset": f"HeartCycle all-method-valid beats. Figure: `{ba['filename']}`; source: `{kind}_detection_status.csv`.",
            "Algorithms": f"Selected pair {ba['a']} versus {ba['b']}; exact catalogue: {algorithms}",
            "Axes and units": f"x = mean {ba['a']}/{ba['b']} time relative to ECG R (ms); y = {ba['a']} minus {ba['b']} time (ms).",
            "N": f"{ba['pair'].subject_id.nunique()} participants and {len(ba['pair'])} paired complete beats.",
            "Key numerical result": f"Participant-balanced bias = {ba['bias']:.1f} ms; descriptive 95% limits of agreement = {ba['low']:.1f} to {ba['high']:.1f} ms.",
            "Interpretation": "The pair quantifies systematic separation between two related threshold-family definitions without mixing ECHO accuracy into agreement.",
            "Limitation": "Repeated beats are correlated within participants; limits are participant-balanced descriptive summaries, not independent-observation inferential limits.",
            "Decision": "Retain this single scientifically related pair; do not generate all possible pairwise Bland–Altman plots.",
        })
        for key in ("clear", "ambiguous"):
            item = metadata[kind][key]; chosen = item["chosen"]
            report(RESULTS/"reports"/kind/item["filename"].replace(".png",".md"), f"{item['code']} {kind} {key} reference-region morphology", {
                "Question": f"What {kind}-point candidate geometry is visible around {ref} in a reproducibly selected {key} morphology example?",
                "Dataset": f"HeartCycle participant {chosen.subject_id}, recording {chosen.recording_id}, beat {chosen.beat_id}. Figure: `{item['filename']}`; source: `{kind}_morphology_features.csv` and detection status.",
                "Algorithms": algorithms,
                "Axes and units": "x = time relative to ECG R (ms); y = filtered dZ/dt (device units/s). Shading is a ±20 ms display neighbourhood around the ECHO event, not an error tolerance.",
                "N": "One deterministic example selected using existing SNR, reference sharpness, and competing-feature measures.",
                "Key numerical result": f"SNR = {chosen.signal_to_noise_ratio_db:.2f} dB; reference sharpness = {chosen.reference_local_sharpness_abs_d2_device_units_s3:.3g} device units/s³; competing features = {chosen.reference_region_competing_features_count:.0f}.",
                "Interpretation": item["descriptor"].capitalize() + "; detector markers show whether methods select the same or different nearby landmarks.",
                "Limitation": "Clear/ambiguous is a relative reproducible selection for illustration, not a validated clinical morphology class.",
                "Decision": "Use the example to interpret the population category analysis, not as standalone evidence of algorithm performance.",
            })
        morph = metadata[kind]["morphology"]
        summary = morph["summary"]
        contrasts = summary.pivot_table(index=["factor","algorithm_id"], columns="category", values="median_participant_error_ms").reset_index()
        report(RESULTS/"reports"/kind/f"F06C_{kind}_morphology_vs_error.md", f"F06C {kind} morphology versus reference error", {
            "Question": f"Are reproducible waveform morphology or signal-quality categories associated with {kind}-point reference timing error?",
            "Dataset": f"HeartCycle strict common {ref} set. Figure: `{morph['filename']}`; source: `{kind}_morphology_features.csv`.",
            "Algorithms": algorithms,
            "Axes and units": "x = outcome-blind within-participant morphology/quality category; y = participant median absolute reference timing error (ms). Coloured lines identify algorithms; faint circular observations are participant-algorithm summaries.",
            "N": "Category-specific counts:\n\n" + markdown_table(morph["counts"], ["factor","category","beats","participants"], 0),
            "Key numerical result": markdown_table(contrasts, digits=1),
            "Interpretation": "Category shifts that recur across definitions support an association between waveform morphology and localization error; differing line patterns indicate definition-specific landmark selection.",
            "Limitation": "Categories are participant-relative median strata, comparisons are descriptive, and no causal or predictive conclusion is supported. Physical impedance units remain unverified.",
            "Decision": "Treat morphology as a mechanistic validation target; do not promote these relative categories to universal thresholds.",
        })


def validate(tables: dict[str, dict[str, pd.DataFrame]], registry: pd.DataFrame) -> list[str]:
    checks = []
    checks.append((registry.candidate_type.value_counts().to_dict() == {"X":13,"B":10}, "exactly 10 B and 13 X definitions"))
    checks.append((len(list((RESULTS/"tables").glob("*/*.csv"))) == 16, "exactly 16 final CSV tables"))
    for kind, n_methods, n_eligible, n_common in (("B",10,1198,986),("X",13,691,371)):
        t=tables[kind]; master=t[f"{kind}_master_beats.csv"]; status=t[f"{kind}_detection_status.csv"]; errors=t[f"{kind}_reference_errors.csv"]
        checks.extend([
            (len(master)==1706 and master.subject_id.nunique()==17 and master.recording_id.nunique()==212, f"{kind} cohort reconciliation"),
            (not master.duplicated(KEYS).any(), f"{kind} unique beat keys"),
            (not status.duplicated(KEYS+["algorithm_id"]).any() and len(status)==1706*n_methods, f"{kind} unique complete beat-algorithm grid"),
            ((status.loc[status.attempted,["valid","missed","invalid"]].sum(axis=1)==1).all(), f"{kind} attempted statuses reconcile"),
            (int(t[f"{kind}_reference_eligibility.csv"].reference_eligible.sum())==n_eligible, f"{kind} reference eligibility verified"),
            ((errors.groupby("algorithm_id").size()==n_common).all() and errors.algorithm_id.nunique()==n_methods, f"{kind} strict common reference set verified"),
        ])
    figures=list((RESULTS/"figures").glob("*/*.png")); reports=list((RESULTS/"reports").glob("*/*.md"))
    checks.append((len(figures)==22, "exactly 22 requested final figures"))
    checks.append((len(reports)==22, "every final figure has one report"))
    stems_f={p.stem for p in figures}; stems_r={p.stem for p in reports}
    checks.append((stems_f==stems_r, "figure/report filename pairing complete"))
    failed=[name for passed,name in checks if not passed]
    if failed: raise RuntimeError("Final validation failed: "+"; ".join(failed))
    return [f"PASS — {name}" for _,name in checks]


def write_audit(tables: dict[str, dict[str, pd.DataFrame]], registry: pd.DataFrame, checks: list[str]) -> None:
    inventory=[]
    for folder in ("tables","figures","reports"):
        inventory.extend(f"- `{p.relative_to(RESULTS).as_posix()}`" for p in sorted((RESULTS/folder).glob("*/*")) if p.is_file())
    b,x=tables["B"],tables["X"]
    text=f"""# FINAL HEARTCYCLE RQ1 AUDIT

## Verified inputs

- HeartCycle only: 17 participants, 212 recordings, 1,706 unique beats.
- Frozen catalogue: 10 B-point and 13 X-point definitions; exact names and provenance are in `config.yaml` and `configs/canonical_method_registry.csv`.
- ECHO AVO eligibility: {int(b['B_reference_eligibility.csv'].reference_eligible.sum()):,} beats; strict common B accuracy: {b['B_reference_errors.csv'].beat_id.nunique():,} beats.
- ECHO AVC eligibility: {int(x['X_reference_eligibility.csv'].reference_eligible.sum()):,} beats; strict common X accuracy: {x['X_reference_errors.csv'].beat_id.nunique():,} beats.

## Completed and corrected

- Preserved the verified 16-table scientific core under `results/tables/B` and `results/tables/X`.
- Replaced the multi-waveform Figure 1 grids with four separate filtered-dZ/dt examples.
- Retained participant-coloured circular-dot reference accuracy plots.
- Used named-method signed-error histograms without embedded statistical strings.
- Replaced 100%-valid bars with missed/invalid failure-only plots and participant bootstrap intervals.
- Separated participant-wise range, definition-versus-peer-median disagreement, and one selected Bland–Altman pair per point type.
- Replaced morphology heatmaps with reference-centred waveform examples and category-on-x error comparisons.

## Scientific findings and boundaries

- Exact algorithm-level reference errors, bias/tails, failure rates, disagreement summaries, and morphology contrasts are in the paired reports.
- Disagreement is never interpreted as ECHO accuracy.
- No clinical timing tolerance was invented.
- No rest/exercise, disease, workload, or cross-dataset contrast was manufactured.
- ECHO events use a separately aligned time base; inherited R matching uncertainty remains distinct from detector error.
- Source impedance calibration is unverified, so waveform axes use device units/s rather than ohm/s.
- Morphology associations are descriptive and participant-relative; they do not establish causality.
- Repeated beats are correlated; Bland–Altman limits are participant-balanced descriptive limits.

## Reproducibility validation

Run `python src/07_generate_figures.py`, then `python -m unittest discover -s tests -v`.

"""+"\n".join(f"- {c}" for c in checks)+"\n\n## Final inventory\n\n"+"\n".join(inventory)+"\n- `reports/FINAL_HEARTCYCLE_RQ1_AUDIT.md`\n"
    (RESULTS/"reports"/"FINAL_HEARTCYCLE_RQ1_AUDIT.md").write_text(text,encoding="utf-8")


def build() -> None:
    tables, registry, config, style = load_tables()
    set_style(style)
    if RESULTS.exists(): shutil.rmtree(RESULTS)
    copy_tables(tables)
    selection=select_examples(tables)
    meta={"F01":plot_f01(selection,tables,registry,config,style)}
    for kind in ("B","X"):
        (RESULTS/"reports"/kind).mkdir(parents=True,exist_ok=True)
        plot_accuracy(kind,tables[kind],registry,style)
        plot_histograms(kind,tables[kind],registry)
        failure=plot_failure(kind,tables[kind],registry)
        subject=plot_subject_disagreement(kind,tables[kind],registry,style)
        definition=plot_definition_disagreement(kind,subject["timing"],registry,style)
        ba=plot_bland_altman(kind,subject["timing"],registry,style)
        clear,ambiguous=morphology_examples(kind,tables[kind])
        clear_meta=plot_morphology_waveform(kind,"clear",clear,tables[kind],config,style)
        ambiguous_meta=plot_morphology_waveform(kind,"ambiguous",ambiguous,tables[kind],config,style)
        morphology=plot_morphology_error(kind,tables[kind],registry,style)
        meta[kind]={"failure":failure,"subject":subject,"definition":definition,"ba":ba,"clear":clear_meta,"ambiguous":ambiguous_meta,"morphology":morphology}
    create_reports(tables,registry,selection,meta)
    checks=validate(tables,registry)
    write_audit(tables,registry,checks)


if __name__ == "__main__":
    build()
