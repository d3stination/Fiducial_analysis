"""Frozen Phase 4 B/X candidate definitions and independent detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class BXSignals:
    time_s: np.ndarray
    dzdt: np.ndarray
    d2: np.ndarray
    d3: np.ndarray
    sampling_rate_hz: float


@dataclass(frozen=True)
class Candidate:
    sample_index: int | None
    timestamp_s: float | None
    status: str
    reason: str
    search_start_sample: int | None
    search_stop_sample_exclusive: int | None
    interpolation: str


@dataclass(frozen=True)
class CandidateDefinition:
    name: str
    candidate_type: str
    category: str
    signal: str
    exact_definition: str
    search_region: str
    parameters: str
    implementation_status: str
    unresolved_ambiguity: str = ""


def _missing(reason: str, start=None, stop=None, status="MISSING") -> Candidate:
    return Candidate(None, None, status, reason, start, stop, "none")


def _native(signals: BXSignals, index: int, start: int, stop: int) -> Candidate:
    return Candidate(index, float(signals.time_s[index]), "DETECTED", "", start, stop, "native_sample")


def _nearest_index(time: np.ndarray, value: float) -> int:
    right = int(np.searchsorted(time, value, side="left"))
    if right <= 0:
        return 0
    if right >= time.size:
        return time.size - 1
    return right if abs(time[right] - value) < abs(value - time[right - 1]) else right - 1


def _interpolated(signals: BXSignals, value: float, start: int, stop: int) -> Candidate:
    return Candidate(
        _nearest_index(signals.time_s, value), float(value), "DETECTED", "",
        start, stop, "linear_interpolation_nearest_native_sample_index",
    )


def _crossing(
    signals: BXSignals, values: np.ndarray, start: int, stop: int,
    level: float, direction: str, select: str,
) -> Candidate:
    if stop - start < 2:
        return _missing("search_window_fewer_than_two_samples", start, stop)
    segment = values[start:stop]
    hits = (
        np.flatnonzero((segment[:-1] < level) & (segment[1:] >= level))
        if direction == "up"
        else np.flatnonzero((segment[:-1] > level) & (segment[1:] <= level))
    )
    if not hits.size:
        return _missing(f"no_{direction}ward_threshold_crossing", start, stop)
    left = start + int(hits[-1] if select == "last" else hits[0])
    y0, y1 = float(values[left]), float(values[left + 1])
    if y1 == y0:
        return _native(signals, left, start, stop)
    fraction = (level - y0) / (y1 - y0)
    timestamp = float(signals.time_s[left] + fraction * (signals.time_s[left + 1] - signals.time_s[left]))
    return _interpolated(signals, timestamp, start, stop)


def _time_window(signals: BXSignals, start_s: float, stop_s: float, limit_stop: int) -> tuple[int, int]:
    start = int(np.searchsorted(signals.time_s, start_s, side="left"))
    stop = int(np.searchsorted(signals.time_s, stop_s, side="right"))
    return max(0, start), min(limit_stop, stop)


def detect_B_last_local_minimum(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del nr
    start, stop = r, c + 1
    minima, _ = find_peaks(-s.dzdt[start:stop])
    qualified = [start + int(i) for i in minima if s.d3[start + int(i)] > 0]
    return _native(s, qualified[-1], start, stop) if qualified else _missing("no_local_minimum_with_positive_second_derivative", start, stop)


def detect_B_d3_last_downward_zero_Cminus180_to_Cminus10(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del nr
    start, stop = _time_window(s, max(s.time_s[r], s.time_s[c] - .180), s.time_s[c] - .010, c + 1)
    return _crossing(s, s.d3, start, stop, 0.0, "down", "last")


def detect_B_d3_max_Cminus150_to_Cminus30(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del nr
    start, stop = _time_window(s, max(s.time_s[r], s.time_s[c] - .150), s.time_s[c] - .030, c + 1)
    if stop <= start:
        return _missing("empty_Cminus150_to_Cminus30_window", start, stop)
    return _native(s, start + int(np.argmax(s.d3[start:stop])), start, stop)


def _b_percentage(s: BXSignals, r: int, c: int, fraction: float) -> Candidate:
    return _crossing(s, s.dzdt, r, c + 1, fraction * float(s.dzdt[c]), "up", "last")


def detect_B_dzdt_45pct_upstroke(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del nr
    return _b_percentage(s, r, c, .45)


def detect_B_dzdt_25pct_upstroke(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del nr
    return _b_percentage(s, r, c, .25)


def detect_B_d2_max_Cminus150_to_Cminus30(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del nr
    start, stop = _time_window(s, max(s.time_s[r], s.time_s[c] - .150), s.time_s[c] - .030, c + 1)
    if stop <= start:
        return _missing("empty_Cminus150_to_Cminus30_window", start, stop)
    return _native(s, start + int(np.argmax(s.d2[start:stop])), start, stop)


def detect_B_d2_last_upward_zero_Cminus180_to_Cminus10(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del nr
    start, stop = _time_window(s, max(s.time_s[r], s.time_s[c] - .180), s.time_s[c] - .010, c + 1)
    return _crossing(s, s.d2, start, stop, 0.0, "up", "last")


def detect_B_55pct_RC(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del nr
    return _interpolated(s, float(s.time_s[r] + .55 * (s.time_s[c] - s.time_s[r])), r, c + 1)


def detect_B_adaptive_baseline_10pct_upstroke(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del nr
    level = float(s.dzdt[r] + .10 * (s.dzdt[c] - s.dzdt[r]))
    return _crossing(s, s.dzdt, r, c + 1, level, "up", "last")


def detect_B_15pct_upstroke(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del nr
    return _b_percentage(s, r, c, .15)


def detect_B_morphology_consensus_40ms(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del s, r, c, nr
    return _missing("approved_component_list_tie_break_and_minimum_count_unresolved", status="NOT_IMPLEMENTED_AMBIGUOUS")


def detect_B_adaptive_multirepresentation_consensus_35ms(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del s, r, c, nr
    return _missing("approved_signal_representations_components_and_aggregation_unresolved", status="NOT_IMPLEMENTED_AMBIGUOUS")


def detect_X_first_local_minimum(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    start, stop = c + 1, nr
    minima, _ = find_peaks(-s.dzdt[start:stop])
    return _native(s, start + int(minima[0]), start, stop) if minima.size else _missing("no_post_C_local_minimum", start, stop)


def detect_X_global_minimum(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    start, stop = c, nr
    return _native(s, start + int(np.argmin(s.dzdt[start:stop])), start, stop) if stop > start else _missing("empty_C_to_next_R_window", start, stop)


def _x_plus30(s: BXSignals, c: int, nr: int) -> tuple[int, int]:
    return _time_window(s, s.time_s[c] + .030, s.time_s[nr], nr)


def detect_X_d3_first_upward_zero_Cplus30_to_nextR(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    start, stop = _x_plus30(s, c, nr)
    return _crossing(s, s.d3, start, stop, 0.0, "up", "first")


def detect_X_d3_min_C_to_nextR(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    return _native(s, c + int(np.argmin(s.d3[c:nr])), c, nr) if nr > c else _missing("empty_C_to_next_R_window", c, nr)


def _x_threshold(s: BXSignals, c: int, nr: int, fraction: float) -> Candidate:
    trough = detect_X_global_minimum(s, 0, c, nr)
    if trough.sample_index is None:
        return _missing("X_global_minimum_unavailable", c, nr)
    level = fraction * float(s.dzdt[trough.sample_index])
    return _crossing(s, s.dzdt, c, trough.sample_index + 1, level, "down", "first")


def detect_X_dzdt_45pct_downstroke(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    return _x_threshold(s, c, nr, .45)


def detect_X_dzdt_25pct_downstroke(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    return _x_threshold(s, c, nr, .25)


def detect_X_d2_first_upward_zero_Cplus30_to_nextR(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    start, stop = _x_plus30(s, c, nr)
    return _crossing(s, s.d2, start, stop, 0.0, "up", "first")


def detect_X_second_derivative_after_minimum(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    trough = detect_X_global_minimum(s, 0, c, nr)
    if trough.sample_index is None:
        return _missing("X_global_minimum_unavailable", c, nr)
    start, stop = trough.sample_index + 1, nr
    maxima, _ = find_peaks(s.d2[start:stop])
    return _native(s, start + int(maxima[0]), start, stop) if maxima.size else _missing("no_second_derivative_local_maximum_after_X_global_minimum", start, stop)


def detect_X_d2_min_C_to_nextR(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    return _native(s, c + int(np.argmin(s.d2[c:nr])), c, nr) if nr > c else _missing("empty_C_to_next_R_window", c, nr)


def detect_X_Weissler_HR(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del s, r, c, nr
    return _missing("verified_biological_sex_unavailable", status="UNAVAILABLE_REQUIRED_METADATA")


def detect_X_adaptive_half_descent_to_trough(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    trough = detect_X_global_minimum(s, 0, c, nr)
    if trough.sample_index is None:
        return _missing("X_global_minimum_unavailable", c, nr)
    level = .5 * (float(s.dzdt[c]) + float(s.dzdt[trough.sample_index]))
    return _crossing(s, s.dzdt, c, trough.sample_index + 1, level, "down", "first")


def detect_X_rr_adaptive_global_minimum(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    rr_s = float(s.time_s[nr] - s.time_s[r])
    start, stop = _time_window(s, s.time_s[c], min(s.time_s[c] + .40 * rr_s, s.time_s[nr]), nr)
    return _native(s, start + int(np.argmin(s.dzdt[start:stop])), start, stop) if stop > start else _missing("empty_RR_adaptive_post_C_window", start, stop)


def detect_X_15pct_downstroke(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del r
    return _x_threshold(s, c, nr, .15)


def detect_X_morphology_consensus_60ms(s: BXSignals, r: int, c: int, nr: int) -> Candidate:
    del s, r, c, nr
    return _missing("approved_component_list_tie_break_and_minimum_count_unresolved", status="NOT_IMPLEMENTED_AMBIGUOUS")


DEFINITIONS = (
    CandidateDefinition("B_last_local_minimum", "B", "Local Extrema", "dZ/dt", "Last local minimum between R and C where the first derivative is zero and the second derivative is positive.", "R to C", "last qualifying minimum", "IMPLEMENTED"),
    CandidateDefinition("B_d3_last_downward_zero_Cminus180_to_Cminus10", "B", "Derivative", "d3Z/dt3", "Last downward third-derivative zero crossing closest to C.", "max(R,C-180 ms) to C-10 ms", "downward crossing; last", "IMPLEMENTED"),
    CandidateDefinition("B_d3_max_Cminus150_to_Cminus30", "B", "Derivative", "d3Z/dt3", "Maximum third-derivative value.", "max(R,C-150 ms) to C-30 ms", "global maximum", "IMPLEMENTED"),
    CandidateDefinition("B_dzdt_45pct_upstroke", "B", "Slope", "dZ/dt", "First point encountered searching backward from C where dZ/dt reaches 45% of C amplitude.", "R to C", "45% of C amplitude; last upward crossing in forward time", "IMPLEMENTED"),
    CandidateDefinition("B_dzdt_25pct_upstroke", "B", "Slope", "dZ/dt", "First point encountered searching backward from C where dZ/dt reaches 25% of C amplitude.", "R to C", "25% of C amplitude; last upward crossing in forward time", "IMPLEMENTED"),
    CandidateDefinition("B_d2_max_Cminus150_to_Cminus30", "B", "Curvature", "d2Z/dt2", "Maximum second-derivative value.", "max(R,C-150 ms) to C-30 ms", "global maximum", "IMPLEMENTED"),
    CandidateDefinition("B_d2_last_upward_zero_Cminus180_to_Cminus10", "B", "Curvature", "d2Z/dt2", "Last second-derivative crossing from negative to positive.", "max(R,C-180 ms) to C-10 ms", "upward crossing; last", "IMPLEMENTED"),
    CandidateDefinition("B_55pct_RC", "B", "Adaptive", "R and C", "B = R + 0.55 x (C-R).", "R to C", "0.55 fraction", "IMPLEMENTED"),
    CandidateDefinition("B_adaptive_baseline_10pct_upstroke", "B", "Adaptive", "dZ/dt", "Last upward crossing of dZ/dt at baseline plus 10% of baseline-to-C amplitude; baseline is dZ/dt at R.", "R to C", "baseline=dZ/dt(R); 10%", "IMPLEMENTED"),
    CandidateDefinition("B_15pct_upstroke", "B", "Threshold", "dZ/dt", "First point encountered searching backward from C where dZ/dt reaches 15% of C amplitude.", "R to C", "15% of C amplitude; last upward crossing in forward time", "IMPLEMENTED"),
    CandidateDefinition("B_morphology_consensus_40ms", "B", "Consensus", "multiple", "Median of candidates in the densest 40 ms cluster.", "R to C", "40 ms cluster", "NOT_IMPLEMENTED_AMBIGUOUS", "Approved component list, tie-breaking, and minimum candidate count require confirmation."),
    CandidateDefinition("B_adaptive_multirepresentation_consensus_35ms", "B", "Consensus", "multiple", "Combine approved multiple representations within 35 ms using spatial averaging.", "R to C", "35 ms tolerance", "NOT_IMPLEMENTED_AMBIGUOUS", "Component representations and aggregation are unresolved; wavelets are not approved."),
    CandidateDefinition("X_first_local_minimum", "X", "Local Extrema", "dZ/dt", "First local minimum after C.", "C to next R", "first post-C local minimum", "IMPLEMENTED"),
    CandidateDefinition("X_global_minimum", "X", "Local Extrema", "dZ/dt", "Absolute minimum of dZ/dt.", "C to next R", "global minimum", "IMPLEMENTED"),
    CandidateDefinition("X_d3_first_upward_zero_Cplus30_to_nextR", "X", "Derivative", "d3Z/dt3", "First third-derivative crossing from negative to positive.", "C+30 ms to next R", "upward crossing; first", "IMPLEMENTED"),
    CandidateDefinition("X_d3_min_C_to_nextR", "X", "Derivative", "d3Z/dt3", "Minimum third-derivative value.", "C to next R", "global minimum", "IMPLEMENTED"),
    CandidateDefinition("X_dzdt_45pct_downstroke", "X", "Slope", "dZ/dt", "First descending point where dZ/dt reaches 45% of the amplitude at X_global_minimum.", "C to X_global_minimum", "45% of X_global_minimum amplitude; first downward crossing", "IMPLEMENTED"),
    CandidateDefinition("X_dzdt_25pct_downstroke", "X", "Slope", "dZ/dt", "First descending point where dZ/dt reaches 25% of the amplitude at X_global_minimum.", "C to X_global_minimum", "25% of X_global_minimum amplitude; first downward crossing", "IMPLEMENTED"),
    CandidateDefinition("X_d2_first_upward_zero_Cplus30_to_nextR", "X", "Curvature", "d2Z/dt2", "First second-derivative crossing from negative to positive.", "C+30 ms to next R", "upward crossing; first", "IMPLEMENTED"),
    CandidateDefinition("X_second_derivative_after_minimum", "X", "Curvature", "d2Z/dt2", "First local maximum of the second derivative after X_global_minimum.", "X_global_minimum to next R", "first local maximum", "IMPLEMENTED"),
    CandidateDefinition("X_d2_min_C_to_nextR", "X", "Curvature", "d2Z/dt2", "Minimum second-derivative value.", "C to next R", "global minimum", "IMPLEMENTED"),
    CandidateDefinition("X_Weissler_HR", "X", "Adaptive", "B and HR", "Male: X=B+(-0.0017xHR+0.413)s; female: X=B+(-0.0016xHR+0.418)s.", "formula, bounded by next R for QC", "verified biological sex and B dependency required", "IMPLEMENTED_METADATA_GATED", "Verified biological sex is unavailable; B dependency/source validation remains unresolved."),
    CandidateDefinition("X_adaptive_half_descent_to_trough", "X", "Adaptive", "dZ/dt", "First downward crossing halfway between C amplitude and X_global_minimum amplitude.", "C to X_global_minimum", "50% of C-to-trough descent", "IMPLEMENTED"),
    CandidateDefinition("X_rr_adaptive_global_minimum", "X", "Adaptive", "dZ/dt", "Absolute minimum from C to C+0.4xRR, bounded by next R.", "C to min(C+0.4xRR,next R)", "0.4 x RR", "IMPLEMENTED"),
    CandidateDefinition("X_15pct_downstroke", "X", "Threshold", "dZ/dt", "First descending point where dZ/dt reaches 15% of X_global_minimum amplitude.", "C to X_global_minimum", "15% of X_global_minimum amplitude; first downward crossing", "IMPLEMENTED"),
    CandidateDefinition("X_morphology_consensus_60ms", "X", "Consensus", "multiple", "Median of candidates in the selected 60 ms cluster.", "C to next R", "60 ms cluster", "NOT_IMPLEMENTED_AMBIGUOUS", "Approved component list, tie-breaking, and minimum candidate count require confirmation."),
)


DETECTORS: dict[str, Callable[[BXSignals, int, int, int], Candidate]] = {
    definition.name: globals()[f"detect_{definition.name}"] for definition in DEFINITIONS
}
