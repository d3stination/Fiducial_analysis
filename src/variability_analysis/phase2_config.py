"""Frozen configuration for Phase 2 pilot preprocessing and R/C validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Phase2Config:
    """Approved Phase 2 parameters expressed in physical units."""

    icg_filter_type: str = "butterworth_bandpass_sosfiltfilt"
    icg_filter_low_hz: float = 0.8
    icg_filter_high_hz: float = 30.0
    icg_filter_order: int = 3
    icg_filter_phase: str = "zero_phase"
    r_detector: str = "pan_tompkins_sampling_rate_adapted_offline"
    r_bandpass_low_hz: float = 5.0
    r_bandpass_high_hz: float = 15.0
    r_bandpass_order: int = 3
    r_bandpass_phase: str = "zero_phase_offline_adaptation"
    r_derivative: str = "pan_tompkins_five_point"
    r_integration_window_ms: float = 150.0
    r_refractory_ms: float = 200.0
    r_t_wave_window_ms: float = 360.0
    r_learning_window_s: float = 2.0
    r_searchback_rr_factor: float = 1.66
    r_adaptive_update_weight: float = 0.125
    r_threshold_fraction: float = 0.25
    r_secondary_threshold_fraction: float = 0.5
    r_localization_window_ms: float = 150.0
    reference_r_match_tolerance_ms: float = 120.0
    c_method: str = "C_baseline_globalmax_QC60to200"
    c_search_start_ms: float = 0.0
    c_search_stop_ms: float = 300.0
    c_valid_min_rc_ms: float = 60.0
    c_valid_max_rc_ms: float = 200.0
    plot_max_beats: int = 8
    plot_dpi: int = 300
    config_version: str = "phase2-pilot-1"
    pan_tompkins_reference_doi: str = "10.1109/TBME.1985.325532"

    def to_dict(self) -> dict:
        """Return a JSON-serializable parameter record."""

        return asdict(self)

