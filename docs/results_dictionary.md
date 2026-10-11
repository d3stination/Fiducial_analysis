# Results dictionary

Blank numeric cells mean unavailable or not applicable. They are not zero. Counts use the denominator named in the column. Stored scientific values retain full exported precision.

## algorithm_disagreement.csv

One participant and either one method-specific peer difference or one all-method range summary.

Rows: 2,525. Columns: 10.

| Column | Definition |
|---|---|
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |
| `condition` | Acquisition condition within the dataset. |
| `point_type` | Fiducial family, B or X. |
| `subject_id` | Pseudonymous participant identifier used for clustering and pairing. |
| `algorithm_id` | Stable frozen method ID. ALL_METHOD_RANGE denotes an across-method range. |
| `candidate_name` | Frozen implementation name. |
| `observation_type` | Observation type. |
| `common_valid_beat_n` | Common valid beat n, as an integer count. |
| `participant_median_disagreement_ms` | Participant median disagreement, in milliseconds. |
| `participant_p95_disagreement_ms` | Participant p95 disagreement, in milliseconds. |

## condition_comparison_summary.csv

One dataset, point type, paired outcome and method, summarized across matched participants.

Rows: 96. Columns: 17.

| Column | Definition |
|---|---|
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |
| `point_type` | Fiducial family, B or X. |
| `outcome` | Outcome. |
| `algorithm_id` | Stable frozen method ID. ALL_METHOD_RANGE denotes an across-method range. |
| `normal_condition` | Normal condition. |
| `stress_condition` | Stress condition. |
| `unit` | Unit. |
| `matched_participant_n` | Matched participant n, as an integer count. |
| `median_stress_minus_normal` | Median stress minus normal. |
| `bootstrap_95ci_low` | Lower bound of the stated 95% interval, in the outcome unit. |
| `bootstrap_95ci_high` | Upper bound of the stated 95% interval, in the outcome unit. |
| `q25_stress_minus_normal` | Q25 stress minus normal. |
| `q75_stress_minus_normal` | Q75 stress minus normal. |
| `positive_change_n` | Positive change n, as an integer count. |
| `negative_change_n` | Negative change n, as an integer count. |
| `zero_change_n` | Zero change n, as an integer count. |
| `interpretation` | Interpretation. |

## condition_comparisons.csv

One matched participant, outcome and method. ALL_METHOD_RANGE denotes the all-method spread outcome.

Rows: 2,016. Columns: 11.

| Column | Definition |
|---|---|
| `point_type` | Fiducial family, B or X. |
| `outcome` | Outcome. |
| `subject_id` | Pseudonymous participant identifier used for clustering and pairing. |
| `algorithm_id` | Stable frozen method ID. ALL_METHOD_RANGE denotes an across-method range. |
| `normal_condition` | Normal condition. |
| `stress_condition` | Stress condition. |
| `normal_value` | Normal value. |
| `stress_value` | Stress value. |
| `stress_minus_normal` | Stress minus normal. |
| `unit` | Unit. |
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |

## condition_morphology_associations.csv

One dataset, point type, algorithm or all-method outcome, paired reliability outcome and feature-change test.

Rows: 672. Columns: 19.

| Column | Definition |
|---|---|
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |
| `point_type` | Fiducial family, B or X. |
| `algorithm_id` | Stable frozen method ID. ALL_METHOD_RANGE denotes an across-method range. |
| `candidate_name` | Frozen implementation name. |
| `outcome` | Outcome. |
| `outcome_unit` | Outcome unit. |
| `feature` | Feature. |
| `matched_participant_n` | Matched participant n, as an integer count. |
| `spearman_rho` | Participant-level Spearman rank correlation. |
| `bootstrap_95ci_low` | Lower bound of the stated 95% interval, in the outcome unit. |
| `bootstrap_95ci_high` | Upper bound of the stated 95% interval, in the outcome unit. |
| `permutation_p` | Two-sided participant-level permutation p value. |
| `direction_definition` | Direction definition. |
| `bh_q_value` | Benjamini-Hochberg adjusted p value within the stated analysis family. |
| `exploratory_supported` | True only when the prespecified bootstrap-direction and within-family BH criteria are met. |
| `evidence_classification` | Evidence classification. |
| `causal_guardrail` | Causal guardrail. |
| `permutation_method` | Permutation unit, tail and exact or Monte Carlo procedure. |
| `bootstrap_interpretation` | Required qualification for the reported bootstrap interval. |

## dataset_algorithm_results.csv

One dataset, condition, point type and frozen algorithm.

Rows: 115. Columns: 40.

| Column | Definition |
|---|---|
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |
| `cohort` | Recorded cohort label without causal interpretation. |
| `condition` | Acquisition condition within the dataset. |
| `point_type` | Fiducial family, B or X. |
| `algorithm_id` | Stable frozen method ID. ALL_METHOD_RANGE denotes an across-method range. |
| `algorithm_name` | Frozen implementation name. |
| `exact_algorithm_and_parameter_version` | Exact algorithm and parameter version. |
| `participant_n` | Participant n, as an integer count. |
| `recording_n` | Recording n, as an integer count. |
| `total_beat_n` | Total beat n, as an integer count. |
| `attempted_beat_n` | Attempted beat n, as an integer count. |
| `valid_detection_n` | Valid detection n, as an integer count. |
| `missed_detection_n` | Missed detection n, as an integer count. |
| `invalid_detection_n` | Invalid detection n, as an integer count. |
| `upstream_ineligible_n` | Upstream ineligible n, as an integer count. |
| `success_rate_pct` | Success rate pct, in percent. |
| `attempted_failure_rate_pct` | Attempted failure rate pct, in percent. |
| `reference_type` | Reference provenance. ECHO and expert ICG annotations remain distinct. |
| `reference_matched_beat_n` | Reference matched beat n, as an integer count. |
| `strict_common_reference_beat_n` | Strict common reference beat n, as an integer count. |
| `median_signed_reference_error_ms` | Median signed reference error, in milliseconds. |
| `participant_level_median_absolute_reference_error_ms` | Participant level median absolute reference error, in milliseconds. |
| `p90_absolute_reference_error_ms` | P90 absolute reference error, in milliseconds. |
| `p95_absolute_reference_error_ms` | P95 absolute reference error, in milliseconds. |
| `participant_median_peer_disagreement_ms` | Participant median peer disagreement, in milliseconds. |
| `common_valid_beat_n` | Common valid beat n, as an integer count. |
| `analysis_status` | Availability and comparability status for the result row. |
| `selected_for_detailed_case_study` | Selected for detailed case study. |
| `case_study_role` | Case study role. |
| `selection_rationale` | Selection rationale. |
| `verified_primary_failure_reasons` | Verified primary failure reasons. |
| `attempted_failure_denominator_n` | Attempted failure denominator n, as an integer count. |
| `failed_or_ineligible_n` | Failed or ineligible n, as an integer count. |
| `unresolved_proportion` | Unresolved proportion. |
| `morphology_supported_association_n` | Morphology supported association n, as an integer count. |
| `morphology_test_n` | Morphology test n, as an integer count. |
| `morphology_participant_n` | Morphology participant n, as an integer count. |
| `matched_condition_changes` | Matched condition changes. |
| `selected_waveform_cases` | Selected waveform cases. |
| `supplementary_status` | Supplementary status. |

## dataset_reference_audit.csv

One audited dataset and reference source record.

Rows: 10. Columns: 20.

| Column | Definition |
|---|---|
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |
| `cohort` | Recorded cohort label without causal interpretation. |
| `condition` | Acquisition condition within the dataset. |
| `candidate_type` | Fiducial family, B or X. |
| `participant_n` | Participant n, as an integer count. |
| `recording_n` | Recording n, as an integer count. |
| `total_beat_n` | Total beat n, as an integer count. |
| `eligible_beat_n` | Eligible beat n, as an integer count. |
| `upstream_ineligible_beat_n` | Upstream ineligible beat n, as an integer count. |
| `valid_detection_events` | Valid detection events. |
| `missed_detection_events` | Missed detection events. |
| `invalid_detection_events` | Invalid detection events. |
| `reference_type` | Reference provenance. ECHO and expert ICG annotations remain distinct. |
| `reference_eligible_or_matched_beat_n` | Reference eligible or matched beat n, as an integer count. |
| `strict_common_reference_beat_n` | Strict common reference beat n, as an integer count. |
| `paired_participant_n` | Paired participant n, as an integer count. |
| `sampling_frequency_hz_min` | Sampling frequency hz min. |
| `sampling_frequency_hz_max` | Sampling frequency hz max. |
| `signal_units` | Signal units. |
| `reference_status` | Reference status. |

## detection_failure_mechanisms.csv

One dataset, condition, algorithm and mutually exclusive primary failure reason.

Rows: 690. Columns: 23.

| Column | Definition |
|---|---|
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |
| `dataset_id` | Stable internal dataset identifier. |
| `condition` | Acquisition condition within the dataset. |
| `point_type` | Fiducial family, B or X. |
| `algorithm_id` | Stable frozen method ID. ALL_METHOD_RANGE denotes an across-method range. |
| `candidate_name` | Frozen implementation name. |
| `primary_reason` | Mutually exclusive verified failure reason from the approved hierarchy. |
| `reason_record_n` | Reason record n, as an integer count. |
| `reason_participant_n` | Reason participant n, as an integer count. |
| `reason_unique_beat_n` | Reason unique beat n, as an integer count. |
| `attempted_failure_reason_n` | Attempted failure reason n, as an integer count. |
| `total_record_n` | Total record n, as an integer count. |
| `total_participant_n` | Total participant n, as an integer count. |
| `attempted_denominator_n` | Attempted denominator n, as an integer count. |
| `valid_n` | Valid n, as an integer count. |
| `original_missed_n` | Original missed n, as an integer count. |
| `original_invalid_n` | Original invalid n, as an integer count. |
| `upstream_ineligible_n` | Upstream ineligible n, as an integer count. |
| `failed_or_ineligible_n` | Failed or ineligible n, as an integer count. |
| `reason_proportion_of_failed_or_ineligible` | Reason proportion of failed or ineligible. |
| `reason_proportion_of_attempted_failures` | Reason proportion of attempted failures. |
| `unresolved_n` | Unresolved n, as an integer count. |
| `unresolved_proportion` | Unresolved proportion. |

## morphology_associations.csv

One dataset, condition, point type, algorithm, outcome and prespecified waveform feature test.

Rows: 1,794. Columns: 32.

| Column | Definition |
|---|---|
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |
| `dataset_id` | Stable internal dataset identifier. |
| `condition` | Acquisition condition within the dataset. |
| `point_type` | Fiducial family, B or X. |
| `algorithm_id` | Stable frozen method ID. ALL_METHOD_RANGE denotes an across-method range. |
| `candidate_name` | Frozen implementation name. |
| `outcome` | Outcome. |
| `outcome_column` | Outcome column. |
| `effect_metric` | Effect metric. |
| `minimum_participant_data` | Minimum participant data. |
| `reference_type` | Reference provenance. ECHO and expert ICG annotations remain distinct. |
| `comparison_set` | Comparison set. |
| `feature` | Feature. |
| `feature_label` | Feature label. |
| `feature_units` | Feature units. |
| `available_observation_n` | Available observation n, as an integer count. |
| `available_participant_n` | Available participant n, as an integer count. |
| `participant_effect_n` | Participant effect n, as an integer count. |
| `participant_effect_observation_n` | Participant effect observation n, as an integer count. |
| `median_participant_effect` | Median participant effect. |
| `q25_participant_effect` | Q25 participant effect. |
| `q75_participant_effect` | Q75 participant effect. |
| `bootstrap_95ci_low` | Lower bound of the stated 95% interval, in the outcome unit. |
| `bootstrap_95ci_high` | Upper bound of the stated 95% interval, in the outcome unit. |
| `wilcoxon_raw_p` | Wilcoxon raw p. |
| `direction` | Direction. |
| `attempted_denominator_n` | Attempted denominator n, as an integer count. |
| `failure_n` | Failure n, as an integer count. |
| `multiplicity_family` | Multiplicity family. |
| `bh_q_value` | Benjamini-Hochberg adjusted p value within the stated analysis family. |
| `exploratory_supported_after_bh` | Exploratory supported after bh. |
| `interpretation_guardrail` | Interpretation guardrail. |

## paired_method_agreement.csv

One dataset, condition, point type and predeclared algorithm pair.

Rows: 10. Columns: 17.

| Column | Definition |
|---|---|
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |
| `point_type` | Fiducial family, B or X. |
| `condition` | Acquisition condition within the dataset. |
| `algorithm_a` | Algorithm a. |
| `algorithm_b` | Algorithm b. |
| `participants` | Participants. |
| `paired_beats` | Paired beats, as an integer count. |
| `participant_balanced_bias_ms` | Participant balanced bias, in milliseconds. |
| `bias_ci_low` | Lower bound of the stated 95% interval, in the outcome unit. |
| `bias_ci_high` | Upper bound of the stated 95% interval, in the outcome unit. |
| `participant_balanced_loa_low_ms` | Participant balanced loa low, in milliseconds. |
| `participant_balanced_loa_high_ms` | Participant balanced loa high, in milliseconds. |
| `loa_low_ci_low` | Lower bound of the stated 95% interval, in the outcome unit. |
| `loa_low_ci_high` | Upper bound of the stated 95% interval, in the outcome unit. |
| `loa_high_ci_low` | Lower bound of the stated 95% interval, in the outcome unit. |
| `loa_high_ci_high` | Upper bound of the stated 95% interval, in the outcome unit. |
| `weighting` | Weighting. |

## reference_accuracy.csv

One participant, reference type, condition and algorithm on the strict common-reference sample.

Rows: 1,449. Columns: 12.

| Column | Definition |
|---|---|
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |
| `point_type` | Fiducial family, B or X. |
| `condition` | Acquisition condition within the dataset. |
| `reference_type` | Reference provenance. ECHO and expert ICG annotations remain distinct. |
| `subject_id` | Pseudonymous participant identifier used for clustering and pairing. |
| `algorithm_id` | Stable frozen method ID. ALL_METHOD_RANGE denotes an across-method range. |
| `candidate_name` | Frozen implementation name. |
| `strict_common_reference_beat_n` | Strict common reference beat n, as an integer count. |
| `median_signed_error_ms` | Median signed error, in milliseconds. |
| `median_absolute_error_ms` | Median absolute error, in milliseconds. |
| `p90_absolute_error_ms` | P90 absolute error, in milliseconds. |
| `p95_absolute_error_ms` | P95 absolute error, in milliseconds. |

## waveform_case_evidence.csv

One reproducibly selected dataset and Figure 1 case category.

Rows: 12. Columns: 13.

| Column | Definition |
|---|---|
| `dataset` | Dataset label: HeartCycle, IITB or ReBeat. |
| `panel` | Panel. |
| `category` | Category. |
| `candidate_type` | Fiducial family, B or X. |
| `subject_id` | Pseudonymous participant identifier used for clustering and pairing. |
| `recording_id` | Stable recording identifier. |
| `beat_id` | Stable R-to-next-R beat identifier. |
| `condition` | Acquisition condition within the dataset. |
| `selection_criterion` | Selection criterion. |
| `timing_range_ms` | Timing range, in milliseconds. |
| `failed_or_ineligible_algorithm_n` | Failed or ineligible algorithm n, as an integer count. |
| `verified_failure_reasons` | Verified failure reasons. |
| `reference_present` | Reference present. |

## Denominator rules

`total_beat_n` includes all beat-method records. `upstream_ineligible_n` is excluded from `attempted_beat_n`. Attempted records reconcile exactly as valid plus missed plus invalid. Reference metrics use strict all-method common-reference beats. Disagreement uses all-method common-valid beats and must be interpreted with the excluded-beat count.

## Derived analytical inputs

Files under `data/derived/` preserve event-level or participant-level evidence used to reproduce the canonical summaries. They are not additional competing result tables. Their original sources and hashes are listed in the manifests.
