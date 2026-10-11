# Methodology

## Signal processing and alignment

The frozen pipeline reads each recording without changing the source file. ECG R events are detected with the sampling-rate-adapted offline Pan-Tompkins implementation. ICG is filtered with a third-order zero-phase Butterworth bandpass from 0.8 to 30 Hz. The C candidate is the retained global maximum in the frozen R-to-C search region, with 60 to 200 ms treated as a quality-control interval rather than a new detector rule.

Each beat spans one detected ECG R event to the next. All B and X timings use the detected R as the time origin. B definitions operate before C. X definitions operate after C and before the next R. Sampling frequency, indices and event timestamps remain in the exported evidence.

## Frozen B and X definitions

The 10 B definitions and 13 X definitions are implemented in `src/variability_analysis/bx_candidates.py`. `configs/canonical_method_registry.csv` is the readable registry. `phase4.py` executes each definition and applies the frozen boundary and order checks. `phase4_finalize.py` resolves X10 after its B8 dependency is available.

No detector was changed during release preparation. The byte-level hashes for `bx_candidates.py`, `phase4.py` and `signal_processing.py` are checked in every validation run.

## Detection statuses and verified failure reasons

An upstream-ineligible beat is kept separate from a method attempt. Attempted outcomes are valid, missed or invalid. The mutually exclusive primary failure hierarchy is:

1. Upstream dependency or ineligibility.
2. Unusable search window.
3. No qualifying feature or crossing.
4. Rejected boundary or order constraint.
5. Other verified reason.
6. Unresolved.

The assigned reason must follow the recorded execution path. Acquisition warnings remain contextual quality-control fields unless the frozen method explicitly records an acquisition-based rejection.

## Reference matching and accuracy

HeartCycle ECHO R events are matched to detected ECG R events within the frozen 120 ms tolerance. AVO and AVC must be unique and fall within the matched detected R-to-R interval. B timing error is detected B minus ECHO AVO. X timing error is detected X minus ECHO AVC.

ReBeat expert B and X annotations use their verified one-based coordinate mapping. They are not described as independent ECHO references. IITB has no independent B/X timing reference and is excluded from reference-accuracy claims.

Reference comparisons use strict all-method common-reference samples. Each participant contributes a median absolute error for each method. Signed error and P90/P95 absolute-error tails are also retained. No tolerance-exceedance percentage is reported because no approved clinical or analytical threshold exists.

## Disagreement and coverage

Within-beat range is the maximum valid method timing minus the minimum valid method timing on a beat where every method for that point type is valid. Method-specific peer disagreement is the absolute difference from the median of the other methods. These measures quantify consensus, not accuracy.

Common-valid beat counts and excluded beats are reported with disagreement. Lower disagreement among retained X beats is not interpreted as improved reliability without considering detection failures and common-valid coverage.

## Morphology and quality analyses

The six prespecified features are peak or turning-point sharpness, local waveform slope, signal amplitude, signal-to-noise ratio, competing extrema or inflection points, and adjacent-beat morphological variability. Feature strata use outcome-blind within-participant medians. Equal-to-median observations are assigned to the low category.

All 10 B and 13 X definitions remain in the all-method screen. Outcomes are strict common-reference error when a comparable reference exists, attempted detection failure, and method-specific or all-method disagreement. Participant effects are estimated within participant before participant-level summaries and bootstrap intervals are calculated. The 1,794-test table preserves supported and unsupported findings.

## Paired condition analyses

IITB compares the same 18 participants at rest and post-exercise. ReBeat compares the same 24 participants at baseline and cognitive workload. Separate recordings are compared through participant summaries, not beat-to-beat pairing. Changes are stress condition minus baseline condition.

Condition-morphology associations use participant-level feature changes and reliability changes. Spearman statistics use participant as the permutation unit. P values are two-sided and BH q values are calculated within dataset, point type and outcome families. These are exploratory associations, not causal effects.

## Bland-Altman analysis

Predeclared related algorithm pairs are compared on identical valid beats. Bias and empirical limits of agreement use equal participant weight. Confidence intervals use participant-clustered bootstrap resampling. Agreement between methods is not independent-reference accuracy.
