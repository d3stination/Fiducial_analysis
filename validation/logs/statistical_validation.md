# Targeted statistical validation

The audit checked all 672 condition-morphology tests. The pre-correction IITB X3 beat-to-beat variability versus failure-change row had rho 0.3973597071, permutation p 0.00009999, BH q 0.009099, bootstrap interval 0.317375 to 0.692017 and n=18. Seventeen of the 18 failure-rate changes were zero.

The original permutation implementation compared a SciPy Spearman statistic with vectorised tied-rank correlations using an exact floating-point boundary. Algebraically equal permutations differed at about 1e-16, so none met the strict boundary and the plus-one Monte Carlo formula returned 1/10,001. This was a numerical comparison bug, not evidence for an alternative test.

The corrected participant-label Monte Carlo result was p 0.108889. Exhaustive enumeration of the 18 distinct placements of the sole non-zero outcome gave the official two-sided p 2/18 = 0.111111. Rho was unchanged. The corrected BH q was 0.9604357211. The participant bootstrap interval was 0.398182 to 0.656044 after re-ranking every resample and excluding degenerate resamples; it is conditional and does not provide independent support.

Across the 672-test family, 73 permutation p-values and 602 bootstrap intervals changed after the numerical-boundary and resample-ranking corrections. One support classification changed. Supported results fell from 15 to 14. No other support classification changed.

Source-linked evidence is in `data/derived/iitb_x3_participant_pairs.csv`, `data/derived/condition_morphology_statistical_audit.csv` and `data/derived/final_targeted_validation_summary.csv`.
