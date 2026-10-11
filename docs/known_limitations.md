# Known limitations

- HeartCycle ECHO and ReBeat expert ICG annotations are different reference types and are not pooled as equivalent ground truth.
- IITB has no independent B/X timing reference.
- Common-reference and common-valid analyses exclude beats where one or more methods fail. Coverage must be interpreted with the timing distribution.
- No formally approved timing-error tolerance exists. Percentage-above-tolerance results are unavailable.
- The six morphology analyses are exploratory. Association does not establish a physiological cause of detector failure.
- Condition recordings differ in RR interval, motion exposure, acquisition context and other physiology. Paired changes do not isolate one causal mechanism.
- The IITB X3 bootstrap interval is conditional on non-degenerate resamples because 17 of 18 failure changes were zero.
- Raw HDF5 and MAT recordings are not redistributed. A full source replay requires authorized local access.
- X10 uses a provisional average-coefficient Weissler relation and depends on B8. It is not a height-adjusted subject-specific equation.
- The repository has no open-source licence pending confirmation of code ownership and third-party permissions.
