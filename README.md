# ECG ICG fiducial variability analysis

This repository contains the verified analyses for Publication 1 Research Question 1: Which B-point and X-point localisation approaches are supported by available references, and what factors contribute to their variability and failure?

The study compares 10 frozen B definitions and 13 frozen X definitions in HeartCycle, IITB-ICG and ReBeatICG. It separates reference accuracy from algorithm disagreement. HeartCycle uses ECHO AVO and AVC event references, ReBeat uses expert ICG annotations, and IITB has no independent B/X timing reference. No result supports a universally best method.

## Repository contents

- `src/variability_analysis/` contains the frozen preprocessing and detector code used for source replay.
- `src/statistics/` contains the participant-aware morphology and paired-condition analyses.
- `scripts/` contains release consolidation, HeartCycle replay and validation entry points.
- `results/tables/` contains the authoritative scientific tables.
- `results/figures/` contains 66 accepted panels. Each panel has PNG, PDF, plotting CSV and caption files with the same stem.
- `results/reports/` contains the 12-page supervisor report.
- `data/derived/` contains the participant-level and event-level records needed to verify final statistics. It does not contain raw recordings.
- `manifests/` records source hashes, output hashes, provenance and validation results.

## Datasets

Raw recordings are not redistributed.

- HeartCycle: local HDF5 recordings with separate ECHO event channels used for AVO and AVC alignment.
- IITB-ICG: rest and post-exercise recordings. Dataset access and acquisition notes are described in `docs/dataset_setup.md`.
- ReBeatICG: baseline and cognitive-workload recordings from Zenodo record 4725433, with expert ICG annotations kept distinct from ECHO.

Copy `configs/paths.example.yaml` to `configs/paths.local.yaml`, set the local data root, and keep that local file untracked.

## Python environment

Python 3.12 is the supported version used for the independent release validation.

```powershell
python3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` records the exact versions installed during the successful isolated replay.

## Tested commands

Validate the packaged results, detector hashes, statistics, figure metadata and raw-data exclusion:

```powershell
.\.venv\Scripts\python.exe scripts\validate_results.py
```

Replay HeartCycle from the external HDF5 files and compare R, C, B1 to B10, X1 to X13, ECHO alignment and final reference metrics:

```powershell
.\.venv\Scripts\python.exe scripts\replay_heartcycle.py --data-root "D:\path\to\Variability Analysis"
```

Replay all 84 IITB and ReBeat MAT recordings and compare every frozen candidate status, sample, search window and timing value:

```powershell
.\.venv\Scripts\python.exe scripts\replay_iitb_rebeat.py --data-root "D:\path\to\Variability Analysis"
```

Rebuild the clean result package from the preserved working project:

```powershell
.\.venv\Scripts\python.exe scripts\consolidate_results.py --source-project "D:\path\to\verified-working-project" --data-root "D:\path\to\Variability Analysis"
.\.venv\Scripts\python.exe scripts\validate_results.py
```

Run regression tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Frozen fiducial definitions

The full definitions, parameters, dependency fields and implementation status are in `configs/canonical_method_registry.csv`.

- B1 uses the last qualifying local minimum from R to C.
- B2 and B3 use third-derivative criteria before C.
- B4, B5 and B10 use 45%, 25% and 15% dZ/dt upstroke crossings.
- B6 and B7 use second-derivative maximum or zero-crossing criteria.
- B8 is the fixed 55% R-to-C timing definition.
- B9 is the adaptive baseline plus 10% upstroke crossing.
- X1 and X2 use the first local or global post-C minimum.
- X3 and X4 use third-derivative criteria.
- X5, X6 and X13 use 45%, 25% and 15% downstroke crossings.
- X7, X8 and X9 use second-derivative criteria.
- X10 is the provisional Weissler heart-rate relation dependent on B8.
- X11 uses half the C-to-trough descent.
- X12 uses the global minimum in a C-to-C+0.4RR window bounded by the next R.

## Results

The primary 115-row table combines valid, missed, invalid and upstream-ineligible counts in one row per dataset, condition, point type and algorithm. Separate tables are used only when the observation unit changes, such as participant-level reference accuracy, paired condition changes, morphology tests or Bland-Altman agreement.

The corrected IITB X3 beat-to-beat variability association remains unsupported: rho 0.39736, exact permutation p 0.11111, BH q 0.96044, n=18. Seventeen participants had zero failure-rate change. Its positive bootstrap interval is conditional on non-degenerate resamples and is not independent evidence of support.

See `docs/scientific_findings.md` for the supported interpretation and `docs/known_limitations.md` for boundaries on the claims.

## Final report

The supervisor report is at `results/reports/P1_Final_Supervisor_RQ1_Report.docx`. It remains 12 A4 pages, uses Times New Roman, and has no title page, table of contents or reference page.

No Git remote is configured by the release preparation. No licence is included because ownership and third-party licensing have not been confirmed.
