# Reproducibility

## Environment

The independent release validation uses Python 3.12 in a virtual environment created without system site packages. `requirements.txt` states supported ranges. `requirements-lock.txt` records the exact successful installation.

## Execution order

1. Configure the external data root.
2. Install the isolated environment.
3. Run `scripts/validate_results.py` to verify packaged outputs.
4. Run `scripts/replay_heartcycle.py --data-root <path>` for the upstream HeartCycle replay.
5. Run `scripts/replay_iitb_rebeat.py --data-root <path>` for all IITB and ReBeat recordings.
6. Run `python -m unittest discover -s tests -v` for regression tests.

`scripts/run_pipeline.py` exposes the same operations through one entry point.

## Deterministic settings

The participant bootstrap and Monte Carlo permutation seed is 20261009. Standard bootstrap runs use 4,000 resamples. Condition-morphology Monte Carlo tests use 10,000 participant-label permutations. The corrected IITB X3 test uses the exact 18 distinct placements of its sole non-zero failure change.

Floating-point comparisons use exact sample and status matching, 51 ns for inherited serialized candidate timestamps, and 1e-8 ms for final reference metrics. Integer counts, identifiers and status labels must match exactly.

## Frozen hashes

The validator checks these SHA-256 values:

- `bx_candidates.py`: `b5581c121542731b4b34ce8b33eb4b420ce33b43c93b02cf3ef4a97116a057cd`
- `phase4.py`: `2351684a1c03a8a53fb66f4da1ea4fbb25c162aefad392212bfc5333f5099140`
- `signal_processing.py`: `0a5b6741e50ec47773b4a8367825b428592a80c076a37bcf1ccb40a49ceaab2e`

The complete source manifest is generated at `manifests/source_hashes.csv`.

## External-data boundary

The repository includes participant-level and selected event-level analytical evidence. It excludes raw recordings and the 206 MB all-dataset Phase 4 export. The HeartCycle replay generates fresh Phase 1 to Phase 4 outputs in a temporary local folder, compares scientific values and removes the temporary files unless `--keep-work` is supplied.

IITB and ReBeat raw replays use the same frozen package and require their external MAT files. The isolated validation reproduced 84 recordings, 20,654 beats and 475,042 method-beat records with zero differing status/count cells. HeartCycle reproduced 1,706 beats, 39,238 B/X records, ECHO alignment and all 23 final reference summaries. Machine-readable summaries are stored under `validation/`.
