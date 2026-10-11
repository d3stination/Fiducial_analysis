# Clean reproducibility validation

Environment: Python 3.12.14 virtual environment with `include-system-site-packages = false`. Exact packages are in `requirements-lock.txt`.

Commands:

```powershell
.\.venv\Scripts\python.exe scripts\consolidate_results.py --source-project <verified-project> --data-root <external-data-root>
.\.venv\Scripts\python.exe scripts\replay_iitb_rebeat.py --data-root <external-data-root>
.\.venv\Scripts\python.exe scripts\replay_heartcycle.py --data-root <external-data-root>
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\validate_results.py
```

IITB/ReBeat: 84 recordings, 20,654 beats, 475,042 candidate records and 92 dataset-condition-point-method strata. The canonical keys matched and zero status/count cells differed.

HeartCycle: 1,706 Phase 2 rows, 1,706 Phase 3 rows, 39,238 Phase 4 candidate rows, 1,706 reference-alignment rows and 23 reference-summary rows. Keys, status classes, QC classes and sample indices matched exactly. Candidate timestamps matched within 51 ns. Shared reference-alignment values matched within 0.0000005 ms. Final reference metrics matched within 0.000001 ms. All six replay checks passed.

Regression tests: 10 passed, 0 failed. Release scientific/packaging checks: 23 passed, 0 failed. Raw datasets remain external and are not redistributed.
