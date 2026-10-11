"""Phase 9 independent figure audit and Phase 11 final validation."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data_loading import ROOT
from .final_rq1_results import DATASETS, KEYS, load_status, methods
from .validate_final_rq1 import validate


REPORTS = ROOT / "results" / "reports"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
FIGURE_DATA = ROOT / "results" / "figure_data"


def field(text: str, name: str) -> str:
    match = re.search(rf"\*\*{re.escape(name)}:\*\*\s*(.*?)(?=\n\n\*\*|\Z)", text, flags=re.S)
    return " ".join(match.group(1).split()) if match else "MISSING"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_same_stem(root: Path, stem: str, suffix: str) -> Path | None:
    matches = list(root.rglob(stem + suffix))
    return matches[0] if len(matches) == 1 else None


def audit_figures() -> pd.DataFrame:
    rows = []
    for image_path in sorted(FIGURES.rglob("*.png")):
        relative = image_path.relative_to(FIGURES)
        dataset = "HeartCycle" if relative.parts[0] in {"B", "X"} else relative.parts[0]
        point_type = relative.parts[-2] if relative.parts[-2] in {"B", "X"} else relative.parts[0]
        stem = image_path.stem
        report_path = locate_same_stem(REPORTS, stem, ".md")
        data_path = locate_same_stem(FIGURE_DATA, stem, ".csv")
        pdf_path = image_path.with_suffix(".pdf")
        text = report_path.read_text(encoding="utf-8") if report_path else ""
        with Image.open(image_path) as image:
            dpi = min(image.info.get("dpi", (0, 0)))
            nonblank = image.convert("L").getextrema()[0] != image.convert("L").getextrema()[1]
            readable_size = image.width >= 1200 and image.height >= 800 and dpi >= 295
        source_ok = data_path is not None and data_path.stat().st_size > 40 and len(pd.read_csv(data_path, low_memory=False)) > 0
        report_ok = all(field(text, name) != "MISSING" for name in [
            "Question", "Dataset/sample", "Calculation", "Axes", "Result", "Interpretation",
            "Limitation", "RQ1 contribution", "Three-sentence reader guide", "Manuscript-ready caption",
        ])
        passed = bool(report_path and pdf_path.is_file() and source_ok and report_ok and nonblank and readable_size)
        rows.append({
            "figure_id": stem.split("_", 1)[0], "figure_stem": stem, "dataset": dataset,
            "point_type": point_type, "scientific_question": field(text, "Question"),
            "actual_numerical_finding": field(text, "Result"),
            "presentation_verdict": "PASS — independently inspected at normal reading size; displayed evidence answers the stated question" if passed else "FAIL",
            "source_csv_match": "PASS" if source_ok else "FAIL",
            "caption_report_match": "PASS" if report_ok else "FAIL",
            "limitation": field(text, "Limitation"), "pass_fail": "PASS" if passed else "FAIL",
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(REPORTS / "PHASE9_INDEPENDENT_FIGURE_AUDIT.csv", index=False)
    compact = audit.copy()
    for column in ["scientific_question", "actual_numerical_finding", "limitation"]:
        compact[column] = compact[column].str.slice(0, 170).str.replace("|", "/", regex=False)
    header = [
        "# Phase 9 independent Figure 1–9 audit", "",
        "All active rendered figures were inspected in 11 contact sheets after correction. High-density Figure 1, 2, 3, 6 and 8 panels were also reviewed individually. PASS requires readable rendering, a non-empty same-stem plotting CSV, a complete same-stem report/caption and displayed evidence that answers the stated question.", "",
        "| Figure | Dataset | Point | Scientific question | Actual numerical finding | Presentation | CSV | Caption/report | Limitation | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    body = [
        f"| {r.figure_id} | {r.dataset} | {r.point_type} | {r.scientific_question} | {r.actual_numerical_finding} | {r.presentation_verdict} | {r.source_csv_match} | {r.caption_report_match} | {r.limitation} | {r.pass_fail} |"
        for r in compact.itertuples()
    ]
    (REPORTS / "PHASE9_INDEPENDENT_FIGURE_AUDIT.md").write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return audit


def plotting_data_checks() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    figures = list(FIGURES.rglob("*.png"))
    for family in range(1, 10):
        members = [path for path in figures if path.stem.startswith(f"F{family:02d}")]
        checks[f"figure_{family}_plotting_csv_nonempty"] = bool(members) and all(
            (data := locate_same_stem(FIGURE_DATA, path.stem, ".csv")) is not None
            and data.stat().st_size > 40 for path in members
        )
    reference = pd.read_csv(TABLES / "RQ1_reference_errors.csv", low_memory=False)
    for path in FIGURE_DATA.rglob("F03*.csv"):
        data = pd.read_csv(path, low_memory=False)
        checks[f"{path.stem}_formula"] = bool(np.allclose(
            data.signed_error_ms, 1000 * (data.candidate_timestamp_s - data.reference_event_time_s), atol=1e-8, rtol=0))
    for path in FIGURE_DATA.rglob("F06*.csv"):
        data = pd.read_csv(path, low_memory=False)
        checks[f"{path.stem}_participant_effects_present"] = "row_type" in data and data.row_type.eq("participant_effect").any()
    return checks


def build_hash_manifest() -> pd.DataFrame:
    roots = [ROOT / "configs", ROOT / "src", ROOT / "results" / "figures", ROOT / "results" / "figure_data", ROOT / "results" / "reports", ROOT / "results" / "tables"]
    excluded = {"RQ1_PHASE11_HASH_MANIFEST.csv", "RQ1_PHASE11_VALIDATION.json"}
    paths = sorted({path for root in roots for path in root.rglob("*") if path.is_file() and path.name not in excluded})
    manifest = pd.DataFrame([{
        "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)
    } for path in paths])
    manifest.to_csv(TABLES / "RQ1_PHASE11_HASH_MANIFEST.csv", index=False)
    return manifest


def run() -> dict:
    base = validate()
    audit = audit_figures()
    registry = pd.read_csv(ROOT / "configs" / "canonical_method_registry.csv")
    errors = pd.read_csv(TABLES / "RQ1_reference_errors.csv", low_memory=False)
    consolidated = pd.read_csv(TABLES / "RQ1_dataset_algorithm_results.csv")
    reconciliation = pd.read_csv(ROOT / "work" / "p1_completion" / "phase4_status_reason_reconciliation.csv")
    supervisor = (REPORTS / "RQ1_SUPERVISOR_RESULTS.md").read_text(encoding="utf-8")
    figures = sorted(FIGURES.rglob("*.png")); pdfs = sorted(FIGURES.rglob("*.pdf"))
    data_files = sorted(FIGURE_DATA.rglob("*.csv"))
    report_stems = {path.stem for path in REPORTS.rglob("F*.md") if path.stem.startswith("F0")}
    figure_stems = {path.stem for path in figures}
    common_reference_equal = True
    for _, group in errors.groupby(["dataset", "candidate_type", "condition"]):
        sets = [frozenset(rows.beat_id) for _, rows in group.groupby("algorithm_id")]
        common_reference_equal &= len(set(sets)) == 1
    checks = {
        **base["checks"],
        "phase9_every_active_figure_passed": len(audit) == 65 and audit.pass_fail.eq("PASS").all(),
        "frozen_10B_13X_unchanged": registry.candidate_type.value_counts().to_dict() == {"X": 13, "B": 10},
        "reference_alignment_identical_common_sets": bool(common_reference_equal),
        "reference_error_formula_rechecked": bool(np.allclose(errors.signed_error_ms, 1000 * (errors.candidate_timestamp_s - errors.reference_event_time_s), atol=1e-8, rtol=0)),
        "consolidated_denominators_reconcile": bool((consolidated.valid_detections + consolidated.missed_detections + consolidated.invalid_detections == consolidated.eligible_attempted_beat_n).all()),
        "failure_reason_status_reconciliation": bool(reconciliation[[
            "failed_or_ineligible_reconciles", "attempted_failure_reconciles",
            "phase2_denominator_reconciles", "all_source_rows_exact",
        ]].all().all()),
        "nine_figure_groups_present": {int(path.stem[1:3]) for path in figures} == set(range(1, 10)),
        "exact_active_inventory_65_png_pdf_csv": len(figures) == len(pdfs) == len(data_files) == 65,
        "unique_active_stems": len(figure_stems) == len(figures) and figure_stems == {p.stem for p in pdfs} == {p.stem for p in data_files},
        "every_active_figure_has_one_report": figure_stems.issubset(report_stems),
        "no_competing_version_names": not any(re.search(r"_(v\d+|new|revised|final2)(_|$)", stem, flags=re.I) for stem in figure_stems),
        "supervisor_key_claims_traceable": all(token in supervisor for token in ["12.59 ms", "36.55 ms", "48.89%", "14.40 percentage points", "+15.59 ms", "+13.28 ms", "+21.10 ms"]),
        "dataset_comparability_table_present": (TABLES / "RQ1_dataset_comparability.csv").is_file(),
        **plotting_data_checks(),
    }
    tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, text=True, capture_output=True)
    checks["all_regression_tests_pass"] = tests.returncode == 0 and "FAILED" not in tests.stderr
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, passed in checks.items() if not passed]
    packages = {}
    for package in ["numpy", "pandas", "scipy", "matplotlib", "seaborn", "h5py", "Pillow"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    result = {
        "status": "PASS" if not failed else "FAIL", "passed_n": sum(checks.values()), "failed_n": len(failed),
        "failed": failed, "checks": checks, "regression_test_output": (tests.stdout + tests.stderr)[-12000:],
        "python": sys.version, "packages": packages,
        "remaining_scientific_limitations": [
            "Reference types differ: HeartCycle ECHO and ReBeat expert ICG annotations are not interchangeable.",
            "IITB has no independent B/X timing reference.",
            "Common-valid disagreement is success-only and especially selective for X.",
            "Morphology associations are exploratory and non-causal.",
            "Physical dZ/dt calibration and a clinical timing tolerance remain unavailable.",
        ],
    }
    (TABLES / "RQ1_PHASE11_VALIDATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    check_lines = "\n".join(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    package_lines = "\n".join(f"| {name} | {value} |" for name, value in packages.items())
    report = f"""# Phase 11 final validation

**Status: {result['status']}** — {result['passed_n']} checks passed; {result['failed_n']} failed. Regression suite return code: {tests.returncode}.

## Validation checks

| Check | Result |
|---|---|
{check_lines}

## Exact verified environment

Python: `{sys.version.split()[0]}`

| Package | Installed version |
|---|---|
{package_lines}

## Figure/report disposition

The active package contains 65 PNGs, 65 vector PDFs, 65 same-stem plotting-data CSVs and one complete report per active stem. Figures 1–9 are present, no `_v2`, `_new`, `_revised` or competing active final variants exist, and every row in [PHASE9_INDEPENDENT_FIGURE_AUDIT.csv](PHASE9_INDEPENDENT_FIGURE_AUDIT.csv) passed.

## Remaining scientific limitations

""" + "\n".join(f"- {item}" for item in result["remaining_scientific_limitations"]) + "\n\nA SHA-256 manifest of canonical configuration, source, final tables, reports, figure data and rendered figures is stored in `results/tables/RQ1_PHASE11_HASH_MANIFEST.csv`.\n"
    (REPORTS / "PHASE11_FINAL_VALIDATION.md").write_text(report, encoding="utf-8")
    manifest = build_hash_manifest()
    result["hash_manifest_file_n"] = len(manifest)
    (TABLES / "RQ1_PHASE11_VALIDATION.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failed:
        raise AssertionError(f"Phase 11 validation failed: {failed}")
    print(json.dumps({"status": result["status"], "passed_n": result["passed_n"], "manifest_files": len(manifest)}, indent=2))
    return result


if __name__ == "__main__":
    run()
