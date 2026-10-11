"""Scientific and packaging checks for the clean P1 release."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MANIFESTS = ROOT / "manifests"
EXPECTED_HASHES = {
    "src/variability_analysis/bx_candidates.py": "b5581c121542731b4b34ce8b33eb4b420ce33b43c93b02cf3ef4a97116a057cd",
    "src/variability_analysis/phase4.py": "2351684a1c03a8a53fb66f4da1ea4fbb25c162aefad392212bfc5333f5099140",
    "src/variability_analysis/signal_processing.py": "0a5b6741e50ec47773b4a8367825b428592a80c076a37bcf1ccb40a49ceaab2e",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(rows: list[dict], name: str, passed: bool, observed: object, expected: object) -> None:
    rows.append({"check": name, "passed": bool(passed), "observed": str(observed), "expected": str(expected)})


def duplicate_count(frame: pd.DataFrame, key: list[str]) -> int:
    return int(frame.duplicated(key).sum())


def figure_audit() -> pd.DataFrame:
    rows = []
    for png in sorted(FIGURES.rglob("*.png")):
        pdf, csv, report = png.with_suffix(".pdf"), png.with_suffix(".csv"), png.with_suffix(".md")
        text = report.read_text(encoding="utf-8") if report.exists() else ""
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            width, height = image.size
        lower = text.lower()
        sample = "participant" in lower and ("beat" in lower or "n=" in lower)
        result = "result" in lower or "had a median" in lower
        limitation = "limitation" in lower
        rows.append({
            "figure_stem": png.stem,
            "scientific_group": png.parent.name,
            "png_exists": png.exists(),
            "pdf_exists": pdf.exists(),
            "plotting_csv_exists": csv.exists(),
            "caption_exists": report.exists(),
            "png_dpi_x": float(dpi[0]) if dpi else 0,
            "png_dpi_y": float(dpi[1]) if dpi else 0,
            "width_px": width,
            "height_px": height,
            "caption_mentions_sample": sample,
            "caption_mentions_result": result,
            "caption_has_explicit_limitation": limitation,
            "acceptance_pass": bool(pdf.exists() and csv.exists() and report.exists() and sample and result and limitation and min(dpi) >= 299),
        })
    return pd.DataFrame(rows)


def source_manifest() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        expected = EXPECTED_HASHES.get(relative, "")
        rows.append({
            "path": relative,
            "sha256": digest,
            "frozen_expected_sha256": expected,
            "frozen_hash_match": digest == expected if expected else "not_applicable",
        })
    return pd.DataFrame(rows)


def output_provenance() -> pd.DataFrame:
    rows = []
    for folder in [ROOT / "results", ROOT / "data/derived", ROOT / "docs", ROOT / "configs"]:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            generator = "scripts/consolidate_results.py" if relative.startswith(("results/tables/", "results/figures/", "data/derived/")) else "maintained release documentation"
            rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path), "provenance": generator})
    return pd.DataFrame(rows)


def sensitive_path_hits() -> list[str]:
    hits = []
    pattern = re.compile(r"[A-Za-z]:[\\/](?:Users|Documents|Downloads)[\\/]", re.IGNORECASE)
    extensions = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".toml"}
    ignored = {"source_to_release_inventory.csv"}
    for path in ROOT.rglob("*"):
        if any(part in {".venv", ".git"} for part in path.relative_to(ROOT).parts):
            continue
        if not path.is_file() or path.suffix.lower() not in extensions or path.name in ignored or path.stat().st_size > 25_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            hits.append(path.relative_to(ROOT).as_posix())
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=ROOT / "manifests/validation_summary.json")
    args = parser.parse_args()
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    summary = pd.read_csv(TABLES / "dataset_algorithm_results.csv")
    check(checks, "dataset algorithm strata", len(summary) == 115, len(summary), 115)
    check(checks, "dataset labels", set(summary.dataset) == {"HeartCycle", "IITB", "ReBeat"}, sorted(summary.dataset.unique()), "HeartCycle, IITB, ReBeat")
    check(checks, "all B methods", set(summary.loc[summary.point_type.eq("B"), "algorithm_id"]) == {f"B{i}" for i in range(1, 11)}, summary.loc[summary.point_type.eq("B"), "algorithm_id"].nunique(), 10)
    check(checks, "all X methods", set(summary.loc[summary.point_type.eq("X"), "algorithm_id"]) == {f"X{i}" for i in range(1, 14)}, summary.loc[summary.point_type.eq("X"), "algorithm_id"].nunique(), 13)
    identity = summary.total_beat_n.eq(summary.attempted_beat_n + summary.upstream_ineligible_n)
    attempts = summary.attempted_beat_n.eq(summary.valid_detection_n + summary.missed_detection_n + summary.invalid_detection_n)
    check(checks, "status denominators reconcile", bool((identity & attempts).all()), int((~(identity & attempts)).sum()), 0)
    check(checks, "dataset summary unique", duplicate_count(summary, ["dataset", "condition", "point_type", "algorithm_id"]) == 0, duplicate_count(summary, ["dataset", "condition", "point_type", "algorithm_id"]), 0)

    morphology = pd.read_csv(TABLES / "morphology_associations.csv")
    check(checks, "six-feature morphology tests", len(morphology) == 1794, len(morphology), 1794)
    check(checks, "six morphology features", morphology.feature.nunique() == 6, morphology.feature.nunique(), 6)

    condition = pd.read_csv(TABLES / "condition_morphology_associations.csv")
    check(checks, "condition morphology tests", len(condition) == 672, len(condition), 672)
    check(checks, "condition morphology supported count", int(condition.exploratory_supported.sum()) == 14, int(condition.exploratory_supported.sum()), 14)
    x3 = condition.loc[
        condition.dataset.eq("IITB") & condition.point_type.eq("X") & condition.algorithm_id.eq("X3")
        & condition.outcome.eq("failure_rate_pct") & condition.feature.eq("beat_to_beat_variability")
    ]
    check(checks, "IITB X3 row unique", len(x3) == 1, len(x3), 1)
    if len(x3) == 1:
        row = x3.iloc[0]
        check(checks, "IITB X3 rho", np.isclose(row.spearman_rho, 0.3973597071195132), row.spearman_rho, 0.3973597071195132)
        check(checks, "IITB X3 exact permutation p", np.isclose(row.permutation_p, 1 / 9), row.permutation_p, 1 / 9)
        check(checks, "IITB X3 q", np.isclose(row.bh_q_value, 0.9604357211337692), row.bh_q_value, 0.9604357211337692)
        check(checks, "IITB X3 unsupported", not bool(row.exploratory_supported), row.exploratory_supported, False)
        check(checks, "IITB X3 bootstrap caution", "conditional" in str(row.bootstrap_interpretation).lower() and "17 of 18" in str(row.bootstrap_interpretation), row.bootstrap_interpretation, "conditional non-degenerate-resample limitation")

    failures = pd.read_csv(TABLES / "detection_failure_mechanisms.csv")
    check(checks, "failure reason strata", failures.groupby(["dataset", "condition", "point_type", "algorithm_id"]).ngroups == 115, failures.groupby(["dataset", "condition", "point_type", "algorithm_id"]).ngroups, 115)
    check(checks, "failure reason reconciliation", bool(failures.groupby(["dataset", "condition", "point_type", "algorithm_id"]).reason_record_n.sum().eq(failures.groupby(["dataset", "condition", "point_type", "algorithm_id"]).failed_or_ineligible_n.first()).all()), "checked", "all reason totals equal failed or ineligible totals")

    figures = figure_audit()
    figures.to_csv(MANIFESTS / "figure_acceptance.csv", index=False)
    check(checks, "accepted figure panels", len(figures) == 66 and bool(figures.acceptance_pass.all()), f"{int(figures.acceptance_pass.sum())}/{len(figures)}", "66/66")

    hashes = source_manifest()
    hashes.to_csv(MANIFESTS / "source_hashes.csv", index=False)
    frozen = hashes.loc[hashes.frozen_expected_sha256.ne("")]
    check(checks, "frozen detector hashes", len(frozen) == 3 and frozen.frozen_hash_match.astype(str).str.lower().eq("true").all(), f"{int(frozen.frozen_hash_match.astype(str).str.lower().eq('true').sum())}/{len(frozen)}", "3/3")

    raw = [
        path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".h5", ".mat"}
        and ".venv" not in path.relative_to(ROOT).parts
    ]
    check(checks, "no raw recordings", not raw, raw[:5], "none")
    private_paths = sensitive_path_hits()
    check(checks, "no private absolute paths", not private_paths, private_paths, "none")
    check(checks, "final Word report present", (ROOT / "results/reports/P1_Final_Supervisor_RQ1_Report.docx").exists(), (ROOT / "results/reports/P1_Final_Supervisor_RQ1_Report.docx").exists(), True)

    validation = pd.DataFrame(checks)
    validation.to_csv(MANIFESTS / "validation_summary.csv", index=False)
    output_provenance().to_csv(MANIFESTS / "output_provenance.csv", index=False)
    payload = {"passed": bool(validation.passed.all()), "checks": checks}
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(validation.to_string(index=False))
    if not validation.passed.all():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
