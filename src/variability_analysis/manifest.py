"""Explicit dataset manifest for the six approved cohort/condition groups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MANIFEST_VERSION = "phase1-2026-10-01"


@dataclass(frozen=True)
class DatasetSpec:
    """One immutable dataset/cohort/condition entry in the Phase 1 manifest."""

    dataset_id: str
    dataset_name: str
    cohort: str
    condition: str
    paired_group: str
    relative_directory: Path
    pattern: str
    loader: str
    expected_files: int
    expected_subjects: int
    annotation_directory: Path | None = None
    task_code: str = ""

    def directory(self, project_root: Path) -> Path:
        """Resolve the raw-data directory without modifying it."""

        return Path(project_root) / self.relative_directory

    def discover(self, project_root: Path) -> list[Path]:
        """Return deterministically ordered source files for this entry."""

        directory = self.directory(project_root)
        return sorted(directory.glob(self.pattern)) if directory.is_dir() else []


def dataset_manifest(project_root: Path) -> tuple[DatasetSpec, ...]:
    """Return the approved manifest; ``project_root`` is validated by callers."""

    del project_root
    return (
        DatasetSpec(
            "heartcycle_healthy", "HeartCycle", "healthy", "healthy",
            "heartcycle_healthy", Path("Heart Cycle"), "*.h5", "heartcycle",
            212, 17,
        ),
        DatasetSpec(
            "rebeat_baseline", "ReBeatICG", "healthy", "baseline",
            "rebeat", Path("ReBeatICG/Raw/Baseline"), "*.mat", "rebeat",
            24, 24, Path("ReBeatICG/Annotations/Baseline"), "BL",
        ),
        DatasetSpec(
            "rebeat_cognitive", "ReBeatICG", "healthy",
            "cognitive_workload", "rebeat",
            Path("ReBeatICG/Raw/Cognetive Workload"), "*.mat", "rebeat",
            24, 24, Path("ReBeatICG/Annotations/Cognetive Workload"), "CW",
        ),
        DatasetSpec(
            "iitb_normal", "IITB ICG", "healthy", "normal", "iitb_healthy",
            Path("IITB ICG/Normal Data"), "*.mat", "iitb", 18, 18,
        ),
        DatasetSpec(
            "iitb_post_exercise", "IITB ICG", "healthy", "post_exercise",
            "iitb_healthy", Path("IITB ICG/Excerise"), "*.mat", "iitb",
            18, 18,
        ),
        DatasetSpec(
            "iitb_cvd", "IITB ICG", "cardiovascular_disorders", "cvd",
            "iitb_cvd", Path("IITB ICG/cardiovascular disorders"), "*.mat",
            "iitb", 22, 22,
        ),
    )

