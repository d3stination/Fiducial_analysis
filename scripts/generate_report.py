"""Install the verified supervisor report in the clean release.

The report is copied, not rewritten. Public metadata scrubbing and visual
render checks are performed by the documented release procedure.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_project / "results/reports/P1_Final_Supervisor_RQ1_Report.docx"
    target = ROOT / "results/reports/P1_Final_Supervisor_RQ1_Report.docx"
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(target)


if __name__ == "__main__":
    main()
