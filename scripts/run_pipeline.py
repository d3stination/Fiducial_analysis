"""Run the supported P1 release workflows."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, help="Verified working project used to rebuild the clean package")
    parser.add_argument("--data-root", type=Path, help="External Variability Analysis root containing raw datasets")
    parser.add_argument("--consolidate", action="store_true", help="Rebuild canonical tables and accepted figure package")
    parser.add_argument("--heartcycle-replay", action="store_true", help="Replay HeartCycle from HDF5 inputs")
    parser.add_argument("--iitb-rebeat-replay", action="store_true", help="Replay IITB and ReBeat from MAT inputs")
    parser.add_argument("--validate", action="store_true", help="Run release scientific checks")
    args = parser.parse_args()
    if not any([args.consolidate, args.heartcycle_replay, args.iitb_rebeat_replay, args.validate]):
        args.validate = True
    if args.consolidate:
        if not args.source_project:
            parser.error("--consolidate requires --source-project")
        command = [sys.executable, "scripts/consolidate_results.py", "--source-project", str(args.source_project)]
        if args.data_root:
            command.extend(["--data-root", str(args.data_root)])
        run(command)
    if args.heartcycle_replay:
        if not args.data_root:
            parser.error("--heartcycle-replay requires --data-root")
        run([sys.executable, "scripts/replay_heartcycle.py", "--data-root", str(args.data_root)])
    if args.iitb_rebeat_replay:
        if not args.data_root:
            parser.error("--iitb-rebeat-replay requires --data-root")
        run([sys.executable, "scripts/replay_iitb_rebeat.py", "--data-root", str(args.data_root)])
    if args.validate:
        run([sys.executable, "scripts/validate_results.py"])


if __name__ == "__main__":
    main()
