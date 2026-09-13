"""CLI to run the validation-phase eval harness against data/eval/*.json.

The validation phase is CLI/script-only (see design doc) - this is that
script. It loads the fixed eval cases, runs each through both the
claim-diff mechanism and the control pipeline via eval_harness, and
prints/saves a machine-checkable report (Success Criterion 1: counts,
never a percentage - the sample is too small for a rate to mean
anything).

Usage:
    python -m backend.run_eval_cli
    python -m backend.run_eval_cli --cases synthetic-er-red-flag-direct-conflict
    python -m backend.run_eval_cli --output data/eval/results/latest.json

Requires ANTHROPIC_API_KEY and GOOGLE_API_KEY in .env - this makes real,
billed calls to both providers.
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import eval_harness

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
CASE_FILES = [DATA_DIR / "synthetic_cases.json", DATA_DIR / "historical_cases.json"]


def filter_cases(
    cases: List[Dict[str, Any]], case_ids: Optional[List[str]]
) -> List[Dict[str, Any]]:
    if not case_ids:
        return cases
    wanted = set(case_ids)
    return [c for c in cases if c.get("id") in wanted]


def format_report(report: Dict[str, Any]) -> str:
    total = report["total_cases"]
    lines = [
        f"Eval cases: {total}",
        f"Retained -- new mechanism: {report['new_mechanism_retained_count']}/{total}"
        f"   control: {report['control_retained_count']}/{total}",
        "",
    ]
    for case in report["cases"]:
        new_r = "RETAINED" if case["new_mechanism"]["retained"] else "lost"
        ctrl_r = "RETAINED" if case["control"]["retained"] else "lost"
        lines.append(f"[{case['case_source']}] {case['case_id']}")
        lines.append(f"  new mechanism: {new_r}    control: {ctrl_r}")
    return "\n".join(lines)


def check_api_keys() -> None:
    """Fail fast, before any (billed) call, if the direct-call client can't run."""
    missing = [name for name in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY") if not os.getenv(name)]
    if missing:
        raise SystemExit(
            f"Missing required API key(s) in .env: {', '.join(missing)}. "
            "The direct-call client needs both to run a live eval."
        )


async def main_async(case_ids: Optional[List[str]], output_path: Path) -> Dict[str, Any]:
    check_api_keys()

    cases = eval_harness.load_eval_cases(*CASE_FILES)
    cases = filter_cases(cases, case_ids)
    if not cases:
        raise SystemExit("No matching eval cases found in data/eval/.")

    print(f"Running {len(cases)} eval case(s) against live models...\n")
    report = await eval_harness.run_eval_harness(cases)
    print(format_report(report))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {output_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", help="Comma-separated case ids to run (default: all)")
    parser.add_argument("--output", help="Path to save the full JSON report")
    args = parser.parse_args()

    case_ids = args.cases.split(",") if args.cases else None
    output_path = (
        Path(args.output)
        if args.output
        else DATA_DIR / "results" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    asyncio.run(main_async(case_ids, output_path))


if __name__ == "__main__":
    main()
