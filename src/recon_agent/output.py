from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .models import ReconciliationResult, RollForwardSchedule


def export_reconciliations(path: Path, results: Iterable[ReconciliationResult]) -> None:
    fieldnames = [
        "account",
        "period",
        "gl_balance",
        "subledger_balance",
        "variance",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "account": result.account,
                    "period": result.period,
                    "gl_balance": f"{result.gl_balance:.2f}",
                    "subledger_balance": f"{result.subledger_balance:.2f}",
                    "variance": f"{result.variance:.2f}",
                    "notes": " | ".join(result.notes),
                }
            )


def export_roll_forward(path: Path, schedule: RollForwardSchedule) -> None:
    fieldnames = [
        "account",
        "period",
        "opening_balance",
        "activity",
        "adjustments",
        "closing_balance",
        "commentary",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in schedule.as_rows():
            writer.writerow(row)
