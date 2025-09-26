from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List


@dataclass
class LedgerBalance:
    """Represents a balance for a general ledger account for a specific period."""

    account: str
    period: str
    amount: float


@dataclass
class Transaction:
    """Represents a transactional line item associated to a GL account."""

    account: str
    booked_at: date
    description: str
    debit: float = 0.0
    credit: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def net(self) -> float:
        return self.debit - self.credit


@dataclass
class ReconciliationResult:
    """Stores reconciliation output for a single account and period."""

    account: str
    period: str
    gl_balance: float
    subledger_balance: float
    variance: float
    unresolved_transactions: List[Transaction] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class RollForwardLine:
    account: str
    period: str
    opening_balance: float
    activity: float
    adjustments: float
    closing_balance: float
    commentary: str = ""


@dataclass
class RollForwardSchedule:
    account: str
    lines: List[RollForwardLine] = field(default_factory=list)

    def as_rows(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for line in self.lines:
            rows.append(
                {
                    "account": line.account,
                    "period": line.period,
                    "opening_balance": f"{line.opening_balance:.2f}",
                    "activity": f"{line.activity:.2f}",
                    "adjustments": f"{line.adjustments:.2f}",
                    "closing_balance": f"{line.closing_balance:.2f}",
                    "commentary": line.commentary,
                }
            )
        return rows
