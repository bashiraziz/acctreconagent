from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List

from .models import LedgerBalance, Transaction


@dataclass
class DataSourceConfig:
    name: str
    path: Path
    account_column: str
    amount_column: str
    period_column: str | None = None


class BaseDataSource:
    """Abstract interface for loading data from ERP, GL exports, or subledgers."""

    config: DataSourceConfig

    def __init__(self, config: DataSourceConfig):
        self.config = config

    def load_balances(self) -> Iterable[LedgerBalance]:
        raise NotImplementedError

    def load_transactions(self) -> Iterable[Transaction]:
        raise NotImplementedError


class CSVBalanceDataSource(BaseDataSource):
    """Simple CSV reader for balances and transaction exports."""

    def __init__(self, config: DataSourceConfig, date_column: str | None = None):
        super().__init__(config)
        self.date_column = date_column

    def _rows(self) -> Iterator[dict[str, str]]:
        with self.config.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield row

    def load_balances(self) -> List[LedgerBalance]:
        balances: List[LedgerBalance] = []
        for row in self._rows():
            account = row[self.config.account_column]
            period = row.get(self.config.period_column or "period", "")
            amount = float(row[self.config.amount_column])
            balances.append(LedgerBalance(account=account, period=period, amount=amount))
        return balances

    def load_transactions(self) -> List[Transaction]:
        if not self.date_column:
            return []
        transactions: List[Transaction] = []
        for row in self._rows():
            account = row[self.config.account_column]
            period = row.get(self.config.period_column or "period", "")
            amount = float(row[self.config.amount_column])
            transactions.append(
                Transaction(
                    account=account,
                    booked_at=_parse_csv_date(row[self.date_column]),
                    description=row.get("description", ""),
                    debit=max(amount, 0.0),
                    credit=max(-amount, 0.0),
                    metadata={"period": period},
                )
            )
        return transactions


def _parse_csv_date(raw: str):
    from datetime import datetime

    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {raw}")
