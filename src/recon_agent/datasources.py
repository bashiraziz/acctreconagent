from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .models import LedgerBalance, Transaction


@dataclass
class DataSourceConfig:
    name: str
    path: Path
    account_column: str | None = None
    amount_column: str | None = None
    period_column: str | None = None
    column_aliases: Dict[str, List[str]] = field(default_factory=dict)


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

    def load_balances(self) -> List[LedgerBalance]:
        balances: List[LedgerBalance] = []
        with self.config.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            resolver = _ColumnResolver(headers, self.config, explicit_date=self.date_column)
            for row in reader:
                account = resolver.account(row)
                period = resolver.period(row)
                amount = resolver.amount(row)
                balances.append(LedgerBalance(account=account, period=period, amount=amount))
        return balances

    def load_transactions(self) -> List[Transaction]:
        if not self.date_column:
            return []
        transactions: List[Transaction] = []
        with self.config.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            resolver = _ColumnResolver(headers, self.config, explicit_date=self.date_column)
            for row in reader:
                account = resolver.account(row)
                period = resolver.period(row)
                amount = resolver.amount(row)
                booked_at = resolver.booked_at(row)
                description = resolver.description(row)
                transactions.append(
                    Transaction(
                        account=account,
                        booked_at=booked_at,
                        description=description,
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


DEFAULT_COLUMN_ALIASES: Dict[str, List[str]] = {
    "account": ["account", "accountcode", "accountnumber", "glaccount", "mainaccount"],
    "amount": ["amount", "balance", "endingbalance", "closingbalance", "balanceamount"],
    "period": ["period", "fiscalperiod", "accountingperiod", "month"],
    "date": ["booked_at", "transactiondate", "posteddate", "postdate", "date", "glpostdate"],
    "description": ["description", "memo", "details", "narration", "reference"],
}


class _ColumnResolver:
    """Resolves canonical column names using explicit config and alias fallbacks."""

    def __init__(self, headers: Sequence[str], config: DataSourceConfig, explicit_date: str | None):
        if not headers:
            raise ValueError("CSV file is missing a header row.")
        self._header_lookup = {
            self._normalize(header): header
            for header in headers
            if header is not None
        }
        self._aliases = _merge_aliases(DEFAULT_COLUMN_ALIASES, config.column_aliases)
        self._account_column = self._resolve_column(config.account_column, "account", required=True)
        self._amount_column = self._resolve_column(config.amount_column, "amount", required=True)
        self._period_column = self._resolve_column(config.period_column, "period", required=False)
        self._date_column = self._resolve_column(explicit_date, "date", required=False)
        self._description_column = self._resolve_column(None, "description", required=False)

    def account(self, row: dict[str, str]) -> str:
        return str(row.get(self._account_column, "")).strip()

    def amount(self, row: dict[str, str]) -> float:
        return _coerce_float(row.get(self._amount_column))

    def period(self, row: dict[str, str]) -> str:
        if not self._period_column:
            return ""
        return str(row.get(self._period_column, "")).strip()

    def booked_at(self, row: dict[str, str]):
        if not self._date_column:
            raise ValueError("Date column could not be resolved for transaction export.")
        raw = row.get(self._date_column)
        if raw is None:
            raise ValueError(f"Row missing date value for column '{self._date_column}'.")
        return _parse_csv_date(str(raw))

    def description(self, row: dict[str, str]) -> str:
        if not self._description_column:
            return ""
        return str(row.get(self._description_column, "")).strip()

    def _resolve_column(self, explicit: str | None, alias_key: str, *, required: bool) -> str | None:
        if explicit:
            normalized = self._normalize(explicit)
            if normalized in self._header_lookup:
                return self._header_lookup[normalized]
            raise ValueError(f"Configured column '{explicit}' not found in CSV headers.")

        for candidate in self._aliases.get(alias_key, []):
            normalized = self._normalize(candidate)
            if normalized in self._header_lookup:
                return self._header_lookup[normalized]
        if required:
            raise ValueError(
                f"Could not resolve required '{alias_key}' column. Provide an explicit column name or alias."
            )
        return None

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())


def _merge_aliases(
    defaults: Dict[str, List[str]], overrides: Dict[str, List[str]] | None,
) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {key: list(values) for key, values in defaults.items()}
    if not overrides:
        return merged
    for field, names in overrides.items():
        field_key = field.lower()
        merged.setdefault(field_key, [])
        for name in names:
            normalized = name.lower()
            if not any(existing.lower() == normalized for existing in merged[field_key]):
                merged[field_key].append(name)
    return merged


def _coerce_float(raw_value: object) -> float:
    if raw_value is None:
        return 0.0
    text = str(raw_value).strip()
    if not text:
        return 0.0
    cleaned = text.replace(",", "")
    try:
        return float(cleaned)
    except ValueError as exc:
        raise ValueError(f"Could not convert '{raw_value}' to float.") from exc
