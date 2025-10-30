from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from .models import LedgerBalance, ReconciliationResult, Transaction


class ReconciliationEngine:
    """Compares GL balances to subledger activity and flags variances."""

    def __init__(self, materiality_threshold: float = 1.0):
        self.materiality_threshold = materiality_threshold

    def reconcile(
        self,
        gl_balances: Iterable[LedgerBalance],
        subledger_balances: Iterable[LedgerBalance],
        transactions: Iterable[Transaction] | None = None,
    ) -> List[ReconciliationResult]:
        gl_map = self._group_balances(gl_balances)
        sub_map = self._group_balances(subledger_balances)
        txn_map = self._group_transactions(transactions or [])

        account_periods = set(gl_map.keys()) | set(sub_map.keys()) | set(txn_map.keys())
        results: List[ReconciliationResult] = []
        for account, period in sorted(account_periods):
            gl_amount = gl_map.get((account, period), 0.0)
            sub_amount = sub_map.get((account, period), 0.0)
            variance = gl_amount - sub_amount

            ledger_txns = txn_map.get((account, period), {})
            gl_txns = ledger_txns.get("GL", [])
            sub_txns = ledger_txns.get("Subledger", [])
            other_txns: List[Transaction] = []
            for ledger_name, txns in ledger_txns.items():
                if ledger_name not in {"GL", "Subledger"}:
                    other_txns.extend(txns)

            notes: List[str] = []
            if abs(variance) > self.materiality_threshold:
                notes.append(
                    f"Variance of {variance:.2f} exceeds threshold {self.materiality_threshold:.2f}."
                )

            if gl_txns or sub_txns:
                gl_total = sum(txn.net for txn in gl_txns)
                sub_total = sum(txn.net for txn in sub_txns)
                notes.append(
                    f"GL detail net {gl_total:.2f}; Subledger detail net {sub_total:.2f}."
                )

            unresolved: List[Transaction] = []
            if abs(variance) > self.materiality_threshold:
                unresolved = gl_txns + sub_txns + other_txns

            results.append(
                ReconciliationResult(
                    account=account,
                    period=period,
                    gl_balance=gl_amount,
                    subledger_balance=sub_amount,
                    variance=variance,
                    unresolved_transactions=unresolved,
                    gl_transactions=gl_txns,
                    subledger_transactions=sub_txns,
                    notes=notes,
                )
            )
        return results

    @staticmethod
    def _group_balances(balances: Iterable[LedgerBalance]) -> Dict[Tuple[str, str], float]:
        grouped: Dict[Tuple[str, str], float] = defaultdict(float)
        for balance in balances:
            grouped[(balance.account, balance.period)] += balance.amount
        return grouped

    @classmethod
    def _group_transactions(
        cls, transactions: Iterable[Transaction],
    ) -> Dict[Tuple[str, str], Dict[str, List[Transaction]]]:
        grouped: Dict[Tuple[str, str], Dict[str, List[Transaction]]] = defaultdict(dict)
        for txn in transactions:
            period = txn.metadata.get("period", "")
            ledger = cls._normalize_ledger_label(txn.metadata.get("ledger"))
            bucket = grouped.setdefault((txn.account, period), {})
            bucket.setdefault(ledger, []).append(txn)
        return grouped

    @staticmethod
    def _normalize_ledger_label(label: str | None) -> str:
        if not label:
            return "Other"
        cleaned = label.strip().lower()
        if cleaned in {"gl", "general ledger", "general_ledger", "general-ledger"}:
            return "GL"
        if cleaned in {"subledger", "sub-ledger", "sl", "sub"}:
            return "Subledger"
        return label.strip() or "Other"
