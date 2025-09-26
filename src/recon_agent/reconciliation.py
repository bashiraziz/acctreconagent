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

        accounts = set(gl_map.keys()) | set(sub_map.keys())
        results: List[ReconciliationResult] = []
        for account, period in sorted(accounts):
            gl_amount = gl_map.get((account, period), 0.0)
            sub_amount = sub_map.get((account, period), 0.0)
            variance = gl_amount - sub_amount
            unresolved = txn_map.get((account, period), []) if abs(variance) > self.materiality_threshold else []
            notes: List[str] = []
            if abs(variance) > self.materiality_threshold:
                notes.append(
                    f"Variance of {variance:.2f} exceeds threshold {self.materiality_threshold:.2f}."
                )
            results.append(
                ReconciliationResult(
                    account=account,
                    period=period,
                    gl_balance=gl_amount,
                    subledger_balance=sub_amount,
                    variance=variance,
                    unresolved_transactions=unresolved,
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

    @staticmethod
    def _group_transactions(
        transactions: Iterable[Transaction],
    ) -> Dict[Tuple[str, str], List[Transaction]]:
        grouped: Dict[Tuple[str, str], List[Transaction]] = defaultdict(list)
        for txn in transactions:
            period = txn.metadata.get("period", "")
            grouped[(txn.account, period)].append(txn)
        return grouped
