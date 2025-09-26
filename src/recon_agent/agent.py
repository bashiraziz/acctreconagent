from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

from .models import LedgerBalance, ReconciliationResult, RollForwardSchedule, Transaction
from .reconciliation import ReconciliationEngine
from .rollforward import RollForwardBuilder


class InsightGenerator(Protocol):
    def summarize(self, reconciliations: List[ReconciliationResult]) -> str:
        ...


@dataclass
class AgentConfig:
    materiality_threshold: float = 1.0
    adjustments_name: str = "adjustments"


class ReconciliationAgent:
    """Coordinates data ingestion, reconciliation, roll forwards, and insights."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        insight_generator: InsightGenerator | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.insight_generator = insight_generator
        self.reconciliation_engine = ReconciliationEngine(
            materiality_threshold=self.config.materiality_threshold
        )
        self.roll_forward_builder = RollForwardBuilder(
            adjustments_name=self.config.adjustments_name
        )

    def run(
        self,
        gl_balances: Iterable[LedgerBalance],
        subledger_balances: Iterable[LedgerBalance],
        transactions: Iterable[Transaction],
        ordered_periods: List[str],
        activity_by_period: dict[str, float],
        adjustments_by_period: dict[str, float] | None = None,
    ) -> tuple[List[ReconciliationResult], RollForwardSchedule, str | None]:
        reconciliations = self.reconciliation_engine.reconcile(
            gl_balances=gl_balances,
            subledger_balances=subledger_balances,
            transactions=transactions,
        )

        accounts = {balance.account for balance in gl_balances}
        account = next(iter(accounts)) if accounts else "unknown"
        roll_forward = self.roll_forward_builder.build(
            account=account,
            ordered_periods=ordered_periods,
            balances=gl_balances,
            activity_by_period=activity_by_period,
            adjustments_by_period=adjustments_by_period,
        )

        summary: str | None = None
        if self.insight_generator:
            summary = self.insight_generator.summarize(reconciliations)
        return reconciliations, roll_forward, summary
