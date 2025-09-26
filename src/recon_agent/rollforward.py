from __future__ import annotations

from typing import Iterable, List

from .models import LedgerBalance, RollForwardLine, RollForwardSchedule


class RollForwardBuilder:
    """Creates roll forward schedules from balances and activity."""

    def __init__(self, adjustments_name: str = "adjustments"):
        self.adjustments_name = adjustments_name

    def build(
        self,
        account: str,
        ordered_periods: List[str],
        balances: Iterable[LedgerBalance],
        activity_by_period: dict[str, float],
        adjustments_by_period: dict[str, float] | None = None,
    ) -> RollForwardSchedule:
        balance_map = {(balance.account, balance.period): balance.amount for balance in balances}
        schedule = RollForwardSchedule(account=account)
        adjustments_by_period = adjustments_by_period or {}

        opening_balance = 0.0
        for period in ordered_periods:
            closing = balance_map.get((account, period), opening_balance)
            activity = activity_by_period.get(period, 0.0)
            adjustments = adjustments_by_period.get(period, 0.0)
            line = RollForwardLine(
                account=account,
                period=period,
                opening_balance=opening_balance,
                activity=activity,
                adjustments=adjustments,
                closing_balance=closing,
            )
            schedule.lines.append(line)
            opening_balance = closing
        return schedule
