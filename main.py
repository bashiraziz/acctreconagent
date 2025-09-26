from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from recon_agent import (
    AgentConfig,
    GeminiConfig,
    GeminiInsightGenerator,
    GeminiLLM,
    LedgerBalance,
    ReconciliationAgent,
    SimpleInsightGenerator,
    Transaction,
)


def _select_insight_generator() -> SimpleInsightGenerator:
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        try:
            llm = GeminiLLM(GeminiConfig())
            return GeminiInsightGenerator(llm)
        except Exception as exc:  # noqa: BLE001
            print(f"Gemini setup failed ({exc}); falling back to simple insights.")
    return SimpleInsightGenerator()


def main() -> None:
    gl_balances = [
        LedgerBalance(account="1000", period="2025-07", amount=10500.0),
        LedgerBalance(account="1000", period="2025-08", amount=9800.0),
    ]
    subledger_balances = [
        LedgerBalance(account="1000", period="2025-07", amount=10300.0),
        LedgerBalance(account="1000", period="2025-08", amount=9800.0),
    ]
    transactions = [
        Transaction(
            account="1000",
            booked_at=date(2025, 7, 15),
            description="Manual adjustment",
            debit=200.0,
            credit=0.0,
            metadata={"period": "2025-07"},
        )
    ]
    agent = ReconciliationAgent(
        config=AgentConfig(materiality_threshold=10.0),
        insight_generator=_select_insight_generator(),
    )
    ordered_periods = ["2025-07", "2025-08"]
    activity = {"2025-07": 500.0, "2025-08": -700.0}
    adjustments = {"2025-07": 200.0}
    reconciliations, roll_forward, summary = agent.run(
        gl_balances=gl_balances,
        subledger_balances=subledger_balances,
        transactions=transactions,
        ordered_periods=ordered_periods,
        activity_by_period=activity,
        adjustments_by_period=adjustments,
    )

    print("Reconciliation results:")
    for result in reconciliations:
        print(
            f"Account {result.account} period {result.period} variance {result.variance:.2f}"
        )
        if result.notes:
            for note in result.notes:
                print(f"  - {note}")

    print("\nRoll forward schedule:")
    for line in roll_forward.lines:
        print(
            f"{line.period}: opening {line.opening_balance:.2f} activity {line.activity:.2f} "
            f"adjustments {line.adjustments:.2f} closing {line.closing_balance:.2f}"
        )

    if summary:
        print("\nInsights:")
        print(summary)


if __name__ == "__main__":
    main()
