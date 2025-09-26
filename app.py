from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from recon_agent import (  # type: ignore  # pylint: disable=wrong-import-position
    AgentConfig,
    LedgerBalance,
    ReconciliationAgent,
    ReconciliationResult,
    RollForwardSchedule,
    SimpleInsightGenerator,
    Transaction,
)


def _read_csv(upload: UploadedFile) -> pd.DataFrame:
    contents = upload.read()
    upload.seek(0)
    return pd.read_csv(io.StringIO(contents.decode("utf-8")))


def _balances_from_df(df: pd.DataFrame) -> List[LedgerBalance]:
    df = df.rename(columns=str.lower)
    required = {"account", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required balance columns: {', '.join(sorted(missing))}")
    return [
        LedgerBalance(
            account=str(row.get("account")),
            period=str(row.get("period", "")),
            amount=float(row.get("amount")),
        )
        for _, row in df.iterrows()
    ]


def _transactions_from_df(df: pd.DataFrame) -> List[Transaction]:
    df = df.rename(columns=str.lower)
    required = {"account", "booked_at"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {', '.join(sorted(missing))}")
    debit_col = "debit" if "debit" in df.columns else None
    credit_col = "credit" if "credit" in df.columns else None
    amount_col = "amount" if "amount" in df.columns else None

    transactions: List[Transaction] = []
    for _, row in df.iterrows():
        debit = float(row.get(debit_col, 0.0)) if debit_col else 0.0
        credit = float(row.get(credit_col, 0.0)) if credit_col else 0.0
        if amount_col and not debit_col and not credit_col:
            net = float(row.get(amount_col, 0.0))
            debit = max(net, 0.0)
            credit = max(-net, 0.0)
        booked = pd.to_datetime(row.get("booked_at")).date()
        period = str(row.get("period", booked.strftime("%Y-%m")))
        transactions.append(
            Transaction(
                account=str(row.get("account")),
                booked_at=booked,
                description=str(row.get("description", "")),
                debit=debit,
                credit=credit,
                metadata={"period": period},
            )
        )
    return transactions


def _periods_from_balances(balances: List[LedgerBalance]) -> List[str]:
    periods = sorted({balance.period for balance in balances if balance.period})
    return periods or [date.today().strftime("%Y-%m")]


def _activity_from_transactions(transactions: List[Transaction]) -> dict[str, float]:
    activity: dict[str, float] = {}
    for txn in transactions:
        period = txn.metadata.get("period", "")
        activity[period] = activity.get(period, 0.0) + txn.net
    return activity


def _render_results(
    results: List[ReconciliationResult],
    schedule: RollForwardSchedule,
    summary: str | None,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    results_df: Optional[pd.DataFrame] = None
    schedule_df: Optional[pd.DataFrame] = None

    if results:
        st.subheader("Reconciliation Results")
        results_df = pd.DataFrame(
            {
                "Account": [r.account for r in results],
                "Period": [r.period for r in results],
                "GL Balance": [r.gl_balance for r in results],
                "Subledger Balance": [r.subledger_balance for r in results],
                "Variance": [r.variance for r in results],
                "Notes": [" | ".join(r.notes) for r in results],
            }
        )
        st.dataframe(results_df)
    if schedule.lines:
        st.subheader("Roll Forward Schedule")
        schedule_df = pd.DataFrame(
            {
                "Period": [line.period for line in schedule.lines],
                "Opening": [line.opening_balance for line in schedule.lines],
                "Activity": [line.activity for line in schedule.lines],
                "Adjustments": [line.adjustments for line in schedule.lines],
                "Closing": [line.closing_balance for line in schedule.lines],
            }
        )
        st.dataframe(schedule_df)
    if summary:
        st.subheader("Insights")
        st.write(summary)

    return results_df, schedule_df


st.set_page_config(page_title="GL Reconciliation Agent", page_icon=":bar_chart:", layout="wide")
st.title("GL Reconciliation & Roll Forward Agent")
st.write(
    "Upload GL and subledger exports to automatically reconcile balances, generate roll forward "
    "schedules, and capture narrative insights."
)

with st.sidebar:
    st.header("Controls")
    threshold = st.number_input(
        "Materiality threshold",
        min_value=0.0,
        value=10.0,
        step=1.0,
        help="Variances above this amount are flagged for review.",
    )

col1, col2 = st.columns(2)
with col1:
    gl_file = st.file_uploader(
        "General Ledger balances CSV",
        type=["csv"],
        help="Required columns: account, period (YYYY-MM), amount",
    )
with col2:
    subledger_file = st.file_uploader(
        "Subledger balances CSV",
        type=["csv"],
        help="Required columns: account, period (YYYY-MM), amount",
    )

transactions_file = st.file_uploader(
    "Detailed transactions CSV (optional)",
    type=["csv"],
    help="Columns: account, booked_at, description, amount or debit/credit, optional period",
)

submit = st.button("Run reconciliation", type="primary")

if submit:
    if not gl_file or not subledger_file:
        st.error("Please provide both GL balances and subledger balances.")
    else:
        try:
            gl_df = _read_csv(gl_file)
            sub_df = _read_csv(subledger_file)
            txn_df = _read_csv(transactions_file) if transactions_file else None

            gl_balances = _balances_from_df(gl_df)
            sub_balances = _balances_from_df(sub_df)
            transactions = _transactions_from_df(txn_df) if txn_df is not None else []

            ordered_periods = _periods_from_balances(gl_balances)
            activity = _activity_from_transactions(transactions)

            agent = ReconciliationAgent(
                config=AgentConfig(materiality_threshold=threshold),
                insight_generator=SimpleInsightGenerator(),
            )
            results, schedule, summary = agent.run(
                gl_balances=gl_balances,
                subledger_balances=sub_balances,
                transactions=transactions,
                ordered_periods=ordered_periods,
                activity_by_period=activity,
                adjustments_by_period=None,
            )

            results_df, schedule_df = _render_results(results, schedule, summary)

            if results_df is not None:
                st.download_button(
                    "Download reconciliation csv",
                    results_df.to_csv(index=False).encode("utf-8"),
                    file_name="reconciliation_results.csv",
                    mime="text/csv",
                )
            if schedule_df is not None:
                st.download_button(
                    "Download roll forward csv",
                    schedule_df.to_csv(index=False).encode("utf-8"),
                    file_name="roll_forward_schedule.csv",
                    mime="text/csv",
                )

        except Exception as exc:  # noqa: BLE001
            st.error(f"Error while processing files: {exc}")
