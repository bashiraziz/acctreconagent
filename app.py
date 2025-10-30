from __future__ import annotations

import io
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_env_file(paths: Iterable[Path], *, override: bool = False) -> None:
    for path in paths:
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key.startswith("#"):
                continue
            cleaned = value.strip().strip("'\"")
            if override or key not in os.environ:
                os.environ[key] = cleaned


env_candidates = [Path(".env"), Path(".env.local")]
if load_dotenv is not None:
    load_dotenv(override=False)
else:
    _load_env_file(env_candidates, override=False)

from recon_agent import (  # type: ignore  # pylint: disable=wrong-import-position
    AgentConfig,
    LedgerBalance,
    ReconciliationAgent,
    ReconciliationResult,
    RollForwardSchedule,
    SimpleInsightGenerator,
    Transaction,
)

REPO_URL = os.getenv("ACCTRECON_REPO_URL", "https://github.com/your-org/acctreconagent")
README_URL = os.getenv("ACCTRECON_README_URL", f"{REPO_URL}#readme")
DOCS_URL = os.getenv("ACCTRECON_DOCS_URL", f"{REPO_URL}/blob/main/docs/openai_agents.md")

DEFAULT_COLUMN_ALIASES: Dict[str, Dict[str, List[str]]] = {
    "balances": {
        "account": ["account", "accountcode", "glaccount", "mainaccount"],
        "amount": ["amount", "balance", "endingbalance", "balanceamount"],
        "period": ["period", "fiscalperiod", "accountingperiod", "month"]
    },
    "transactions": {
        "account": ["account", "accountcode", "glaccount"],
        "booked_at": ["booked_at", "postingdate", "posteddate", "postdate", "transactiondate", "date"],
        "debit": ["debit", "debitamount", "debitlc"],
        "credit": ["credit", "creditamount", "creditlc"],
        "amount": ["amount", "netamount", "net", "balance"],
        "description": ["description", "memo", "narration", "details", "reference"],
        "period": ["period", "fiscalperiod", "accountingperiod", "month"]
    },
    "detail": {
        "account": ["account", "accountcode", "glaccount", "accountnumber"],
        "booked_at": ["booked_at", "postingdate", "posteddate", "postdate", "transactiondate", "glpostdate", "date"],
        "debit": ["debit", "debitamount", "debitlc"],
        "credit": ["credit", "creditamount", "creditlc"],
        "amount": ["amount", "netamount", "amountlc", "balance", "endingbalancelc"],
        "period": ["period", "fiscalperiod", "accountingperiod"],
        "description": ["description", "memo", "narration", "details"],
        "document_number": ["documentnumber", "docnumber", "documentno", "reference"],
        "document_date": ["documentdate", "docdate", "invoicedate"],
        "source": ["source", "journalsource", "journal", "entrytype", "category"]
    }
}


def _normalize_column_name(column: str) -> str:
    return ''.join(ch for ch in column.lower() if ch.isalnum())


def _merge_aliases(
    defaults: Dict[str, Dict[str, List[str]]],
    overrides: Dict[str, Dict[str, List[str]]] | None,
) -> Dict[str, Dict[str, List[str]]]:
    merged: Dict[str, Dict[str, List[str]]] = {
        section: {field: list(values) for field, values in fields.items()}
        for section, fields in defaults.items()
    }
    if not overrides:
        return merged
    for section, fields in overrides.items():
        section_key = section.lower()
        section_target = merged.setdefault(section_key, {})
        for field, names in fields.items():
            field_key = field.lower()
            section_target.setdefault(field_key, [])
            for name in names:
                if name not in section_target[field_key]:
                    section_target[field_key].append(name)
    return merged


def _with_aliases(
    section: str,
    field: str,
    base_candidates: List[str],
    aliases: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    section_key = section.lower()
    field_key = field.lower()
    extras = aliases.get(section_key, {}).get(field_key, [])
    seen: set[str] = set()
    ordered: List[str] = []
    for candidate in base_candidates + extras:
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)
    return ordered


def _parse_alias_overrides(raw_text: str) -> Dict[str, Dict[str, List[str]]]:
    raw_text = raw_text.strip()
    if not raw_text:
        return {}
    try:
        return _parse_alias_json(raw_text)
    except ValueError as json_error:
        try:
            return _parse_alias_lines(raw_text)
        except ValueError as line_error:
            raise ValueError(
                "Invalid alias definitions. Use JSON or lines like "
                "'Balances account column = GL Account, Account Code' or "
                "'Transactions posting date = Posting Date'. "
                f"({line_error})"
            ) from json_error


def _parse_alias_json(raw_text: str) -> Dict[str, Dict[str, List[str]]]:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid alias JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Alias overrides must be an object mapping")
    parsed: Dict[str, Dict[str, List[str]]] = {}
    for section, fields in data.items():
        if not isinstance(fields, dict):
            raise ValueError(f"Alias section '{section}' must map to an object of field arrays")
        section_key = str(section).lower()
        parsed_section: Dict[str, List[str]] = {}
        for field, names in fields.items():
            if isinstance(names, str):
                parsed_section[str(field).lower()] = [names]
            elif isinstance(names, list):
                parsed_section[str(field).lower()] = [str(name) for name in names]
            else:
                raise ValueError(
                    f"Alias list for '{section}.{field}' must be a string or list of strings"
                )
        parsed[section_key] = parsed_section
    return parsed


def _parse_alias_lines(raw_text: str) -> Dict[str, Dict[str, List[str]]]:
    parsed: Dict[str, Dict[str, List[str]]] = {}
    for idx, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            left, right = line.split(":", 1)
        elif "=" in line:
            left, right = line.split("=", 1)
        else:
            raise ValueError(f"Line {idx}: expected ':' or '=' to separate column aliases.")
        section, field = _interpret_alias_key(left.strip(), line_number=idx)
        aliases = [alias.strip() for alias in right.split(",") if alias.strip()]
        if not aliases:
            raise ValueError(f"Line {idx}: provide at least one alias after the delimiter.")
        section_entry = parsed.setdefault(section, {})
        field_aliases = section_entry.setdefault(field, [])
        for alias in aliases:
            if alias not in field_aliases:
                field_aliases.append(alias)
    if not parsed:
        raise ValueError("No alias mappings found.")
    return parsed


def _interpret_alias_key(raw_key: str, *, line_number: int) -> tuple[str, str]:
    key = raw_key.strip()
    if not key:
        raise ValueError(f"Line {line_number}: missing column name before delimiter.")
    if "." in key:
        section, field = (part.strip().lower() for part in key.split(".", 1))
        if not section or not field:
            raise ValueError(f"Line {line_number}: expected text on both sides of '.'.")
        return section, field

    tokens = [token.lower() for token in re.split(r"[\s_\-/]+", key) if token]
    if len(tokens) < 2:
        raise ValueError(
            f"Line {line_number}: please specify a section and column "
            f"(example: 'Balances account column')."
        )
    section = tokens[0]
    field = "_".join(tokens[1:])
    return section, field


def _read_tabular(upload: UploadedFile) -> pd.DataFrame:
    upload.seek(0, io.SEEK_END)
    size = upload.tell()
    upload.seek(0)
    if size > MAX_UPLOAD_BYTES:
        raise ValueError(f"File '{upload.name}' exceeds 20 MB limit.")

    suffix = Path(upload.name or "").suffix.lower()
    raw = upload.read()
    upload.seek(0)

    if suffix in {".csv", ""}:
        return pd.read_csv(io.StringIO(raw.decode("utf-8-sig")))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(io.BytesIO(raw))
    raise ValueError(f"Unsupported file format for '{upload.name}'. Please upload CSV or Excel.")


MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def _balances_from_df(df: pd.DataFrame, aliases: Dict[str, Dict[str, List[str]]]) -> List[LedgerBalance]:
    df = df.rename(columns=str.lower)
    column_map = {_normalize_column_name(col): col for col in df.columns}
    account_col = _find_first_column(
        column_map, _with_aliases("balances", "account", ["account"], aliases)
    )
    amount_col = _find_first_column(
        column_map, _with_aliases("balances", "amount", ["amount"], aliases)
    )
    if not account_col or not amount_col:
        missing = []
        if not account_col:
            missing.append("account")
        if not amount_col:
            missing.append("amount")
        raise ValueError(
            "Missing required balance columns: " + ", ".join(missing)
        )
    period_col = _find_first_column(
        column_map, _with_aliases("balances", "period", ["period"], aliases)
    )

    balances: List[LedgerBalance] = []
    for _, row in df.iterrows():
        period_value = str(row.get(period_col, "")) if period_col else ""
        balances.append(
            LedgerBalance(
                account=str(row.get(account_col)),
                period=period_value,
                amount=float(row.get(amount_col, 0.0)),
            )
        )
    return balances


def _find_first_column(column_map: Dict[str, str], candidates: List[str]) -> str | None:
    for candidate in candidates:
        normalized = _normalize_column_name(candidate)
        if normalized in column_map:
            return column_map[normalized]
    return None


def _coerce_float(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return 0.0
        value = cleaned
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _derive_period(raw_period: Any, booked_at: date) -> str:
    if raw_period not in (None, ""):
        text = str(raw_period)
        match = re.search(r"(\d{2,4})[-/](\d{2})", text)
        if match:
            year_raw, month_raw = match.groups()
            year = int(year_raw)
            if year < 100:
                year += 2000 if year < 70 else 1900
            month = int(month_raw)
            if 1 <= month <= 12:
                return f"{year:04d}-{month:02d}"
        parsed = pd.to_datetime(str(raw_period), errors="coerce")
        if not pd.isna(parsed):
            return parsed.to_period("M").strftime("%Y-%m")
    return booked_at.strftime("%Y-%m")


def _transactions_from_df(
    df: pd.DataFrame | None,
    aliases: Dict[str, Dict[str, List[str]]],
) -> List[Transaction]:
    if df is None or df.empty:
        return []
    df = df.rename(columns=str.lower)
    column_map = {_normalize_column_name(col): col for col in df.columns}
    account_col = _find_first_column(
        column_map, _with_aliases("transactions", "account", ["account"], aliases)
    )
    date_col = _find_first_column(
        column_map, _with_aliases("transactions", "booked_at", ["booked_at"], aliases)
    )
    if not account_col or not date_col:
        raise ValueError("Missing required transaction columns: account or posting date")
    debit_col = _find_first_column(
        column_map, _with_aliases("transactions", "debit", ["debit"], aliases)
    )
    credit_col = _find_first_column(
        column_map, _with_aliases("transactions", "credit", ["credit"], aliases)
    )
    amount_col = _find_first_column(
        column_map, _with_aliases("transactions", "amount", ["amount"], aliases)
    )
    description_col = _find_first_column(
        column_map, _with_aliases("transactions", "description", ["description"], aliases)
    )
    period_col = _find_first_column(
        column_map, _with_aliases("transactions", "period", ["period"], aliases)
    )

    transactions: List[Transaction] = []
    for _, row in df.iterrows():
        booked_at = row.get(date_col)
        booked_date = pd.to_datetime(booked_at).date()
        debit = _coerce_float(row.get(debit_col)) if debit_col else 0.0
        credit = _coerce_float(row.get(credit_col)) if credit_col else 0.0
        if not debit_col and not credit_col and amount_col:
            net_amount = _coerce_float(row.get(amount_col))
            debit = max(net_amount, 0.0)
            credit = max(-net_amount, 0.0)
        metadata_period = str(row.get(period_col, "")) if period_col else ""
        description = str(row.get(description_col, "")) if description_col else ""
        transactions.append(
            Transaction(
                account=str(row.get(account_col)),
                booked_at=booked_date,
                description=description,
                debit=debit,
                credit=credit,
                metadata={"period": metadata_period or booked_date.strftime("%Y-%m")},
            )
        )
    return transactions


def _transactions_from_detail_df(
    df: pd.DataFrame | None,
    ledger_label: str,
    aliases: Dict[str, Dict[str, List[str]]],
) -> List[Transaction]:
    if df is None or df.empty:
        return []

    df = df.rename(columns=str.lower)
    column_map = {_normalize_column_name(col): col for col in df.columns}
    account_col = _find_first_column(
        column_map, _with_aliases("detail", "account", ["account"], aliases)
    )
    date_col = _find_first_column(
        column_map, _with_aliases("detail", "booked_at", ["booked_at"], aliases)
    )
    if not account_col or not date_col:
        raise ValueError("Detail files must include account and posting date columns.")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    document_date_col = _find_first_column(
        column_map, _with_aliases("detail", "document_date", ["documentdate"], aliases)
    )
    if document_date_col:
        df[document_date_col] = pd.to_datetime(df[document_date_col], errors="coerce")

    debit_col = _find_first_column(
        column_map, _with_aliases("detail", "debit", ["debit"], aliases)
    )
    credit_col = _find_first_column(
        column_map, _with_aliases("detail", "credit", ["credit"], aliases)
    )
    amount_col = _find_first_column(
        column_map, _with_aliases("detail", "amount", ["amount"], aliases)
    )
    period_col = _find_first_column(
        column_map, _with_aliases("detail", "period", ["period"], aliases)
    )
    description_col = _find_first_column(
        column_map, _with_aliases("detail", "description", ["description"], aliases)
    )
    document_number_col = _find_first_column(
        column_map, _with_aliases("detail", "document_number", ["documentnumber"], aliases)
    )
    source_col = _find_first_column(
        column_map, _with_aliases("detail", "source", ["source"], aliases)
    )

    transactions: List[Transaction] = []
    for _, row in df.iterrows():
        booked_raw = row.get(date_col)
        if pd.isna(booked_raw):
            continue
        booked_date = pd.to_datetime(booked_raw).date()
        period_value = row.get(period_col) if period_col else None
        period = _derive_period(period_value, booked_date)

        debit = _coerce_float(row.get(debit_col)) if debit_col else 0.0
        credit = _coerce_float(row.get(credit_col)) if credit_col else 0.0
        if not debit_col and not credit_col and amount_col:
            net_amount = _coerce_float(row.get(amount_col))
            if net_amount >= 0:
                debit = net_amount
            else:
                credit = -net_amount

        metadata: Dict[str, str] = {"period": period, "ledger": ledger_label}
        if period_col and row.get(period_col) not in (None, ""):
            metadata["source_period"] = str(row.get(period_col))
        if document_number_col and row.get(document_number_col) not in (None, ""):
            metadata["document_number"] = str(row.get(document_number_col))
        if source_col and row.get(source_col) not in (None, ""):
            metadata["source"] = str(row.get(source_col))
        if document_date_col and not pd.isna(row.get(document_date_col)):
            metadata["document_date"] = pd.to_datetime(row.get(document_date_col)).date().isoformat()

        description = str(row.get(description_col, "")) if description_col else ""
        transactions.append(
            Transaction(
                account=str(row.get(account_col)),
                booked_at=booked_date,
                description=description,
                debit=debit,
                credit=credit,
                metadata=metadata,
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

with st.expander("How to use this app", expanded=False):
    st.markdown(
        "\n".join(
            [
                "- **Step 1**: Upload GL and subledger balance files (CSV or Excel) in the first row. "
                "Required columns: account, period (YYYY-MM), amount. Use `YYYY-MM` so periods sort correctly.",
                "- **Step 2 (optional)**: Provide a detailed transactions file with account, posting date, "
                "description, and amount/debit/credit to prefill activity.",
                "- **Step 3 (optional)**: Upload GL detail and subledger detail files for richer drill-down.",
                "- **Step 4**: Adjust the materiality threshold in the sidebar and click **Run reconciliation**.",
                "- Use the **Column aliases** panel if your headers differ (e.g., `Balances account column = GL Account`).",
                "- Each file can be up to 20 MB.",
            ]
        )
    )

with st.sidebar:
    st.header("Controls")
    st.markdown("### Resources")
    st.markdown(
        "\n".join(
            [
                f"- [Project repository]({REPO_URL})",
                f"- [README / Quickstart]({README_URL})",
                f"- [Detailed docs]({DOCS_URL})",
            ]
        )
    )
    threshold = st.number_input(
        "Materiality threshold",
        min_value=0.0,
        value=10.0,
        step=1.0,
        help="Variances above this amount are flagged for review.",
    )
    with st.expander("Column aliases", expanded=False):
        st.write(
            "Provide alias overrides if your source headers differ. "
            "You can paste JSON or write plain lines like "
            "'Balances account column = GL Account, Account Code'."
        )
        alias_text = st.text_area(
            "Alias overrides",
            key="alias_overrides",
            placeholder=(
                "Balances account column = GL Account, Account Code\n"
                "Transactions posting date = Posting Date"
            ),
        )
        alias_error: Optional[str] = None
        try:
            user_aliases = _parse_alias_overrides(alias_text)
        except ValueError as exc:  # noqa: F841 - feedback to user
            alias_error = str(exc)
            user_aliases = {}
        if alias_error:
            st.error(alias_error)
        column_aliases = _merge_aliases(DEFAULT_COLUMN_ALIASES, user_aliases)

default_cols = st.columns(2)
with default_cols[0]:
    gl_file = st.file_uploader(
        "General Ledger balances file",
        type=["csv", "xlsx", "xls"],
        help="Required columns: account, period (YYYY-MM), amount. Max size 20 MB. CSV or Excel.",
    )
with default_cols[1]:
    subledger_file = st.file_uploader(
        "Subledger balances file",
        type=["csv", "xlsx", "xls"],
        help="Required columns: account, period (YYYY-MM), amount. Max size 20 MB. CSV or Excel.",
    )

transactions_file = st.file_uploader(
    "Detailed transactions file (optional)",
    type=["csv", "xlsx", "xls"],
    help="Columns: account, posting date, description, amount or debit/credit, optional period. Max size 20 MB. CSV or Excel.",
)

detail_cols = st.columns(2)
with detail_cols[0]:
    gl_detail_file = st.file_uploader(
        "GL detail file (optional)",
        type=["csv", "xlsx", "xls"],
        help="Include account, posting date, and either debit/credit or amount columns. Max size 20 MB. CSV or Excel.",
        key="gl_detail_csv",
    )
with detail_cols[1]:
    subledger_detail_file = st.file_uploader(
        "Subledger detail file (optional)",
        type=["csv", "xlsx", "xls"],
        help="Include account, posting date, and either debit/credit or amount columns. Max size 20 MB. CSV or Excel.",
        key="subledger_detail_csv",
    )

submit = st.button("Run reconciliation", type="primary")

if submit:
    if not gl_file or not subledger_file:
        st.error("Please provide both GL balances and subledger balances.")
    else:
        try:
            gl_df = _read_tabular(gl_file)
            sub_df = _read_tabular(subledger_file)
            txn_df = _read_tabular(transactions_file) if transactions_file else None
            gl_detail_df = _read_tabular(gl_detail_file) if gl_detail_file else None
            subledger_detail_df = _read_tabular(subledger_detail_file) if subledger_detail_file else None

            gl_balances = _balances_from_df(gl_df, column_aliases)
            sub_balances = _balances_from_df(sub_df, column_aliases)
            transactions = _transactions_from_df(txn_df, column_aliases)

            gl_detail_transactions = _transactions_from_detail_df(
                gl_detail_df,
                "GL",
                column_aliases,
            )
            subledger_detail_transactions = _transactions_from_detail_df(
                subledger_detail_df,
                "Subledger",
                column_aliases,
            )

            ordered_periods = _periods_from_balances(gl_balances)
            activity_source = transactions or gl_detail_transactions or subledger_detail_transactions
            activity = _activity_from_transactions(activity_source)

            agent = ReconciliationAgent(
                config=AgentConfig(materiality_threshold=threshold),
                insight_generator=SimpleInsightGenerator(),
            )
            results, schedule, summary = agent.run(
                gl_balances=gl_balances,
                subledger_balances=sub_balances,
                transactions=transactions + gl_detail_transactions + subledger_detail_transactions,
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
