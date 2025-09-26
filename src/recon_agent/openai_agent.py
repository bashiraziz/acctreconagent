from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from .agent import ReconciliationAgent
from .models import LedgerBalance, Transaction


_LEDGER_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "account": {"type": "string"},
        "period": {"type": "string"},
        "amount": {"type": "number"},
    },
    "required": ["account", "amount"],
}


@dataclass
class OpenAIAgentConfig:
    """Configuration for orchestrating through the OpenAI Agents SDK."""

    model: str = "gpt-4.1-mini"
    name: str = "GL Reconciliation Supervisor"
    instructions: str = (
        "You coordinate accounting reconciliations. When the user requests work, call the "
        "`run_reconciliation` tool with the provided data to produce results and respond with "
        "a concise summary plus next steps."
    )
    api_key: str | None = None
    agent_id: str | None = None

    def resolve_api_key(self) -> str:
        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set. Provide api_key or export the environment variable.")
        return key


class _ReconciliationTool:
    """Tool surface that exposes the reconciliation agent to OpenAI Agents."""

    name = "run_reconciliation"
    description = (
        "Execute a GL vs subledger reconciliation and roll-forward schedule generation. "
        "Expects arrays of balance and transaction objects with ISO date strings."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "gl_balances": {"type": "array", "items": _LEDGER_SCHEMA},
            "subledger_balances": {"type": "array", "items": _LEDGER_SCHEMA},
            "transactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "account": {"type": "string"},
                        "booked_at": {"type": "string", "description": "ISO date"},
                        "description": {"type": "string"},
                        "debit": {"type": "number"},
                        "credit": {"type": "number"},
                        "amount": {"type": "number"},
                        "period": {"type": "string"},
                    },
                    "required": ["account", "booked_at"],
                },
            },
            "ordered_periods": {
                "type": "array",
                "items": {"type": "string"},
            },
            "activity_by_period": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
            "adjustments_by_period": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
        },
        "required": ["gl_balances", "subledger_balances"],
    }

    def __init__(self, agent: ReconciliationAgent) -> None:
        self.agent = agent

    @staticmethod
    def _parse_balances(raw: Iterable[Dict[str, Any]]) -> List[LedgerBalance]:
        balances: List[LedgerBalance] = []
        for item in raw:
            balances.append(
                LedgerBalance(
                    account=str(item.get("account")),
                    period=str(item.get("period", "")),
                    amount=float(item.get("amount", 0.0)),
                )
            )
        return balances

    @staticmethod
    def _parse_transactions(raw: Optional[Iterable[Dict[str, Any]]]) -> List[Transaction]:
        if not raw:
            return []
        txns: List[Transaction] = []
        for item in raw:
            booked = date.fromisoformat(str(item.get("booked_at")))
            debit = float(item.get("debit", 0.0))
            credit = float(item.get("credit", 0.0))
            if "amount" in item and "debit" not in item and "credit" not in item:
                net = float(item.get("amount", 0.0))
                debit = max(net, 0.0)
                credit = max(-net, 0.0)
            txns.append(
                Transaction(
                    account=str(item.get("account")),
                    booked_at=booked,
                    description=str(item.get("description", "")),
                    debit=debit,
                    credit=credit,
                    metadata={"period": str(item.get("period", ""))},
                )
            )
        return txns

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        gl_balances = self._parse_balances(kwargs["gl_balances"])
        sub_balances = self._parse_balances(kwargs["subledger_balances"])
        transactions = self._parse_transactions(kwargs.get("transactions"))
        ordered_periods: List[str] = list(kwargs.get("ordered_periods") or [])
        if not ordered_periods:
            ordered_periods = sorted({b.period for b in gl_balances if b.period})
        activity: Dict[str, float] = dict(kwargs.get("activity_by_period") or {})
        if not activity and transactions:
            for txn in transactions:
                period = txn.metadata.get("period", "")
                activity[period] = activity.get(period, 0.0) + txn.net
        adjustments: Dict[str, float] | None = kwargs.get("adjustments_by_period")

        results, schedule, summary = self.agent.run(
            gl_balances=gl_balances,
            subledger_balances=sub_balances,
            transactions=transactions,
            ordered_periods=ordered_periods,
            activity_by_period=activity,
            adjustments_by_period=adjustments,
        )

        return {
            "reconciliations": [
                {
                    "account": result.account,
                    "period": result.period,
                    "gl_balance": result.gl_balance,
                    "subledger_balance": result.subledger_balance,
                    "variance": result.variance,
                    "notes": result.notes,
                }
                for result in results
            ],
            "roll_forward": [
                {
                    "period": line.period,
                    "opening_balance": line.opening_balance,
                    "activity": line.activity,
                    "adjustments": line.adjustments,
                    "closing_balance": line.closing_balance,
                }
                for line in schedule.lines
            ],
            "insights": summary,
        }


class OpenAIAgentOrchestrator:
    """Helper to stand up an OpenAI Agent that can call the reconciliation tool."""

    def __init__(self, agent: ReconciliationAgent, config: OpenAIAgentConfig | None = None) -> None:
        self.config = config or OpenAIAgentConfig()
        self.client = OpenAI(api_key=self.config.resolve_api_key())
        self.tool = _ReconciliationTool(agent)
        self.agent_id = self.config.agent_id or self._ensure_agent()

    def _ensure_agent(self) -> str:
        created = self.client.agents.create(
            model=self.config.model,
            name=self.config.name,
            instructions=self.config.instructions,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": self.tool.name,
                        "description": self.tool.description,
                        "parameters": self.tool.parameters,
                    },
                }
            ],
        )
        return created.id

    def run(self, user_prompt: str, tool_payload: Optional[Dict[str, Any]] = None) -> str:
        thread = self.client.threads.create()
        self.client.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_prompt,
        )
        run = self.client.threads.runs.create(
            thread_id=thread.id,
            agent_id=self.agent_id,
            override={
                "run": {
                    "metadata": tool_payload or {},
                }
            },
        )
        return self._poll_thread(thread.id, run.id)

    def _poll_thread(self, thread_id: str, run_id: str) -> str:
        while True:
            run = self.client.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.status == "completed":
                messages = self.client.threads.messages.list(thread_id=thread_id)
                for message in reversed(messages.data):
                    if message.role == "assistant":
                        return "\n".join(part.text.value for part in message.content if part.type == "text")
                return "Agent run completed with no assistant message."
            if run.status == "requires_action":
                self._handle_required_actions(thread_id, run)
            elif run.status in {"failed", "cancelled", "expired"}:
                raise RuntimeError(f"Agent run ended with status: {run.status}")
            time.sleep(1)

    def _handle_required_actions(self, thread_id: str, run: Any) -> None:
        tool_outputs = []
        required = run.required_action.submit_tool_outputs
        for call in required.tool_calls:
            if call.type != "function":
                continue
            if call.function.name == self.tool.name:
                arguments = json.loads(call.function.arguments)
                result = self.tool(**arguments)
                tool_outputs.append({
                    "tool_call_id": call.id,
                    "output": json.dumps(result),
                })
        if tool_outputs:
            self.client.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )
        else:
            raise RuntimeError("Agent requested unsupported tool calls.")
