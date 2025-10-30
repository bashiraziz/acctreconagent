from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
import time

from datetime import date
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from .agent import ReconciliationAgent
from .models import LedgerBalance, Transaction



@dataclass
class AgentRunOutput:
    message: str
    tool_output: Dict[str, Any] | None
    messages_by_role: Dict[str, str] | None = None


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


@dataclass
class OrchestratedAgent:
    """Definition of an Agent instance that participates in orchestration."""

    role: str
    config: OpenAIAgentConfig
    uses_reconciliation_tool: bool = False


@dataclass
class OpenAIMultiAgentConfig:
    """Holds configuration for coordinating multiple Agents."""

    agents: List[OrchestratedAgent] = field(default_factory=list)
    api_key: str | None = None
    primary_role: str | None = None

    def resolve_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        for orchestrated in self.agents:
            if orchestrated.config.api_key:
                return orchestrated.config.resolve_api_key()
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set. Provide api_key or export the environment variable.")
        return key

    def resolve_primary_role(self) -> str:
        if self.primary_role:
            return self.primary_role
        if not self.agents:
            raise ValueError("At least one agent must be configured for orchestration.")
        return self.agents[0].role


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
                        "metadata": {"type": "object"},
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
    def _serialize_transaction(txn: Transaction) -> Dict[str, Any]:
        return {
            "account": txn.account,
            "booked_at": txn.booked_at.isoformat(),
            "description": txn.description,
            "debit": txn.debit,
            "credit": txn.credit,
            "net": txn.net,
            "metadata": txn.metadata,
        }

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
            metadata = dict(item.get("metadata") or {})
            period_value = str(item.get("period", metadata.get("period", "")))
            if period_value:
                metadata.setdefault("period", period_value)
            txns.append(
                Transaction(
                    account=str(item.get("account")),
                    booked_at=booked,
                    description=str(item.get("description", "")),
                    debit=debit,
                    credit=credit,
                    metadata=metadata,
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
                    "unresolved_transactions": [
                        self._serialize_transaction(txn) for txn in result.unresolved_transactions
                    ],
                    "gl_transactions": [
                        self._serialize_transaction(txn) for txn in result.gl_transactions
                    ],
                    "subledger_transactions": [
                        self._serialize_transaction(txn) for txn in result.subledger_transactions
                    ],
                }
                for result in results
            ],
            "roll_forward": {
                "account": schedule.account,
                "lines": [
                    {
                        "account": line.account,
                        "period": line.period,
                        "opening_balance": line.opening_balance,
                        "activity": line.activity,
                        "adjustments": line.adjustments,
                        "closing_balance": line.closing_balance,
                        "commentary": line.commentary,
                    }
                    for line in schedule.lines
                ],
            },
            "insights": summary,
        }


class OpenAIAgentOrchestrator:
    """Helper to stand up an OpenAI Agent that can call the reconciliation tool."""

    def __init__(
        self,
        agent: ReconciliationAgent,
        config: OpenAIAgentConfig | OpenAIMultiAgentConfig | None = None,
    ) -> None:
        self._multi_config = self._normalize_config(config)
        self.client = OpenAI(api_key=self._multi_config.resolve_api_key())
        self.tool = _ReconciliationTool(agent)
        self._tool_spec = {
            "type": "function",
            "function": {
                "name": self.tool.name,
                "description": self.tool.description,
                "parameters": self.tool.parameters,
            },
        }
        self._agent_ids = self._ensure_agents()
        self._primary_role = self._multi_config.resolve_primary_role()
        self._last_tool_output: Dict[str, Any] | None = None

    @staticmethod
    def _build_default_multi_config(config: OpenAIAgentConfig | None) -> OpenAIMultiAgentConfig:
        supervisor_config = config or OpenAIAgentConfig()
        reviewer_config = OpenAIAgentConfig(
            model=supervisor_config.model,
            name="GL Reconciliation Reviewer",
            instructions=(
                "You review reconciliation outputs and craft a concise executive summary. "
                "Highlight key variances, unresolved items, and recommended next steps."
            ),
        )
        return OpenAIMultiAgentConfig(
            agents=[
                OrchestratedAgent(
                    role="supervisor",
                    config=supervisor_config,
                    uses_reconciliation_tool=True,
                ),
                OrchestratedAgent(
                    role="reviewer",
                    config=reviewer_config,
                    uses_reconciliation_tool=False,
                ),
            ],
        )

    def _normalize_config(
        self,
        config: OpenAIAgentConfig | OpenAIMultiAgentConfig | None,
    ) -> OpenAIMultiAgentConfig:
        if isinstance(config, OpenAIMultiAgentConfig):
            if not config.agents:
                raise ValueError("OpenAIMultiAgentConfig must define at least one agent.")
            return config
        return self._build_default_multi_config(config)

    def _ensure_agents(self) -> Dict[str, str]:
        agent_ids: Dict[str, str] = {}
        for orchestrated in self._multi_config.agents:
            cfg = orchestrated.config
            if cfg.agent_id:
                agent_ids[orchestrated.role] = cfg.agent_id
                continue
            tools = [self._tool_spec] if orchestrated.uses_reconciliation_tool else []
            created = self.client.agents.create(
                model=cfg.model,
                name=cfg.name,
                instructions=cfg.instructions,
                tools=tools,
            )
            agent_ids[orchestrated.role] = created.id
        return agent_ids

    def run(self, user_prompt: str, tool_payload: Optional[Dict[str, Any]] = None) -> AgentRunOutput:
        self._last_tool_output = None
        messages_by_role: Dict[str, str] = {}

        primary_message = self._run_agent_for_role(
            role=self._primary_role,
            prompt=user_prompt,
            metadata=tool_payload or {},
            allow_tool=True,
        )
        messages_by_role[self._primary_role] = primary_message
        tool_output = self._last_tool_output

        follow_up_context = self._build_follow_up_context(
            user_prompt=user_prompt,
            supervisor_message=primary_message,
            tool_output=tool_output,
        )

        for orchestrated in self._multi_config.agents:
            if orchestrated.role == self._primary_role:
                continue
            prompt = follow_up_context
            follow_message = self._run_agent_for_role(
                role=orchestrated.role,
                prompt=prompt,
                allow_tool=orchestrated.uses_reconciliation_tool,
            )
            messages_by_role[orchestrated.role] = follow_message

        return AgentRunOutput(
            message=primary_message,
            tool_output=tool_output,
            messages_by_role=messages_by_role,
        )

    def _run_agent_for_role(
        self,
        role: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        allow_tool: bool = False,
    ) -> str:
        agent_id = self._agent_ids[role]
        thread = self.client.threads.create()
        self.client.threads.messages.create(thread_id=thread.id, role="user", content=prompt)
        run_kwargs: Dict[str, Any] = {
            "thread_id": thread.id,
            "agent_id": agent_id,
        }
        if metadata:
            run_kwargs["override"] = {"run": {"metadata": metadata}}
        run = self.client.threads.runs.create(**run_kwargs)
        return self._poll_thread(thread.id, run.id, allow_tool=allow_tool)

    def _poll_thread(self, thread_id: str, run_id: str, allow_tool: bool) -> str:
        while True:
            run = self.client.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.status == "completed":
                messages = self.client.threads.messages.list(thread_id=thread_id)
                for message in reversed(messages.data):
                    if message.role == "assistant":
                        return "\n".join(part.text.value for part in message.content if part.type == "text")
                return "Agent run completed with no assistant message."
            if run.status == "requires_action":
                if not allow_tool:
                    raise RuntimeError("Agent requested tool outputs but tools are disabled for this role.")
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
                self._last_tool_output = result
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

    @staticmethod
    def _build_follow_up_context(
        user_prompt: str,
        supervisor_message: str,
        tool_output: Optional[Dict[str, Any]],
    ) -> str:
        sections = [
            "You are reviewing the results of a general ledger reconciliation workflow.",
            f"Original user request:\n{user_prompt}",
        ]
        sections.append(f"Supervisor response:\n{supervisor_message}")
        if tool_output is not None:
            formatted_tool = json.dumps(tool_output, indent=2, default=str)
            sections.append("Reconciliation tool output (JSON):\n" + formatted_tool)
        sections.append(
            "Provide your analysis based on this context. Reference key figures and suggest actionable next steps."
        )
        return "\n\n".join(sections)
