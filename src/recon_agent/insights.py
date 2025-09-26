from __future__ import annotations

from typing import List

from .llm import GeminiLLM
from .models import ReconciliationResult


class SimpleInsightGenerator:
    """Creates a human friendly summary highlighting variances."""

    def summarize(self, reconciliations: List[ReconciliationResult]) -> str:
        highlights = []
        for result in reconciliations:
            if abs(result.variance) > 0:
                highlights.append(
                    f"Account {result.account} period {result.period}: variance {result.variance:.2f}."
                )
        if not highlights:
            return "All accounts reconciled within threshold."
        return "\n".join(highlights)


class GeminiInsightGenerator:
    """Uses Gemini to generate narrative accounting commentary."""

    def __init__(
        self,
        llm: GeminiLLM,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt or (
            "You are a senior accounting analyst. Craft concise narrative commentary "
            "for GL account reconciliations, focusing on material variances and next steps."
        )

    def summarize(self, reconciliations: List[ReconciliationResult]) -> str:
        if not reconciliations:
            return "No reconciliations to summarize."
        bullet_lines = []
        for result in reconciliations:
            bullet_lines.append(
                " - Account {account} period {period}: GL {gl:.2f}, Subledger {sub:.2f}, "
                "variance {var:.2f}.".format(
                    account=result.account,
                    period=result.period,
                    gl=result.gl_balance,
                    sub=result.subledger_balance,
                    var=result.variance,
                )
            )
        prompt = f"{self.system_prompt}\n\nReconciliation data:\n" + "\n".join(bullet_lines)
        try:
            response = self.llm.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            return (
                "Gemini insight generation failed; falling back to structured highlights. "
                f"Reason: {exc}"
            )
        return response.strip() or "Gemini did not return any commentary."
