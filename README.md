# Recon Agent

Agentic toolkit for automating GL reconciliations and roll forward schedules.

## Features
- Data models for GL balances, subledger activity, and roll forward output.
- Modular reconciliation engine to flag material variances.
- Roll forward builder to assemble period-over-period schedules.
- Agent orchestrator for coordinating reconciliation, scheduling, and insights.
- CSV export helpers and a Streamlit UI for downstream workflows.
- OpenAI Agents SDK orchestration with Gemini-powered narrative commentary.

## Setup (uv)
1. Install uv: `pip install uv` (skip if already installed).
2. Copy `.env.sample` to `.env` and populate your API keys.
3. Sync dependencies and editable package: `uv sync`.
4. Run the CLI demo: `uv run python main.py`.

Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) to enable Gemini insights. Set `OPENAI_API_KEY` to use the OpenAI Agents SDK orchestrator.

## Environment Variables
You can manage secrets locally by copying the sample env file:
```powershell
Copy-Item .env.sample .env
```
Fill in the following values:
- `OPENAI_API_KEY` – required for the OpenAI Agents SDK.
- `GEMINI_API_KEY` or `GOOGLE_API_KEY` – required for Gemini commentary.

## Streamlit UI (MVP)
1. Launch the app: `uv run streamlit run app.py`.
2. Upload CSVs with the following headers:
   - GL balances: `account`, `period` (YYYY-MM), `amount`.
   - Subledger balances: `account`, `period`, `amount`.
   - Transactions (optional): `account`, `booked_at` (date), plus `amount` or `debit`/`credit`, optional `period`.
3. Adjust the materiality slider and click **Run reconciliation** to view results and export CSV outputs.

## OpenAI Agents Orchestration
1. Export credentials (PowerShell example):
   ```powershell
   Copy-Item .env.sample .env  # then open .env and edit values
   . .\.venv\Scripts\Activate.ps1  # optional if you want manual activation
   $env:OPENAI_API_KEY = "sk-..."
   $env:GEMINI_API_KEY = "..."  # or GOOGLE_API_KEY
   ```
2. Instantiate the orchestrator in Python:
   ```python
   from recon_agent import (
       AgentConfig,
       GeminiConfig,
       GeminiInsightGenerator,
       GeminiLLM,
       OpenAIAgentConfig,
       OpenAIAgentOrchestrator,
       ReconciliationAgent,
   )

   reconciliation = ReconciliationAgent(
       config=AgentConfig(materiality_threshold=10),
       insight_generator=GeminiInsightGenerator(GeminiLLM(GeminiConfig())),
   )
   orchestrator = OpenAIAgentOrchestrator(reconciliation, OpenAIAgentConfig())
   reply = orchestrator.run("Reconcile account 1000 with the provided data", tool_payload={
       "gl_balances": [...],
       "subledger_balances": [...],
       "transactions": [...],
   })
   print(reply)
   ```
3. Provide JSON payloads that match the tool schema (see `src/recon_agent/openai_agent.py`).

## Quickstart
1. Swap the sample data in `main.py` with your GL and subledger extracts.
2. Adapt or plug in data sources from `src/recon_agent/datasources.py`.
3. Export results with `src/recon_agent/output.py` or extend with your workflow tooling.

## Next Steps
- Integrate real data loaders (SQL, APIs, spreadsheets).
- Add task queue or workflow scheduling for month-end automation.
- Plug in an LLM-powered insight generator for narrative commentary.
- Capture approvals and reconciliation status in a datastore.

