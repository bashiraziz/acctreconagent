# Recon Agent

Agentic toolkit for automating GL reconciliations and roll forward schedules.

## Features
- Data models for GL balances, subledger activity, and roll forward output.
- Modular reconciliation engine to flag material variances.
- Roll forward builder to assemble period-over-period schedules.
- Agent orchestrator for coordinating reconciliation, scheduling, and insights.
- CSV export helpers and a Streamlit UI for downstream workflows.
- Flexible column alias mapping to accommodate varied export headers.
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
2. Upload CSV or Excel files with the following headers:
   - GL balances: `account`, `period` (YYYY-MM), `amount`.
   - Subledger balances: `account`, `period`, `amount`.
   - Transactions (optional): `account`, `posting_date` (date), plus `amount` or `debit`/`credit`, optional `period`.
3. Adjust the materiality slider and click **Run reconciliation** to view results and export CSV outputs.
4. Use the **Column aliases** expander to override headers with JSON or plain lines like `Balances account column = GL Account, Account Code` or `Transactions posting date = Posting Date`.
5. Each upload supports CSV/XLSX files up to 20 MB (enforced via `.streamlit/config.toml`).
6. Optional uploads:
   - **Detailed transactions file** - single activity extract (GL or subledger) containing account, posting date, amount, and description so the app can build period activity automatically. See `data/demo/detailed_transactions.csv` for a template.
   - **GL detail file** - full general-ledger journal detail with document numbers, sources, etc.; used to trace GL-side variances. Template: `data/demo/gl_detail.csv`.
   - **Subledger detail file** - supporting subledger activity export (AP, AR, inventory, etc.) for variance drill-down. Template: `data/demo/subledger_detail.csv`.

## OpenAI Agents Orchestration
This package exposes the reconciliation engine as a tool that OpenAI Agents can call. The helper in `src/recon_agent/openai_agent.py` stands up a small, multi-role agent team (supervisor + reviewer by default), feeds the reconciliation results back to OpenAI, and returns a structured response for you to consume.

1. **Set credentials**  
   ```powershell
   Copy-Item .env.sample .env  # edit OPENAI_API_KEY, GEMINI_API_KEY / GOOGLE_API_KEY
   . .\.venv\Scripts\Activate.ps1  # optional manual activation
   $env:OPENAI_API_KEY = "sk-..."
   $env:GEMINI_API_KEY = "..."      # or GOOGLE_API_KEY
   ```

2. **Instantiate the orchestrator**  
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

   orchestrator = OpenAIAgentOrchestrator(reconciliation)  # default supervisor + reviewer roles
   ```

   Pass a custom `OpenAIMultiAgentConfig` if you need to rename roles, change models, or reuse pre-created agent IDs.

3. **Build the payload**  
   Provide JSON that matches the tool schema. Required arrays use the fields:
   - `gl_balances`: each item needs `account`, optional `period` (YYYY-MM), `amount`.
   - `subledger_balances`: same shape as `gl_balances`.
   - `transactions` (optional): `account`, `booked_at` (ISO date), plus either `amount` or the `debit`/`credit` pair. Additional metadata is passed through untouched.
   - Optional roll-forward hints: `ordered_periods`, `activity_by_period`, `adjustments_by_period`.

4. **Run the agents**  
   ```python
   reply = orchestrator.run(
       "Reconcile account 1000 for Q4",
       tool_payload={
           "gl_balances": [{"account": "1000", "period": "2024-10", "amount": 120000}],
           "subledger_balances": [{"account": "1000", "period": "2024-10", "amount": 118500}],
           "transactions": [
               {
                   "account": "1000",
                   "booked_at": "2024-10-15",
                   "description": "Inventory adjustment",
                   "amount": -1500,
               }
           ],
       },
   )
   ```
   The supervisor agent invokes the reconciliation tool with your payload; the reviewer sees the supervisor’s notes plus the JSON output and drafts an executive summary.

5. **Consume the results**  
   `reply` is an `AgentRunOutput` dataclass with three pieces of information:
   - `message`: the supervisor’s final answer.
   - `messages_by_role`: every agent’s message, keyed by role (e.g., `"reviewer"`).
   - `tool_output`: raw reconciliation data—variance tables, roll-forward lines, unresolved transactions, and generated insights ready for downstream reporting.

Tips:
- Call `OpenAIAgentConfig(api_key="...")` or `OpenAIMultiAgentConfig(api_key="...")` if you do not want to rely on environment variables.
- Set `uses_reconciliation_tool=False` on a role if it should never trigger the reconciliation run (e.g., commentary-only agents).
- Attach the returned `tool_output` to your audit records or feed it into `src/recon_agent/output.py` for exports.
- Reference the full walkthrough in `docs/openai_agents.md` and the hands-on notebook at `notebooks/openai_agents_demo.ipynb`.

## Quickstart
1. Swap the sample data in `main.py` with your GL and subledger extracts.
2. Adapt or plug in data sources from `src/recon_agent/datasources.py` (use `DataSourceConfig.column_aliases` to map your header names).
3. Export results with `src/recon_agent/output.py` or extend with your workflow tooling.

## Next Steps
- Integrate real data loaders (SQL, APIs, spreadsheets).
- Add task queue or workflow scheduling for month-end automation.
- Plug in an LLM-powered insight generator for narrative commentary.
- Capture approvals and reconciliation status in a datastore.

