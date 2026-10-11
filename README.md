# Inventory Intelligence System

An agentic AI + RAG assistant for supply chain inventory management. Ask in plain language
for demand forecasts, reorder alerts, supplier recommendations, or demand anomalies; a
LangGraph router sends the question to a specialist agent that combines classical ML models,
SQL lookups, and document retrieval.

Modeled on the operations of an Indian kitchen-hardware manufacturer (Uninox Houseware).
The datasets are small and illustrative, so treat every figure the app returns as sample data.

**API docs:** `https://inventory-intelligence.onrender.com/docs`
**Live app:** `https://inventory-intelligence-app.streamlit.app/` 

Both run on free tiers, so the first request after a quiet period can take about a minute
while the services wake up.

---

## What you can ask

| Question | What happens |
|---|---|
| "Forecast demand for TBP-001 next 3 months" | XGBoost forecast with 90% intervals |
| "Which SKUs need reordering?" | Compares stock against a reorder point computed from demand and supplier lead time |
| "Who is the best supplier for RSH-001?" | Looks up supplier terms in Postgres, ranks suppliers by lead time and on-time rate, and adds catalog context from the vector store |
| "Any unusual demand this quarter?" | Isolation Forest flags outlier months |

---

## Results

All numbers were measured on this project's data. The sample sizes are small, and each note says what it does and doesn't show.

| Area | Result | Note |
|---|---|---|
| Demand forecast | Mean RMSE **138.7** vs **180.6** for a last-month baseline (**23% lower**) | 3-fold chronological cross-validation, 416 rows, 16 SKUs. The model beat the baseline in every fold (184.9/114.8/116.3 vs 192.0/149.8/200.0). The margin is much smaller in the first fold. |
| Reorder logic | Scan-all went from 0 to **2 of 23** SKUs flagged | Both code paths now share one reorder-point calculation. Before, "scan all" used a static table column while single-SKU lookups computed it. |
| Intent routing | 100% intent and 100% SKU-extraction accuracy | 12 labeled queries |
| Retrieval | 57% hit-rate (4 of 7 checkable queries) | Pure semantic search is weak at exact SKU codes. Metadata filtering is the likely fix. |
| Groundedness (LLM-as-judge) | 91.7% on the latest run | 12 queries. The judge is non-deterministic, so recent runs ranged from roughly 83% to 100%. Look at the trend, not a single run. |
| Tests | **43** automated tests, all passing | API and agent layers run against mocked services |
| Alerts latency | Median 5.1 s uncached, 0.8 s cached | Free-tier Render, 3 and 5 requests |
| Chat latency | Median 7.4 s | 8 requests, including the LLM call and retrieval |
| Memory | About 218 MB resident locally | Measured on Windows. The 512 MB free-tier limit is a design target, and Render's free tier doesn't show memory usage, so it hasn't been confirmed there. |

---

## Architecture

```
Streamlit Cloud (frontend)
        |  HTTPS + JWT bearer token
        v
Render (FastAPI, free tier, single worker)
        |
        |-- Supabase Postgres ------ SKUs, suppliers, demand, POs, users
        |-- Chroma Cloud ----------- vector store (PO + supplier-catalog chunks)
        |-- Supabase Storage ------- trained model files (S3-compatible API)
        |-- Upstash Redis (opt.) --- alert cache, rate limiting
        '-- Groq ------------------- LLM for answer synthesis and intent fallback

GitHub Actions (nightly, 02:00 UTC)
        '-- retrain models -> run eval harness -> upload models to Storage
```

Render's free tier has no persistent disk, so nothing stateful lives in the API container.
It rebuilds its working state from Postgres, Chroma Cloud, and Storage on every cold start.

### How a query flows
1. **Router** scores the question against keyword lists for four intents. Ties go to an
   explicit priority order, and the LLM classifies only when no keyword matches.
2. **Agent** (demand, reorder, supplier, anomaly, or a general-help node) calls its tools:
   forecast, stock and reorder-point check, supplier lookup, anomaly detection, and
   document retrieval.
3. **Synthesiser** has the LLM write the answer from the tool output and retrieved context,
   with an instruction not to relabel historical periods as "current". With
   `DEMO_MODE=true` it returns the tool output directly and needs no API key.

---

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq (default `openai/gpt-oss-120b`, set with `GROQ_MODEL`) |
| Vector store | Chroma Cloud with chromadb's built-in ONNX MiniLM embeddings (no PyTorch) |
| Forecasting | XGBoost (lag, rolling, and calendar features; recursive multi-step) |
| Anomaly detection | Isolation Forest (`contamination=0.05`, unsupervised) |
| Database | PostgreSQL on Supabase via SQLAlchemy 2.0 |
| Model storage | Supabase Storage via the S3 API (boto3) |
| Cache / rate limits | Upstash Redis (REST), with an in-process fallback |
| Auth | JWT (PyJWT) with Argon2id password hashing |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit (API-only, no direct database access) |
| CI | GitHub Actions |
| Python | 3.11 |

---

## Project structure

```
.
|-- config.py                      Settings (pydantic-settings)
|-- requirements-api.txt           Render: lean, inference only
|-- requirements-train.txt         GitHub Actions: adds mlflow, boto3
|-- requirements-dev.txt           Local: adds data generation, ingestion, pytest
|-- runtime.txt / .python-version  Python 3.11 pin
|
|-- api/                           FastAPI app, schemas, rate limiting, routes
|-- auth/                          Password hashing, JWT creation/verification
|-- db/                            SQLAlchemy models, session, CSV-to-Postgres migration
|-- cache/                         Redis wrapper (cache + rate limit)
|-- storage/                       S3-compatible model storage
|-- orchestrator/                  LangGraph graph, router, tools, reorder logic, LLM factory
|-- agents/                        demand / reorder / supplier / anomaly nodes
|-- knowledge/
|   |-- vector_store/              Chroma embed + retrieve (cached client, retries)
|   '-- feature_store/             Feature engineering, XGBoost, Isolation Forest
|-- ingestion/                     PDF parse -> chunk -> MinHash dedupe
|-- data_generation/               Synthetic PO and catalog PDFs, tabular extension
|-- eval/                          Intent, retrieval, and groundedness evaluation
|-- scripts/                       ingest_docs, train_models, create_user, setup
|-- frontend/                      Streamlit app and its own requirements.txt
|-- tests/                         43 tests (pytest)
'-- .github/workflows/retrain.yml  Nightly retrain + eval
```

---

## Local setup

```bash
git clone https://github.com/Rishu0200/Inventory-Intelligence.git
cd Inventory-Intelligence
python -m venv venv
venv\Scripts\activate                  # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                   # fill in values (see below)
```

**Input data.** `data/raw/*.csv` is gitignored, so a fresh clone doesn't include it. The
migration reads five files from `data/raw/`: `demand_history.csv`, `inventory_history.csv`,
`purchase_orders.csv`, `supplier_directory.csv`, `supplier_terms.csv`. Dates are
day-first (DD-MM-YYYY).

```bash
python -m db.migrate_csv_to_db            # CSVs -> Postgres (safe to re-run)
python -m data_generation.generate_pos
python -m data_generation.generate_supplier_docs
python -m scripts.ingest_docs             # PDFs -> Chroma
python scripts/train_models.py            # models -> Storage (exits non-zero on failure)
python -m scripts.create_user             # create a login

uvicorn api.main:app --reload --port 8000     # terminal 1
streamlit run frontend/app.py                 # terminal 2
```

With no cloud credentials set, the app falls back to local SQLite, an in-process cache, and
local model files, so you can run a reduced version offline. `DEMO_MODE=true` skips the
LLM entirely.

---

## Configuration

Settings come from `.env`; see `.env.example` for the full annotated list. These are the
ones that cause problems:

| Variable | Notes |
|---|---|
| `JWT_SECRET_KEY` | Required. Generate with `python -c "import secrets; print(secrets.token_hex(32))"`. If it's empty, login is refused. |
| `DATABASE_URL` | Use the Supabase **Session Pooler** string (the direct connection is IPv6-only). Make the driver explicit: `postgresql+psycopg2://...?sslmode=require`. |
| `CHROMA_CLOUD_TENANT` / `CHROMA_CLOUD_DATABASE` | The tenant is the UUID. The database is the name you chose. They're easy to swap by accident. |
| `R2_*` | Names are historical. Any S3-compatible provider works (Supabase Storage here). |
| `REDIS_URL` / `REDIS_TOKEN` | Optional. Use Upstash's REST URL and token, not the `rediss://` string. |
| `GROQ_MODEL` | Groq retires models. If you get `model_not_found`, check their deprecations page. |

If you change the embedding model, wipe and re-ingest both Chroma collections. Vectors from
different models aren't comparable, and Chroma rejects the mismatch.

---

## API

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/ping` | none | Health: models loaded, vector-store chunk count |
| POST | `/api/auth/login` | none | Form fields `username` (email) and `password`; returns a JWT. Limited to 10 attempts/min per email. |
| POST | `/api/query` | JWT | Natural-language query. Limited to 10/min per user and 25/min overall. |
| POST | `/api/query/stream` | JWT | Same, as server-sent events. It streams the finished answer word by word; it is not token streaming. |
| GET | `/api/alerts?refresh=` | JWT | Reorder and anomaly alerts, cached for 10 minutes |
| GET | `/api/forecast/{sku_id}?horizon=` | JWT | Forecast with 90% intervals (1-12 months) |
| GET | `/api/forecast?horizon=` | JWT | Forecast for every SKU (1-6 months) |
| GET | `/api/skus`, `/api/inventory`, `/api/demand/{sku_id}` | JWT | Data endpoints used by the dashboard |

Interactive docs are at `/docs`, and the **Authorize** button logs in with your account.

---

## Testing and evaluation

```bash
python -m pytest tests/ -q                # 43 tests, external services mocked

python -m eval.intent_eval                # fast, no LLM
python -m eval.retrieval_eval             # needs Chroma populated
python -m eval.run_eval                   # full harness, logs to the eval_runs table
```

The harness runs nightly in CI, so quality can be tracked over time. The groundedness judge
is given the same context the synthesiser used (tool output plus retrieved documents).
Early on it saw only the retrieved documents and reported a misleadingly low score.

---

## Deployment

**API (Render).** Build command `pip install -r requirements-api.txt`. Start command
`uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1`. Health check path `/ping`.
Set `PYTHON_VERSION=3.11.9` and add every secret from `.env.example` in the dashboard.
Check the build log for the Python version it actually used.

**Frontend (Streamlit Community Cloud).** Main file `frontend/app.py`, which reads
`frontend/requirements.txt`. The only secret is `API_BASE_URL = "https://<your-render-url>"`.

**Nightly job (GitHub Actions).** Add the same secrets as repository secrets
(`DATABASE_URL`, `R2_*`, `GROQ_API_KEY`, `CHROMA_CLOUD_*`). You can trigger it manually from
the Actions tab. A failed training stage now fails the run.

---

## Problems found and fixed

- **Date parsing:** pandas' default inference misparsed or nulled 210 of 224 day-first PO dates. The migration now forces day-first.
- **Inconsistent reorder points:** the scan-all path and the single-SKU path used different definitions. A single shared module now decides.
- **Forecast features:** multi-step forecasts advanced only the first lag, leaving calendar and rolling features frozen. They now advance every step.
- **Forecast validation:** features were stacked SKU by SKU, so cross-validation wasn't chronological. Rows are now sorted by period first.
- **Safety stock:** it didn't scale to supplier lead time. It now uses daily deviation times the square root of lead time.
- **Silent failures:** a pandas merge with overlapping column names raised a `KeyError` that a bare `except` swallowed, so supplier recommendations were always empty. They now use an explicit SQL join.
- **Session expiry:** objects were read after their session closed. The session factory now sets `expire_on_commit=False`.
- **Green CI on failure:** the training script swallowed errors. It now exits non-zero.

---

## Known limitations

- **Small, illustrative data:** 512 demand records over about 2.5 years, 16 SKUs with demand history. Real seasonality and promotions aren't captured.
- **Forecast horizon:** it counts from the last month in the data (April 2025), not from today.
- **Reorder points:** a normal-demand, 95% service-level approximation. SKUs with no demand history, such as raw materials, fall back to the static table value.
- **Retrieval:** weak on exact SKU codes (57% on the checkable queries). Structured questions are answered from SQL, and RAG supplies supporting document context.
- **Latency:** chat answers took about 7 s at the median, and cold starts take about a minute on both free tiers.
- **Not load-tested:** only sequential requests were measured, and memory on Render is unverified.
- **Auth:** one 12-hour access token with no refresh or revocation. Login limiting is keyed by email, so someone could briefly lock out a colleague's account.
- **Operations:** no admin UI (users are created with `scripts/create_user.py`) and a single API worker.
- **Evaluation:** 12 labeled queries and a non-deterministic LLM judge, so small score changes are noise.
- **Anomaly detection:** unsupervised, with no labeled anomalies to measure against.

## Roadmap

- Write thumbs up/down feedback to the `query_logs` table and use it to grow the eval set.
- Filter retrieval by SKU metadata to improve the hit-rate.
- Load-test concurrent users and check memory on Render.
- Share one model-loading path across the API modules (the model is currently loaded in three places).
- Refresh tokens and an admin page for user management.

---

## Author

Built by Rishabh Anand. [https://www.linkedin.com/in/rishabh-anand-844097211/](#) | [GitHub](https://github.com/Rishu0200)

## License

MIT