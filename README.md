# 📦 Inventory Intelligence System

> **An end-to-end Agentic AI + RAG system for supply chain management**
> Built on real Uninox Houseware data — demand forecasting, reorder alerts, supplier intelligence, and anomaly detection through a natural language interface.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7)](https://render.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Live Demo** → `https://inventory-intelligence-ui.onrender.com`
**API Docs** → `https://inventory-intelligence-api.onrender.com/docs`

---

## The Problem

Inventory mismanagement costs Indian manufacturers 25–30% of working capital in either excess stock or stockout losses. At Aashi Enterprises (Uninox Houseware), I observed this firsthand — reorder decisions were made on gut feel, supplier selection was tribal knowledge locked in people's heads, and demand spikes were only noticed after stockouts had already happened.

This project is my answer: a production-ready intelligent inventory system that any warehouse manager can query in plain language to get actionable, data-backed answers.

---

## What It Can Do

| Query | What happens |
|---|---|
| *"Forecast demand for TBP-001 next 3 months"* | XGBoost predicts with 90% confidence intervals |
| *"Which SKUs need reordering right now?"* | Scans all 23 SKUs against computed ROP, returns prioritised alerts |
| *"Who is the best supplier for RSH-001?"* | RAG retrieves catalog terms, ranks by lead time and on-time rate |
| *"Any unusual demand patterns this quarter?"* | Isolation Forest flags outliers with root-cause context from PO history |

---

## Architecture

```
DATA SOURCES (6 CSVs — 512 demand records, 224 POs, 12 suppliers)
         |
         +---> RAG Pipeline                    ML Pipeline
               PDF Generation                 Feature Engineering
               (207 PDFs via Faker+ReportLab) (lag-1/3/6, rolling stats)
               Parse -> Chunk -> Deduplicate   XGBoost Demand Forecast
               ChromaDB (700+ chunks)          Isolation Forest Anomaly
                         |                              |
                         +----------+  +----------------+
                                    |  |
                          LangGraph Orchestrator
                          router.py | graph.py | tools.py
                                    |
                     +--------------+--------------+
                     |         |         |         |
                  Demand  Reorder  Supplier  Anomaly
                  Agent   Agent    Agent     Agent
                                    |
                             FastAPI (Render)
                          /query | /alerts | /forecast
                                    |
                          Streamlit Dashboard (Render)
                    Chat UI | Forecast Charts | Stock Table
```

---

## Tech Stack

| Layer | Technology | Why This Choice |
|---|---|---|
| Agentic AI | LangGraph 0.2 | StateGraph with conditional routing — not a linear chain |
| LLM | Groq + Llama 3.3 70B | Free tier, 6K tokens/min, fastest inference available |
| RAG | ChromaDB + all-MiniLM-L6-v2 | Local persistence, no API key, 22MB embedding model |
| Demand Forecasting | XGBoost + ARIMA | XGBoost for multi-feature; ARIMA for univariate baseline |
| Anomaly Detection | Isolation Forest | Unsupervised, handles unlabelled inventory data |
| Experiment Tracking | MLflow | Local tracking, no hosted service needed |
| API | FastAPI + Uvicorn | Async, auto OpenAPI docs, Pydantic v2 validation |
| Frontend | Streamlit + Plotly | Fastest path from API to interactive dashboard |
| Data Synthesis | Faker + ReportLab + SDV | Generates realistic PDFs and correlated tabular data |
| Deployment | Render free tier | Docker-based, CI/CD from GitHub, zero cost |

---

## Project by the Numbers

| Metric | Value |
|---|---|
| Python files | 42 |
| Lines of code | 3,503 |
| SKUs tracked | 23 (16 finished goods + 7 raw materials) |
| Demand records | 512 (Sep 2022 – Apr 2025) |
| Purchase orders | 224 |
| Suppliers | 12 |
| Generated PDF documents | 207 (200 POs + 7 catalogs) |
| ChromaDB chunks | 700+ |
| LangGraph nodes | 6 |
| LangChain @tools | 6 |
| API endpoints | 6 |
| Jupyter notebooks | 3 |
| Pytest test cases | 22 |
| Monthly deployment cost | Rs 0 |

---

## Project Structure

```
inventory-intelligence/
├── config.py                     Central settings — imported everywhere
├── requirements.txt              Local dev (full stack)
├── render.yaml                   Both services configured
├── .streamlit/config.toml        Streamlit production settings
├── data/
│   ├── raw/                      6 Uninox Houseware source CSVs
│   ├── synthetic/                207 generated PDFs (local only)
│   └── processed/                Committed to repo for Render
│       ├── chroma_store/         Pre-built vector embeddings (~8 MB)
│       ├── models/               Trained XGBoost + IsoForest (~5 MB)
│       └── features.parquet      Pre-computed feature matrix
├── data_generation/              Faker + ReportLab + SDV scripts
├── ingestion/                    PDF parser > chunker > deduplicator
├── knowledge/
│   ├── vector_store/             ChromaDB embedder + retriever
│   └── feature_store/            Feature engineering + ML models + MLflow
├── agents/                       4 LangGraph agent nodes
├── orchestrator/                 StateGraph + router + tools + LLM factory
├── api/                          FastAPI: main, schemas, 3 route files
├── frontend/                     Streamlit dashboard + requirements
├── scripts/                      setup.py, ingest_docs.py, train_models.py
├── tests/                        22 pytest tests
└── notebooks/                    EDA, forecasting, anomaly detection
```

---

## Quick Start

```bash
git clone https://github.com/Rishu0200/Inventory-Intelligence.git
cd inventory-intelligence
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Add GROQ_API_KEY from console.groq.com
python scripts/setup.py     # ~5 min: generates PDFs, embeds, trains models
uvicorn api.main:app --reload --port 8000   # Terminal 1
streamlit run frontend/app.py               # Terminal 2
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/ping` | Health — model status + ChromaDB chunk count |
| POST | `/api/query` | Natural language query |
| POST | `/api/query/stream` | Streaming SSE version |
| GET | `/api/alerts` | All reorder + anomaly alerts |
| GET | `/api/forecast/{sku_id}` | Demand forecast + confidence intervals |
| GET | `/api/forecast` | Forecast for all SKUs |

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | groq / ollama / openai / gemini |
| `GROQ_API_KEY` | — | Free from console.groq.com |
| `DEMO_MODE` | `false` | true = rule-based, no LLM calls |
| `API_BASE_URL` | `http://localhost:8000` | Set to Render URL for frontend |

---

## Challenges and Solutions

**1. No real supply chain data available**
Supply chain PO documents and demand history are proprietary. Built a complete synthetic data pipeline using Faker for realistic Indian supplier details (GSTIN, city, contact names), ReportLab for structured PDF generation, and SDV GaussianCopulaSynthesizer for statistically correlated tabular data that preserves demand-stockout correlation and lead-time distributions.

**2. RAM exceeded Render free tier (512 MB)**
Full requirements install sdv which pulls PyTorch (2.1 GB disk, ~300 MB RAM). Total estimated runtime RAM was ~650 MB. Solution: three requirements files — requirements.txt for local dev, requirements-api.txt for Render API (16 lean packages, ~420 MB), and frontend/requirements.txt for Streamlit. Each environment gets exactly what it needs.

**3. ChromaDB wiped on every Render redeploy**
Render free tier uses ephemeral disk. Solution: run ingestion locally once, commit the pre-built chroma_store (~8 MB) and trained model artifacts to the GitHub repo. Updated .gitignore to not ignore these. Render deploys from the committed knowledge base on every push.

**4. mlflow ImportError crashing production API**
import mlflow at the top of demand_model.py caused ImportError on Render because mlflow is not in requirements-api.txt. Solution: moved mlflow to a lazy import inside train_and_save() only, wrapped in try/except. The production API never calls train_and_save() — only forecast_sku() and load_model() which are mlflow-free.

**5. Render cold start killing demo experience**
Free tier spins down after 15 minutes. Solution: UptimeRobot pings /ping every 14 minutes for the API and /healthz for the frontend. Background thread model loading ensures health check responds instantly while models load in parallel.

**6. LLM cost for a portfolio project**
OpenAI charges per token. Multiple agent calls per query make it expensive for a free demo. Solution: llm_factory.py supports Groq (free: 30 req/min, 6K tokens/min), Ollama (local), OpenAI, and Gemini. Default is Groq llama-3.3-70b-versatile — faster than GPT-3.5 and completely free.

**7. Intent routing failing on multi-intent queries**
Pure keyword matching breaks on conversational queries like "I want to order RSH-001, who is the cheapest supplier?" Solution: two-tier routing — keyword scoring first (fast, free, handles ~85% of queries), LLM fallback only for ambiguous cases.

---

## Known Limitations

**Data** — Synthetic demand patterns lack real-world complexity: Diwali spikes, GST quarter-end effects, and promotional events are not captured. A production deployment needs actual ERP data export.

**Static knowledge base** — Adding new documents requires local re-ingestion and a redeploy. No real-time /upload endpoint exists.

**No API authentication** — The REST API is publicly accessible. A production system needs JWT tokens or API key middleware.

**Single worker** — Render free tier runs one worker (0.1 vCPU). High concurrent traffic causes timeouts.

**Fixed anomaly contamination** — Isolation Forest contamination=0.05 was set empirically. Production would need dynamic tuning based on observed anomaly rates.

**ARIMA for simple seasonality** — ARIMA(1,1,1) baseline does not capture multi-seasonal patterns. Facebook Prophet or hierarchical time-series models would improve accuracy.

**ChromaDB committed to repo** — Works for 207 documents (~8 MB). Does not scale to thousands of documents — a hosted vector DB would be needed.

**No feedback loop** — Cannot learn from whether recommendations were acted upon. A production system would log query → action → outcome for continuous improvement.

---

## Running Tests

```bash
pytest tests/ -v    # 22 tests, ~4s
```

---

## About the Author

**Rishabh Anand** — Data Scientist with real supply chain domain expertise. Former Supply Chain and Operations Executive , managing supplier payments, raw material procurement, and international export documentation (Commercial Invoices, LC, SAFTA certificates). GATE 2023 qualified (90th percentile, ECE). This project bridges two years of hands-on supply chain experience with applied ML and agentic AI.

Email: Rishabhanand0200@gmail.com

---

## License

MIT — free to use, fork, and build upon.
