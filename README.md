# Retail Insights Assistant
### GenAI-Powered Multi-Agent Retail Analytics System

---

## Overview

A production-grade, multi-agent AI assistant that analyzes retail sales data, generates
automated business insights, and answers ad-hoc analytical questions in natural language.

Built with **LangGraph**, **Groq (llama)**, **FAISS**, **HuggingFace Embeddings** and **Streamlit**.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              LangGraph DAG                  │
│                                             │
│  [Agent 1] Query Resolver                   │
│   ├── FAISS metadata routing                │
│   └── Groq LLM intent + filter extraction  │
│                    │                        │
│  [Agent 2] Data Extractor                   │
│   ├── Load routed CSV(s)                    │
│   └── Pandas filter + stats                │
│                    │                        │
│  [Agent 3] Validator                        │
│   ├── Groq LLM answer generation           │
│   └── Response validation + fallback       │
└─────────────────────────────────────────────┘
    │
    ▼
Final Response → Streamlit UI
```

---

## Project Structure

```
retail_insights_assistant/
├── config/
│   ├── settings.py          # Pydantic env config loader
│   └── prompts.py           # All LLM prompt templates
├── data/
│   └── raw/                 # Place CSV files here
├── ingestion/
│   ├── loader.py            # CSV loader + metadata generator
│   └── embedder.py          # HuggingFace embedding pipeline
├── vectorstore/
│   └── faiss_store.py       # FAISS index build/load/query
├── agents/
│   ├── query_resolver.py    # Agent 1: intent + FAISS routing
│   ├── data_extractor.py    # Agent 2: Pandas data extraction
│   └── validator.py         # Agent 3: LLM answer + validation
├── graph/
│   └── dag.py               # LangGraph DAG orchestration
├── memory/
│   └── conversation.py      # Multi-turn chat memory
├── summarizer/
│   └── summarize.py         # Summarization mode
├── ui/
│   └── app.py               # Streamlit UI
├── tests/
│   └── test_pipeline.py     # Smoke tests
├── logs/                    # Auto-created at runtime
├── vectorstore/faiss_index/ # Auto-created at runtime
├── .env                     # Secret keys (never commit)
├── .env.example             # Safe template for repo
├── requirements.txt
└── README.md
```

---

## Supported Datasets

| File | Description |
|---|---|
| `Amazon Sale Report.csv` | Orders, categories, quantities, amounts, shipping |
| `Cloud Warehouse Compersion Chart.csv` | Shiprocket vs INCREFF comparison |
| `Expense IIGF.csv` | Expense and income tracking |
| `International Sale Report.csv` | International orders, customers, gross amounts |
| `May-2022.csv` | Platform-wise MRP pricing (May 2022) |
| `P L March 2021.csv` | Profit & Loss with platform MRPs |
| `Sale Report.csv` | SKU inventory with stock, size, color |

---

## Setup

### 1. Clone / unzip the project

```bash
cd retail_insights_assistant
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### 5. Add datasets

```
Place all CSV files inside:  data/raw/
```

### 6. Create required directories

```bash
mkdir -p data/raw logs vectorstore/faiss_index
```

---

## Running the App

### Streamlit UI (recommended)

```bash
streamlit run ui/app.py
```

Opens at: `http://localhost:8501`

### Run tests

```bash
pytest tests/test_pipeline.py -v
```

---

## Usage

### Chat Q&A Mode
### Summarize Mode


---

## Key Design Decisions

### Why metadata-only FAISS indexing?
Row-level embedding of large CSVs is slow, memory-heavy, and imprecise for structured
tabular data. FAISS indexes only 7 metadata descriptors for fast routing, then Pandas
handles deterministic structured queries. Best of both worlds.

### Why LangGraph over plain LangChain?
LangGraph enforces a typed, acyclic DAG with explicit state. Every node has a defined
input/output contract (GraphState). This prevents hidden state bugs and makes the
pipeline fully debuggable and auditable — critical for production.

### Why Groq?
Groq inference is significantly faster than OpenAI for the same LLaMA3 model tier.
For a data Q&A assistant where latency matters (interactive UI), Groq provides
near-real-time response at lower cost.

### Why HuggingFace `all-MiniLM-L6-v2`?
Lightweight (22M params), runs on CPU, produces 384-dim vectors. Sufficient for
semantic routing over 7 metadata documents. No GPU or API cost required.

### Why Pandas over PySpark?
Dataset sizes in this assignment are moderate (thousands of rows). Pandas is simpler,
faster to set up, and sufficient. PySpark would be introduced at 100GB+ scale
(see Scaling section).

---

## Scaling to 100GB+

| Layer | Current | At 100GB Scale |
|---|---|---|
| Ingestion | Pandas | PySpark / Databricks / Dask |
| Storage | Local CSV | AWS S3 / Azure Data Lake / GCS |
| Query Layer | Pandas | DuckDB / BigQuery / Snowflake |
| Vector Index | FAISS (local) | Pinecone / Weaviate / pgvector |
| LLM Orchestration | Direct Groq | LangChain + prompt caching + batching |
| Serving | Streamlit | FastAPI + async workers |
| Monitoring | Loguru + LangSmith | Datadog / Grafana + LangSmith |

**Strategy:**
1. Ingest raw data into a Data Lake (S3/GCS) as Parquet
2. Use DuckDB or BigQuery as analytical query layer
3. Extract statistical aggregates per partition → embed as metadata chunks in Pinecone
4. LLM receives aggregate context (not raw rows) — keeps token usage bounded
5. Prompt caching for repeated query patterns reduces cost significantly

---

## Assumptions

- CSV filenames match exactly as listed in `CSV_REGISTRY` inside `ingestion/loader.py`
- All CSV files are placed in `data/raw/` before startup
- Internet access is available for Groq API and HuggingFace model download on first run
- HuggingFace embedding model is downloaded to local cache on first run (~90MB)

---

## Limitations

- FAISS index routing depends on quality of metadata descriptions — ambiguous queries
  may route to a suboptimal CSV
- Pandas aggregations are limited to in-memory operations (not suitable for 100GB+)
- No persistent user session storage — conversation memory resets on Streamlit refresh
- LLM responses are non-deterministic (temperature=0 reduces but does not eliminate this)

---

## Possible Improvements

- Add DuckDB as a structured query engine for complex aggregations
- Persist conversation memory to Redis or SQLite for cross-session continuity
- Add LLM confidence scoring to trigger automatic fallback
- Introduce streaming LLM responses in Streamlit (`st.write_stream`)
- Add authentication layer for multi-user deployments
- Replace FAISS with Pinecone for cloud-native, persistent vector storage
- Add unit tests for each individual agent node

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — LLaMA3-70B-8192 |
| Orchestration | LangGraph 0.4.x |
| Chaining | LangChain 0.3.x |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS (CPU) |
| Data Layer | Pandas 2.x |
| UI | Streamlit 1.x |
| Observability | LangSmith + Loguru |
| Config | Pydantic Settings |
| Retry | Tenacity |

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key |
| `GROQ_MODEL_NAME` | Model name (default: `llama3-70b-8192`) |
| `HF_TOKEN` | HuggingFace token for embeddings |
| `LANGCHAIN_API_KEY` | LangSmith observability key |
| `LANGCHAIN_PROJECT` | LangSmith project name |
| `DATA_DIR` | Path to CSV files (default: `data/raw`) |
| `FAISS_INDEX_PATH` | FAISS persistence path |
| `LOG_LEVEL` | Logging level (default: `INFO`) |
| `APP_ENV` | `development` or `production` |
