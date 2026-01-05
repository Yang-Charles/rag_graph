# RAG Graph — Multimodal Retrieval Scaffold

Lightweight scaffold combining multimodal retrieval and a simple knowledge graph.

Core components

- Milvus (`pymilvus`) — full-text (BM25), semantic, and image-vector retrieval
- Knowledge graph — NetworkX (default) and optional Neo4j backing
- FastAPI backend exposing search APIs
- Vue frontend for quick comparison and visualization

This README provides a concise quickstart, API usage, and notes for BM25 / KG integration.

## Quickstart (local development)

Prerequisites: Python 3.10+, Docker (for Milvus), pip.

1) Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

2) Start Milvus (recommended via Docker Compose)

- Download an official standalone compose file or create a `docker-compose.milvus.yml` and run:

```powershell
docker compose -f docker-compose.milvus.yml up -d
```

Milvus default ports: gRPC `19530`, HTTP/monitor `19121`.

3) Start the backend

```powershell
$env:PYTHONPATH='.'; uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

4) Open the frontend

- Open `frontend/index.html` in your browser (or serve the `frontend` folder as static files).

5) (Optional) Load sample data

```powershell
python scripts/load_sample_data.py
```

## API (important)

- POST `/api/search` (form)
  - `query` (string, required)
  - `image` (file, optional)
  - `methods` (csv string, optional) — subset of `fulltext,semantic,image,kg,fused`
  - `topk` (int, optional, default=5)

The backend executes only requested methods to reduce cost. Response contains requested keys, e.g. `semantic`, `fulltext`, `kg`, and optional `fused` (RRF fused results).

## Milvus & BM25 notes

- To enable BM25 full-text support, create a `VARCHAR` field with `enable_analyzer=true` and add a BM25 index on that field:

```python
# example
coll.create_index(field_name="text", index_params={"index_type": "BM25", "params": {"k1": 1.2, "b": 0.75}})
```

- Alternatively you may generate sparse BM25-weighted vectors externally and insert them into a `SPARSE_FLOAT_VECTOR` field.

See official doc: https://milvus.io/docs/zh/full-text-search.md

## Knowledge Graph (KG)

- Default: in-memory NetworkX KG for development (`backend/app/services/kg_service.py`).
- Optional: Neo4j integration — configure `KGService.connect_neo4j(uri, user, password)` and install `neo4j` driver.

Insert example (Neo4j mode):

```python
nodes = [{"id":"doc:1","labels":["Document"],"props":{"text":"示例文本"}}]
edges = [{"from":"doc:1","to":"org:1","rel":"BELONGS_TO","props":{}}]
kg.kg_insert_nodes_edges(nodes, edges)
```

## Development notes

- Two backend search flavors exist:
  - `search.py`: calls per-method search functions
  - `search_v2.py`: uses `multi_vector_search` to run multiple Milvus modalities in one call
- Frontend now sends `methods` and `topk` so the backend only runs requested retrievals.
- Heavy dependencies (`torch`, `sentence-transformers`, `pymilvus`) require time and disk space to install.

## Useful commands

- Install deps: `python -m pip install -r requirements.txt`
- Start Milvus: `docker compose -f docker-compose.milvus.yml up -d`
- Start backend: `uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000`

## Contributing

PRs and issues welcome. For production deployments consult Milvus and Neo4j official docs.

---
_Concise quickstart — consult in-code docs and upstream docs for production guidance._
# RAG Graph — Multimodal RAG with Milvus + NetworkX + FastAPI + Vue

This repository provides a scaffold for a Retrieval-Augmented Generation (RAG) system combining:
- Milvus (pymilvus==2.6) for full-text / semantic / image vector retrieval
- NetworkX for a local knowledge graph (swapable for Neo4j)
- FastAPI backend
- Vue frontend (minimal)

Quick start (after installing dependencies):

安装 Milvus
Milvus 在 Milvus 资源库中提供了 Docker Compose 配置文件。要使用 Docker Compose 安装 Milvus，只需运行

Download the configuration file
wget https://github.com/milvus-io/milvus/releases/download/v2.6.8/milvus-standalone-docker-compose.yml -O docker-compose.yml
Start Milvus
sudo docker compose up -d



1. Install Python deps:

```powershell
python -m pip install -r requirements.txt
```

2. Run the backend:

```powershell
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

3. Open `frontend/index.html` in a browser.

Notes:
- The Milvus service code is scaffolded for multi-vector search; tune index/search params per your Milvus deployment following https://milvus.io/docs/zh/multi-vector-search.md
- Use `scripts/load_sample_data.py` to load example data into Milvus and the NetworkX graph.