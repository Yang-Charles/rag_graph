
# RAG Graph — 多模态检索与知识图谱脚手架

> 轻量级原型，支持文本/图像/知识图谱多检索，核心依赖：Milvus、NetworkX/Neo4j、FastAPI、Vue

---

## 1. 项目简介
- 快速搭建「文本+图像+知识图谱」RAG 原型
- 支持本地开发与生产扩展，代码量极简
- 检索方式可选：全文（BM25）、语义、图像、知识图谱、融合

---

## 2. 系统架构

| 组件      | 作用                       | 可替换方案               |
|-----------|----------------------------|--------------------------|
| Milvus    | 全文/语义/图像向量检索     | Elasticsearch, PgVector  |
| NetworkX  | 内存知识图谱               | Neo4j（已支持）          |
| FastAPI   | 检索 API                   | Flask, Spring            |
| Vue       | 检索对比前端               | React, Streamlit         |

---

## 3. 快速开始

### 3.1 环境准备
- Python ≥ 3.10
- Docker（Milvus 必需）

### 3.2 安装依赖
```bash
python -m pip install -r requirements.txt
```

### 3.3 启动 Milvus（Docker Compose）
```bash
# 下载官方 Compose 文件
wget https://github.com/milvus-io/milvus/releases/download/v2.6.8/milvus-standalone-docker-compose.yml -O docker-compose.yml
# 启动
docker compose up -d
# 检查容器是否启动并运行
docker compose ps
# Milvus WebUI:
 http://127.0.0.1:9091/webui/
```

### 3.4 启动后端
```bash
# Windows
$env:PYTHONPATH='.'; uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
# macOS / Linux
PYTHONPATH=. uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3.5 启动前端
- 直接用浏览器打开 frontend/index.html
- 或用任意静态文件服务器托管 frontend/

### 3.6 （可选）加载示例数据
```bash
python scripts/load_sample_data.py
```

---

## 4. API 说明

POST /api/search

| 字段      | 类型     | 必填 | 说明                               |
|-----------|----------|------|------------------------------------|
| query     | string   | ✓    | 检索文本                           |
| image     | file     | ×    | 图像文件（可选）                   |
| methods   | csv      | ×    | 检索方式：fulltext, semantic, image, kg, fused（可多选） |
| topk      | int      | ×    | 返回条数，默认 5                   |

**返回示例**
```json
{
  "fulltext": [...],
  "semantic": [...],
  "kg": [...],
  "fused": [...] // RRF 融合，仅 methods 含 fused 时返回
}
```

---

## 5. 知识图谱（KG）

| 模式       | 优点        | 启用方式                                                            |
| -------- | --------- | --------------------------------------------------------------- |
| NetworkX | 零配置、秒启动   | 默认                                                              |
| Neo4j    | 事务、可视化、生产 | 安装 `neo4j` 驱动后调用 `KGService.connect_neo4j(uri, user, password)` |

## 6. 开发进阶

| 文件             | 说明                                    |
|------------------|-----------------------------------------|
| search.py        | 各检索方法独立调用                      |
| search_v2.py     | 单次 multi_vector_search 聚合多向量      |
| kg_service.py    | 支持 NetworkX/Neo4j 双后端              |
| frontend/        | 支持动态勾选 methods & topk，后端按需执行 |

---

## 7. 常用命令速查

```bash
# 安装依赖
python -m pip install -r requirements.txt
# 启动 Milvus
docker compose up -d
# 启动后端
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
# 加载示例数据
python scripts/load_sample_data.py
```

---

## 9. 贡献与生产化
Issue / PR 欢迎
生产部署请直接参考
– Milvus 官方文档
– Neo4j 官方文档
