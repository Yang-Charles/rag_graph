 RAG Graph — Multimodal Retrieval Scaffold

> 轻量级脚手架，结合多模态检索与简易知识图谱  
> 核心：Milvus + NetworkX + FastAPI + Vue

---

## 1. 项目定位
- 快速搭建「文本+图像+知识图谱」的 RAG 原型
- 本地开发 → 生产扩展均可复用
- 代码量最小化，官方文档链接最大化

---

## 2. 系统架构

| 组件        | 作用                             | 可替换方案               |
|-------------|----------------------------------|--------------------------|
| Milvus      | 全文(BM25)、语义、图像向量检索   | Elasticsearch、PgVector |
| NetworkX    | 内存级知识图谱                   | Neo4j（已留接口）        |
| FastAPI     | 检索 API                         | Flask、Spring            |
| Vue         | 对比式可视化前端                 | React、Streamlit         |

---

## 3. 5 分钟本地启动

### 3.1 前置条件
- Python ≥ 3.10
- Docker（仅 Milvus 需要）

### 3.2 安装 Python 依赖
```bash
python -m pip install -r requirements.txt

3.3 启动 Milvus（Docker Compose 一键）
# 下载官方 Compose 文件
wget https://github.com/milvus-io/milvus/releases/download/v2.6.8/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动
sudo docker compose up -d
Milvus-standalone容器使用默认设置为本地19530端口提供服务，并将其数据映射到当前文件夹中的volumes/milvus。

# 验证
docker compose ps
# WebUI: http://127.0.0.1:9091/webui/


3.4 启动后端
# Windows
$env:PYTHONPATH='.'; uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# macOS / Linux
PYTHONPATH=. uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000


3.5 打开前端
浏览器直接打开 frontend/index.html
或
任意静态文件服务器托管 frontend/ 目录
3.6（可选）灌入示例数据
bash
复制
python scripts/load_sample_data.py
4. API 一览
| 字段      | 类型     | 必填 | 说明                               |
| ------- | ------ | -- | -------------------------------- |
| query   | string | ✓  | 检索句子                             |
| image   | file   | ×  | 图像文件                             |
| methods | csv    | ×  | 全文 / 语义 / 图像 / 知识图谱 / fused（可组合） |
| topk    | int    | ×  | 默认 5                             |


返回示例
{
  "fulltext": [...],
  "semantic": [...],
  "kg": [...],
  "fused": [...]      // RRF 融合结果，仅当 methods 包含 fused 时返回
}


6. 知识图谱（KG）
| 模式       | 优点        | 启用方式                                                            |
| -------- | --------- | --------------------------------------------------------------- |
| NetworkX | 零配置、秒启动   | 默认                                                              |
| Neo4j    | 事务、可视化、生产 | 安装 `neo4j` 驱动后调用 `KGService.connect_neo4j(uri, user, password)` |


7. 开发进阶
| 文件             | 说明                                    |
| -------------- | ------------------------------------- |
| `search.py`    | 各方法独立调用                               |
| `search_v2.py` | 单次 `multi_vector_search` 聚合多向量，减少 RPC |
| 前端             | 已支持动态勾选 methods & topk，后端按需执行         |



8. 常用命令速查
bash
复制
# 安装依赖
python -m pip install -r requirements.txt

# 启动 Milvus（若已改文件名）
docker compose -f docker-compose.milvus.yml up -d

# 启动后端
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

# 加载示例数据
python scripts/load_sample_data.py

9. 贡献与生产化
Issue / PR 欢迎
生产部署请直接参考
– Milvus 官方文档
– Neo4j 官方文档
