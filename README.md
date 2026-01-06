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

### Neo4j 安装部署方法

#### Docker 方式（推荐）

1. **使用 Docker Compose 安装**：
   ```bash
   # 创建项目目录并准备数据卷
   mkdir -p ~/neo4j/{data,logs,plugins,import,conf}
   cd ~/neo4j
   
   # 启动 Neo4j
   cd neo4j_server
   docker compose up -d          # 旧版本 CLI 用 docker-compose up -d
   ```

2. **访问 Neo4j**：
   - Web UI: http://localhost:7474
   - 默认用户名: `neo4j`
   - 密码: `yourStrongPassword`（首次登录后需要修改）

3. **可选增强配置**：
   - 在 `./conf/neo4j.conf` 中添加内存和JVM调优参数：
   ```
   dbms.memory.heap.initial_size=1G
   dbms.memory.heap.max_size=4G
   dbms.memory.pagecache.size=2G
   ```


#### 项目中使用 Neo4j

1. **安装依赖** （已包含在requirements.txt） ：
   ```bash
   pip install neo4j
   ```

2. **配置连接**：
   在代码中使用 `KGService` 连接到 Neo4j：
   ```python
   from backend.app.services.kg_service import KGService
   
   # 连接到 Neo4j
   kg_service = KGService(
       neo4j_uri="bolt://localhost:7687",
       user="neo4j",
       password="yourStrongPassword"
   )
   ```

3. **环境变量配置**（推荐）：
   创建 `.env` 文件配置连接信息：
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=yourStrongPassword
   ```

#### 安全配置

1. **修改默认密码** ：
   - 首次启动后必须修改默认密码
   - 使用强密码策略

2. **启用认证和授权** ：
   - 配置用户权限
   - 使用角色基础访问控制

3. **网络配置** ：
   - 限制访问端口
   - 使用防火墙规则

#### 常用管理命令

```bash
# 检查 Neo4j 状态
docker exec -it neo4j-container neo4j status

# 进入容器
docker exec -it neo4j-container bash

# 运行 Cypher 查询
docker exec -it neo4j-container cypher-shell -u neo4j -p yourStrongPassword

# 备份数据库
docker exec -it neo4j-container neo4j-admin backup --from=neo4j://localhost:7687 --to=/backups/

# 恢复数据库
docker exec -it neo4j-container neo4j-admin load --from=/path/to/backup --force
```

| 端口 | 用途 | 说明 |
|------|------|------|
| 7474 | HTTP | Web UI 和 REST API |
| 7473 | HTTPS | 加密 Web UI |
| 7687 | Bolt | 客户端连接协议 |
| 7688 | Bolt Routing | 集群路由 |
---

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
