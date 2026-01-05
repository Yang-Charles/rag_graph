from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from backend.app.services.milvus_service_v2 import MilvusService
from backend.app.services.kg_service import KGService
from backend.app.services.reranker import rrf_fuse
import asyncio

router = APIRouter()

milvus = MilvusService()
kg = KGService()


@router.post("/search")
async def search(query: str = Form(...), image: Optional[UploadFile] = File(None)):
    image_bytes = None
    if image:
        image_bytes = await image.read()

    # Run milvus multi-vector (semantic + image + fulltext) in thread
    milvus_task = asyncio.to_thread(milvus.multi_vector_search, query, image_bytes, 10)
    kg_task = asyncio.to_thread(kg.search_entities, query)

    milvus_res, kg_res = await asyncio.gather(milvus_task, kg_task)

    # milvus_res contains keys semantic,image,fulltext
    fulltext_res = milvus_res.get('fulltext', [])
    semantic_res = milvus_res.get('semantic', [])
    image_res = milvus_res.get('image', [])

    # Fuse all lists (including KG) using RRF
    fused = rrf_fuse([fulltext_res, semantic_res, image_res, kg_res])

    return {
        "fulltext": fulltext_res,
        "semantic": semantic_res,
        "image": image_res,
        "kg": kg_res,
        "fused": fused,
    }
