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

    # Run milvus hybrid search in thread
    milvus_task = asyncio.to_thread(milvus.hybrid_search, query, image_bytes, 10)
    kg_task = asyncio.to_thread(kg.search_entities, query)

    milvus_res, kg_res = await asyncio.gather(milvus_task, kg_task)

    # Fuse milvus results with KG using RRF
    fused = rrf_fuse([milvus_res, kg_res])

    return {
        "milvus": milvus_res,
        "kg": kg_res,
        "fused": fused,
    }
