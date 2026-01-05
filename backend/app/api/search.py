from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from backend.app.services.milvus_service_v2 import MilvusService
from backend.app.services.kg_service import KGService
from backend.app.services.reranker import rrf_fuse
import asyncio

router = APIRouter()

# Instantiate services (simple singletons for scaffold)
milvus = MilvusService()
kg = KGService()


@router.post("/search")
async def search(
    query: str = Form(...),
    image: Optional[UploadFile] = File(None),
    methods: Optional[str] = Form(None),
    topk: int = Form(5),
):
    """Search endpoint.

    `methods` is an optional comma-separated list of methods to run:
      fulltext,semantic,image,kg,fused
    `topk` controls how many results to request per method.
    The handler will only run requested methods to reduce cost.
    """
    image_bytes = None
    if image:
        image_bytes = await image.read()

    # parse requested methods
    if methods:
        requested = {m.strip() for m in methods.split(",") if m.strip()}
    else:
        requested = {"fulltext", "semantic", "image", "kg", "fused"}

    # Determine which milvus methods are requested
    milvus_methods = {"fulltext", "semantic", "image"} & requested

    tasks = []
    milvus_task = None
    # If more than one milvus modality requested, use multi_vector_search to reduce overhead
    if len(milvus_methods) > 1:
        milvus_task = asyncio.to_thread(milvus.multi_vector_search, query, image_bytes, topk)
        tasks.append(milvus_task)
    else:
        # call individual milvus methods only if requested
        if "fulltext" in milvus_methods:
            tasks.append(asyncio.to_thread(milvus.search_fulltext, query, topk))
        if "semantic" in milvus_methods:
            tasks.append(asyncio.to_thread(milvus.search_semantic, query, topk))
        if "image" in milvus_methods:
            tasks.append(asyncio.to_thread(milvus.search_image, image_bytes, topk))

    kg_task = None
    if "kg" in requested:
        kg_task = asyncio.to_thread(kg.search_entities, query, topk)
        tasks.append(kg_task)

    # Run only the required tasks
    results = await asyncio.gather(*tasks) if tasks else []

    # Collect results into a dict to return only requested keys
    resp = {}

    # If milvus_task used multi_vector_search, its result is a dict
    idx = 0
    if milvus_task is not None:
        milvus_res = results[idx]
        idx += 1
        for k in ("fulltext", "semantic", "image"):
            if k in requested:
                resp[k] = milvus_res.get(k, [])
    else:
        # individual milvus calls were appended in order fulltext, semantic, image
        for k in ("fulltext", "semantic", "image"):
            if k in milvus_methods:
                resp[k] = results[idx]
                idx += 1

    # KG result
    if kg_task is not None:
        resp["kg"] = results[idx] if idx < len(results) else []

    # fused (RRF) - compute only if requested
    if "fused" in requested:
        # prepare lists in fixed order expected by rrf_fuse
        fulltext_res = resp.get("fulltext", [])
        semantic_res = resp.get("semantic", [])
        image_res = resp.get("image", [])
        kg_res = resp.get("kg", [])
        resp["fused"] = rrf_fuse([fulltext_res, semantic_res, image_res, kg_res])

    return resp
