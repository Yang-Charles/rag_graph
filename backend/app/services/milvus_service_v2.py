"""
Improved Milvus service (v2) that:
- creates collection with multiple vector fields
- creates indexes for vector fields
- inserts documents with text and image vectors
- performs separate searches for text_vector, image_vector and a simple fulltext fallback

Note: For true multi-vector AnnSearchRequest usage follow Milvus docs; here we emulate multi-vector
by searching each vector field and returning structured results for downstream fusion.
"""
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, Index
from typing import List, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import io

class MilvusService:
    def __init__(self, host="127.0.0.1", port="19530", collection_name="multimodal_docs"):
        self.host = host
        self.port = port
        self.conn = None
        self.collection_name = collection_name
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")

    def connect(self):
        if self.conn is None:
            connections.connect(host=self.host, port=self.port)
            self.conn = True

    def create_collection(self, dim_text=384, dim_image=512):
        self.connect()
        if utility.has_collection(self.collection_name):
            return

        fields = [
            FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=dim_text),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=dim_image),
        ]

        schema = CollectionSchema(fields, description="Multimodal collection")
        coll = Collection(self.collection_name, schema)

        # Create indexes for vector fields
        try:
            idx_params_text = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}}
            Index(collection_name=self.collection_name, field_name="text_vector", index_params=idx_params_text)

            idx_params_image = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}}
            Index(collection_name=self.collection_name, field_name="image_vector", index_params=idx_params_image)
        except Exception:
            pass

    def insert_documents(self, docs: List[dict]):
        self.connect()
        coll = Collection(self.collection_name)
        texts = [d.get("text", "") for d in docs]
        text_vecs = self.text_model.encode(texts, convert_to_numpy=True).astype(np.float32)

        img_vecs = []
        for d in docs:
            b = d.get("image_bytes")
            if b:
                # Placeholder: using zeros; replace with CLIP embedding
                img_vecs.append(np.zeros(512, dtype=np.float32))
            else:
                img_vecs.append(np.zeros(512, dtype=np.float32))

        ids = [d.get("id") for d in docs]
        entities = [ids, texts, text_vecs.tolist(), np.stack(img_vecs).tolist()]
        coll.insert(entities)
        coll.flush()

    def search_semantic(self, query: str, topk=10) -> List[Tuple[Any, float, str]]:
        self.connect()
        coll = Collection(self.collection_name)
        qvec = self.text_model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        res = coll.search([qvec.tolist()], "text_vector", param=search_params, limit=topk, output_fields=["doc_id", "text"])
        out = []
        for hits in res:
            for h in hits:
                out.append((h.entity.get("doc_id"), float(h.score), "semantic"))
        return out

    def search_image(self, image_bytes: bytes, topk=10) -> List[Tuple[Any, float, str]]:
        self.connect()
        coll = Collection(self.collection_name)
        # TODO: compute CLIP image embedding; currently placeholder zeros
        qvec = np.zeros(512, dtype=np.float32)
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        res = coll.search([qvec.tolist()], "image_vector", param=search_params, limit=topk, output_fields=["doc_id", "text"])
        out = []
        for hits in res:
            for h in hits:
                out.append((h.entity.get("doc_id"), float(h.score), "image"))
        return out

    def search_fulltext(self, query: str, topk=10) -> List[Tuple[Any, float, str]]:
        self.connect()
        coll = Collection(self.collection_name)
        # Simple LIKE query as a fallback; Milvus full-text requires text index setup
        expr = f"text like '%{query.replace("'", "\\'")}%'")"
        try:
            qres = coll.query(expr=expr, output_fields=["doc_id", "text"]) or []
        except Exception:
            qres = []
        out = []
        for r in qres[:topk]:
            out.append((r.get("doc_id"), 1.0, "fulltext"))
        return out

    def multi_vector_search(self, query: str, image_bytes: bytes = None, topk=10) -> dict:
        """Run per-modality searches and return per-method results.

        This function runs semantic, image (if provided), and fulltext searches and
        returns a dict suitable for downstream fusion.
        """
        semantic = self.search_semantic(query, topk=topk)
        image = []
        if image_bytes:
            image = self.search_image(image_bytes, topk=topk)
        fulltext = self.search_fulltext(query, topk=topk)
        return {"semantic": semantic, "image": image, "fulltext": fulltext}
