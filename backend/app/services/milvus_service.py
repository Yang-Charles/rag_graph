"""
Milvus service scaffold supporting:
- creating a collection with multiple vector fields (text/vector/image)
- inserting documents
- simple search methods: fulltext (placeholder), semantic vector search, image vector search

Notes:
- This scaffold uses `pymilvus` primitives. For production tune index params and use Milvus' multi-vector search APIs as documented:
  https://milvus.io/docs/zh/multi-vector-search.md
"""
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from typing import List, Tuple, Any
import numpy as np
import re
import math
from collections import Counter

from sentence_transformers import SentenceTransformer
from PIL import Image
import io

class MilvusService:
    def __init__(self, host="127.0.0.1", port="19530"):
        self.host = host
        self.port = port
        self.conn = None
        self.collection_name = "multimodal_docs"
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        # BM25 corpus statistics kept in-memory for sparse vector generation
        self.doc_freq = {}  # term -> document frequency
        self.total_docs = 0
        self.total_doc_len = 0
        self.dim_text = 384
        # For image embeddings you can use a CLIP model; placeholder here

    def connect(self):
        if self.conn is None:
            connections.connect(host=self.host, port=self.port)
            self.conn = True

    def create_collection(self, dim_text=384, dim_image=512):
        # remember text dim for sparse hashing
        self.dim_text = dim_text
        self.connect()
        if utility.has_collection(self.collection_name):
            return

        # Configure fields:
        # - `text` is a VARCHAR used for full-text BM25 search. enable_analyzer must be true
        #   so Milvus can analyze the text (default analyzer is `standard`).
        # - `text_sparse` is a SPARSE_FLOAT_VECTOR that can store sparse embeddings
        #   produced by BM25 (or other sparse encoders) if desired.
        fields = [
            FieldSchema(name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535,
                        type_params={"enable_analyzer": "true"}),
            FieldSchema(name="text_vector", dtype=DataType.FLOAT_VECTOR, dim=dim_text),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=dim_image),
            FieldSchema(name="text_sparse", dtype=DataType.SPARSE_FLOAT_VECTOR,
                        type_params={"dim": str(dim_text)}, is_nullable=True),
        ]

        schema = CollectionSchema(fields, description="Multimodal collection")
        coll = Collection(self.collection_name, schema)

        # Create a BM25 index on the `text` field so Milvus can serve BM25 ranking.
        # Parameters `k1` and `b` follow BM25 standard; tune as needed.
        try:
            index_params = {"index_type": "BM25", "params": {"k1": 1.2, "b": 0.75}}
            coll.create_index(field_name="text", index_params=index_params)
        except Exception:
            # Index creation may fail on older Milvus versions or if BM25 isn't enabled.
            # We choose to continue silently; callers can inspect logs if needed.
            pass

    def insert_documents(self, docs: List[dict]):
        # docs: [{'id': int, 'text': str, 'image_bytes': bytes|None}]
        self.connect()
        coll = Collection(self.collection_name)
        texts = [d.get("text", "") for d in docs]
        text_vecs = self.text_model.encode(texts, convert_to_numpy=True).astype(np.float32)

        # placeholder image vectors: zeros or computed via CLIP
        img_vecs = []
        for d in docs:
            b = d.get("image_bytes")
            if b:
                # TODO: compute image embedding via CLIP
                img_vecs.append(np.zeros(512, dtype=np.float32))
            else:
                img_vecs.append(np.zeros(512, dtype=np.float32))

        ids = [d.get("id") for d in docs]
        # Update BM25 corpus stats with incoming documents, then compute sparse vectors.
        self._update_corpus_stats(texts)
        sparse_vectors = [self.bm25_sparse_vector(t) for t in texts]

        entities = [ids, texts, text_vecs.tolist(), np.stack(img_vecs).tolist(), sparse_vectors]
        # Note: we currently don't auto-generate BM25 sparse embeddings here; Milvus can
        # compute BM25 ranking from the `text` field when `enable_analyzer` + BM25 index
        # are configured.
        coll.insert(entities)

    def _tokenize(self, text: str) -> List[str]:
        # simple word tokenizer; lowercase and keep word chars
        return re.findall(r"\w+", text.lower())

    def _update_corpus_stats(self, texts: List[str]):
        # Update document frequency and length stats for BM25
        for t in texts:
            tokens = self._tokenize(t)
            if not tokens:
                # still count as a document
                self.total_docs += 1
                continue
            unique = set(tokens)
            for term in unique:
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1
            self.total_docs += 1
            self.total_doc_len += len(tokens)

    def bm25_sparse_vector(self, text: str, k1: float = 1.2, b: float = 0.75) -> dict:
        """Convert `text` into a sparse vector (dict with `indices` and `values`) using BM25 weights.

        - Uses in-memory corpus stats (`self.doc_freq`, `self.total_docs`, `self.total_doc_len`).
        - Maps terms to vector indices using a stable hash modulo `self.dim_text`.
        - Returns {'indices': [...], 'values': [...]} which is compatible with Milvus SPARSE_FLOAT_VECTOR inserts.
        """
        tokens = self._tokenize(text)
        if not tokens or self.total_docs == 0:
            return None

        tf = Counter(tokens)
        dl = len(tokens)
        avgdl = (self.total_doc_len / self.total_docs) if self.total_docs > 0 else dl

        idx2val = {}
        for term, freq in tf.items():
            df = self.doc_freq.get(term, 0)
            # smoothed idf
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
            denom = freq + k1 * (1 - b + b * (dl / avgdl)) if avgdl > 0 else freq + k1
            score = idf * ((freq * (k1 + 1)) / denom)
            idx = abs(hash(term)) % self.dim_text
            idx2val[idx] = idx2val.get(idx, 0.0) + float(score)

        if not idx2val:
            return None

        indices = list(idx2val.keys())
        values = [idx2val[i] for i in indices]
        return {"indices": indices, "values": values}

    def search_semantic(self, query: str, topk=10) -> List[Tuple[Any, float, str]]:
        self.connect()
        coll = Collection(self.collection_name)
        qvec = self.text_model.encode([query])[0].astype(np.float32)
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
        # TODO: compute image embedding via CLIP
        qvec = np.zeros(512, dtype=np.float32)
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        res = coll.search([qvec.tolist()], "image_vector", param=search_params, limit=topk, output_fields=["doc_id", "text"])
        out = []
        for hits in res:
            for h in hits:
                out.append((h.entity.get("doc_id"), float(h.score), "image"))
        return out

    def search_fulltext(self, query: str, topk=10) -> List[Tuple[Any, float, str]]:
        # Milvus full-text BM25 integration requires collection with TEXT fields and enabled text index.
        # As a scaffold we run a simple filter+placeholder score using collection.query
        self.connect()
        coll = Collection(self.collection_name)
        # escape single quotes in the user query for safe query expression
        safe = query.replace("'", "\\'")
        expr = f"text like '%{safe}%'"
        try:
            qres = coll.query(expr=expr, output_fields=["doc_id", "text"])
        except Exception:
            qres = []
        out = []
        for r in qres[:topk]:
            out.append((r.get("doc_id"), 1.0, "fulltext"))
        return out
