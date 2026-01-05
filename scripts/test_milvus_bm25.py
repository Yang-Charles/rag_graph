#!/usr/bin/env python3
"""Quick test for Milvus BM25 sparse-vector integration.

This script will:
- create the collection via `MilvusService`
- insert a few sample documents
- run a full-text search and a semantic search

Requires a running Milvus server accessible at the host/port used below.
"""
from backend.app.services.milvus_service import MilvusService


def main():
    svc = MilvusService(host="127.0.0.1", port="19530")
    svc.create_collection()

    docs = [
        {"id": 1, "text": "The quick brown fox jumps over the lazy dog."},
        {"id": 2, "text": "A fast fox and a quick dog play in the yard."},
        {"id": 3, "text": "Python programming language and code examples."},
    ]

    print("Inserting sample documents...")
    svc.insert_documents(docs)

    print("Fulltext search for 'quick fox':")
    try:
        res = svc.search_fulltext("quick fox", topk=5)
        print(res)
    except Exception as e:
        print("Fulltext search failed:", e)

    print("Semantic search for 'programming language':")
    try:
        res2 = svc.search_semantic("programming language", topk=5)
        print(res2)
    except Exception as e:
        print("Semantic search failed:", e)


if __name__ == "__main__":
    main()
