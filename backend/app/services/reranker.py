"""
Reranker utilities: implements Reciprocal Rank Fusion (RRF)
to combine multiple ranked lists into a single fused ranking.
"""
from typing import List, Tuple, Any

def rrf_fuse(ranked_lists: List[List[Tuple[Any, float, str]]], k: int = 60) -> List[Tuple[Any, float]]:
    """Fuse multiple ranked lists using RRF.

    ranked_lists: list of lists, each element is (doc_id, score, source)
    Returns list of (doc_id, fused_score) sorted desc.
    """
    scores = {}
    for rl in ranked_lists:
        for rank, item in enumerate(rl, start=1):
            doc_id = item[0]
            scores.setdefault(doc_id, 0.0)
            scores[doc_id] += 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused
