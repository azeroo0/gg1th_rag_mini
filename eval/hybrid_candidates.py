"""
Hybrid 후보(Top-8) 캐시.

C(Hybrid)와 C+D(Hybrid+Reranking) 평가가 Qwen Reranking 효과만 비교할 수
있도록, 동일한 Hybrid Top-8 후보 집합을 한 번 계산해 캐시 파일에 저장하고
재사용한다. Dense/BM25 검색을 C+D 평가에서 중복 수행하지 않기 위함이다.
"""

import json
from pathlib import Path

from rag.hybrid_retriever import COLLECTION_NAME_V2, get_bm25_index, hybrid_search_by_vector

BASE_DIR = Path(__file__).resolve().parent.parent
CANDIDATES_CACHE_PATH = (
    BASE_DIR / "eval" / "results" / "hybrid" / "candidates_top8.json"
)

CANDIDATE_TOP_K = 8


def build_candidates(questions, question_vectors, client, bm25_index=None, top_k=CANDIDATE_TOP_K):
    if bm25_index is None:
        bm25_index = get_bm25_index(collection_name=COLLECTION_NAME_V2, client=client)

    candidates = {}

    for question in questions:
        results = hybrid_search_by_vector(
            question,
            question_vectors[question],
            top_k=top_k,
            collection_name=COLLECTION_NAME_V2,
            client=client,
            bm25_index=bm25_index,
        )
        candidates[question] = results

    return candidates


def save_candidates(candidates, path=CANDIDATES_CACHE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return path


def load_candidates(path=CANDIDATES_CACHE_PATH):
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def get_or_build_candidates(
    questions,
    question_vectors,
    client,
    bm25_index=None,
    top_k=CANDIDATE_TOP_K,
    force_rebuild=False,
):
    """
    캐시에 모든 질문이 있으면 그대로 반환하고, 없으면 새로 만들어 캐시에 저장한다.
    """

    if not force_rebuild:
        cached = load_candidates()

        if cached is not None and set(questions).issubset(set(cached.keys())):
            return cached

    candidates = build_candidates(
        questions, question_vectors, client, bm25_index=bm25_index, top_k=top_k
    )

    save_candidates(candidates)

    return candidates
