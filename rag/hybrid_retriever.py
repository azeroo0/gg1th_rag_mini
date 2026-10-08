"""
Hybrid Retrieval: Dense + BM25 + RRF.

beopjeong_law_v2(206 Child)를 대상으로 Dense Retrieval과 BM25 Retrieval을
각각 수행한 뒤, Reciprocal Rank Fusion(RRF)으로 결합한다.

Dense와 BM25가 반드시 같은 206개 Child를 검색하도록, BM25 색인은
Qdrant의 beopjeong_law_v2 컬렉션을 scroll pagination으로 전체 읽어
구축한다. (별도 텍스트 파일이나 legal_chunks_v2.json을 직접 읽지 않는다.)

중복 이슈: 같은 조문(parent_id)에서 항 전체 Child와 호/목/정의 세부
Child가 함께 검색될 수 있다(예: 제2조제4호 전체 Child, 가목~카목 Child,
정의 전용 Child가 모두 '제2조'에 속함). RRF 기본 결과는 이를 그대로
반영하므로 Top-N 안에 같은 조문의 Child가 여러 개 들어갈 수 있다.
이 문제를 완화하기 위한 Parent 다양화는 선택 기능(diversify_parent)으로
제공하며, 기본 RRF 결과(diversify_parent=False)는 그대로 유지한다.
"""

import re
import time
from collections import defaultdict

from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

from common.qdrant import get_qdrant_client
from rag.retriever import (
    COLLECTION_NAME_V2,
    dense_search,
    dense_search_by_vector,
)

SCROLL_BATCH = 100

DEFAULT_DENSE_TOP_K = 20
DEFAULT_BM25_TOP_K = 20
DEFAULT_RRF_K = 60
DEFAULT_FINAL_TOP_K = 5

RESULT_FIELDS = [
    "parent_id",
    "article",
    "article_title",
    "paragraph",
    "item",
    "subitem",
    "granularity",
    "source_text",
    "text",
    "document_type",
    "chunk_role",
]

# 조문ㆍ항ㆍ호ㆍ목 번호는 형태소 분석으로 쪼개지면 검색에 불리하므로
# 정규식으로 통째로 추출해 보존한다.
LEGAL_REFERENCE_PATTERN = re.compile(
    r"제\d+조(?:의\d+)?|제\d+항|제\d+호|[가-하]목"
)

# 명사ㆍ어근ㆍ외래어ㆍ숫자 위주로 남기고 조사ㆍ어미는 제외한다.
KEEP_TAGS = ("NNG", "NNP", "NNB", "VV", "VA", "XR", "SL", "SN", "SH")

_kiwi = None

def get_kiwi():
    global _kiwi

    if _kiwi is None:
        _kiwi = Kiwi()

    return _kiwi

def normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def tokenize(text):
    """
    한국어 법률 텍스트를 BM25 색인/질의에 쓸 토큰 목록으로 바꾼다.

    조문 번호는 정규식으로 보존하고, 나머지는 kiwipiepy 형태소 분석으로
    명사ㆍ어근 위주 토큰을 추출한다. 질의와 문서에 동일한 함수를 쓴다.
    """

    text = normalize_text(text)

    if not text:
        return []

    legal_tokens = LEGAL_REFERENCE_PATTERN.findall(text)
    remainder = LEGAL_REFERENCE_PATTERN.sub(" ", text)

    kiwi = get_kiwi()
    tokens = [token.lower() for token in legal_tokens]

    for token in kiwi.tokenize(remainder):
        if token.tag in KEEP_TAGS:
            tokens.append(token.form.lower())

    return tokens

def fetch_all_points(client=None, collection_name=COLLECTION_NAME_V2):
    """
    Qdrant 컬렉션의 전체 Point를 scroll pagination으로 누락 없이 읽는다.
    """

    if client is None:
        client = get_qdrant_client()

    points = []
    offset = None

    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        points.extend(batch)

        if offset is None:
            break

    return points

class BM25Index:
    """
    Qdrant 컬렉션 전체를 읽어 만든 BM25 색인.

    Dense가 검색하는 것과 동일한 컬렉션(동일한 206개 Child)을 대상으로
    하므로, Dense/BM25 후보 집합의 Child ID 공간이 항상 일치한다.
    """

    def __init__(self, collection_name=COLLECTION_NAME_V2, client=None):
        self.collection_name = collection_name

        points = fetch_all_points(client=client, collection_name=collection_name)

        self.payloads = []
        corpus_tokens = []

        for point in points:
            payload = point.payload or {}
            self.payloads.append(payload)

            text = payload.get("text") or payload.get("source_text") or ""
            corpus_tokens.append(tokenize(text))

        self.size = len(self.payloads)
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, question, top_k=DEFAULT_BM25_TOP_K):
        query_tokens = tokenize(question)
        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        results = []

        for rank, index in enumerate(ranked_indices[:top_k], start=1):
            payload = self.payloads[index]

            result = {
                "rank": rank,
                "score": float(scores[index]),
                "id": payload.get("id", ""),
            }
            result.update(
                {field: payload.get(field, "") for field in RESULT_FIELDS}
            )
            results.append(result)

        return results

_bm25_index_cache = {}

def get_bm25_index(collection_name=COLLECTION_NAME_V2, client=None, force_reload=False):
    """
    BM25 색인을 프로세스 내에서 재사용한다(컬렉션별 1회 구축).
    """

    if force_reload or collection_name not in _bm25_index_cache:
        _bm25_index_cache[collection_name] = BM25Index(
            collection_name=collection_name, client=client
        )

    return _bm25_index_cache[collection_name]

def reciprocal_rank_fusion(dense_results, bm25_results, k=DEFAULT_RRF_K):
    """
    score(d) = sum(1 / (k + rank_i(d)))

    Dense와 BM25 순위를 Child ID 기준으로 결합하고 중복을 제거한다.
    """

    rrf_scores = defaultdict(float)
    dense_rank_map = {}
    bm25_rank_map = {}
    payload_map = {}

    for result in dense_results:
        chunk_id = result["id"]
        dense_rank_map[chunk_id] = result["rank"]
        rrf_scores[chunk_id] += 1.0 / (k + result["rank"])
        payload_map[chunk_id] = result

    for result in bm25_results:
        chunk_id = result["id"]
        bm25_rank_map[chunk_id] = result["rank"]
        rrf_scores[chunk_id] += 1.0 / (k + result["rank"])

        if chunk_id not in payload_map:
            payload_map[chunk_id] = result

    fused = []

    for chunk_id, score in rrf_scores.items():
        payload = payload_map[chunk_id]

        entry = {"id": chunk_id}
        entry.update(
            {field: payload.get(field, "") for field in RESULT_FIELDS}
        )
        entry["dense_rank"] = dense_rank_map.get(chunk_id)
        entry["bm25_rank"] = bm25_rank_map.get(chunk_id)
        entry["rrf_score"] = score

        fused.append(entry)

    fused.sort(key=lambda item: item["rrf_score"], reverse=True)

    for rank, item in enumerate(fused, start=1):
        item["rank"] = rank

    return fused

def apply_parent_diversification(fused_results, max_per_parent=1):
    """
    같은 parent_id(조문)에서 나온 Child를 최대 max_per_parent개로 제한한다.

    RRF 점수 순서를 그대로 따르면서 초과분만 건너뛰므로,
    기본 RRF 순위 자체는 변경하지 않고 별도 결과로 제공한다.
    """

    parent_counts = defaultdict(int)
    diversified = []

    for item in fused_results:
        parent_id = item.get("parent_id", "")

        if parent_counts[parent_id] >= max_per_parent:
            continue

        parent_counts[parent_id] += 1
        diversified.append(dict(item))

    for rank, item in enumerate(diversified, start=1):
        item["rank"] = rank

    return diversified

def hybrid_search_by_vector(
    question,
    query_vector,
    top_k=DEFAULT_FINAL_TOP_K,
    dense_top_k=DEFAULT_DENSE_TOP_K,
    bm25_top_k=DEFAULT_BM25_TOP_K,
    rrf_k=DEFAULT_RRF_K,
    collection_name=COLLECTION_NAME_V2,
    client=None,
    bm25_index=None,
    diversify_parent=False,
    max_per_parent=1,
):
    """
    이미 만들어 둔 질문 벡터로 Hybrid Search를 수행한다.

    평가처럼 질문이 많을 때 Embedding API를 다시 호출하지 않도록
    query_vector를 주입받는다. question 텍스트는 BM25에 그대로 쓴다.
    """

    if client is None:
        client = get_qdrant_client()

    dense_results = dense_search_by_vector(
        query_vector,
        top_k=dense_top_k,
        collection_name=collection_name,
        client=client,
    )

    if bm25_index is None:
        bm25_index = get_bm25_index(collection_name=collection_name, client=client)

    bm25_results = bm25_index.search(question, top_k=bm25_top_k)

    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=rrf_k)

    if diversify_parent:
        fused = apply_parent_diversification(fused, max_per_parent=max_per_parent)

    return fused[:top_k]

def hybrid_search(
    question,
    top_k=DEFAULT_FINAL_TOP_K,
    dense_top_k=DEFAULT_DENSE_TOP_K,
    bm25_top_k=DEFAULT_BM25_TOP_K,
    rrf_k=DEFAULT_RRF_K,
    collection_name=COLLECTION_NAME_V2,
    embeddings=None,
    client=None,
    bm25_index=None,
    diversify_parent=False,
    max_per_parent=1,
):
    """
    질문 텍스트를 받아 Embedding을 새로 만들고 Hybrid Search를 수행한다.

    질문이 많은 배치 실행에서는 질문 벡터를 미리 만들어 두고
    hybrid_search_by_vector를 쓰는 편이 API 호출 수가 적다.
    """

    if client is None:
        client = get_qdrant_client()

    dense_results = dense_search(
        question,
        top_k=dense_top_k,
        collection_name=collection_name,
        embeddings=embeddings,
        client=client,
    )

    if bm25_index is None:
        bm25_index = get_bm25_index(collection_name=collection_name, client=client)

    bm25_results = bm25_index.search(question, top_k=bm25_top_k)

    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=rrf_k)

    if diversify_parent:
        fused = apply_parent_diversification(fused, max_per_parent=max_per_parent)

    return fused[:top_k]

if __name__ == "__main__":
    question = "고영향 인공지능이란 무엇인가요?"

    print("질문:", question)

    started = time.monotonic()
    results = hybrid_search(question, top_k=5)
    elapsed = time.monotonic() - started

    for result in results:
        print(f"\n[{result['rank']}] rrf_score={result['rrf_score']:.5f}")
        print("Dense rank:", result["dense_rank"], "| BM25 rank:", result["bm25_rank"])
        print("Article:", result["article"], result["paragraph"], result["item"], result["subitem"])
        print("Content:", result["source_text"][:200])

    print(f"\n소요 시간: {elapsed:.2f}초")
