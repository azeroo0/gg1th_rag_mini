import time

from common.ai_model import get_embedding_model
from common.qdrant import get_qdrant_client

# 기존 Dense Baseline 컬렉션. 기본값을 변경하면 기존 기능이 예고 없이
# 다른 데이터를 사용하게 되므로 기본값은 v1을 유지한다.
COLLECTION_NAME = "beopjeong_law_v1"

# 정의 전용 Child를 추가한 비교 실험용 컬렉션
COLLECTION_NAME_V2 = "beopjeong_law_v2"

def dense_search_by_vector(
    query_vector,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
    client=None,
):
    """
    이미 만들어 둔 질문 벡터로 Dense Retrieval을 수행한다.

    Embedding API를 호출하지 않으므로, 같은 질문을 여러 컬렉션에
    검색할 때 질문 벡터를 재사용할 수 있다.
    검색 결과 형식은 dense_search와 동일하다.
    """

    if client is None:
        client = get_qdrant_client()

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []

    for rank, point in enumerate(response.points, start=1):
        payload = point.payload or {}

        results.append({
            "rank": rank,
            "score": float(point.score),
            "id": payload.get("id", ""),
            "parent_id": payload.get("parent_id", ""),
            "article": payload.get("article", ""),
            "article_title": payload.get("article_title", ""),
            "paragraph": payload.get("paragraph", ""),
            "item": payload.get("item", ""),
            "subitem": payload.get("subitem", ""),
            "granularity": payload.get("granularity", ""),
            "chunk_role": payload.get("chunk_role", ""),
            "document_type": payload.get("document_type", ""),
            "text": payload.get("text", ""),
            "source_text": payload.get("source_text", ""),
        })

    return results

def dense_search(
    question: str,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
    embeddings=None,
    client=None,
):
    """
    Dense Retrieval을 수행한다.

    collection_name으로 검색 대상 컬렉션(v1/v2)을 선택할 수 있고,
    embeddings/client를 주입하면 반복 호출 시 재사용할 수 있다.
    인자를 생략하면 기존 동작과 동일하게 v1을 검색한다.

    질문이 많을 때는 질문 벡터를 미리 만들어 두고
    dense_search_by_vector를 쓰는 편이 API 호출 수가 적다.
    """

    if embeddings is None:
        embeddings = get_embedding_model()

    query_vector = embeddings.embed_query(question)

    return dense_search_by_vector(
        query_vector,
        top_k=top_k,
        collection_name=collection_name,
        client=client,
    )

# Embedding API 게이트웨이의 분당 요청 제한(30 RPM)을 넘지 않도록
# 호출 간 최소 간격을 두고, 429 응답에는 지수 백오프로 재시도한다.
# 평가처럼 질문이 많은 배치 실행에서만 사용하고, 단건 검색은 dense_search를 쓴다.
MIN_CALL_INTERVAL = 2.2
MAX_RETRIES = 5
RETRY_BASE_DELAY = 10.0

# 재시도할 가치가 있는 일시적 오류만 다시 시도한다.
# (컬렉션 이름 오류 등 설정 오류는 즉시 올려보낸다.)
RETRYABLE_ERROR_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
}

_last_call_time = 0.0

def is_retryable(error):
    if type(error).__name__ in RETRYABLE_ERROR_NAMES:
        return True

    message = str(error).lower()

    return "rate limit" in message or "429" in message

def dense_search_with_retry(
    question: str,
    top_k: int = 5,
    collection_name: str = COLLECTION_NAME,
    embeddings=None,
    client=None,
    verbose: bool = False,
):
    """
    dense_search를 호출 간격 제한과 재시도를 적용해 실행한다.
    검색 결과 형식은 dense_search와 동일하다.
    """

    global _last_call_time

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        elapsed = time.monotonic() - _last_call_time
        wait = MIN_CALL_INTERVAL - elapsed

        if wait > 0:
            time.sleep(wait)

        try:
            results = dense_search(
                question,
                top_k=top_k,
                collection_name=collection_name,
                embeddings=embeddings,
                client=client,
            )
            _last_call_time = time.monotonic()

            return results
        except Exception as error:
            _last_call_time = time.monotonic()
            last_error = error

            if not is_retryable(error):
                raise

            if attempt == MAX_RETRIES:
                break

            delay = RETRY_BASE_DELAY * attempt

            if verbose:
                print(
                    f"검색 재시도 {attempt}/{MAX_RETRIES - 1} "
                    f"({delay:.0f}초 대기): {type(error).__name__}"
                )

            time.sleep(delay)

    raise RuntimeError(
        f"검색 재시도 {MAX_RETRIES}회 실패: {last_error}"
    ) from last_error

# XML 항번호는 원문자(①②③…)로 기록되므로 조문 경로 표기용 숫자로 변환한다.
CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

def format_article_path(result):
    """
    검색 결과를 '제2조제4호바목' 형태의 조문 경로 문자열로 만든다.
    """

    parts = [result.get("article", "")]

    paragraph = str(result.get("paragraph", "")).strip()
    item = str(result.get("item", "")).strip()
    subitem = str(result.get("subitem", "")).strip()

    if paragraph:
        if paragraph in CIRCLED_NUMBERS:
            paragraph = str(CIRCLED_NUMBERS.index(paragraph) + 1)

        parts.append(f"제{paragraph}항")

    if item:
        parts.append(f"제{item.rstrip('.')}호")

    if subitem:
        parts.append(f"{subitem.rstrip('.')}목")

    path_text = "".join(part for part in parts if part)

    if result.get("chunk_role") == "definition_only":
        path_text += "(정의)"

    return path_text

if __name__ == "__main__":
    question = "고영향 인공지능이란 무엇인가요?"

    print("질문:", question)

    results = dense_search(question, top_k=5)

    for result in results:
        print(f"\n[{result['rank']}]")
        print("Score:", round(result["score"], 4))
        print("Article:", result["article"])
        print("Paragraph:", result["paragraph"])
        print("Item:", result["item"])
        print("Subitem:", result["subitem"])
        print("Granularity:", result["granularity"])
        print("Chunk role:", result["chunk_role"])
        print("Content:", result["source_text"][:300])