"""
법정갈건호 RAG API.

POST /ask: 질문을 받아 Hybrid(C) 또는 Hybrid+Reranking(C+D) 검색 결과를
근거로 Grounded Answer를 생성한다.
GET /health: API 서버와 Qdrant 연결 상태를 확인한다.

Embedding/LLM/Qdrant 클라이언트는 요청마다 새로 만들지 않고 모듈
전역에서 재사용한다. Reranking 모델은 hybrid_rerank 모드로 처음
요청이 들어올 때만 지연 로딩된다(rag/reranker.py의 지연 로딩 구조 재사용).
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from common.ai_model import get_embedding_model, get_llm_model
from common.qdrant import get_qdrant_client
from rag.pipeline import MODES, generate_answer
from rag.chunker import LAW_EFFECTIVE_DATE, LAW_NAME, LAW_NUMBER

app = FastAPI(title="법정갈건호 RAG API")

DEFAULT_MODE = "hybrid"

# 요청마다 다시 만들지 않고 재사용하는 전역 리소스.
_resources = {
    "qdrant_client": None,
    "embeddings": None,
    "llm": None,
}


def get_shared_client():
    if _resources["qdrant_client"] is None:
        _resources["qdrant_client"] = get_qdrant_client()

    return _resources["qdrant_client"]


def get_shared_embeddings():
    if _resources["embeddings"] is None:
        _resources["embeddings"] = get_embedding_model()

    return _resources["embeddings"]


def get_shared_llm():
    if _resources["llm"] is None:
        _resources["llm"] = get_llm_model(max_tokens=1024)

    return _resources["llm"]


class AskRequest(BaseModel):
    question: str = Field(..., description="사용자 질문")
    mode: Optional[str] = Field(
        None, description=f"검색 모드. {MODES} 중 하나 (기본값 {DEFAULT_MODE})"
    )


class SourceItem(BaseModel):
    law_name: str = ""
    article: str = ""
    article_title: str = ""
    paragraph: str = ""
    item: str = ""
    subitem: str = ""
    content: str = ""
    effective_date: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    mode: str
    retrieval_count: int
    law_name: str
    law_number: str
    effective_date: str
    elapsed_seconds: float


@app.get("/")
def root():
    return {
        "message": "RAG API"
    }


@app.get("/health")
def health():
    """
    API 서버 상태와 Qdrant 연결 상태를 확인한다.
    """

    health_status = {
        "status": "ok",
        "api": "ok",
        "qdrant": "unknown",
    }

    try:
        client = get_shared_client()
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        health_status["qdrant"] = "ok"
        health_status["qdrant_collections"] = collection_names
    except Exception as error:
        health_status["status"] = "degraded"
        health_status["qdrant"] = "error"
        health_status["qdrant_error"] = str(error)

        return JSONResponse(status_code=503, content=health_status)

    return health_status


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = (request.question or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="질문이 비어 있습니다.")

    mode = request.mode or DEFAULT_MODE

    if mode not in MODES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 mode입니다: {mode} (지원: {list(MODES)})",
        )

    try:
        result = generate_answer(
            question,
            mode=mode,
            client=get_shared_client(),
            embeddings=get_shared_embeddings(),
            llm=get_shared_llm(),
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"답변 생성 중 오류가 발생했습니다: {error}",
        ) from error

    sources = result.get("sources") or []

    return AskResponse(
        answer=result["answer"],
        sources=[SourceItem(**source) for source in sources],
        mode=result["mode"],
        retrieval_count=result["retrieval_count"],
        law_name=result.get("law_name", LAW_NAME),
        law_number=result.get("law_number", LAW_NUMBER),
        effective_date=result.get("effective_date", LAW_EFFECTIVE_DATE),
        elapsed_seconds=result["elapsed_seconds"],
    )
