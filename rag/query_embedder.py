"""
질문 임베딩 최적화 모듈.

평가는 같은 질문을 v1ㆍv2 두 컬렉션에 대해 검색하므로, 질문당 Embedding
API를 두 번 호출하면 분당 요청 제한(30 RPM)에 쉽게 걸린다.
이 모듈은 다음 방식으로 호출 수를 줄인다.

- 중복 질문 제거: 같은 질문은 한 번만 임베딩한다.
- 배치 처리: embed_documents()로 여러 질문을 한 번에 임베딩한다.
- 디스크 캐시: 모델명, Base URL 식별자, 질문 텍스트로 키를 만들어
  재실행 시 API를 호출하지 않는다.
- Retry-After 우선 적용: 429 응답의 Retry-After 헤더를 먼저 따르고,
  헤더가 없으면 제한된 횟수로 지수 백오프한다.

캐시에는 모델명, Base URL의 해시 식별자, 질문 텍스트, 벡터만 저장하며
API 키는 로그와 캐시에 저장하지 않는다.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

from common.ai_model import get_embedding_model
from common.config import BASE_URL, EMBEDDING_MODEL
from rag.retriever import is_retryable

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = BASE_DIR / "data" / "cache" / "query_vectors.json"

# 배치 크기는 5에서 시작하고, 429가 발생하면 줄이고
# 연속 성공하면 상한까지 다시 늘린다.
INITIAL_BATCH_SIZE = 5
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 16
GROWTH_AFTER_SUCCESSES = 3

# 분당 30요청 제한을 넘지 않도록 API 호출 사이 최소 간격을 둔다.
MIN_CALL_INTERVAL = 2.2

MAX_RETRIES = 5
RETRY_BASE_DELAY = 10.0
MAX_RETRY_DELAY = 120.0

CACHE_VERSION = 1

def base_url_identifier(base_url=BASE_URL):
    """
    Base URL을 식별자로 바꾼다.

    Base URL에 토큰이 포함된 환경에서도 원문이 캐시나 로그에 남지 않도록
    해시 앞부분만 사용한다.
    """

    raw = (base_url or "default").encode("utf-8")

    return hashlib.sha256(raw).hexdigest()[:12]

def cache_key(question, model=EMBEDDING_MODEL, base_url_id=None):
    """
    모델명, Base URL 식별자, 질문 텍스트를 반영한 캐시 키를 만든다.
    """

    if base_url_id is None:
        base_url_id = base_url_identifier()

    payload = "\n".join([
        f"v{CACHE_VERSION}",
        model or "",
        base_url_id,
        question,
    ])

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def parse_retry_after(error):
    """
    429 응답에서 대기 시간을 초 단위로 읽는다. 없으면 None.

    Retry-After 헤더를 우선 보고, 없으면 밀리초 헤더와
    오류 메시지의 '1분', '30초' 같은 표현을 확인한다.
    """

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)

    if headers is not None:
        raw = headers.get("retry-after")

        if raw:
            try:
                return min(float(raw), MAX_RETRY_DELAY)
            except (TypeError, ValueError):
                pass

        raw_ms = headers.get("retry-after-ms")

        if raw_ms:
            try:
                return min(float(raw_ms) / 1000.0, MAX_RETRY_DELAY)
            except (TypeError, ValueError):
                pass

    message = str(error)

    match = re.search(r"(\d+(?:\.\d+)?)\s*(ms|밀리초)", message)

    if match:
        return min(float(match.group(1)) / 1000.0, MAX_RETRY_DELAY)

    match = re.search(r"(\d+(?:\.\d+)?)\s*(초|s\b|sec|second)", message)

    if match:
        return min(float(match.group(1)), MAX_RETRY_DELAY)

    match = re.search(r"(\d+(?:\.\d+)?)\s*(분|min|minute)", message)

    if match:
        return min(float(match.group(1)) * 60.0, MAX_RETRY_DELAY)

    return None

class QueryEmbedder:
    """
    질문 벡터를 중복 제거, 배치, 캐시, 재시도와 함께 생성한다.
    """

    def __init__(
        self,
        embeddings=None,
        model=EMBEDDING_MODEL,
        cache_path=DEFAULT_CACHE_PATH,
        batch_size=INITIAL_BATCH_SIZE,
        use_cache=True,
        verbose=True,
        min_call_interval=MIN_CALL_INTERVAL,
        retry_base_delay=RETRY_BASE_DELAY,
        sleep_fn=time.sleep,
    ):
        self.embeddings = embeddings
        self.model = model
        self.cache_path = Path(cache_path) if cache_path else None
        self.batch_size = max(MIN_BATCH_SIZE, min(batch_size, MAX_BATCH_SIZE))
        self.initial_batch_size = self.batch_size
        self.use_cache = use_cache and self.cache_path is not None
        self.verbose = verbose
        self.base_url_id = base_url_identifier()

        # 호출 간격과 대기 함수를 주입할 수 있게 하여
        # 테스트에서 실제로 기다리지 않도록 한다.
        self.min_call_interval = min_call_interval
        self.retry_base_delay = retry_base_delay
        self.sleep_fn = sleep_fn

        self._cache = self._load_cache()
        self._last_call_time = 0.0
        self._consecutive_successes = 0

        self.stats = {
            "unique_questions": 0,
            "duplicate_questions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "api_calls": 0,
            "api_batches": 0,
            "rate_limit_retries": 0,
            "other_retries": 0,
            "embedding_seconds": 0.0,
            "final_batch_size": self.batch_size,
        }

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    def _load_cache(self):
        if not self.use_cache or not self.cache_path.exists():
            return {}

        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            self._log(f"질문 벡터 캐시를 읽을 수 없어 새로 만듭니다: {error}")

            return {}

        entries = data.get("entries", {})

        return {
            key: value for key, value in entries.items()
            if isinstance(value, dict) and value.get("vector")
        }

    def _save_cache(self):
        if not self.use_cache:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "cache_version": CACHE_VERSION,
            "model": self.model,
            "base_url_id": self.base_url_id,
            "entries": self._cache,
        }

        # 쓰기 중 중단되어도 기존 캐시가 깨지지 않도록 임시 파일에 쓰고 교체한다.
        temp_path = self.cache_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temp_path, self.cache_path)

    def _get_embeddings(self):
        if self.embeddings is None:
            self.embeddings = get_embedding_model()

        return self.embeddings

    def _throttle(self):
        wait = self.min_call_interval - (
            time.monotonic() - self._last_call_time
        )

        if wait > 0:
            self.sleep_fn(wait)

    def _shrink_batch(self):
        previous = self.batch_size
        self.batch_size = max(MIN_BATCH_SIZE, self.batch_size // 2)
        self._consecutive_successes = 0

        if self.batch_size != previous:
            self._log(
                f"배치 크기 조정: {previous} -> {self.batch_size} (API 제한 대응)"
            )

    def _grow_batch(self):
        self._consecutive_successes += 1

        if self._consecutive_successes < GROWTH_AFTER_SUCCESSES:
            return

        if self.batch_size >= MAX_BATCH_SIZE:
            return

        previous = self.batch_size
        self.batch_size = min(MAX_BATCH_SIZE, self.batch_size + 1)
        self._consecutive_successes = 0

        self._log(f"배치 크기 조정: {previous} -> {self.batch_size}")

    def _embed_batch(self, texts):
        """
        한 배치를 임베딩한다. 429는 Retry-After를 우선 적용해 재시도한다.
        """

        embeddings = self._get_embeddings()
        last_error = None

        for attempt in range(1, MAX_RETRIES + 1):
            self._throttle()

            try:
                started = time.monotonic()
                vectors = embeddings.embed_documents(texts)

                self._last_call_time = time.monotonic()
                self.stats["api_calls"] += 1
                self.stats["api_batches"] += 1
                self.stats["embedding_seconds"] += (
                    self._last_call_time - started
                )

                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"Embedding 개수 불일치: {len(vectors)} != {len(texts)}"
                    )

                self._grow_batch()

                return vectors
            except Exception as error:
                self._last_call_time = time.monotonic()
                last_error = error

                if not is_retryable(error):
                    raise

                rate_limited = "RateLimitError" in type(error).__name__ or (
                    "rate limit" in str(error).lower()
                )

                if rate_limited:
                    self.stats["rate_limit_retries"] += 1
                    self._shrink_batch()
                else:
                    self.stats["other_retries"] += 1

                if attempt == MAX_RETRIES:
                    break

                delay = parse_retry_after(error)
                source = "Retry-After"

                if delay is None:
                    delay = min(
                        self.retry_base_delay * attempt, MAX_RETRY_DELAY
                    )
                    source = "지수 백오프"

                self._log(
                    f"임베딩 재시도 {attempt}/{MAX_RETRIES - 1} "
                    f"({source} {delay:.1f}초 대기): {type(error).__name__}"
                )

                self.sleep_fn(delay)

        raise RuntimeError(
            f"임베딩 재시도 {MAX_RETRIES}회 실패: {last_error}"
        ) from last_error

    def embed_questions(self, questions):
        """
        질문 목록을 받아 {질문: 벡터} 사전을 돌려준다.

        같은 질문은 한 번만 임베딩하며, 캐시에 있으면 API를 호출하지 않는다.
        """

        started = time.monotonic()

        unique = list(dict.fromkeys(questions))

        self.stats["unique_questions"] = len(unique)
        self.stats["duplicate_questions"] = len(questions) - len(unique)

        vectors = {}
        pending = []

        for question in unique:
            key = cache_key(question, self.model, self.base_url_id)
            entry = self._cache.get(key)

            if entry is not None:
                vectors[question] = entry["vector"]
                self.stats["cache_hits"] += 1
            else:
                pending.append((question, key))

        self.stats["cache_misses"] = len(pending)

        self._log(
            f"질문 {len(questions)}개 (고유 {len(unique)}개) | "
            f"캐시 적중 {self.stats['cache_hits']}개 | "
            f"임베딩 필요 {len(pending)}개"
        )

        done = 0

        while done < len(pending):
            batch = pending[done:done + self.batch_size]
            texts = [question for question, _ in batch]

            batch_vectors = self._embed_batch(texts)

            for (question, key), vector in zip(batch, batch_vectors):
                vectors[question] = vector
                self._cache[key] = {
                    "model": self.model,
                    "base_url_id": self.base_url_id,
                    "question": question,
                    "vector": vector,
                }

            done += len(batch)

            self._log(
                f"임베딩 진행: {done}/{len(pending)} "
                f"(배치 {len(batch)}개, API 호출 {self.stats['api_calls']}회)"
            )

            self._save_cache()

        self.stats["final_batch_size"] = self.batch_size
        self.stats["total_seconds"] = round(time.monotonic() - started, 2)
        self.stats["embedding_seconds"] = round(
            self.stats["embedding_seconds"], 2
        )

        missing = [q for q in unique if q not in vectors]

        if missing:
            raise RuntimeError(
                f"질문 벡터를 만들지 못했습니다: {len(missing)}개"
            )

        return vectors

    def report(self):
        """
        호출 통계를 사람이 읽을 수 있는 줄 목록으로 만든다.
        """

        stats = self.stats

        saved = (
            stats["duplicate_questions"] + stats["cache_hits"]
        )

        return [
            f"고유 질문: {stats['unique_questions']}개"
            f" (중복 제거 {stats['duplicate_questions']}개)",
            f"캐시 적중: {stats['cache_hits']}개"
            f" / 캐시 미적중: {stats['cache_misses']}개",
            f"Embedding API 호출: {stats['api_calls']}회"
            f" (배치 {stats['api_batches']}회, 최종 배치 크기"
            f" {stats['final_batch_size']})",
            f"재시도: RateLimit {stats['rate_limit_retries']}회"
            f" / 기타 {stats['other_retries']}회",
            f"임베딩 소요 시간: {stats['embedding_seconds']}초"
            f" (대기 포함 {stats.get('total_seconds', 0)}초)",
            f"절약한 임베딩 요청: {saved}개",
        ]
