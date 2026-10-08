"""
질문 임베딩 최적화(QueryEmbedder) 검증 테스트.

Embedding API를 호출하지 않도록 가짜 임베딩 객체를 주입하여
중복 제거, 배치 처리, 캐시, 재시도 동작을 확인한다.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.query_embedder import (
    INITIAL_BATCH_SIZE,
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    QueryEmbedder,
    base_url_identifier,
    cache_key,
    parse_retry_after,
)

class FakeRateLimitError(Exception):
    """
    openai.RateLimitError와 같은 이름을 쓰는 가짜 예외.
    is_retryable은 예외 클래스 이름으로 판단한다.
    """

    def __init__(self, message="rate limit exceeded", response=None):
        super().__init__(message)
        self.response = response

FakeRateLimitError.__name__ = "RateLimitError"

class FakeResponse:
    def __init__(self, headers):
        self.headers = headers

class FakeEmbeddings:
    """
    호출 기록을 남기는 가짜 임베딩 모델.
    """

    def __init__(self, fail_times=0, dimension=4):
        self.calls = []
        self.fail_times = fail_times
        self.dimension = dimension

    def embed_documents(self, texts):
        self.calls.append(list(texts))

        if self.fail_times > 0:
            self.fail_times -= 1

            raise FakeRateLimitError()

        return [
            [float(len(text)) + i for i in range(self.dimension)]
            for text in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def make_embedder(cache_dir, embeddings, **kwargs):
    options = {
        "embeddings": embeddings,
        "model": "test-embedding-model",
        "cache_path": Path(cache_dir) / "query_vectors.json",
        "verbose": False,
        # 테스트에서는 실제로 대기하지 않는다.
        "min_call_interval": 0.0,
        "retry_base_delay": 0.0,
        "sleep_fn": lambda seconds: None,
    }
    options.update(kwargs)

    return QueryEmbedder(**options)

class CacheKeyTest(unittest.TestCase):
    def test_key_depends_on_question(self):
        self.assertNotEqual(
            cache_key("질문 A", "model", "abc"),
            cache_key("질문 B", "model", "abc"),
        )

    def test_key_depends_on_model(self):
        self.assertNotEqual(
            cache_key("질문", "model-a", "abc"),
            cache_key("질문", "model-b", "abc"),
        )

    def test_key_depends_on_base_url_identifier(self):
        self.assertNotEqual(
            cache_key("질문", "model", "abc"),
            cache_key("질문", "model", "xyz"),
        )

    def test_key_is_stable(self):
        self.assertEqual(
            cache_key("질문", "model", "abc"),
            cache_key("질문", "model", "abc"),
        )

    def test_base_url_identifier_is_hashed(self):
        secret_url = "https://gateway.example.com/v1/token-abc123"
        identifier = base_url_identifier(secret_url)

        self.assertNotIn("token-abc123", identifier)
        self.assertNotIn("gateway.example.com", identifier)
        self.assertEqual(len(identifier), 12)
        self.assertEqual(identifier, base_url_identifier(secret_url))

class RetryAfterTest(unittest.TestCase):
    def test_header_seconds(self):
        error = FakeRateLimitError(
            response=FakeResponse({"retry-after": "7"})
        )

        self.assertEqual(parse_retry_after(error), 7.0)

    def test_header_milliseconds(self):
        error = FakeRateLimitError(
            response=FakeResponse({"retry-after-ms": "2500"})
        )

        self.assertEqual(parse_retry_after(error), 2.5)

    def test_header_takes_priority_over_message(self):
        error = FakeRateLimitError(
            "try again in 60 seconds",
            response=FakeResponse({"retry-after": "3"}),
        )

        self.assertEqual(parse_retry_after(error), 3.0)

    def test_korean_message_minutes(self):
        error = FakeRateLimitError(
            "요청 빈도 초과입니다. 잠시 후(1분) 다시 시도해주세요."
        )

        self.assertEqual(parse_retry_after(error), 60.0)

    def test_no_hint_returns_none(self):
        self.assertIsNone(parse_retry_after(FakeRateLimitError("오류")))

class EmbedQuestionsTest(unittest.TestCase):
    def test_duplicate_questions_embedded_once(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            embeddings = FakeEmbeddings()
            embedder = make_embedder(cache_dir, embeddings)

            questions = ["질문1", "질문2", "질문1", "질문2", "질문3"]
            vectors = embedder.embed_questions(questions)

            embedded = [text for call in embeddings.calls for text in call]

            self.assertEqual(sorted(embedded), ["질문1", "질문2", "질문3"])
            self.assertEqual(embedder.stats["unique_questions"], 3)
            self.assertEqual(embedder.stats["duplicate_questions"], 2)
            self.assertEqual(set(vectors), {"질문1", "질문2", "질문3"})

    def test_batch_size_starts_at_five(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            embeddings = FakeEmbeddings()
            embedder = make_embedder(cache_dir, embeddings)

            self.assertEqual(embedder.batch_size, INITIAL_BATCH_SIZE)

            embedder.embed_questions([f"질문{i}" for i in range(12)])

            self.assertEqual(len(embeddings.calls[0]), INITIAL_BATCH_SIZE)
            self.assertEqual(embedder.stats["api_calls"], 3)

    def test_batch_size_shrinks_on_rate_limit(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            embeddings = FakeEmbeddings(fail_times=1)
            embedder = make_embedder(cache_dir, embeddings)

            embedder.embed_questions([f"질문{i}" for i in range(5)])

            self.assertEqual(embedder.stats["rate_limit_retries"], 1)
            self.assertLess(embedder.batch_size, INITIAL_BATCH_SIZE)
            self.assertGreaterEqual(embedder.batch_size, MIN_BATCH_SIZE)

    def test_batch_size_never_exceeds_max(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            embeddings = FakeEmbeddings()
            embedder = make_embedder(cache_dir, embeddings)

            embedder.embed_questions([f"질문{i}" for i in range(200)])

            self.assertLessEqual(embedder.batch_size, MAX_BATCH_SIZE)

    def test_cache_hit_avoids_api_call(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            first_embeddings = FakeEmbeddings()
            first = make_embedder(cache_dir, first_embeddings)
            questions = ["질문1", "질문2", "질문3"]
            first_vectors = first.embed_questions(questions)

            self.assertEqual(first.stats["cache_hits"], 0)
            self.assertGreater(first.stats["api_calls"], 0)

            second_embeddings = FakeEmbeddings()
            second = make_embedder(cache_dir, second_embeddings)
            second_vectors = second.embed_questions(questions)

            self.assertEqual(second.stats["api_calls"], 0)
            self.assertEqual(second.stats["cache_hits"], 3)
            self.assertEqual(second_embeddings.calls, [])
            self.assertEqual(first_vectors, second_vectors)

    def test_cache_miss_on_model_change(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            make_embedder(
                cache_dir, FakeEmbeddings()
            ).embed_questions(["질문1"])

            other_embeddings = FakeEmbeddings()
            other = make_embedder(
                cache_dir, other_embeddings, model="other-model"
            )
            other.embed_questions(["질문1"])

            self.assertEqual(other.stats["cache_hits"], 0)
            self.assertEqual(other.stats["api_calls"], 1)

    def test_cache_file_has_no_credentials(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            embedder = make_embedder(cache_dir, FakeEmbeddings())
            embedder.embed_questions(["질문1"])

            raw = (Path(cache_dir) / "query_vectors.json").read_text(
                encoding="utf-8"
            )
            data = json.loads(raw)

            self.assertNotIn("api_key", raw.lower())
            self.assertNotIn("authorization", raw.lower())
            self.assertNotIn("base_url\":", raw)

            entry = next(iter(data["entries"].values()))

            self.assertEqual(
                set(entry), {"model", "base_url_id", "question", "vector"}
            )

    def test_no_cache_option_skips_file(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            embedder = make_embedder(
                cache_dir, FakeEmbeddings(), use_cache=False
            )
            embedder.embed_questions(["질문1"])

            self.assertFalse(
                (Path(cache_dir) / "query_vectors.json").exists()
            )

    def test_corrupted_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_path = Path(cache_dir) / "query_vectors.json"
            cache_path.write_text("깨진 내용", encoding="utf-8")

            embedder = make_embedder(cache_dir, FakeEmbeddings())
            vectors = embedder.embed_questions(["질문1"])

            self.assertIn("질문1", vectors)
            self.assertEqual(embedder.stats["api_calls"], 1)

    def test_non_retryable_error_is_raised_immediately(self):
        class BrokenEmbeddings:
            def __init__(self):
                self.calls = 0

            def embed_documents(self, texts):
                self.calls += 1

                raise ValueError("설정 오류")

        with tempfile.TemporaryDirectory() as cache_dir:
            embeddings = BrokenEmbeddings()
            embedder = make_embedder(cache_dir, embeddings)

            with self.assertRaises(ValueError):
                embedder.embed_questions(["질문1"])

            self.assertEqual(embeddings.calls, 1)

    def test_report_lines_are_produced(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            embedder = make_embedder(cache_dir, FakeEmbeddings())
            embedder.embed_questions(["질문1", "질문1", "질문2"])

            report = embedder.report()

            self.assertTrue(report)
            self.assertTrue(all(isinstance(line, str) for line in report))

class VectorSearchContractTest(unittest.TestCase):
    def test_dense_search_by_vector_signature(self):
        import inspect

        from rag.retriever import dense_search_by_vector

        signature = inspect.signature(dense_search_by_vector)

        self.assertEqual(
            list(signature.parameters),
            ["query_vector", "top_k", "collection_name", "client"],
        )
        self.assertNotIn("embeddings", signature.parameters)

    def test_dense_search_still_accepts_question(self):
        import inspect

        from rag.retriever import dense_search

        signature = inspect.signature(dense_search)

        self.assertEqual(
            list(signature.parameters),
            ["question", "top_k", "collection_name", "embeddings", "client"],
        )
        self.assertEqual(
            signature.parameters["collection_name"].default,
            "beopjeong_law_v1",
        )

if __name__ == "__main__":
    unittest.main()
