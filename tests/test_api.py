"""
FastAPI 앱(app/main.py) 단위 테스트.

실제 Qdrant/Embedding/LLM 호출 없이, generate_answer와 Qdrant 클라이언트를
가짜 객체로 대체해 요청/응답 스키마와 예외 처리를 검증한다.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient

import app.main as main_module

FAKE_ANSWER = {
    "answer": "고영향 인공지능은 제2조제4호에 정의되어 있습니다.",
    "sources": [
        {
            "law_name": "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법",
            "article": "제2조",
            "article_title": "정의",
            "paragraph": "",
            "item": "4",
            "subitem": "",
            "content": "4. \"고영향 인공지능\"이란 ...",
            "effective_date": "20260721",
        }
    ],
    "mode": "hybrid",
    "retrieval_count": 4,
    "rerank": None,
    "elapsed_seconds": 1.23,
    "law_name": "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법",
    "law_number": "21311",
    "effective_date": "20260721",
}


class FakeCollections:
    def __init__(self, names):
        self.collections = [type("C", (), {"name": n})() for n in names]


class FakeQdrantClient:
    def get_collections(self):
        return FakeCollections(["beopjeong_law_v1", "beopjeong_law_v2"])


class FailingQdrantClient:
    def get_collections(self):
        raise RuntimeError("connection refused")


class AskEndpointTest(unittest.TestCase):
    def setUp(self):
        main_module._resources["qdrant_client"] = FakeQdrantClient()
        main_module._resources["embeddings"] = object()
        main_module._resources["llm"] = object()
        self.client = TestClient(main_module.app)

    def test_ask_returns_answer_and_sources(self):
        with patch.object(
            main_module, "generate_answer", return_value=dict(FAKE_ANSWER)
        ):
            response = self.client.post(
                "/ask", json={"question": "고영향 인공지능이란 무엇인가요?"}
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["mode"], "hybrid")
        self.assertEqual(len(body["sources"]), 1)
        self.assertIn("고영향", body["answer"])

    def test_ask_accepts_question_only_request(self):
        # 기존 question-only 요청(mode 없이)이 그대로 동작해야 한다.
        with patch.object(
            main_module, "generate_answer", return_value=dict(FAKE_ANSWER)
        ) as mock_generate:
            response = self.client.post(
                "/ask", json={"question": "고영향 인공지능이란 무엇인가요?"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_generate.call_args.kwargs["mode"], "hybrid")

    def test_ask_with_hybrid_rerank_mode(self):
        fake = dict(FAKE_ANSWER)
        fake["mode"] = "hybrid_rerank"

        with patch.object(main_module, "generate_answer", return_value=fake) as mock_generate:
            response = self.client.post(
                "/ask",
                json={
                    "question": "고영향 인공지능이란 무엇인가요?",
                    "mode": "hybrid_rerank",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_generate.call_args.kwargs["mode"], "hybrid_rerank")

    def test_blank_question_returns_400(self):
        response = self.client.post("/ask", json={"question": "   "})

        self.assertEqual(response.status_code, 400)

    def test_missing_question_returns_422(self):
        response = self.client.post("/ask", json={})

        self.assertEqual(response.status_code, 422)

    def test_invalid_mode_returns_400(self):
        response = self.client.post(
            "/ask",
            json={"question": "질문", "mode": "not_a_real_mode"},
        )

        self.assertEqual(response.status_code, 400)

    def test_no_sources_returns_empty_list_not_missing(self):
        fake = dict(FAKE_ANSWER)
        fake["sources"] = []
        fake["retrieval_count"] = 0

        with patch.object(main_module, "generate_answer", return_value=fake):
            response = self.client.post("/ask", json={"question": "관련 없는 질문"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["sources"], [])

    def test_pipeline_exception_returns_502(self):
        with patch.object(
            main_module, "generate_answer", side_effect=RuntimeError("llm down")
        ):
            response = self.client.post(
                "/ask", json={"question": "고영향 인공지능이란 무엇인가요?"}
            )

        self.assertEqual(response.status_code, 502)


class HealthEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main_module.app)

    def test_health_ok_when_qdrant_reachable(self):
        main_module._resources["qdrant_client"] = FakeQdrantClient()

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["qdrant"], "ok")
        self.assertIn("beopjeong_law_v2", body["qdrant_collections"])

    def test_health_degraded_when_qdrant_unreachable(self):
        main_module._resources["qdrant_client"] = FailingQdrantClient()

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual(body["status"], "degraded")
        self.assertEqual(body["qdrant"], "error")


if __name__ == "__main__":
    unittest.main()
