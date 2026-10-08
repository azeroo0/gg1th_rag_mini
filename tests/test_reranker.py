"""
Qwen Reranker(rag/reranker.py) 단위 테스트.

실제 모델 로딩/추론은 느리고 GPU/torch 설치에 의존하므로, _score_batch를
가짜 함수로 대체해 rerank() 정렬ㆍ필드 부여 로직만 검증한다.
실제 모델 로딩까지 확인하는 통합 테스트는 RUN_RERANKER_TESTS=1 환경변수가
있을 때만 실행한다(torch/transformers 다운로드 및 GPU 추론 필요).
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.reranker import QwenReranker, is_reranker_available


class RerankLogicTest(unittest.TestCase):
    """
    _load/_score_batch를 가짜로 대체해 rerank()의 정렬ㆍ필드 로직만 확인한다.
    """

    def make_reranker_with_fake_scores(self, score_map):
        reranker = QwenReranker()
        reranker._model = object()  # _load()가 다시 로딩하지 않도록 표시

        def fake_score(query, documents, instruction=None):
            return [score_map[doc] for doc in documents]

        reranker.score = fake_score

        return reranker

    def test_rerank_sorts_by_score_descending(self):
        candidates = [
            {"id": "A", "rank": 1, "text": "doc-a"},
            {"id": "B", "rank": 2, "text": "doc-b"},
            {"id": "C", "rank": 3, "text": "doc-c"},
        ]

        reranker = self.make_reranker_with_fake_scores(
            {"doc-a": 0.2, "doc-b": 0.9, "doc-c": 0.5}
        )

        reranked = reranker.rerank("query", candidates, text_field="text")

        self.assertEqual([r["id"] for r in reranked], ["B", "C", "A"])
        self.assertEqual(reranked[0]["rerank_rank"], 1)
        self.assertEqual(reranked[0]["original_rank"], 2)

    def test_rerank_preserves_original_fields(self):
        candidates = [{"id": "A", "rank": 1, "text": "doc-a", "parent_id": "P1"}]
        reranker = self.make_reranker_with_fake_scores({"doc-a": 0.7})

        reranked = reranker.rerank("query", candidates, text_field="text")

        self.assertEqual(reranked[0]["parent_id"], "P1")
        self.assertAlmostEqual(reranked[0]["rerank_score"], 0.7)

    def test_rerank_empty_candidates_returns_empty(self):
        reranker = self.make_reranker_with_fake_scores({})

        self.assertEqual(reranker.rerank("query", [], text_field="text"), [])


class LazyLoadingTest(unittest.TestCase):
    def test_model_not_loaded_on_construction(self):
        reranker = QwenReranker()

        self.assertFalse(reranker.is_loaded())
        self.assertFalse(reranker.stats["loaded"])


@unittest.skipUnless(
    os.environ.get("RUN_RERANKER_TESTS") == "1",
    "RUN_RERANKER_TESTS=1일 때만 실행 (실제 모델 다운로드/추론 필요)",
)
class RealModelTest(unittest.TestCase):
    def test_real_model_scores_relevant_higher(self):
        self.assertTrue(is_reranker_available())

        reranker = QwenReranker()

        query = "고영향 인공지능이란 무엇인가요?"
        relevant = (
            "고영향 인공지능이란 사람의 생명, 신체의 안전 및 기본권에 중대한 "
            "영향을 미치거나 위험을 초래할 우려가 있는 인공지능시스템이다."
        )
        irrelevant = "제1조(목적) 이 법은 인공지능의 건전한 발전을 위한 것이다."

        scores = reranker.score(query, [relevant, irrelevant])

        self.assertGreater(scores[0], scores[1])
        self.assertTrue(reranker.is_loaded())


if __name__ == "__main__":
    unittest.main()
