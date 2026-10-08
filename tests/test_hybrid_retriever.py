"""
Hybrid Retrieval(rag/hybrid_retriever.py) 단위 테스트.

실제 Qdrant/Embedding API를 호출하지 않도록 BM25Index와 Dense 검색
결과를 가짜 데이터로 대체하고, RRF 계산ㆍ토큰화ㆍParent 다양화
로직만 검증한다.
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.hybrid_retriever import (
    apply_parent_diversification,
    reciprocal_rank_fusion,
    tokenize,
)


class TokenizeTest(unittest.TestCase):
    def test_preserves_article_reference_tokens(self):
        tokens = tokenize("제2조제4호에서 고영향 인공지능을 정의한다.")

        self.assertIn("제2조", tokens)
        self.assertIn("제4호", tokens)

    def test_empty_text_returns_empty_list(self):
        self.assertEqual(tokenize(""), [])
        self.assertEqual(tokenize(None), [])

    def test_question_and_document_use_same_function(self):
        # 질문과 문서에 동일한 정규화ㆍ토큰화가 적용되어 공통 토큰이
        # 나오는지 확인한다(형태소 분석 결과라 완전 일치 문자열은 아닐 수 있음).
        question_tokens = set(tokenize("고영향 인공지능이란 무엇인가요?"))
        document_tokens = set(tokenize("고영향 인공지능이란 사람의 생명에 영향을 미치는 것이다."))

        self.assertTrue(question_tokens & document_tokens)
        self.assertIn("인공", question_tokens)
        self.assertIn("인공", document_tokens)


class ReciprocalRankFusionTest(unittest.TestCase):
    def make_result(self, chunk_id, rank, **extra):
        base = {
            "id": chunk_id,
            "rank": rank,
            "parent_id": f"parent-{chunk_id}",
            "article": "제2조",
            "article_title": "",
            "paragraph": "",
            "item": "",
            "subitem": "",
            "granularity": "item",
            "source_text": "text",
            "text": "text",
            "document_type": "main",
            "chunk_role": "",
        }
        base.update(extra)

        return base

    def test_rrf_formula(self):
        dense = [self.make_result("A", 1), self.make_result("B", 2)]
        bm25 = [self.make_result("B", 1), self.make_result("A", 2)]

        fused = reciprocal_rank_fusion(dense, bm25, k=60)

        scores = {item["id"]: item["rrf_score"] for item in fused}

        # A: dense rank1 + bm25 rank2 = 1/61 + 1/62
        # B: dense rank2 + bm25 rank1 = 1/62 + 1/61
        self.assertAlmostEqual(scores["A"], scores["B"])
        self.assertAlmostEqual(scores["A"], 1 / 61 + 1 / 62)

    def test_deduplicates_by_chunk_id(self):
        dense = [self.make_result("A", 1)]
        bm25 = [self.make_result("A", 1)]

        fused = reciprocal_rank_fusion(dense, bm25, k=60)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["dense_rank"], 1)
        self.assertEqual(fused[0]["bm25_rank"], 1)

    def test_only_in_one_side_still_included(self):
        dense = [self.make_result("A", 1)]
        bm25 = [self.make_result("B", 1)]

        fused = reciprocal_rank_fusion(dense, bm25, k=60)
        ids = {item["id"] for item in fused}

        self.assertEqual(ids, {"A", "B"})

        by_id = {item["id"]: item for item in fused}
        self.assertIsNone(by_id["A"]["bm25_rank"])
        self.assertIsNone(by_id["B"]["dense_rank"])

    def test_rank_assigned_by_score_order(self):
        dense = [self.make_result("A", 1), self.make_result("B", 5)]
        bm25 = [self.make_result("A", 10), self.make_result("B", 1)]

        fused = reciprocal_rank_fusion(dense, bm25, k=60)

        # B: dense rank5(1/65) + bm25 rank1(1/61) vs A: dense rank1(1/61) + bm25 rank10(1/70)
        self.assertEqual(fused[0]["rank"], 1)
        self.assertEqual(len(fused), 2)


class ParentDiversificationTest(unittest.TestCase):
    def test_caps_results_per_parent(self):
        fused = [
            {"id": "1", "parent_id": "P1", "rrf_score": 0.9},
            {"id": "2", "parent_id": "P1", "rrf_score": 0.8},
            {"id": "3", "parent_id": "P2", "rrf_score": 0.7},
        ]

        diversified = apply_parent_diversification(fused, max_per_parent=1)

        self.assertEqual([item["id"] for item in diversified], ["1", "3"])
        self.assertEqual(diversified[0]["rank"], 1)
        self.assertEqual(diversified[1]["rank"], 2)

    def test_does_not_mutate_input(self):
        fused = [{"id": "1", "parent_id": "P1", "rrf_score": 0.9}]

        apply_parent_diversification(fused, max_per_parent=1)

        self.assertNotIn("rank", fused[0])


if __name__ == "__main__":
    unittest.main()
