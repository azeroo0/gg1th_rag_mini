"""
Grounded Answer Pipeline(rag/pipeline.py) 단위 테스트.

실제 Qdrant/Embedding/LLM 호출 없이, retrieve()와 LLM을 가짜로 대체해
Parent Context Expansion과 출처 구성 로직만 검증한다.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.pipeline import build_context, empty_answer, generate_answer, get_parent_map


def make_result(**overrides):
    base = {
        "id": "21311-20260721-제2조-p1-i4-definition",
        "parent_id": "21311-20260721-제2조",
        "article": "제2조",
        "article_title": "정의",
        "paragraph": "",
        "item": "4.",
        "subitem": "",
        "granularity": "item",
        "chunk_role": "definition_only",
        "source_text": "4. \"고영향 인공지능\"이란 ...",
        "text": "제2조(정의)\n제4호 고영향 인공지능\n\n4. \"고영향 인공지능\"이란 ...",
        "document_type": "main",
    }
    base.update(overrides)

    return base


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content="테스트 답변"):
        self.content = content
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return FakeResponse(self.content)


class BuildContextTest(unittest.TestCase):
    def test_uses_real_parent_map(self):
        parent_map = get_parent_map()
        self.assertIn("21311-20260721-제2조", parent_map)

    def test_deduplicates_parent_block_but_keeps_all_sources(self):
        results = [
            make_result(id="A", item="4.", subitem=""),
            make_result(id="B", item="4.", subitem="가", granularity="subitem"),
        ]

        context_text, sources = build_context(results, max_sources=4)

        self.assertEqual(len(sources), 2)
        # 같은 parent_id(제2조)이므로 Context 블록은 1개여야 한다.
        self.assertEqual(context_text.count("[근거"), 1)

    def test_long_parent_falls_back_to_child_text(self):
        # 제2조 Parent 원문은 길어서(MAX_PARENT_CHARS 초과) 조문 전체 대신
        # Child의 text(관련 항목 발췌)가 쓰여야 한다.
        results = [make_result()]

        context_text, _ = build_context(results)

        self.assertIn("발췌", context_text)
        self.assertNotIn("가.", context_text)  # 가목 이하 열거 목록이 없어야 함

    def test_sources_include_required_fields(self):
        results = [make_result()]

        _, sources = build_context(results)

        source = sources[0]

        for field in (
            "law_name", "article", "article_title", "paragraph",
            "item", "subitem", "content", "effective_date",
        ):
            self.assertIn(field, source)

        self.assertEqual(source["law_name"], "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법")
        self.assertEqual(source["effective_date"], "20260721")


class GenerateAnswerTest(unittest.TestCase):
    def test_empty_results_returns_refusal_without_llm_call(self):
        fake_llm = FakeLLM()

        with patch("rag.pipeline.retrieve", return_value=([], None)):
            result = generate_answer(
                "전혀 관련 없는 질문", mode="hybrid", llm=fake_llm
            )

        self.assertEqual(result["sources"], [])
        self.assertIsNone(fake_llm.last_messages)
        self.assertIn("보류", result["answer"])

    def test_blank_question_raises_value_error(self):
        with self.assertRaises(ValueError):
            generate_answer("   ", mode="hybrid", llm=FakeLLM())

    def test_invalid_mode_raises_value_error(self):
        with patch("rag.pipeline.retrieve", side_effect=ValueError("알 수 없는 mode")):
            with self.assertRaises(ValueError):
                generate_answer("질문", mode="not_a_mode", llm=FakeLLM())

    def test_llm_receives_context_and_question(self):
        fake_llm = FakeLLM(content="고영향 인공지능은 ... [제2조제4호]")
        results = [make_result()]

        with patch("rag.pipeline.retrieve", return_value=(results, None)):
            result = generate_answer(
                "고영향 인공지능이란 무엇인가요?", mode="hybrid", llm=fake_llm
            )

        self.assertIsNotNone(fake_llm.last_messages)
        user_content = fake_llm.last_messages[1]["content"]

        self.assertIn("[법률 근거]", user_content)
        self.assertIn("고영향 인공지능이란 무엇인가요?", user_content)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["law_number"], "21311")

    def test_sources_never_missing_key_when_present(self):
        fake_llm = FakeLLM()
        results = [make_result()]

        with patch("rag.pipeline.retrieve", return_value=(results, None)):
            result = generate_answer("질문", mode="hybrid", llm=fake_llm)

        self.assertTrue(len(result["sources"]) >= 1)

    def test_rerank_info_is_passed_through(self):
        fake_llm = FakeLLM()
        results = [make_result()]
        rerank_info = {"candidate_count": 8, "reranker_model": "Qwen/Qwen3-Reranker-0.6B"}

        with patch("rag.pipeline.retrieve", return_value=(results, rerank_info)):
            result = generate_answer("질문", mode="hybrid_rerank", llm=fake_llm)

        self.assertEqual(result["rerank"], rerank_info)


class EmptyAnswerTest(unittest.TestCase):
    def test_contains_law_version_fields(self):
        payload = empty_answer("hybrid")

        self.assertEqual(payload["law_number"], "21311")
        self.assertEqual(payload["sources"], [])


if __name__ == "__main__":
    unittest.main()
