"""
LLM Judge(eval/judge.py) 단위 테스트.

실제 LLM 호출 없이 가짜 LLM로 응답 파싱ㆍ프롬프트 구성만 검증한다.
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from eval.judge import JudgeParseError, build_judge_user_prompt, judge_answer, parse_judge_response


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content):
        self.content = content
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return FakeResponse(self.content)


class ParseJudgeResponseTest(unittest.TestCase):
    def test_parses_valid_json(self):
        text = (
            '{"correctness": 0.8, "faithfulness": 0.9, '
            '"hallucination_rate": 0.1, "reasoning": "근거를 잘 반영함"}'
        )

        result = parse_judge_response(text)

        self.assertEqual(result["correctness"], 0.8)
        self.assertEqual(result["faithfulness"], 0.9)
        self.assertEqual(result["hallucination_rate"], 0.1)

    def test_extracts_json_from_surrounding_text(self):
        text = (
            "다음은 평가 결과입니다.\n"
            '{"correctness": 1.0, "faithfulness": 1.0, "hallucination_rate": 0.0}\n'
            "추가 설명"
        )

        result = parse_judge_response(text)

        self.assertEqual(result["correctness"], 1.0)

    def test_missing_json_raises(self):
        with self.assertRaises(JudgeParseError):
            parse_judge_response("평가 결과를 JSON으로 드릴 수 없습니다.")

    def test_malformed_json_raises(self):
        with self.assertRaises(JudgeParseError):
            parse_judge_response("{correctness: 0.8 malformed}")

    def test_parses_refused_field(self):
        text = (
            '{"correctness": 0.0, "faithfulness": 0.0, '
            '"hallucination_rate": 0.0, "refused": true}'
        )

        result = parse_judge_response(text)

        self.assertTrue(result["refused"])

    def test_refused_defaults_to_false(self):
        text = '{"correctness": 0.8, "faithfulness": 0.8, "hallucination_rate": 0.1}'

        result = parse_judge_response(text)

        self.assertFalse(result["refused"])


class JudgeAnswerTest(unittest.TestCase):
    def test_includes_question_points_sources_answer_in_prompt(self):
        fake_llm = FakeLLM(
            '{"correctness": 0.5, "faithfulness": 0.5, "hallucination_rate": 0.5}'
        )

        judge_answer(
            question="고영향 인공지능이란 무엇인가요?",
            answer_points=["사람의 생명에 중대한 영향을 미치는 AI"],
            answer_text="고영향 인공지능은 ...",
            source_contents=["4. 고영향 인공지능이란 ..."],
            llm=fake_llm,
        )

        user_content = fake_llm.last_messages[1]["content"]

        self.assertIn("고영향 인공지능이란 무엇인가요?", user_content)
        self.assertIn("사람의 생명에 중대한 영향을 미치는 AI", user_content)
        self.assertIn("고영향 인공지능은 ...", user_content)

    def test_result_includes_judge_metadata(self):
        fake_llm = FakeLLM(
            '{"correctness": 0.5, "faithfulness": 0.5, "hallucination_rate": 0.5}'
        )

        result = judge_answer("질문", [], "답변", [], llm=fake_llm)

        self.assertIn("judge_model", result)
        self.assertIn("prompt_version", result)


if __name__ == "__main__":
    unittest.main()
