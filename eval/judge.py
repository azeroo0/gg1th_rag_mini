"""
LLM Judge: 정답성(correctness)ㆍ근거충실성(faithfulness)ㆍ환각비율(hallucination_rate) 평가.

고정된 평가 프롬프트와 temperature=0을 사용해 재현성을 확보하고,
평가 결과를 캐시해 동일 질문ㆍ답변을 반복 평가하지 않는다
(LLM_BASE_URL 게이트웨이의 분당 30요청 제한을 고려한 비용/시간 최적화).

지표 정의(계산 기준):
- correctness(정답성): 답변이 Golden Set의 answer_points(정답 요점)를
  얼마나 포함하는지 0.0~1.0으로 Judge가 판단한 값.
- faithfulness(근거충실성): 답변의 주장들이 [제공된 근거](실제 검색된
  Child의 source_text)에 의해 뒷받침되는 정도 0.0~1.0.
- hallucination_rate(환각비율): 답변 주장 중 [제공된 근거]로 뒷받침되지
  않는 주장의 비율 0.0~1.0 (faithfulness와 독립적으로 Judge가 추정).
- refused(거절 여부): 답변이 질문의 핵심 내용에 대한 실질적 정보 제공을
  거부하거나 보류했는지 여부(true/false). 일부 세부사항만 모른다고
  밝히면서도 핵심 질문에는 실질적으로 답했다면 false로 판단한다.

correctness/faithfulness/hallucination_rate는 answerable=true 질문에만
의미가 있다(거절 응답에는 판단할 실질적 주장이 없으므로 집계에서 N/A로
둔다). refused는 모든 질문에 대해 판단한다(거절 정밀도/재현율 계산용).
추가 API 호출 없이 이 Judge 호출 한 번으로 네 항목을 함께 얻는다.
"""

import json
import re
import time

from common.ai_model import get_llm_model
from common.config import MODEL as LLM_MODEL_NAME

JUDGE_MODEL = LLM_MODEL_NAME
JUDGE_TEMPERATURE = 0
JUDGE_MAX_TOKENS = 512
JUDGE_PROMPT_VERSION = "v1"

JUDGE_SYSTEM_PROMPT = """당신은 한국 법률 QA 시스템의 답변 품질을 채점하는 평가자다.
아래 [질문], [필수 포함 사항], [제공된 근거], [평가할 답변]을 보고
다른 설명 없이 JSON 객체만 출력한다.

- correctness: 답변이 [필수 포함 사항]의 내용을 얼마나 포함하는지 0.0(전혀 없음)~1.0(완전히 포함)
- faithfulness: 답변의 주장들이 [제공된 근거]에 의해 실제로 뒷받침되는 정도 0.0~1.0
- hallucination_rate: 답변의 주장 중 [제공된 근거]로 뒷받침되지 않는 주장의 비율 0.0(없음)~1.0(전부)
- refused: 답변이 질문의 핵심 내용에 대한 실질적 정보 제공을 거부하거나
  보류했는지 여부(true/false). 일부 세부사항만 모른다고 밝히면서도
  핵심 질문에는 실질적으로 답했다면 false로 판단한다.

JSON 형식: {"correctness": 0.0, "faithfulness": 0.0, "hallucination_rate": 0.0, "refused": false, "reasoning": "짧은 한국어 설명 1~2문장"}"""


class JudgeParseError(RuntimeError):
    pass


def build_judge_user_prompt(question, answer_points, answer_text, source_contents):
    points_text = "\n".join(f"- {point}" for point in answer_points) or "(없음)"
    sources_text = "\n".join(f"- {content}" for content in source_contents) or "(없음)"

    return (
        f"[질문]\n{question}\n\n"
        f"[필수 포함 사항(정답 요점)]\n{points_text}\n\n"
        f"[제공된 근거]\n{sources_text}\n\n"
        f"[평가할 답변]\n{answer_text}"
    )


def parse_judge_response(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise JudgeParseError(f"Judge 응답에서 JSON을 찾을 수 없습니다: {text!r}")

    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError as error:
        raise JudgeParseError(f"Judge 응답 JSON 파싱 실패: {error}") from error

    return {
        "correctness": float(parsed.get("correctness", 0.0)),
        "faithfulness": float(parsed.get("faithfulness", 0.0)),
        "hallucination_rate": float(parsed.get("hallucination_rate", 0.0)),
        "refused": bool(parsed.get("refused", False)),
        "reasoning": str(parsed.get("reasoning", "")),
    }


def judge_answer(question, answer_points, answer_text, source_contents, llm=None):
    """
    답변 1건을 Judge로 평가한다. 반환값에 judge_model/prompt_version을
    함께 담아 추후 재현/추적할 수 있게 한다.
    """

    if llm is None:
        llm = get_llm_model(temperature=JUDGE_TEMPERATURE, max_tokens=JUDGE_MAX_TOKENS)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_judge_user_prompt(
                question, answer_points, answer_text, source_contents
            ),
        },
    ]

    started = time.monotonic()
    response = llm.invoke(messages)
    elapsed = round(time.monotonic() - started, 3)

    result = parse_judge_response(response.content)
    result["judge_model"] = JUDGE_MODEL
    result["prompt_version"] = JUDGE_PROMPT_VERSION
    result["elapsed_seconds"] = elapsed

    return result
