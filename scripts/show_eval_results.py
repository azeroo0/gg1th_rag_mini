"""
기존 평가 결과(JSON)를 읽어 터미널에 표 형식으로 출력한다.

평가를 다시 실행하거나 Embedding API를 호출하지 않으며,
eval/results/<version>/metrics.json을 그대로 읽어서 보여주기만 한다.

출력은 두 가지 표로 구성된다.
- 답변 평가 지표: 정답성, 근거충실성, 인용정밀도, 인용재현율, 환각비율,
  거절정밀도, 거절재현율, 오거절, 놓친거절. eval/evaluate_answers.py가
  만든 <version>/answer_metrics.json에서 읽는다. 아직 계산되지 않았거나
  판단 근거가 부족한 지표는 N/A로 표시한다.
  (과거 코드의 '총나절' 표기는 '놓친거절'(answerable=false인데 답변한
  건수)을 뜻했으므로 이 이름으로 통일했다.)
- 검색 평가 지표: Hit@1, Hit@3, Hit@5, Recall@5, MRR.
  평가 수준(article_level, provision_level, precise_diagnosis)별로
  동일한 형식의 표를 각각 출력한다. <version>/metrics.json에서 읽는다.
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from eval.result_io import RESULTS_DIR

DEFAULT_VERSION = "dense_v2"

RETRIEVAL_LEVELS = [
    ("article_level", "검색 평가 지표 - article_level"),
    ("provision_level", "검색 평가 지표 - provision_level"),
    ("precise_diagnosis", "검색 평가 지표 - precise_diagnosis"),
]

RETRIEVAL_METRICS = [
    ("Hit@1", "hit_at_1", "float"),
    ("Hit@3", "hit_at_3", "float"),
    ("Hit@5", "hit_at_5", "float"),
    ("Recall@5", "recall_at_5", "float"),
    ("MRR", "mrr", "float"),
]

# 답변 평가 지표. 아직 답변 평가 기능이 없어 answer_evaluation 키가
# metrics.json에 없으므로 현재는 전부 N/A로 출력된다.
# 추후 답변 평가가 추가되면 metrics.json에 "answer_evaluation" 객체로
# 아래 key들이 채워진다고 가정하고 그대로 표시한다.
ANSWER_METRICS = [
    ("정답성", "correctness", "float"),
    ("근거충실성", "faithfulness", "float"),
    ("인용정밀도", "citation_precision", "float"),
    ("인용재현율", "citation_recall", "float"),
    ("환각비율", "hallucination_rate", "float"),
    ("거절정밀도", "refusal_precision", "float"),
    ("거절재현율", "refusal_recall", "float"),
    ("오거절", "false_refusal_count", "int"),
    ("놓친거절", "missed_refusal_count", "int"),
]

AVAILABLE_VERSIONS = ["dense_v1", "dense_v2", "hybrid", "hybrid_rerank"]

COLUMN_GAP = 2


def display_width(text):
    width = 0

    for char in text:
        if unicodedata.east_asian_width(char) in ("W", "F"):
            width += 2
        else:
            width += 1

    return width


def format_value(value, value_type):
    if value is None:
        return "N/A"

    if value_type == "int":
        return str(int(value))

    return f"{float(value):.3f}"


def pad_cell(text, width):
    return text + " " * (width - display_width(text) + COLUMN_GAP)


def print_row(cells, widths):
    print("".join(pad_cell(cell, width) for cell, width in zip(cells, widths)))


def print_metric_table(title, labels, value_cells):
    widths = [
        max(display_width(label), display_width(value))
        for label, value in zip(labels, value_cells)
    ]

    print(f"[{title}]")
    print_row(labels, widths)
    print_row(value_cells, widths)
    print()


def load_metrics(version):
    path = RESULTS_DIR / version / "metrics.json"

    if not path.exists():
        print(f"결과 파일이 없습니다: {path}")
        raise SystemExit(2)

    return json.loads(path.read_text(encoding="utf-8"))


def load_answer_evaluation(version):
    """
    eval/evaluate_answers.py가 저장한 answer_metrics.json을 읽는다.
    없으면 빈 dict를 반환하고 호출부에서 전부 N/A로 표시한다.
    """

    path = RESULTS_DIR / version / "answer_metrics.json"

    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))

    return payload.get("answer_evaluation") or {}


def answer_value_cells(answer_evaluation):
    return [
        format_value(answer_evaluation.get(key), value_type)
        for _, key, value_type in ANSWER_METRICS
    ]


def retrieval_value_cells(level_metrics):
    return [
        format_value(level_metrics.get(key), value_type)
        for _, key, value_type in RETRIEVAL_METRICS
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="저장된 평가 결과(metrics.json)를 표 형식으로 출력"
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help=(
            f"출력할 결과 버전 디렉터리 이름 (기본 {DEFAULT_VERSION}, "
            f"사용 가능: {AVAILABLE_VERSIONS})"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    payload = load_metrics(args.version)
    evaluation_metrics = payload.get("evaluation_metrics") or {}
    answer_evaluation = load_answer_evaluation(args.version)

    print(f"===== 평가 결과: {args.version} =====")
    print()

    print_metric_table(
        "답변 평가 지표",
        [label for label, _, _ in ANSWER_METRICS],
        answer_value_cells(answer_evaluation),
    )

    for level_key, title in RETRIEVAL_LEVELS:
        level_metrics = evaluation_metrics.get(level_key) or {}

        print_metric_table(
            title,
            [label for label, _, _ in RETRIEVAL_METRICS],
            retrieval_value_cells(level_metrics),
        )


if __name__ == "__main__":
    main()
