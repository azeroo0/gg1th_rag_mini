"""
Dense v2 / Hybrid / Hybrid+Reranking 최종 비교표를 만든다.

이미 저장된 metrics.json(검색 평가)과 answer_metrics.json(답변 평가)만
읽어서 표로 합치며, 평가를 다시 실행하거나 API를 호출하지 않는다.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from eval.result_io import RESULTS_DIR, now_timestamp, save_text

VERSIONS = ["dense_v2", "hybrid", "hybrid_rerank"]
VERSION_LABELS = {
    "dense_v2": "Dense v2",
    "hybrid": "Hybrid (Dense+BM25+RRF)",
    "hybrid_rerank": "Hybrid+Reranking",
}

RETRIEVAL_METRICS = [
    ("Hit@1", "hit_at_1"),
    ("Hit@3", "hit_at_3"),
    ("Hit@5", "hit_at_5"),
    ("Recall@5", "recall_at_5"),
    ("MRR", "mrr"),
]

ANSWER_METRICS = [
    ("정답성", "correctness"),
    ("근거충실성", "faithfulness"),
    ("인용정밀도", "citation_precision"),
    ("인용재현율", "citation_recall"),
    ("환각비율", "hallucination_rate"),
    ("거절정밀도", "refusal_precision"),
    ("거절재현율", "refusal_recall"),
    ("오거절", "false_refusal_count"),
    ("놓친거절", "missed_refusal_count"),
]


def load_json(path):
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value):
    if value is None:
        return "N/A"

    if isinstance(value, float):
        return f"{value:.3f}"

    return str(value)


def main():
    retrieval = {}
    answers = {}

    for version in VERSIONS:
        metrics_payload = load_json(RESULTS_DIR / version / "metrics.json")
        retrieval[version] = (
            (metrics_payload or {}).get("evaluation_metrics", {}).get("provision_level", {})
        )

        answer_payload = load_json(RESULTS_DIR / version / "answer_metrics.json")
        answers[version] = (answer_payload or {}).get("answer_evaluation", {})

    lines = [
        "# 최종 비교: Dense v2 vs Hybrid vs Hybrid+Reranking",
        "",
        f"- 실행 시각: {now_timestamp()}",
        "- 검색 평가 지표는 provision_level(조문ㆍ항ㆍ호ㆍ목 경로 기준) 기준이다.",
        "",
        "## 검색 성능",
        "",
        "| 지표 | " + " | ".join(VERSION_LABELS[v] for v in VERSIONS) + " |",
        "| --- | " + " | ".join(["---"] * len(VERSIONS)) + " |",
    ]

    for label, key in RETRIEVAL_METRICS:
        row = [label] + [fmt(retrieval[v].get(key)) for v in VERSIONS]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## 답변 품질 (Hybrid / Hybrid+Reranking만 해당, dense_v2는 Answer Pipeline 대상 아님)",
        "",
        "| 지표 | " + " | ".join(VERSION_LABELS[v] for v in VERSIONS) + " |",
        "| --- | " + " | ".join(["---"] * len(VERSIONS)) + " |",
    ]

    for label, key in ANSWER_METRICS:
        row = [label] + [fmt(answers[v].get(key)) for v in VERSIONS]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "주의: 지표 값 차이만으로 성능이 개선되었다고 단정하지 말고,",
        "eval/results/hybrid/summary.md, eval/results/hybrid_rerank/summary.md의",
        "질문별 개선/악화 목록을 함께 확인한다.",
        "",
    ]

    text = "\n".join(lines)

    path = save_text(
        RESULTS_DIR / "comparisons" / "final_comparison.md", text, now_timestamp()
    )

    print(text)
    print("\n저장 파일:", path.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
