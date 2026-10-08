"""
Dense v1(beopjeong_law_v1)과 Dense v2(beopjeong_law_v2)의
검색 결과를 동일 질문ㆍ동일 Embedding 모델로 비교한다.

정의 전용 Child 추가 효과를 확인하기 위한 진단용 스크립트이며,
Golden Set 전체 평가는 eval/evaluate.py가 담당한다.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common.ai_model import get_embedding_model
from common.config import EMBEDDING_MODEL
from common.qdrant import get_qdrant_client
from eval.golden_schema import (
    matches,
    parse_reference,
    result_reference,
)
from eval.result_io import RESULTS_DIR, now_timestamp, save_json, save_text
from rag.query_embedder import QueryEmbedder
from rag.retriever import dense_search_by_vector, format_article_path

V1_COLLECTION = "beopjeong_law_v1"
V2_COLLECTION = "beopjeong_law_v2"

DEFINITION_CHILD_ID = "21311-20260721-제2조-p1-i4-definition"
ARTICLE2_ITEM4_ID = "21311-20260721-제2조-p1-i4"

TOP_K_SHALLOW = 5
TOP_K_DEEP = 30

# 정의 질문(1~3, 5)과 사업자 의무 질문(4)을 함께 확인하여
# 정의 질문 개선이 의무 질문 성능을 떨어뜨리지 않는지 본다.
# expected_paths는 조문 경로로 지정하며, 정답이 지정한 수준까지만 비교한다.
# (예: '제2조제4호'이면 제2조제4호를 가리키는 모든 Child가 적중이다.)
QUESTIONS = [
    {
        "qid": "D1",
        "question": "고영향 인공지능이란 무엇인가요?",
        "intent": "definition",
        "expected_paths": ["제2조제4호"],
    },
    {
        "qid": "D2",
        "question": "고영향 인공지능은 어떤 분야에서 사용되나요?",
        "intent": "definition_scope",
        "expected_paths": ["제2조제4호"],
    },
    {
        "qid": "D3",
        "question": "채용 심사에 사용하는 AI도 고영향 인공지능인가요?",
        "intent": "definition_apply",
        "expected_paths": ["제2조제4호사목"],
    },
    {
        "qid": "D4",
        "question": "고영향 인공지능 사업자는 어떤 조치를 해야 하나요?",
        "intent": "obligation",
        "expected_paths": ["제34조제1항"],
    },
    {
        "qid": "D5",
        "question": "생성형 인공지능이란 무엇인가요?",
        "intent": "definition",
        "expected_paths": ["제2조제5호"],
    },
]

def simplify(result):
    return {
        "rank": result["rank"],
        "score": round(result["score"], 6),
        "id": result["id"],
        "parent_id": result["parent_id"],
        "article_path": format_article_path(result),
        "article": result["article"],
        "paragraph": result["paragraph"],
        "item": result["item"],
        "subitem": result["subitem"],
        "granularity": result["granularity"],
        "chunk_role": result["chunk_role"],
        "source_text": result["source_text"],
    }

def rank_of(results, chunk_id):
    for result in results:
        if result["id"] == chunk_id:
            return result["rank"]

    return None

def expected_rank(results, expected_path):
    """
    조문 경로 정답을 만족하는 검색 결과의 최상위 순위를 반환한다.
    """

    gold = parse_reference(expected_path)

    for result in results:
        if matches(gold, result_reference(result)):
            return result["rank"]

    return None

def run_collection(collection_name, question_vectors, client):
    """
    질문별로 Top-30을 한 번 검색하고 Top-5는 그 앞부분을 사용한다.
    (Top-5는 Top-30의 prefix이므로 동일한 결과다.)

    질문 벡터는 미리 만들어 둔 것을 재사용하므로 v1과 v2 검색이
    같은 벡터를 쓰고, 이 함수는 Embedding API를 호출하지 않는다.
    """

    per_question = []

    for entry in QUESTIONS:
        deep = dense_search_by_vector(
            question_vectors[entry["question"]],
            top_k=TOP_K_DEEP,
            collection_name=collection_name,
            client=client,
        )

        shallow = deep[:TOP_K_SHALLOW]

        per_question.append({
            "qid": entry["qid"],
            "question": entry["question"],
            "intent": entry["intent"],
            "expected_paths": entry["expected_paths"],
            "top_5": [simplify(r) for r in shallow],
            "top_30": [simplify(r) for r in deep],
            "expected_ranks": {
                path: expected_rank(deep, path)
                for path in entry["expected_paths"]
            },
            "definition_child_rank": rank_of(deep, DEFINITION_CHILD_ID),
            "item4_full_child_rank": rank_of(deep, ARTICLE2_ITEM4_ID),
        })

    return per_question

def build_comparison(v1_results, v2_results):
    v1_map = {r["qid"]: r for r in v1_results}
    v2_map = {r["qid"]: r for r in v2_results}

    comparisons = []

    for entry in QUESTIONS:
        qid = entry["qid"]
        v1 = v1_map[qid]
        v2 = v2_map[qid]

        v1_top5_ids = [r["id"] for r in v1["top_5"]]
        v2_top5_ids = [r["id"] for r in v2["top_5"]]

        v1_rank_map = {r["id"]: r["rank"] for r in v1["top_30"]}
        v2_rank_map = {r["id"]: r["rank"] for r in v2["top_30"]}

        rank_shifts = []

        for chunk_id, v1_rank in v1_rank_map.items():
            v2_rank = v2_rank_map.get(chunk_id)

            if v2_rank is not None and v2_rank != v1_rank:
                rank_shifts.append({
                    "id": chunk_id,
                    "v1_rank": v1_rank,
                    "v2_rank": v2_rank,
                    "delta": v2_rank - v1_rank,
                })

        best_expected_v1 = min(
            (r for r in v1["expected_ranks"].values() if r is not None),
            default=None,
        )
        best_expected_v2 = min(
            (r for r in v2["expected_ranks"].values() if r is not None),
            default=None,
        )

        if best_expected_v1 is None and best_expected_v2 is None:
            verdict = "no_expected_hit"
        elif best_expected_v1 is None:
            verdict = "improved"
        elif best_expected_v2 is None:
            verdict = "regressed"
        elif best_expected_v2 < best_expected_v1:
            verdict = "improved"
        elif best_expected_v2 > best_expected_v1:
            verdict = "regressed"
        else:
            verdict = "unchanged"

        comparisons.append({
            "qid": qid,
            "question": entry["question"],
            "intent": entry["intent"],
            "expected_paths": entry["expected_paths"],
            "v1_top5_ids": v1_top5_ids,
            "v2_top5_ids": v2_top5_ids,
            "top5_identical": v1_top5_ids == v2_top5_ids,
            "entered_top5": [
                cid for cid in v2_top5_ids if cid not in v1_top5_ids
            ],
            "left_top5": [
                cid for cid in v1_top5_ids if cid not in v2_top5_ids
            ],
            "v1_expected_ranks": v1["expected_ranks"],
            "v2_expected_ranks": v2["expected_ranks"],
            "best_expected_rank_v1": best_expected_v1,
            "best_expected_rank_v2": best_expected_v2,
            "definition_child_rank_v1": v1["definition_child_rank"],
            "definition_child_rank_v2": v2["definition_child_rank"],
            "item4_full_child_rank_v1": v1["item4_full_child_rank"],
            "item4_full_child_rank_v2": v2["item4_full_child_rank"],
            "rank_shifts_in_top30": sorted(
                rank_shifts, key=lambda item: item["v2_rank"]
            ),
            "verdict": verdict,
        })

    return comparisons

def render_markdown(comparisons, timestamp, v1_results, v2_results):
    v1_map = {r["qid"]: r for r in v1_results}
    v2_map = {r["qid"]: r for r in v2_results}

    lines = [
        "# Dense v1 vs Dense v2 검색 비교",
        "",
        f"- 실행 시각: {timestamp}",
        f"- Embedding 모델: {EMBEDDING_MODEL}",
        f"- v1 컬렉션: {V1_COLLECTION} (Child 205)",
        f"- v2 컬렉션: {V2_COLLECTION} (Child 206, 정의 전용 Child 1개 추가)",
        f"- Top-K: {TOP_K_SHALLOW} / {TOP_K_DEEP}",
        "",
        "v2는 v1의 기존 205개 벡터와 Payload를 그대로 복사했으므로,",
        "두 버전의 차이는 정의 전용 Child 1개 추가에서만 발생한다.",
        "",
        "## 요약",
        "",
        "| 질문 | 유형 | 정답 최상위 순위 v1 | v2 | 판정 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for item in comparisons:
        lines.append(
            "| {qid}. {question} | {intent} | {v1} | {v2} | {verdict} |".format(
                qid=item["qid"],
                question=item["question"],
                intent=item["intent"],
                v1=item["best_expected_rank_v1"] or "미검색",
                v2=item["best_expected_rank_v2"] or "미검색",
                verdict=item["verdict"],
            )
        )

    lines += ["", "## 질문별 상세", ""]

    for item in comparisons:
        lines += [
            f"### {item['qid']}. {item['question']}",
            "",
            f"- 기대 근거 조문: {', '.join(item['expected_paths'])}",
            f"- 정의 전용 Child 순위: v1 {item['definition_child_rank_v1'] or '없음'}"
            f" / v2 {item['definition_child_rank_v2'] or '없음'}",
            f"- Top-5 동일 여부: {item['top5_identical']}",
            f"- Top-5 진입: {item['entered_top5'] or '없음'}",
            f"- Top-5 이탈: {item['left_top5'] or '없음'}",
            "",
            "| 순위 | v1 조문 | v1 점수 | v2 조문 | v2 점수 |",
            "| --- | --- | --- | --- | --- |",
        ]

        v1_top5 = v1_map[item["qid"]]["top_5"]
        v2_top5 = v2_map[item["qid"]]["top_5"]

        for index in range(TOP_K_SHALLOW):
            left = v1_top5[index] if index < len(v1_top5) else None
            right = v2_top5[index] if index < len(v2_top5) else None

            lines.append(
                "| {rank} | {lp} | {ls} | {rp} | {rs} |".format(
                    rank=index + 1,
                    lp=left["article_path"] if left else "-",
                    ls=round(left["score"], 4) if left else "-",
                    rp=right["article_path"] if right else "-",
                    rs=round(right["score"], 4) if right else "-",
                )
            )

        lines.append("")

    return "\n".join(lines)

def build_payload(collection_name, experiment_name, per_question, timestamp):
    return {
        "experiment_name": experiment_name,
        "collection_name": collection_name,
        "embedding_model": EMBEDDING_MODEL,
        "dataset_version": "v1" if collection_name == V1_COLLECTION else "v2",
        "evaluation_type": "diagnostic_questions",
        "total_questions": len(per_question),
        "top_k": TOP_K_DEEP,
        "execution_timestamp": timestamp,
        "retrieval_results": per_question,
        "evaluation_metrics": None,
    }

def main():
    client = get_qdrant_client()

    for collection_name, expected in (
        (V1_COLLECTION, 205),
        (V2_COLLECTION, 206),
    ):
        if not client.collection_exists(collection_name):
            raise RuntimeError(f"컬렉션이 없습니다: {collection_name}")

        count = client.get_collection(collection_name).points_count

        if count != expected:
            raise RuntimeError(
                f"{collection_name} Point 수 불일치: {count} (기대값 {expected})"
            )

    timestamp = now_timestamp()

    print("===== 질문 임베딩 =====")

    embedder = QueryEmbedder(
        embeddings=get_embedding_model(),
        verbose=True,
    )

    question_vectors = embedder.embed_questions(
        [entry["question"] for entry in QUESTIONS]
    )

    v1_results = run_collection(V1_COLLECTION, question_vectors, client)
    v2_results = run_collection(V2_COLLECTION, question_vectors, client)

    comparisons = build_comparison(v1_results, v2_results)

    saved = []

    saved.append(save_json(
        RESULTS_DIR / "dense_v1" / "diagnostic_results.json",
        build_payload(V1_COLLECTION, "dense_v1_diagnostic", v1_results, timestamp),
        timestamp,
    ))
    saved.append(save_json(
        RESULTS_DIR / "dense_v2" / "diagnostic_results.json",
        build_payload(V2_COLLECTION, "dense_v2_diagnostic", v2_results, timestamp),
        timestamp,
    ))

    comparison_payload = {
        "experiment_name": "dense_v1_vs_v2_diagnostic",
        "collection_name": {
            "v1": V1_COLLECTION,
            "v2": V2_COLLECTION,
        },
        "embedding_model": EMBEDDING_MODEL,
        "dataset_version": {"v1": "v1", "v2": "v2"},
        "total_questions": len(QUESTIONS),
        "top_k": {
            "shallow": TOP_K_SHALLOW,
            "deep": TOP_K_DEEP,
        },
        "execution_timestamp": timestamp,
        "retrieval_results": {
            "v1": v1_results,
            "v2": v2_results,
        },
        "evaluation_metrics": {
            "improved": [c["qid"] for c in comparisons if c["verdict"] == "improved"],
            "regressed": [c["qid"] for c in comparisons if c["verdict"] == "regressed"],
            "unchanged": [c["qid"] for c in comparisons if c["verdict"] == "unchanged"],
            "no_expected_hit": [
                c["qid"] for c in comparisons if c["verdict"] == "no_expected_hit"
            ],
        },
        "comparisons": comparisons,
    }

    saved.append(save_json(
        RESULTS_DIR / "comparisons" / "dense_v1_vs_v2.json",
        comparison_payload,
        timestamp,
    ))
    saved.append(save_text(
        RESULTS_DIR / "comparisons" / "dense_v1_vs_v2.md",
        render_markdown(comparisons, timestamp, v1_results, v2_results),
        timestamp,
    ))

    print("\n===== API 호출 요약 =====")

    for line in embedder.report():
        print(line)

    print("\n===== Dense v1 vs v2 비교 =====")
    print("Embedding 모델:", EMBEDDING_MODEL)
    print("실행 시각:", timestamp)

    for item in comparisons:
        print(f"\n[{item['qid']}] {item['question']}")
        print("  유형:", item["intent"])
        print(
            "  정답 최상위 순위:",
            f"v1={item['best_expected_rank_v1']}",
            f"v2={item['best_expected_rank_v2']}",
            f"({item['verdict']})",
        )
        print(
            "  정의 전용 Child 순위:",
            f"v1={item['definition_child_rank_v1']}",
            f"v2={item['definition_child_rank_v2']}",
        )
        print("  Top-5 동일:", item["top5_identical"])

        if item["entered_top5"]:
            print("  Top-5 진입:", item["entered_top5"])

        if item["left_top5"]:
            print("  Top-5 이탈:", item["left_top5"])

    print("\n===== 저장 파일 =====")

    for path in saved:
        print(path.relative_to(BASE_DIR))

if __name__ == "__main__":
    main()
