"""
Golden Test Set 기반 Hybrid Retrieval(Dense + BM25 + RRF) 평가.

평가 기준과 중복 제거 방식은 eval/evaluate.py(Dense v2)와 동일하게
eval/golden_schema.py를 재사용한다. Dense v2와 동일한 질문 벡터를
써서 평가 조건을 맞춘다(data/cache/query_vectors.json 캐시 재사용).

결과는 eval/results/hybrid/에 저장하며 dense_v1ㆍdense_v2 결과는
건드리지 않는다.
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common.ai_model import get_embedding_model
from common.config import EMBEDDING_MODEL
from common.qdrant import get_qdrant_client
from eval.golden_schema import find_golden_set, load_golden_set, reference_depth
from eval.result_io import RESULTS_DIR, now_timestamp, save_json, save_text
from eval.evaluate import aggregate, evaluate_record
from rag.hybrid_retriever import (
    COLLECTION_NAME_V2,
    DEFAULT_BM25_TOP_K,
    DEFAULT_DENSE_TOP_K,
    DEFAULT_RRF_K,
    get_bm25_index,
    hybrid_search_by_vector,
)
from rag.query_embedder import DEFAULT_CACHE_PATH, INITIAL_BATCH_SIZE, QueryEmbedder
from rag.retriever import dense_search_by_vector, format_article_path

EXPERIMENT_NAME = "hybrid_golden_set"
DATASET_VERSION = "legal_chunks_v2.json (Child 206)"
EXPECTED_POINTS = 206

DEFAULT_TOP_K = 5

FIVE_TEST_QUESTIONS = [
    "고영향 인공지능이란 무엇인가요?",
    "생성형 인공지능이란 무엇인가요?",
    "채용 심사에 AI를 사용하면 고영향 인공지능인가요?",
    "고영향 인공지능 사업자의 의무는 무엇인가요?",
    "AI로 생성한 콘텐츠는 표시해야 하나요?",
]

def run_version(records, top_k, question_vectors, client, bm25_index,
                 dense_top_k, bm25_top_k, rrf_k):
    """
    Golden Set 전체에 대해 Hybrid Search를 수행하고 평가한다.

    질문 벡터는 미리 만들어 둔 것을 받으므로 Embedding API를
    새로 호출하지 않는다.
    """

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    retrieval_results = []
    article_eval = []
    provision_eval = []
    precise_eval = []
    unanswerable_results = []

    total = len(records)

    for index, record in enumerate(records, start=1):
        results = hybrid_search_by_vector(
            record["question"],
            question_vectors[record["question"]],
            top_k=top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k,
            collection_name=COLLECTION_NAME_V2,
            client=client,
            bm25_index=bm25_index,
        )

        if index == total or index % 10 == 0:
            print(f"  검색 진행 [hybrid]: {index}/{total}", flush=True)

        retrieval_results.append({
            "qid": record["qid"],
            "question": record["question"],
            "answerable": record["answerable"],
            "category": record["category"],
            "gold_references": [
                {
                    "raw": reference["raw"],
                    "article": reference["article"],
                    "paragraph": reference["paragraph"],
                    "item": reference["item"],
                    "subitem": reference["subitem"],
                    "depth": reference_depth(reference),
                }
                for reference in record["references"]
            ],
            "results": [
                {
                    "rank": result["rank"],
                    "id": result["id"],
                    "parent_id": result["parent_id"],
                    "article_path": format_article_path(result),
                    "article": result["article"],
                    "paragraph": result["paragraph"],
                    "item": result["item"],
                    "subitem": result["subitem"],
                    "granularity": result["granularity"],
                    "chunk_role": result.get("chunk_role", ""),
                    "dense_rank": result["dense_rank"],
                    "bm25_rank": result["bm25_rank"],
                    "rrf_score": round(result["rrf_score"], 6),
                    "source_text": result["source_text"],
                }
                for result in results
            ],
        })

        if not record["answerable"]:
            unanswerable_results.append({
                "qid": record["qid"],
                "question": record["question"],
                "top1_rrf_score": (
                    round(results[0]["rrf_score"], 6) if results else None
                ),
                "top1_article_path": (
                    format_article_path(results[0]) if results else ""
                ),
            })
            continue

        # evaluate_record는 result["score"] 필드로 round()하므로
        # rrf_score를 score 키로도 함께 제공한다.
        results_for_eval = [
            {**result, "score": result["rrf_score"]} for result in results
        ]

        article_item = evaluate_record(record, results_for_eval, "article")
        article_item["qid"] = record["qid"]
        article_eval.append(article_item)

        provision_item = evaluate_record(record, results_for_eval, "provision")
        provision_item["qid"] = record["qid"]
        provision_eval.append(provision_item)

        depths = {reference_depth(r) for r in record["references"]}

        if depths - {"article"}:
            precise_item = evaluate_record(
                record, results_for_eval, "provision", exact=True
            )
            precise_item["qid"] = record["qid"]
            precise_item["gold_depths"] = sorted(depths)
            precise_eval.append(precise_item)

    depth_counts = {}

    for record in answerable:
        for reference in record["references"]:
            depth = reference_depth(reference)
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

    metrics = {
        "article_level": aggregate(article_eval),
        "provision_level": aggregate(provision_eval),
        "precise_diagnosis": aggregate(precise_eval),
        "gold_reference_depth_counts": depth_counts,
        "answerable_question_count": len(answerable),
        "unanswerable_question_count": len(unanswerable),
        "note": (
            "article_level은 조문(제N조) 단위 적중률이며 세부 조항 적중률이 아니다. "
            "precise_diagnosis는 정답이 항ㆍ호ㆍ목까지 지정된 질문만 "
            "조문 경로 완전 일치로 평가한 결과다. "
            "dense_v2와 동일한 Golden Set/질문 벡터/평가 기준을 사용했다."
        ),
    }

    return {
        "retrieval_results": retrieval_results,
        "metrics": metrics,
        "per_question": {
            "article_level": article_eval,
            "provision_level": provision_eval,
            "precise_diagnosis": precise_eval,
        },
        "unanswerable": unanswerable_results,
    }

def rank_diff_for_five_questions(
    question_vectors, top_k, client, bm25_index, dense_top_k, bm25_top_k, rrf_k
):
    """
    5개 지정 질문을 Dense v2와 Hybrid로 각각 직접 검색해 Top-5 순위 차이를
    비교한다. Golden Set 문구와 다를 수 있으므로 Golden Set 평가와는
    별도로 이 질문들의 벡터를 직접 만들어 사용한다.
    """

    comparisons = []

    for question in FIVE_TEST_QUESTIONS:
        vector = question_vectors[question]

        dense_results = dense_search_by_vector(
            vector,
            top_k=top_k,
            collection_name=COLLECTION_NAME_V2,
            client=client,
        )

        hybrid_results = hybrid_search_by_vector(
            question,
            vector,
            top_k=top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            rrf_k=rrf_k,
            collection_name=COLLECTION_NAME_V2,
            client=client,
            bm25_index=bm25_index,
        )

        comparisons.append({
            "question": question,
            "dense_v2_top5": [
                {
                    "rank": r["rank"],
                    "article_path": format_article_path(r),
                    "id": r["id"],
                }
                for r in dense_results
            ],
            "hybrid_top5": [
                {
                    "rank": r["rank"],
                    "article_path": format_article_path(r),
                    "id": r["id"],
                    "dense_rank": r["dense_rank"],
                    "bm25_rank": r["bm25_rank"],
                }
                for r in hybrid_results
            ],
        })

    return comparisons

def render_markdown(payload, five_question_diff, timestamp, params):
    lines = [
        "# Golden Set 평가: Hybrid (Dense + BM25 + RRF)",
        "",
        f"- 실행 시각: {timestamp}",
        f"- 컬렉션: {COLLECTION_NAME_V2}",
        f"- Embedding 모델: {EMBEDDING_MODEL}",
        f"- Dense Top-K: {params['dense_top_k']} / BM25 Top-K: {params['bm25_top_k']}"
        f" / RRF k: {params['rrf_k']} / 최종 Top-K: {params['top_k']}",
        f"- 전체 질문: {payload['total_questions']}",
        f"- answerable 질문: {payload['evaluation_metrics']['answerable_question_count']}",
        f"- answerable=false 질문: "
        f"{payload['evaluation_metrics']['unanswerable_question_count']} "
        "(일반 정답 검색 평가에서 제외)",
        "",
        "## 검색 평가 지표",
        "",
        "| 평가 수준 | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for level in ("article_level", "provision_level", "precise_diagnosis"):
        metrics = payload["evaluation_metrics"][level]
        lines.append(
            "| {level} | {h1} | {h3} | {h5} | {r5} | {mrr} |".format(
                level=level,
                h1=metrics.get("hit_at_1", "-"),
                h3=metrics.get("hit_at_3", "-"),
                h5=metrics.get("hit_at_5", "-"),
                r5=metrics.get("recall_at_5", "-"),
                mrr=metrics.get("mrr", "-"),
            )
        )

    lines += [
        "",
        "article_level은 조문(제N조) 단위 적중률이다. 세부 조항 적중률이 아니다.",
        "precise_diagnosis는 정답이 항ㆍ호ㆍ목까지 지정된 질문만",
        "조문 경로 완전 일치로 평가한 결과다.",
        "",
        "## Dense v2 vs Hybrid 순위 비교 (5개 지정 질문)",
        "",
    ]

    for entry in five_question_diff:
        lines.append(f"### {entry['question']}")
        lines.append("")
        lines.append("Dense v2 Top-5:")

        for r in entry["dense_v2_top5"]:
            lines.append(f"  {r['rank']}. {r['article_path']}")

        lines.append("")
        lines.append("Hybrid Top-5 (dense_rank/bm25_rank):")

        for r in entry["hybrid_top5"]:
            lines.append(
                f"  {r['rank']}. {r['article_path']} "
                f"(dense={r['dense_rank']}, bm25={r['bm25_rank']})"
            )

        lines.append("")

    return "\n".join(lines)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Golden Set 기반 Hybrid Retrieval(C) 평가"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--bm25-top-k", type=int, default=DEFAULT_BM25_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--golden-set", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=INITIAL_BATCH_SIZE)
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--no-cache", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.golden_set:
        golden_path = Path(args.golden_set)

        if not golden_path.exists():
            print("Golden Set 파일이 없습니다:", golden_path)
            raise SystemExit(2)
    else:
        golden_path = find_golden_set()

    if golden_path is None:
        print("Golden Test Set 파일을 찾을 수 없습니다.")
        raise SystemExit(2)

    golden = load_golden_set(golden_path)

    print("Golden Set:", golden["path"])
    print("질문 수:", len(golden["records"]))

    if golden["errors"]:
        print("\n===== Golden Set 해석 오류 =====")

        for error in golden["errors"]:
            print("-", error)

        raise SystemExit(2)

    client = get_qdrant_client()

    if not client.collection_exists(COLLECTION_NAME_V2):
        raise RuntimeError(f"컬렉션이 없습니다: {COLLECTION_NAME_V2}")

    count = client.get_collection(COLLECTION_NAME_V2).points_count

    if count != EXPECTED_POINTS:
        raise RuntimeError(
            f"{COLLECTION_NAME_V2} Point 수 불일치: {count} (기대값 {EXPECTED_POINTS})"
        )

    timestamp = now_timestamp()
    started = time.monotonic()

    print("\n===== 질문 임베딩 (dense_v2와 동일 캐시 재사용) =====")

    embedder = QueryEmbedder(
        embeddings=get_embedding_model(),
        batch_size=args.batch_size,
        cache_path=args.cache_path,
        use_cache=not args.no_cache,
        verbose=True,
    )

    questions = [record["question"] for record in golden["records"]]
    questions += FIVE_TEST_QUESTIONS
    question_vectors = embedder.embed_questions(questions)

    embedding_elapsed = round(time.monotonic() - started, 2)

    print("\n===== BM25 색인 구축 (beopjeong_law_v2 scroll) =====")

    bm25_build_started = time.monotonic()
    bm25_index = get_bm25_index(collection_name=COLLECTION_NAME_V2, client=client)
    bm25_build_elapsed = round(time.monotonic() - bm25_build_started, 2)

    print(f"BM25 색인 문서 수: {bm25_index.size} (소요 {bm25_build_elapsed}초)")

    print("\n===== Hybrid 검색 및 평가 =====")

    search_started = time.monotonic()

    outcome = run_version(
        golden["records"],
        args.top_k,
        question_vectors,
        client,
        bm25_index,
        args.dense_top_k,
        args.bm25_top_k,
        args.rrf_k,
    )

    search_elapsed = round(time.monotonic() - search_started, 2)

    five_question_diff = rank_diff_for_five_questions(
        question_vectors,
        args.top_k,
        client,
        bm25_index,
        args.dense_top_k,
        args.bm25_top_k,
        args.rrf_k,
    )

    params = {
        "dense_top_k": args.dense_top_k,
        "bm25_top_k": args.bm25_top_k,
        "rrf_k": args.rrf_k,
        "top_k": args.top_k,
    }

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset_version": DATASET_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "collection_name": COLLECTION_NAME_V2,
        "retriever_type": "hybrid_dense_bm25_rrf",
        "reranker_model": None,
        "search_parameters": params,
        "execution_timestamp": timestamp,
        "total_questions": len(golden["records"]),
        "golden_set_path": golden["path"],
        "golden_set_field_usage": golden["field_usage"],
        "evaluation_metrics": outcome["metrics"],
        "latency": {
            "embedding_seconds": embedding_elapsed,
            "bm25_index_build_seconds": bm25_build_elapsed,
            "search_and_eval_seconds": search_elapsed,
        },
    }

    retrieval_payload = dict(payload)
    retrieval_payload["retrieval_results"] = outcome["retrieval_results"]
    retrieval_payload["per_question"] = outcome["per_question"]
    retrieval_payload["unanswerable_questions"] = outcome["unanswerable"]
    retrieval_payload["dense_v2_vs_hybrid_five_questions"] = five_question_diff

    saved = []

    saved.append(save_json(
        RESULTS_DIR / "hybrid" / "retrieval_results.json",
        retrieval_payload,
        timestamp,
    ))
    saved.append(save_json(
        RESULTS_DIR / "hybrid" / "metrics.json",
        payload,
        timestamp,
    ))
    saved.append(save_text(
        RESULTS_DIR / "hybrid" / "summary.md",
        render_markdown(payload, five_question_diff, timestamp, params),
        timestamp,
    ))

    total_elapsed = round(time.monotonic() - started, 2)

    print("\n===== API 호출 요약 =====")

    for line in embedder.report():
        print(line)

    print(f"BM25 색인 구축: {bm25_build_elapsed}초 (API 호출 없음)")
    print(f"임베딩 단계: {embedding_elapsed}초")
    print(f"검색ㆍ평가 단계: {search_elapsed}초")
    print(f"전체 소요 시간: {total_elapsed}초")

    print("\n===== 지표 (Hybrid) =====")

    for level in ("article_level", "provision_level", "precise_diagnosis"):
        metrics = outcome["metrics"][level]
        print(
            f"{level:>18}: Hit@1={metrics.get('hit_at_1')} "
            f"Hit@3={metrics.get('hit_at_3')} Hit@5={metrics.get('hit_at_5')} "
            f"Recall@5={metrics.get('recall_at_5')} MRR={metrics.get('mrr')}"
        )

    print("\n===== Dense v2 vs Hybrid 순위 비교 (5개 지정 질문) =====")

    for entry in five_question_diff:
        print(f"\n질문: {entry['question']}")
        print("  Dense v2:", [r["article_path"] for r in entry["dense_v2_top5"]])
        print("  Hybrid  :", [r["article_path"] for r in entry["hybrid_top5"]])

    print("\n===== 저장 파일 =====")

    for path in saved:
        print(path.relative_to(BASE_DIR))

if __name__ == "__main__":
    main()
