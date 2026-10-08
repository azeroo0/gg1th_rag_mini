"""
Golden Test Set 기반 Hybrid+Qwen Reranking(C+D) 평가.

C(Hybrid)와 동일한 Hybrid Top-8 후보 집합(eval/hybrid_candidates.py로
캐시)을 입력으로 Qwen Reranking만 추가해 효과를 비교한다.
- C와 C+D의 Hybrid 후보 집합은 동일하다(캐시 재사용, Dense/BM25 재검색 없음).
- C+D도 C와 같은 질문 벡터를 쓴다(새로 임베딩하지 않음).
- C+D에 별도 청크나 문서를 추가하지 않는다.

평가 기준과 중복 제거 방식은 eval/evaluate.py(Dense v2)/eval/evaluate_hybrid.py와
동일하게 eval/golden_schema.py를 재사용한다.

결과는 eval/results/hybrid_rerank/에 저장하며 hybrid/dense_v1/dense_v2
결과 파일은 건드리지 않는다.
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
from eval.evaluate import aggregate, evaluate_record
from eval.golden_schema import find_golden_set, load_golden_set, reference_depth
from eval.hybrid_candidates import CANDIDATE_TOP_K, get_or_build_candidates
from eval.result_io import RESULTS_DIR, now_timestamp, save_json, save_text
from rag.hybrid_retriever import COLLECTION_NAME_V2, get_bm25_index
from rag.query_embedder import DEFAULT_CACHE_PATH, INITIAL_BATCH_SIZE, QueryEmbedder
from rag.reranker import MODEL_NAME, RerankerUnavailableError, get_reranker
from rag.retriever import format_article_path

EXPERIMENT_NAME = "hybrid_rerank_golden_set"
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


def rerank_record(reranker, record, candidates, top_k):
    """
    한 질문의 Hybrid Top-8 후보를 Reranking하고 평가용 결과 리스트를 만든다.
    """

    reranked = reranker.rerank(
        record["question"], candidates, text_field="source_text"
    )

    for new_rank, item in enumerate(reranked, start=1):
        item["rank"] = new_rank

    return reranked[:top_k]


def run_version(records, top_k, candidates_by_question, reranker, client):
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    retrieval_results = []
    article_eval = []
    provision_eval = []
    precise_eval = []
    unanswerable_results = []
    rank_changes = []

    total = len(records)
    errors = []

    for index, record in enumerate(records, start=1):
        candidates = candidates_by_question[record["question"]]

        try:
            results = rerank_record(reranker, record, candidates, top_k)
        except RerankerUnavailableError as error:
            errors.append({"qid": record["qid"], "error": str(error)})
            results = candidates[:top_k]

        if index == total or index % 10 == 0:
            print(f"  Reranking 진행 [hybrid_rerank]: {index}/{total}", flush=True)

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
                    "hybrid_rank_before_rerank": result.get("original_rank"),
                    "rerank_score": round(result.get("rerank_score", 0.0), 6),
                    "dense_rank": result.get("dense_rank"),
                    "bm25_rank": result.get("bm25_rank"),
                    "rrf_score": round(result.get("rrf_score", 0.0), 6)
                    if result.get("rrf_score") is not None
                    else None,
                    "source_text": result["source_text"],
                }
                for result in results
            ],
        })

        if not record["answerable"]:
            unanswerable_results.append({
                "qid": record["qid"],
                "question": record["question"],
                "top1_article_path": (
                    format_article_path(results[0]) if results else ""
                ),
            })
            continue

        results_for_eval = [
            {**result, "score": result.get("rerank_score", 0.0)}
            for result in results
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

        # Reranking 전(Hybrid 순서)과 후(Reranking 순서) 정답 조문 순위를 비교한다.
        before_results_for_eval = [
            {**candidate, "score": candidate.get("rrf_score", 0.0), "rank": rank}
            for rank, candidate in enumerate(candidates, start=1)
        ]
        before_item = evaluate_record(
            record, before_results_for_eval, "provision"
        )

        before_rank = before_item["first_hit_rank"]
        after_rank = provision_item["first_hit_rank"]

        rank_changes.append({
            "qid": record["qid"],
            "question": record["question"],
            "before_rerank_rank": before_rank,
            "after_rerank_rank": after_rank,
        })

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
            "Hybrid Top-8 후보를 Qwen Reranker로 재정렬한 뒤 Top-5까지 평가했다. "
            "dense_v2/hybrid와 동일한 Golden Set/질문 벡터/평가 기준을 사용했다."
        ),
    }

    # improved/regressed 판정: rank가 작을수록 좋음. None은 '찾지 못함'.
    improved = []
    regressed = []
    unchanged = []

    for change in rank_changes:
        before = change["before_rerank_rank"]
        after = change["after_rerank_rank"]

        if before == after:
            unchanged.append(change)
        elif before is None and after is not None:
            improved.append(change)
        elif before is not None and after is None:
            regressed.append(change)
        elif before is not None and after is not None and after < before:
            improved.append(change)
        else:
            regressed.append(change)

    return {
        "retrieval_results": retrieval_results,
        "metrics": metrics,
        "per_question": {
            "article_level": article_eval,
            "provision_level": provision_eval,
            "precise_diagnosis": precise_eval,
        },
        "unanswerable": unanswerable_results,
        "rank_changes": {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        },
        "errors": errors,
    }


def five_question_before_after(candidates_by_question, reranker, top_k):
    comparisons = []

    for question in FIVE_TEST_QUESTIONS:
        candidates = candidates_by_question[question]
        reranked = reranker.rerank(question, candidates, text_field="source_text")

        comparisons.append({
            "question": question,
            "hybrid_before_rerank": [
                {"rank": rank, "article_path": format_article_path(c), "id": c["id"]}
                for rank, c in enumerate(candidates[:top_k], start=1)
            ],
            "reranked_after": [
                {
                    "rank": rank,
                    "article_path": format_article_path(r),
                    "id": r["id"],
                    "rerank_score": round(r["rerank_score"], 4),
                    "hybrid_rank_before": r["original_rank"],
                }
                for rank, r in enumerate(reranked[:top_k], start=1)
            ],
        })

    return comparisons


def render_markdown(payload, five_question_diff, timestamp, params, reranker_stats):
    lines = [
        "# Golden Set 평가: Hybrid + Qwen Reranking (C+D)",
        "",
        f"- 실행 시각: {timestamp}",
        f"- 컬렉션: {COLLECTION_NAME_V2}",
        f"- Embedding 모델: {EMBEDDING_MODEL}",
        f"- Reranker 모델: {MODEL_NAME}",
        f"- Reranker 디바이스: {reranker_stats.get('device')}",
        f"- Reranker 로딩 시간: {reranker_stats.get('load_seconds')}초",
        f"- Reranker 추론 시간(전체): {round(reranker_stats.get('inference_seconds', 0.0), 2)}초"
        f" / 처리 쌍 수: {reranker_stats.get('pairs_scored')}",
        f"- Hybrid 후보 Top-K: {params['candidate_top_k']} / 최종 평가 Top-K: {params['top_k']}",
        f"- 전체 질문: {payload['total_questions']}",
        "",
        "## 검색 평가 지표 (Reranking 후)",
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

    rank_changes = payload["rank_changes"]

    lines += [
        "",
        "## Reranking 전후 순위 변화 (provision 수준, answerable 질문만)",
        "",
        f"- 개선: {len(rank_changes['improved'])}건",
    ]

    for c in rank_changes["improved"]:
        lines.append(
            f"  - {c['qid']}: {c['before_rerank_rank']} -> {c['after_rerank_rank']}"
        )

    lines.append(f"- 악화: {len(rank_changes['regressed'])}건")

    for c in rank_changes["regressed"]:
        lines.append(
            f"  - {c['qid']}: {c['before_rerank_rank']} -> {c['after_rerank_rank']}"
        )

    lines.append(f"- 변화 없음: {rank_changes['unchanged_count']}건")
    lines.append("")

    lines.append("## Hybrid vs Reranking 순위 비교 (5개 지정 질문)")
    lines.append("")

    for entry in five_question_diff:
        lines.append(f"### {entry['question']}")
        lines.append("")
        lines.append("Hybrid(Reranking 전) Top-5:")

        for r in entry["hybrid_before_rerank"]:
            lines.append(f"  {r['rank']}. {r['article_path']}")

        lines.append("")
        lines.append("Reranking 후 Top-5:")

        for r in entry["reranked_after"]:
            lines.append(
                f"  {r['rank']}. {r['article_path']} "
                f"(score={r['rerank_score']}, hybrid_rank_before={r['hybrid_rank_before']})"
            )

        lines.append("")

    if payload.get("errors"):
        lines.append("## Reranking 오류")
        lines.append("")

        for error in payload["errors"]:
            lines.append(f"- {error['qid']}: {error['error']}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Golden Set 기반 Hybrid+Qwen Reranking(C+D) 평가"
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--candidate-top-k", type=int, default=CANDIDATE_TOP_K)
    parser.add_argument("--golden-set", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=INITIAL_BATCH_SIZE)
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--rebuild-candidates",
        action="store_true",
        help="Hybrid Top-8 후보 캐시를 다시 만든다(기본은 캐시 재사용)",
    )

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

    print("\n===== 질문 임베딩 (dense_v2/hybrid와 동일 캐시 재사용) =====")

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

    print("\n===== Hybrid Top-8 후보 캐시 확인/구축 =====")

    bm25_index = get_bm25_index(collection_name=COLLECTION_NAME_V2, client=client)

    candidates_started = time.monotonic()
    candidates_by_question = get_or_build_candidates(
        questions,
        question_vectors,
        client,
        bm25_index=bm25_index,
        top_k=args.candidate_top_k,
        force_rebuild=args.rebuild_candidates,
    )
    candidates_elapsed = round(time.monotonic() - candidates_started, 2)

    print(f"Hybrid 후보 질문 수: {len(candidates_by_question)} (소요 {candidates_elapsed}초)")

    print("\n===== Qwen Reranker 로딩 =====")

    reranker = get_reranker()

    try:
        reranker._load()
    except RerankerUnavailableError as error:
        print("Reranker를 로딩할 수 없습니다:", error)
        print("Hybrid 기능은 정상이며, Reranking만 미완료 상태로 보고합니다.")
        raise SystemExit(3)

    print("로딩 완료:", reranker.stats)

    print("\n===== Reranking 및 평가 =====")

    rerank_started = time.monotonic()

    outcome = run_version(
        golden["records"], args.top_k, candidates_by_question, reranker, client
    )

    rerank_elapsed = round(time.monotonic() - rerank_started, 2)

    five_question_diff = five_question_before_after(
        candidates_by_question, reranker, args.top_k
    )

    params = {
        "candidate_top_k": args.candidate_top_k,
        "top_k": args.top_k,
    }

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset_version": DATASET_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "collection_name": COLLECTION_NAME_V2,
        "retriever_type": "hybrid_dense_bm25_rrf_plus_qwen_rerank",
        "reranker_model": MODEL_NAME,
        "search_parameters": params,
        "execution_timestamp": timestamp,
        "total_questions": len(golden["records"]),
        "golden_set_path": golden["path"],
        "evaluation_metrics": outcome["metrics"],
        "rank_changes": {
            "improved": outcome["rank_changes"]["improved"],
            "regressed": outcome["rank_changes"]["regressed"],
            "unchanged_count": len(outcome["rank_changes"]["unchanged"]),
        },
        "errors": outcome["errors"],
        "latency": {
            "embedding_seconds": embedding_elapsed,
            "candidate_build_seconds": candidates_elapsed,
            "reranker_load_seconds": reranker.stats.get("load_seconds"),
            "reranker_inference_seconds": round(
                reranker.stats.get("inference_seconds", 0.0), 3
            ),
            "reranking_and_eval_seconds": rerank_elapsed,
        },
    }

    retrieval_payload = dict(payload)
    retrieval_payload["retrieval_results"] = outcome["retrieval_results"]
    retrieval_payload["per_question"] = outcome["per_question"]
    retrieval_payload["unanswerable_questions"] = outcome["unanswerable"]
    retrieval_payload["hybrid_vs_rerank_five_questions"] = five_question_diff

    saved = []

    saved.append(save_json(
        RESULTS_DIR / "hybrid_rerank" / "retrieval_results.json",
        retrieval_payload,
        timestamp,
    ))
    saved.append(save_json(
        RESULTS_DIR / "hybrid_rerank" / "metrics.json",
        payload,
        timestamp,
    ))
    saved.append(save_text(
        RESULTS_DIR / "hybrid_rerank" / "summary.md",
        render_markdown(payload, five_question_diff, timestamp, params, reranker.stats),
        timestamp,
    ))

    total_elapsed = round(time.monotonic() - started, 2)

    print("\n===== 지표 (Hybrid + Reranking) =====")

    for level in ("article_level", "provision_level", "precise_diagnosis"):
        metrics = outcome["metrics"][level]
        print(
            f"{level:>18}: Hit@1={metrics.get('hit_at_1')} "
            f"Hit@3={metrics.get('hit_at_3')} Hit@5={metrics.get('hit_at_5')} "
            f"Recall@5={metrics.get('recall_at_5')} MRR={metrics.get('mrr')}"
        )

    print("\n===== Reranking 전후 순위 변화 =====")
    print("개선:", [c["qid"] for c in outcome["rank_changes"]["improved"]])
    print("악화:", [c["qid"] for c in outcome["rank_changes"]["regressed"]])
    print("변화 없음:", len(outcome["rank_changes"]["unchanged"]))

    print("\n===== 시간 =====")
    print("임베딩:", embedding_elapsed, "초")
    print("후보 구축:", candidates_elapsed, "초")
    print("Reranker 로딩:", reranker.stats.get("load_seconds"), "초")
    print("Reranker 추론:", round(reranker.stats.get("inference_seconds", 0.0), 2), "초")
    print("Reranking+평가:", rerank_elapsed, "초")
    print("전체:", total_elapsed, "초")

    print("\n===== 저장 파일 =====")

    for path in saved:
        print(path.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
