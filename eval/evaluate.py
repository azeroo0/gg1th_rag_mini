"""
Golden Test Set 기반 Dense Retrieval 평가.

동일 질문, 동일 Top-K, 동일 Embedding 모델, 동일 평가 기준으로
v1과 v2 컬렉션을 평가하고 결과를 버전별 파일로 저장한다.

평가 수준을 두 가지로 분리한다.
- article 수준: 조문(제N조) 단위. 조문 ID 기준으로 중복 제거한다.
- provision 수준: 정답이 지정한 항ㆍ호ㆍ목 수준까지 비교한다.
  동일 조문 경로를 가리키는 Child(예: 제2조제4호 전체 Child와
  정의 전용 Child)는 하나로 중복 제거하여 평가를 왜곡하지 않게 한다.

정답이 조 단위로만 주어진 질문은 세부 조항 적중률로 집계하지 않고,
항ㆍ호ㆍ목까지 지정된 질문만 따로 정밀 진단한다.
answerable=false 질문은 일반 정답 검색 평가에서 제외하고 별도로 집계한다.
"""

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import time

from common.ai_model import get_embedding_model
from common.config import EMBEDDING_MODEL
from common.qdrant import get_qdrant_client
from eval.golden_schema import (
    find_golden_set,
    load_golden_set,
    matches,
    reference_depth,
    reference_key,
    result_reference,
)
from eval.result_io import RESULTS_DIR, now_timestamp, save_json, save_text
from rag.query_embedder import (
    DEFAULT_CACHE_PATH,
    INITIAL_BATCH_SIZE,
    QueryEmbedder,
)
from rag.retriever import dense_search_by_vector, format_article_path

VERSIONS = [
    {
        "name": "dense_v1",
        "collection_name": "beopjeong_law_v1",
        "dataset_version": "legal_chunks_v1.json (Child 205)",
        "expected_points": 205,
    },
    {
        "name": "dense_v2",
        "collection_name": "beopjeong_law_v2",
        "dataset_version": "legal_chunks_v2.json (Child 206)",
        "expected_points": 206,
    },
]

DEFAULT_TOP_K = 5
HIT_POSITIONS = [1, 3, 5]
RECALL_POSITION = 5

def deduplicate(results, level):
    """
    검색 결과를 평가 수준에 맞춰 중복 제거하고 순위를 다시 매긴다.
    같은 키가 여러 번 나오면 가장 높은 순위(먼저 나온 것)만 남긴다.
    """

    seen = set()
    reduced = []

    for result in results:
        reference = result_reference(result)
        key = reference_key(reference, level=level)

        if key in seen:
            continue

        seen.add(key)

        reduced.append({
            "rank": len(reduced) + 1,
            "original_rank": result["rank"],
            "score": result["score"],
            "id": result["id"],
            "reference": reference,
            "article_path": format_article_path(result),
            "chunk_role": result.get("chunk_role", ""),
        })

    return reduced

def gold_keys(references, level):
    keys = []

    for reference in references:
        key = reference_key(reference, level=level)

        if key not in keys:
            keys.append(key)

    return keys

def match_gold(gold_references, candidate, level, exact=False):
    """
    중복 제거된 검색 결과 1건이 적중시킨 정답 참조 키 목록을 반환한다.
    """

    hit_keys = []

    for reference in gold_references:
        if level == "article":
            hit = reference["article"] == candidate["reference"]["article"]
        elif exact:
            hit = reference_key(reference) == reference_key(
                candidate["reference"]
            )
        else:
            hit = matches(reference, candidate["reference"])

        if hit:
            key = reference_key(reference, level=level)

            if key not in hit_keys:
                hit_keys.append(key)

    return hit_keys

def evaluate_record(record, results, level, exact=False):
    """
    질문 1건에 대해 지정한 평가 수준의 적중 정보를 계산한다.
    """

    reduced = deduplicate(results, level)
    gold = gold_keys(record["references"], level)

    first_hit_rank = None
    hit_keys_by_rank = {}

    for candidate in reduced:
        hits = match_gold(record["references"], candidate, level, exact=exact)

        if hits:
            hit_keys_by_rank[candidate["rank"]] = hits

            if first_hit_rank is None:
                first_hit_rank = candidate["rank"]

    recalled = set()

    for rank, keys in hit_keys_by_rank.items():
        if rank <= RECALL_POSITION:
            recalled.update(keys)

    hits_at = {
        position: bool(
            first_hit_rank is not None and first_hit_rank <= position
        )
        for position in HIT_POSITIONS
    }

    return {
        "level": level,
        "exact": exact,
        "gold_count": len(gold),
        "first_hit_rank": first_hit_rank,
        "hits": hits_at,
        "recall_at_5": (len(recalled) / len(gold)) if gold else None,
        "reciprocal_rank": (1.0 / first_hit_rank) if first_hit_rank else 0.0,
        "retrieved": [
            {
                "rank": candidate["rank"],
                "original_rank": candidate["original_rank"],
                "score": round(candidate["score"], 6),
                "id": candidate["id"],
                "article_path": candidate["article_path"],
                "chunk_role": candidate["chunk_role"],
                "is_hit": candidate["rank"] in hit_keys_by_rank,
            }
            for candidate in reduced
        ],
    }

def aggregate(per_record):
    """
    질문별 결과를 평균 지표로 집계한다.
    """

    if not per_record:
        return {
            "question_count": 0,
            "hit_at_1": None,
            "hit_at_3": None,
            "hit_at_5": None,
            "recall_at_5": None,
            "mrr": None,
        }

    count = len(per_record)

    recalls = [
        item["recall_at_5"]
        for item in per_record
        if item["recall_at_5"] is not None
    ]

    metrics = {"question_count": count}

    for position in HIT_POSITIONS:
        metrics[f"hit_at_{position}"] = round(
            sum(1 for item in per_record if item["hits"][position]) / count, 4
        )

    metrics["recall_at_5"] = (
        round(sum(recalls) / len(recalls), 4) if recalls else None
    )
    metrics["mrr"] = round(
        sum(item["reciprocal_rank"] for item in per_record) / count, 4
    )

    return metrics

def run_version(version, records, top_k, question_vectors, client):
    """
    한 버전(컬렉션)에 대해 Golden Set 전체를 검색하고 평가한다.

    질문 벡터는 미리 만들어 둔 것을 받아 쓰므로, 이 함수는
    Embedding API를 호출하지 않는다. v1과 v2가 같은 질문 벡터를
    공유하여 평가 조건이 동일하게 유지된다.
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
        results = dense_search_by_vector(
            question_vectors[record["question"]],
            top_k=top_k,
            collection_name=version["collection_name"],
            client=client,
        )

        if index == total or index % 10 == 0:
            print(
                f"  검색 진행 [{version['name']}]: {index}/{total}",
                flush=True,
            )

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
                for result in results
            ],
        })

        if not record["answerable"]:
            unanswerable_results.append({
                "qid": record["qid"],
                "question": record["question"],
                "top1_score": round(results[0]["score"], 6) if results else None,
                "top1_article_path": (
                    format_article_path(results[0]) if results else ""
                ),
            })
            continue

        article_item = evaluate_record(record, results, "article")
        article_item["qid"] = record["qid"]
        article_eval.append(article_item)

        provision_item = evaluate_record(record, results, "provision")
        provision_item["qid"] = record["qid"]
        provision_eval.append(provision_item)

        depths = {reference_depth(r) for r in record["references"]}

        if depths - {"article"}:
            # 항ㆍ호ㆍ목까지 지정된 정답만 정밀 진단 대상으로 삼는다.
            precise_item = evaluate_record(
                record, results, "provision", exact=True
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
            "조문 경로 완전 일치로 평가한 결과다."
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

def build_payload(
    version, golden, top_k, timestamp, outcome, kind, embedding_stats=None
):
    payload = {
        "experiment_name": f"{version['name']}_golden_set",
        "collection_name": version["collection_name"],
        "embedding_model": EMBEDDING_MODEL,
        "dataset_version": version["dataset_version"],
        "golden_set_path": golden["path"],
        "golden_set_field_usage": golden["field_usage"],
        "total_questions": len(golden["records"]),
        "top_k": top_k,
        "execution_timestamp": timestamp,
        "embedding_stats": embedding_stats,
    }

    if kind == "retrieval":
        payload["retrieval_results"] = outcome["retrieval_results"]
        payload["evaluation_metrics"] = outcome["metrics"]
    else:
        payload["retrieval_results"] = None
        payload["evaluation_metrics"] = outcome["metrics"]
        payload["per_question"] = outcome["per_question"]
        payload["unanswerable_questions"] = outcome["unanswerable"]

    return payload

def metric_rows(v1_metrics, v2_metrics):
    rows = []

    for level in ("article_level", "provision_level", "precise_diagnosis"):
        for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "recall_at_5", "mrr"):
            v1_value = v1_metrics[level].get(metric)
            v2_value = v2_metrics[level].get(metric)

            delta = (
                round(v2_value - v1_value, 4)
                if v1_value is not None and v2_value is not None
                else None
            )

            rows.append({
                "level": level,
                "metric": metric,
                "v1": v1_value,
                "v2": v2_value,
                "delta": delta,
            })

    return rows

def question_level_changes(v1_outcome, v2_outcome, level):
    v1_map = {
        item["qid"]: item for item in v1_outcome["per_question"][level]
    }
    v2_map = {
        item["qid"]: item for item in v2_outcome["per_question"][level]
    }

    improved = []
    regressed = []
    unchanged = []

    for qid, v1_item in v1_map.items():
        v2_item = v2_map.get(qid)

        if v2_item is None:
            continue

        v1_rank = v1_item["first_hit_rank"]
        v2_rank = v2_item["first_hit_rank"]

        entry = {
            "qid": qid,
            "v1_first_hit_rank": v1_rank,
            "v2_first_hit_rank": v2_rank,
        }

        if v1_rank == v2_rank:
            unchanged.append(entry)
        elif v1_rank is None:
            improved.append(entry)
        elif v2_rank is None:
            regressed.append(entry)
        elif v2_rank < v1_rank:
            improved.append(entry)
        else:
            regressed.append(entry)

    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged_count": len(unchanged),
    }

def render_markdown(comparison, timestamp, top_k):
    lines = [
        "# Golden Set 평가: Dense v1 vs Dense v2",
        "",
        f"- 실행 시각: {timestamp}",
        f"- Golden Set: {comparison['golden_set_path']}",
        f"- Embedding 모델: {comparison['embedding_model']}",
        f"- Top-K: {top_k}",
        f"- 전체 질문: {comparison['total_questions']}",
        f"- answerable 질문: {comparison['answerable_question_count']}",
        f"- answerable=false 질문: {comparison['unanswerable_question_count']} "
        "(일반 정답 검색 평가에서 제외)",
        "",
        "## 지표 비교",
        "",
        "| 평가 수준 | 지표 | v1 | v2 | 변화 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for row in comparison["metric_rows"]:
        lines.append(
            "| {level} | {metric} | {v1} | {v2} | {delta} |".format(
                level=row["level"],
                metric=row["metric"],
                v1="-" if row["v1"] is None else row["v1"],
                v2="-" if row["v2"] is None else row["v2"],
                delta="-" if row["delta"] is None else f"{row['delta']:+}",
            )
        )

    lines += [
        "",
        "article_level은 조문(제N조) 단위 적중률이다. 세부 조항 적중률이 아니다.",
        "precise_diagnosis는 정답이 항ㆍ호ㆍ목까지 지정된 질문만",
        "조문 경로 완전 일치로 평가한 결과다.",
        "",
        "## 질문별 변화 (provision 수준)",
        "",
    ]

    changes = comparison["question_changes"]["provision_level"]

    lines.append(f"- 개선: {len(changes['improved'])}건")

    for entry in changes["improved"]:
        lines.append(
            f"  - {entry['qid']}: {entry['v1_first_hit_rank']}"
            f" -> {entry['v2_first_hit_rank']}"
        )

    lines.append(f"- 악화: {len(changes['regressed'])}건")

    for entry in changes["regressed"]:
        lines.append(
            f"  - {entry['qid']}: {entry['v1_first_hit_rank']}"
            f" -> {entry['v2_first_hit_rank']}"
        )

    lines.append(f"- 변화 없음: {changes['unchanged_count']}건")
    lines.append("")

    return "\n".join(lines)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Golden Set 기반 Dense Retrieval v1/v2 평가"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"검색 Top-K (기본 {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--golden-set",
        type=str,
        default="",
        help="Golden Set 경로를 직접 지정 (생략 시 자동 탐색)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=INITIAL_BATCH_SIZE,
        help=(
            f"질문 임베딩 배치 크기 (기본 {INITIAL_BATCH_SIZE}, "
            "API 제한에 따라 자동 조정)"
        ),
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=str(DEFAULT_CACHE_PATH),
        help="질문 벡터 캐시 파일 경로",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="질문 벡터 캐시를 사용하지 않고 매번 새로 임베딩한다",
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
        print("다음 경로 중 하나에 파일이 필요합니다.")
        print("  eval/golden_set_v2.jsonl")
        print("  eval/golden_set.jsonl")
        print(
            "파일을 임의로 생성하지 않습니다. "
            "팀원이 작성한 Golden Set을 배치한 뒤 다시 실행하십시오."
        )
        raise SystemExit(2)

    golden = load_golden_set(golden_path)

    print("Golden Set:", golden["path"])
    print("질문 수:", len(golden["records"]))
    print("사용한 필드:", golden["field_usage"])

    if golden["errors"]:
        print("\n===== Golden Set 해석 오류 =====")

        for error in golden["errors"]:
            print("-", error)

        print(
            "\n필드 구조를 확인할 수 없어 평가를 중단합니다. "
            "eval/golden_schema.py의 필드 별칭을 실제 구조에 맞게 조정하십시오."
        )
        raise SystemExit(2)

    if not golden["records"]:
        print("Golden Set에 질문이 없습니다.")
        raise SystemExit(2)

    client = get_qdrant_client()

    for version in VERSIONS:
        if not client.collection_exists(version["collection_name"]):
            raise RuntimeError(
                f"컬렉션이 없습니다: {version['collection_name']}"
            )

        count = client.get_collection(version["collection_name"]).points_count

        if count != version["expected_points"]:
            raise RuntimeError(
                f"{version['collection_name']} Point 수 불일치: "
                f"{count} (기대값 {version['expected_points']})"
            )

    timestamp = now_timestamp()
    started = time.monotonic()

    # 질문을 먼저 한 번에 임베딩하고, v1ㆍv2 검색이 같은 벡터를 재사용한다.
    print("\n===== 질문 임베딩 =====")

    embedder = QueryEmbedder(
        embeddings=get_embedding_model(),
        batch_size=args.batch_size,
        cache_path=args.cache_path,
        use_cache=not args.no_cache,
        verbose=True,
    )

    questions = [record["question"] for record in golden["records"]]
    question_vectors = embedder.embed_questions(questions)

    embedding_elapsed = round(time.monotonic() - started, 2)

    outcomes = {}
    saved = []

    print("\n===== 검색 및 평가 =====")

    search_started = time.monotonic()

    for version in VERSIONS:
        outcome = run_version(
            version, golden["records"], args.top_k, question_vectors, client
        )
        outcomes[version["name"]] = outcome

        saved.append(save_json(
            RESULTS_DIR / version["name"] / "retrieval_results.json",
            build_payload(
                version, golden, args.top_k, timestamp, outcome, "retrieval",
                embedding_stats=embedder.stats,
            ),
            timestamp,
        ))
        saved.append(save_json(
            RESULTS_DIR / version["name"] / "metrics.json",
            build_payload(
                version, golden, args.top_k, timestamp, outcome, "metrics",
                embedding_stats=embedder.stats,
            ),
            timestamp,
        ))

    v1_outcome = outcomes["dense_v1"]
    v2_outcome = outcomes["dense_v2"]

    comparison = {
        "experiment_name": "golden_set_dense_v1_vs_v2",
        "collection_name": {
            "v1": VERSIONS[0]["collection_name"],
            "v2": VERSIONS[1]["collection_name"],
        },
        "embedding_model": EMBEDDING_MODEL,
        "dataset_version": {
            "v1": VERSIONS[0]["dataset_version"],
            "v2": VERSIONS[1]["dataset_version"],
        },
        "golden_set_path": golden["path"],
        "golden_set_field_usage": golden["field_usage"],
        "total_questions": len(golden["records"]),
        "answerable_question_count": (
            v1_outcome["metrics"]["answerable_question_count"]
        ),
        "unanswerable_question_count": (
            v1_outcome["metrics"]["unanswerable_question_count"]
        ),
        "top_k": args.top_k,
        "execution_timestamp": timestamp,
        "evaluation_metrics": {
            "v1": v1_outcome["metrics"],
            "v2": v2_outcome["metrics"],
        },
        "metric_rows": metric_rows(
            v1_outcome["metrics"], v2_outcome["metrics"]
        ),
        "question_changes": {
            level: question_level_changes(v1_outcome, v2_outcome, level)
            for level in ("article_level", "provision_level", "precise_diagnosis")
        },
        "retrieval_results": None,
    }

    saved.append(save_json(
        RESULTS_DIR / "comparisons" / "golden_set_dense_v1_vs_v2.json",
        comparison,
        timestamp,
    ))
    saved.append(save_text(
        RESULTS_DIR / "comparisons" / "golden_set_dense_v1_vs_v2.md",
        render_markdown(comparison, timestamp, args.top_k),
        timestamp,
    ))

    search_elapsed = round(time.monotonic() - search_started, 2)
    total_elapsed = round(time.monotonic() - started, 2)

    print("\n===== API 호출 요약 =====")

    for line in embedder.report():
        print(line)

    print(f"검색 호출: {len(golden['records']) * len(VERSIONS)}회 (Qdrant, API 호출 없음)")
    print(f"임베딩 단계: {embedding_elapsed}초")
    print(f"검색ㆍ평가 단계: {search_elapsed}초")
    print(f"전체 소요 시간: {total_elapsed}초")

    print("\n===== 지표 비교 =====")

    for row in comparison["metric_rows"]:
        print(
            f"{row['level']:>18} {row['metric']:>11}: "
            f"v1={row['v1']} v2={row['v2']} delta={row['delta']}"
        )

    print("\n===== 질문별 변화 (provision 수준) =====")

    changes = comparison["question_changes"]["provision_level"]

    print("개선:", [e["qid"] for e in changes["improved"]])
    print("악화:", [e["qid"] for e in changes["regressed"]])
    print("변화 없음:", changes["unchanged_count"])

    print("\n===== 저장 파일 =====")

    for path in saved:
        print(path.relative_to(BASE_DIR))

if __name__ == "__main__":
    main()
