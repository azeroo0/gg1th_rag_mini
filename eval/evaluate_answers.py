"""
Golden Set 기반 Answer Evaluation (Hybrid / Hybrid+Reranking).

answer_points 등 Golden Set에 있는 정보만으로 평가 가능한 범위를 계산하고,
근거가 부족한 지표는 임의 숫자 대신 N/A(None)로 둔다.

지표 정의(계산 기준):
- 정답성(correctness): LLM Judge가 Golden Set의 answer_points를 답변이
  얼마나 포함하는지 0.0~1.0으로 판단한 값의 평균(answerable 질문만).
- 근거충실성(faithfulness): LLM Judge가 답변의 주장이 실제 검색된 근거
  (sources의 content)로 뒷받침되는 정도를 판단한 값의 평균(answerable만).
- 환각비율(hallucination_rate): LLM Judge가 판단한, 근거로 뒷받침되지
  않는 주장 비율의 평균(answerable만).
- 인용정밀도(citation_precision): 답변의 sources에 실린 조문(article) 중
  Golden Set 정답 조문(answer_articles)과 일치하는 비율의 평균. article
  단위로 중복 제거 후 계산한다(answerable만, sources가 있는 질문만).
- 인용재현율(citation_recall): Golden Set 정답 조문 중 sources에 포함된
  조문의 비율의 평균(answerable만).
- 거절정밀도(refusal_precision): 모델이 거절(LLM Judge의 refused=true)한
  질문 중 실제 answerable=false인 질문의 비율.
- 거절재현율(refusal_recall): answerable=false 질문 중 모델이 올바르게
  거절한 질문의 비율.
- 오거절(false_refusal_count): answerable=true인데 거절한 건수.
- 놓친거절(missed_refusal_count): answerable=false인데 거절하지 않고
  답변한 건수. (과거 코드의 '총나절' 표기를 이 의미로 통일했다.)

거절 여부(refused)는 LLM Judge가 같은 호출에서 판단하므로 별도 API
호출을 추가하지 않는다(eval/judge.py 참고).

실행 시간/비용 최적화를 위해 답변 생성과 Judge 평가 결과를 모두
질문 단위로 캐시하여, 중단 후 재실행 시 이미 끝난 질문은 다시 호출하지
않는다(이어하기 가능).
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from common.ai_model import get_embedding_model, get_llm_model
from common.config import MODEL as LLM_MODEL_NAME
from common.qdrant import get_qdrant_client
from eval.golden_schema import find_golden_set, load_golden_set, matches, reference_key
from eval.judge import JUDGE_PROMPT_VERSION, judge_answer
from eval.result_io import RESULTS_DIR, now_timestamp, save_json, save_text
from rag.pipeline import MODES, generate_answer
from rag.query_embedder import DEFAULT_CACHE_PATH, INITIAL_BATCH_SIZE, QueryEmbedder
from rag.retriever import is_retryable

ANSWER_TOP_K = 4
CANDIDATE_TOP_K = 8

MIN_CALL_INTERVAL = 2.2
MAX_RETRIES = 5
RETRY_BASE_DELAY = 10.0


class Throttle:
    """LLM_BASE_URL 게이트웨이 분당 30요청 제한을 넘지 않도록 호출 간격을 둔다."""

    def __init__(self, min_interval=MIN_CALL_INTERVAL):
        self.min_interval = min_interval
        self.last_call = 0.0

    def wait(self):
        remaining = self.min_interval - (time.monotonic() - self.last_call)

        if remaining > 0:
            time.sleep(remaining)

    def mark(self):
        self.last_call = time.monotonic()


def call_with_retry(fn, throttle, label=""):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        throttle.wait()

        try:
            result = fn()
            throttle.mark()

            return result
        except Exception as error:
            throttle.mark()
            last_error = error

            if not is_retryable(error):
                raise

            if attempt == MAX_RETRIES:
                break

            delay = RETRY_BASE_DELAY * attempt
            print(
                f"  {label} 재시도 {attempt}/{MAX_RETRIES - 1} "
                f"({delay:.0f}초 대기): {type(error).__name__}",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(f"{label} 재시도 {MAX_RETRIES}회 실패: {last_error}") from last_error


def answers_path(mode):
    return RESULTS_DIR / mode / "answers.json"


def judge_path(mode):
    return RESULTS_DIR / mode / "judge_results.json"


def load_json_or_empty(path):
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_answers_for_mode(
    mode, records, question_vectors, client, embeddings, llm, throttle, resume=True
):
    """
    질문별 답변을 생성한다. 이미 캐시에 있는 질문은 다시 호출하지 않는다
    (실행 중 실패해도 기존 완료 결과를 보존하고 재개할 수 있다).
    """

    path = answers_path(mode)
    answers = load_json_or_empty(path) if resume else {}

    total = len(records)
    api_calls = 0

    for index, record in enumerate(records, start=1):
        qid = record["qid"]

        if qid in answers:
            continue

        question = record["question"]
        vector = question_vectors[question]

        def call():
            return generate_answer(
                question,
                mode=mode,
                top_k=ANSWER_TOP_K,
                candidate_top_k=CANDIDATE_TOP_K,
                llm=llm,
                client=client,
                embeddings=embeddings,
                query_vector=vector,
            )

        result = call_with_retry(call, throttle, label=f"[{mode}] {qid} 답변 생성")
        api_calls += 1

        answers[qid] = {
            "qid": qid,
            "question": question,
            "answerable": record["answerable"],
            "answer": result["answer"],
            "sources": result["sources"],
            "mode": result["mode"],
            "retrieval_count": result["retrieval_count"],
            "rerank": result["rerank"],
            "elapsed_seconds": result["elapsed_seconds"],
        }

        write_json(path, answers)

        if index == total or index % 5 == 0:
            print(f"  답변 생성 [{mode}]: {index}/{total} (신규 호출 {api_calls}회)", flush=True)

    return answers, api_calls


def run_judge_for_mode(mode, records, answers, llm, throttle, resume=True):
    """
    답변별 Judge 평가를 수행한다. 캐시에 있는 질문은 다시 평가하지 않는다.
    """

    path = judge_path(mode)
    judged = load_json_or_empty(path) if resume else {}

    total = len(records)
    api_calls = 0
    failures = []

    for index, record in enumerate(records, start=1):
        qid = record["qid"]

        if qid in judged:
            continue

        answer_entry = answers[qid]
        answer_points = record["raw"].get("answer_points") or []
        source_contents = [s["content"] for s in answer_entry["sources"]]

        def call():
            return judge_answer(
                record["question"],
                answer_points,
                answer_entry["answer"],
                source_contents,
                llm=llm,
            )

        try:
            result = call_with_retry(call, throttle, label=f"[{mode}] {qid} Judge")
            api_calls += 1
            judged[qid] = result
        except Exception as error:
            failures.append({"qid": qid, "error": str(error)})
            print(f"  Judge 실패 [{mode}] {qid}: {error}", flush=True)

        write_json(path, judged)

        if index == total or index % 5 == 0:
            print(f"  Judge 평가 [{mode}]: {index}/{total} (신규 호출 {api_calls}회)", flush=True)

    return judged, api_calls, failures


def citation_precision_recall(sources, gold_references):
    """
    sources(이 답변이 실제로 근거로 쓴 Child)의 조문과 Golden Set 정답
    조문(article 단위)을 비교해 인용정밀도/인용재현율을 계산한다.
    """

    if not gold_references:
        return None, None

    gold_article_keys = {reference_key(r, level="article") for r in gold_references}

    source_article_keys = set()

    for source in sources:
        candidate = {
            "article": source.get("article", ""),
            "paragraph": "",
            "item": "",
            "subitem": "",
        }
        source_article_keys.add(reference_key(candidate, level="article"))

    if not source_article_keys:
        return 0.0, 0.0

    matched = source_article_keys & gold_article_keys

    precision = len(matched) / len(source_article_keys)
    recall = len(matched) / len(gold_article_keys)

    return precision, recall


def aggregate_mode_metrics(mode, records, answers, judged):
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    correctness_values = []
    faithfulness_values = []
    hallucination_values = []
    precision_values = []
    recall_values = []

    refused_answerable = 0  # 오거절
    refused_unanswerable = 0  # 올바른 거절
    not_refused_unanswerable = 0  # 놓친거절

    for record in answerable:
        qid = record["qid"]
        judge_result = judged.get(qid)

        if judge_result is not None:
            correctness_values.append(judge_result["correctness"])
            faithfulness_values.append(judge_result["faithfulness"])
            hallucination_values.append(judge_result["hallucination_rate"])

            if judge_result.get("refused"):
                refused_answerable += 1

        sources = answers[qid]["sources"]
        precision, recall = citation_precision_recall(sources, record["references"])

        if precision is not None:
            precision_values.append(precision)
            recall_values.append(recall)

    for record in unanswerable:
        qid = record["qid"]
        judge_result = judged.get(qid)

        if judge_result is None:
            continue

        if judge_result.get("refused"):
            refused_unanswerable += 1
        else:
            not_refused_unanswerable += 1

    total_refused = refused_answerable + refused_unanswerable

    refusal_precision = (
        refused_unanswerable / total_refused if total_refused else None
    )
    refusal_recall = (
        refused_unanswerable / len(unanswerable) if unanswerable else None
    )

    def avg(values):
        return round(sum(values) / len(values), 4) if values else None

    return {
        "correctness": avg(correctness_values),
        "faithfulness": avg(faithfulness_values),
        "citation_precision": avg(precision_values),
        "citation_recall": avg(recall_values),
        "hallucination_rate": avg(hallucination_values),
        "refusal_precision": (
            round(refusal_precision, 4) if refusal_precision is not None else None
        ),
        "refusal_recall": (
            round(refusal_recall, 4) if refusal_recall is not None else None
        ),
        "false_refusal_count": refused_answerable,
        "missed_refusal_count": not_refused_unanswerable,
        "judge_model": LLM_MODEL_NAME,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "answerable_question_count": len(answerable),
        "unanswerable_question_count": len(unanswerable),
        "judged_count": len(judged),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Golden Set 기반 Answer Evaluation (Hybrid/Hybrid+Reranking)"
    )
    parser.add_argument(
        "--modes", nargs="+", choices=list(MODES), default=list(MODES)
    )
    parser.add_argument("--golden-set", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=INITIAL_BATCH_SIZE)
    parser.add_argument("--cache-path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--no-resume", action="store_true")

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

    records = golden["records"]
    resume = not args.no_resume

    client = get_qdrant_client()
    embeddings = get_embedding_model()
    llm = get_llm_model(max_tokens=1024)
    judge_llm = get_llm_model(temperature=0, max_tokens=512)

    throttle = Throttle()

    print("\n===== 질문 임베딩 (기존 캐시 재사용) =====")

    embedder = QueryEmbedder(
        embeddings=embeddings,
        batch_size=args.batch_size,
        cache_path=args.cache_path,
        verbose=True,
    )

    questions = [record["question"] for record in records]
    question_vectors = embedder.embed_questions(questions)

    timestamp = now_timestamp()
    started = time.monotonic()

    total_generation_calls = 0
    total_judge_calls = 0

    mode_metrics = {}

    for mode in args.modes:
        print(f"\n===== 답변 생성 [{mode}] =====")

        answers, generation_calls = generate_answers_for_mode(
            mode, records, question_vectors, client, embeddings, llm, throttle, resume=resume
        )
        total_generation_calls += generation_calls

        print(f"\n===== Judge 평가 [{mode}] =====")

        judged, judge_calls, judge_failures = run_judge_for_mode(
            mode, records, answers, judge_llm, throttle, resume=resume
        )
        total_judge_calls += judge_calls

        metrics = aggregate_mode_metrics(mode, records, answers, judged)
        metrics["judge_failures"] = judge_failures
        mode_metrics[mode] = metrics

        payload = {
            "experiment_name": f"{mode}_answer_evaluation",
            "mode": mode,
            "golden_set_path": golden["path"],
            "total_questions": len(records),
            "execution_timestamp": timestamp,
            "answer_evaluation": metrics,
            "api_calls": {
                "answer_generation_calls": generation_calls,
                "judge_calls": judge_calls,
            },
        }

        save_json(RESULTS_DIR / mode / "answer_metrics.json", payload, timestamp)

    comparison_lines = [
        "# Answer Evaluation 비교: Hybrid vs Hybrid+Reranking",
        "",
        f"- 실행 시각: {timestamp}",
        f"- Judge 모델: {LLM_MODEL_NAME} (prompt_version={JUDGE_PROMPT_VERSION})",
        f"- answerable 질문: {sum(1 for r in records if r['answerable'])} / "
        f"unanswerable 질문: {sum(1 for r in records if not r['answerable'])}",
        "",
        "| 지표 | " + " | ".join(args.modes) + " |",
        "| --- | " + " | ".join(["---"] * len(args.modes)) + " |",
    ]

    metric_keys = [
        "correctness", "faithfulness", "citation_precision", "citation_recall",
        "hallucination_rate", "refusal_precision", "refusal_recall",
        "false_refusal_count", "missed_refusal_count",
    ]

    for key in metric_keys:
        row = [key]

        for mode in args.modes:
            value = mode_metrics[mode].get(key)
            row.append("N/A" if value is None else str(value))

        comparison_lines.append("| " + " | ".join(row) + " |")

    if len(args.modes) == 2:
        save_text(
            RESULTS_DIR / "comparisons" / "answer_evaluation_hybrid_vs_rerank.md",
            "\n".join(comparison_lines),
            timestamp,
        )

    total_elapsed = round(time.monotonic() - started, 2)

    print("\n===== Answer Evaluation 결과 =====")

    for mode in args.modes:
        print(f"\n[{mode}]")

        for key, value in mode_metrics[mode].items():
            if key == "judge_failures":
                continue

            print(f"  {key}: {value}")

    print(f"\nLLM 호출 요약: 답변 생성 {total_generation_calls}회, Judge {total_judge_calls}회")
    print(f"전체 소요 시간: {total_elapsed}초")


if __name__ == "__main__":
    main()
