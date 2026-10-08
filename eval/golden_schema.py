"""
Golden Test Set 로딩 및 정답 조문 참조 정규화.

팀원이 작성한 Golden Set의 필드명이 확정되지 않았으므로,
자주 쓰이는 필드명을 별칭으로 함께 허용하고 실제로 사용한 필드명을
평가 결과에 기록한다. 필드를 해석할 수 없으면 질문을 건너뛰지 않고
오류로 보고한다.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# 우선순위 순서. 앞에 있는 파일이 존재하면 그것을 사용한다.
GOLDEN_SET_CANDIDATES = [
    BASE_DIR / "eval" / "golden_set_v2.jsonl",
    BASE_DIR / "eval" / "golden_set.jsonl",
]

QUESTION_FIELDS = ["question", "query", "q", "질문"]

ANSWERABLE_FIELDS = ["answerable", "is_answerable", "answer_exists"]

# 정답 조문 참조가 담길 수 있는 필드명 후보
REFERENCE_FIELDS = [
    "gold_chunk_ids",
    "gold_ids",
    "gold_article_ids",
    "gold_articles",
    "expected_ids",
    "expected_articles",
    "answer_articles",
    "relevant_articles",
    "references",
    "ground_truth",
    "gold",
    "정답조문",
]

QID_FIELDS = ["qid", "id", "question_id"]

CATEGORY_FIELDS = ["category", "type", "intent", "question_type"]

CIRCLED_NUMBERS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

# '제2조제4호바목', '제34조제1항제2호', '제11조의2' 형태를 분해한다.
PATH_PATTERN = re.compile(
    r"^(?P<article>제\d+조(?:의\d+)?)"
    r"(?:제(?P<paragraph>\d+)항)?"
    r"(?:제(?P<item>\d+)호)?"
    r"(?:(?P<subitem>[가-힣])목)?$"
)

# Child ID 형태: '21311-20260721-제2조-p1-i4', '...-i4-가', '...-i4-definition'
CHUNK_ID_PATTERN = re.compile(
    r"-(?P<article>제\d+조(?:의\d+)?)"
    r"(?:-p(?P<p_index>\d+))?"
    r"(?:-i(?P<item>\d+))?"
    r"(?:-(?P<suffix>[가-힣]|definition|full))?$"
)

class GoldenSetError(Exception):
    pass

def find_golden_set():
    """
    존재하는 Golden Set 파일 경로를 반환한다. 없으면 None.
    """

    for path in GOLDEN_SET_CANDIDATES:
        if path.exists():
            return path

    return None

def pick_field(record, candidates):
    for name in candidates:
        if name in record:
            return name, record[name]

    return None, None

def normalize_paragraph(value):
    text = str(value or "").strip()

    if not text:
        return ""

    if text in CIRCLED_NUMBERS:
        return str(CIRCLED_NUMBERS.index(text) + 1)

    return text

def parse_reference(reference):
    """
    정답 조문 참조 한 건을 (article, paragraph, item, subitem) 구조로 바꾼다.

    지원 형태:
    - '제2조', '제2조제4호', '제2조제4호바목', '제34조제1항제2호'
    - Child ID ('21311-20260721-제2조-p1-i4', '...-i4-바')
    - dict ({'article': '제2조', 'item': '4', ...})
    """

    if isinstance(reference, dict):
        return {
            "article": str(reference.get("article", "")).strip(),
            "paragraph": normalize_paragraph(reference.get("paragraph", "")),
            "item": str(reference.get("item", "")).strip().rstrip("."),
            "subitem": str(reference.get("subitem", "")).strip().rstrip("."),
            "raw": reference,
        }

    text = str(reference).strip()

    if not text:
        raise GoldenSetError("빈 정답 조문 참조")

    compact = text.replace(" ", "")
    match = PATH_PATTERN.match(compact)

    if match:
        return {
            "article": match.group("article"),
            "paragraph": match.group("paragraph") or "",
            "item": match.group("item") or "",
            "subitem": match.group("subitem") or "",
            "raw": text,
        }

    id_match = CHUNK_ID_PATTERN.search(compact)

    if id_match:
        suffix = id_match.group("suffix") or ""

        return {
            "article": id_match.group("article"),
            # Child ID의 pN은 항번호가 아니라 항의 위치 인덱스이므로
            # 항 수준 비교에는 사용하지 않는다.
            "paragraph": "",
            "item": id_match.group("item") or "",
            "subitem": suffix if suffix not in ("definition", "full") else "",
            "raw": text,
        }

    raise GoldenSetError(f"정답 조문 참조를 해석할 수 없습니다: {text!r}")

def reference_depth(reference):
    """
    정답이 어느 수준까지 지정되었는지 반환한다.
    """

    if reference["subitem"]:
        return "subitem"

    if reference["item"]:
        return "item"

    if reference["paragraph"]:
        return "paragraph"

    return "article"

def reference_key(reference, level="full"):
    if level == "article":
        return (reference["article"],)

    return (
        reference["article"],
        reference["paragraph"],
        reference["item"],
        reference["subitem"],
    )

def result_reference(result):
    """
    검색 결과 1건을 정답 참조와 같은 구조로 바꾼다.
    """

    return {
        "article": str(result.get("article", "")).strip(),
        "paragraph": normalize_paragraph(result.get("paragraph", "")),
        "item": str(result.get("item", "")).strip().rstrip("."),
        "subitem": str(result.get("subitem", "")).strip().rstrip("."),
        "raw": result.get("id", ""),
    }

def matches(gold, candidate):
    """
    검색 결과가 정답 조문 범위 안에 있는지 판단한다.

    정답이 지정한 수준까지만 비교하므로, 정답이 '제2조제4호'이면
    제2조제4호 전체 Child와 정의 전용 Child, 가목~카목 Child가 모두 적중이다.
    정답이 '제2조제4호바목'이면 바목만 적중이다.
    """

    if gold["article"] != candidate["article"]:
        return False

    for level in ("paragraph", "item", "subitem"):
        if gold[level] and gold[level] != candidate[level]:
            return False

    return True

def load_golden_set(path):
    """
    JSONL Golden Set을 읽어 평가용 구조로 변환한다.
    """

    records = []
    field_usage = {}
    errors = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{line_number}행 JSON 파싱 실패: {error}")
                continue

            question_field, question = pick_field(raw, QUESTION_FIELDS)

            if not question:
                errors.append(
                    f"{line_number}행 질문 필드를 찾을 수 없습니다. "
                    f"보유 필드: {sorted(raw)}"
                )
                continue

            field_usage["question"] = question_field

            qid_field, qid = pick_field(raw, QID_FIELDS)
            field_usage["qid"] = qid_field

            answerable_field, answerable = pick_field(raw, ANSWERABLE_FIELDS)
            field_usage["answerable"] = answerable_field

            if answerable is None:
                answerable = True

            category_field, category = pick_field(raw, CATEGORY_FIELDS)
            field_usage["category"] = category_field

            reference_field, reference_value = pick_field(raw, REFERENCE_FIELDS)
            field_usage["references"] = reference_field

            if reference_value is None:
                reference_value = []

            if isinstance(reference_value, (str, dict)):
                reference_value = [reference_value]

            references = []

            for entry in reference_value:
                try:
                    references.append(parse_reference(entry))
                except GoldenSetError as error:
                    errors.append(f"{line_number}행 {error}")

            if answerable and not references:
                errors.append(
                    f"{line_number}행 answerable 질문에 정답 조문이 없습니다: "
                    f"{question!r}"
                )

            records.append({
                "qid": str(qid) if qid is not None else f"L{line_number}",
                "question": question,
                "answerable": bool(answerable),
                "category": category or "",
                "references": references,
                "raw": raw,
            })

    return {
        "path": str(path),
        "records": records,
        "field_usage": field_usage,
        "errors": errors,
    }
