"""
청크 품질 분석 스크립트.

legal_chunks.json의 Child 길이를 분석하고,
추가 세분화가 필요할 수 있는 후보를 보고한다.
이 스크립트는 JSON을 수정하지 않는다 (읽기/보고 전용).

주의: 정확한 토크나이저가 없으므로 "추정 토큰 수"는
문자 수 기반의 근사치(대략 2자 ~= 1토큰)이며 공식 토큰 수가 아니다.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks.json"

# 이미 세부 Child로 분할 처리된 (조문, 항위치) 조합은 후보에서 제외한다.
ALREADY_SPLIT = {
    ("제34조", "p1"),
}

# 이미 목 단위로 분할 처리된 (조문, 호 라벨) 조합은 후보에서 제외한다.
ALREADY_SPLIT_ITEM = {
    ("제2조", "4."),
}

LONG_CHUNK_CHAR_THRESHOLD = 300

def estimate_tokens_approx(text):
    """
    공식 토크나이저가 없을 때의 근사치. 2자 당 약 1토큰으로 가정한다.
    (한국어 환경의 일반적인 경험적 비율이며, 실제 임베딩 모델의
    토크나이저 결과와 다를 수 있다.)
    """

    return round(len(text) / 2)

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    children = data["children"]
    main_children = [c for c in children if c["document_type"] == "main"]

    print("===== 전체 Child 길이 통계 (문자 수 기준) =====")

    lengths = sorted(len(c["text"]) for c in main_children)
    total = len(lengths)

    if total:
        print("Child 수:", total)
        print("최소:", lengths[0])
        print("최대:", lengths[-1])
        print("평균: {:.1f}".format(sum(lengths) / total))
        print("중간값:", lengths[total // 2])

    print(
        "\n참고: 아래 '추정 토큰 수'는 공식 토크나이저가 아닌 "
        "문자 수 기반 근사치(2자당 약 1토큰)이다."
    )

    print(f"\n===== 긴 Child 목록 (문자 수 >= {LONG_CHUNK_CHAR_THRESHOLD}) =====")

    long_children = sorted(
        (c for c in main_children if len(c["text"]) >= LONG_CHUNK_CHAR_THRESHOLD),
        key=lambda c: -len(c["text"]),
    )

    for child in long_children:
        print(
            child["id"],
            "| granularity:", child.get("granularity", ""),
            "| chars:", len(child["text"]),
            "| 추정 토큰(근사):", estimate_tokens_approx(child["text"]),
        )

    print("\n===== 추가 세분화 후보 (paragraph 단위, 미분할) =====")
    print(
        "조건: granularity가 paragraph이고, 길이 >= "
        f"{LONG_CHUNK_CHAR_THRESHOLD}자이며 아직 호 단위로 분할되지 않은 항."
    )

    candidates = []

    for child in main_children:
        if child.get("granularity") != "paragraph":
            continue

        if len(child["text"]) < LONG_CHUNK_CHAR_THRESHOLD:
            continue

        article_id = child.get("article", "")
        child_id = child["id"]

        # child_id 패턴: {parent_id}-p{N}  -> p{N} 추출
        suffix = child_id.rsplit("-p", 1)
        p_label = f"p{suffix[1]}" if len(suffix) == 2 else ""

        if (article_id, p_label) in ALREADY_SPLIT:
            continue

        candidates.append(child)

    if not candidates:
        print("(없음)")
    else:
        for child in sorted(candidates, key=lambda c: -len(c["text"])):
            print(
                child["id"],
                "|",
                child.get("article", ""),
                child.get("paragraph", ""),
                "| chars:", len(child["text"]),
                "| 사유: 항 전체 길이가 길어 세부 분할 검토 대상",
            )

    print("\n===== 추가 세분화 후보 (item 단위, 다수 목 보유) =====")
    print("조건: 호 Child이며 목(세부 항목)을 3개 이상 포함하지만 아직 목 단위로 분할되지 않은 경우.")

    item_candidates = []

    for child in main_children:
        if child.get("granularity") != "item":
            continue

        article_id = child.get("article", "")
        item_label = child.get("item", "")

        if (article_id, item_label) in ALREADY_SPLIT_ITEM:
            continue

        subitem_markers = ["가.", "나.", "다.", "라.", "마.", "바.", "사.", "아.", "자.", "차.", "카."]
        marker_hits = sum(1 for marker in subitem_markers if f"\n{marker}" in child["text"])

        if marker_hits >= 3:
            item_candidates.append((child, marker_hits))

    if not item_candidates:
        print("(없음)")
    else:
        for child, hits in sorted(item_candidates, key=lambda pair: -pair[1]):
            print(
                child["id"],
                "|",
                child.get("article", ""),
                child.get("item", ""),
                "| 목 추정 개수:", hits,
                "| chars:", len(child["text"]),
            )

if __name__ == "__main__":
    main()
