import json
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks.json"

LAW_NUMBER = "21311"
LAW_EFFECTIVE_DATE = "20260721"

def normalize(text):
    return re.sub(r"\s+", " ", text).strip()

def get_main_base_children(article_id, parent_id, related_children):
    """
    세부 Child 추가 이전부터 존재하던 '기본 단위' Child만 골라낸다.
    (제2조는 호 단위, 그 외 조문은 항 단위가 기본 단위이며,
    항이 없는 조문은 -full 하나가 기본 단위다.)
    """

    full_id = f"{parent_id}-full"

    if any(child["id"] == full_id for child in related_children):
        return [c for c in related_children if c["id"] == full_id]

    if article_id == "제2조":
        pattern = re.compile(r"^" + re.escape(parent_id) + r"-p\d+-i\d+$")
    else:
        pattern = re.compile(r"^" + re.escape(parent_id) + r"-p\d+$")

    return [c for c in related_children if pattern.match(c["id"])]

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    parents = data["parents"]
    children = data["children"]

    parent_ids = [parent["id"] for parent in parents]
    child_ids = [child["id"] for child in children]
    parent_id_set = set(parent_ids)

    results = []

    def check(label, condition):
        results.append((label, condition))

    # 1. ID 중복
    check("Parent ID 중복 없음", len(parent_ids) == len(set(parent_ids)))
    check("Child ID 중복 없음", len(child_ids) == len(set(child_ids)))

    # 2. Parent 참조 유효성, 빈 텍스트
    invalid_parent_ref = [
        c["id"] for c in children if c["parent_id"] not in parent_id_set
    ]
    empty_text = [c["id"] for c in children if not c["text"].strip()]

    check("모든 parent_id 유효", len(invalid_parent_ref) == 0)
    check("빈 Child 텍스트 없음", len(empty_text) == 0)

    # 3. 본문/부칙 Parent 개수
    main_parents = [p for p in parents if p["document_type"] == "main"]
    supplementary_parents = [
        p for p in parents if p["document_type"] == "supplementary"
    ]

    check("본문 Parent 46개", len(main_parents) == 46)
    check("부칙 Parent 3개", len(supplementary_parents) == 3)

    # 4. 제2조제4호 / 제34조제1항 세부 Child 검사
    article2_item4_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제2조-p1-i4"
    subitem_labels = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카"]
    expected_subitem_ids = [
        f"{article2_item4_id}-{label}" for label in subitem_labels
    ]

    check("제2조제4호 전체 Child 유지", article2_item4_id in child_ids)
    check(
        "제2조제4호 가목~카목(11개) 세부 Child 존재",
        all(cid in child_ids for cid in expected_subitem_ids),
    )

    article34_paragraph1_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제34조-p1"
    expected_item_ids = [
        f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제34조-p1-i{i}" for i in range(1, 7)
    ]

    check("제34조제1항 전체 Child 유지", article34_paragraph1_id in child_ids)
    check(
        "제34조제1항 제1호~제6호(6개) 세부 Child 존재",
        all(cid in child_ids for cid in expected_item_ids),
    )

    # 5. 본문 원문 보존 검사
    main_mismatch = []

    for parent in main_parents:
        related_children = [
            c for c in children if c["parent_id"] == parent["id"]
        ]
        base_children = get_main_base_children(
            parent["article"], parent["id"], related_children
        )

        if not base_children:
            main_mismatch.append(parent["id"])
            continue

        reconstructed = "\n".join(c["source_text"] for c in base_children)

        header = (
            f"{parent['article']}({parent['article_title']})"
            if parent["article_title"]
            else parent["article"]
        )
        body = parent["text"]

        if body.startswith(header + "\n"):
            body = body[len(header) + 1:]

        if normalize(reconstructed) != normalize(body):
            main_mismatch.append(parent["id"])

    check("본문 원문 누락 없음 (Parent-Child 재조합 일치)", len(main_mismatch) == 0)

    # 6. 부칙 원문 보존 검사
    supplementary_mismatch = []

    for parent in supplementary_parents:
        related_children = [
            c for c in children if c["parent_id"] == parent["id"]
        ]

        if not related_children:
            supplementary_mismatch.append(parent["id"])
            continue

        reconstructed = "\n".join(c["source_text"] for c in related_children)

        if normalize(reconstructed) != normalize(parent["text"]):
            supplementary_mismatch.append(parent["id"])

    check("부칙 원문 누락 없음", len(supplementary_mismatch) == 0)

    # 7. 메타데이터 일관성 검사
    law_names = {c["law_name"] for c in parents + children}
    law_numbers = {c["law_number"] for c in parents + children}
    main_effective_dates = {
        c["effective_date"]
        for c in parents + children
        if c["document_type"] == "main"
    }

    check("law_name 일관성", len(law_names) == 1)
    check("law_number 일관성", len(law_numbers) == 1)
    check("본문 effective_date 일관성", len(main_effective_dates) == 1)

    # ----- 출력 -----
    article_counts = Counter(child["article"] for child in children)
    granularity_counts = Counter(
        child.get("granularity", "") for child in children
    )

    print("===== 청킹 검증 결과 =====")
    print("Parent 수:", len(parents))
    print("Child 수:", len(children))
    print("본문 Parent 수:", len(main_parents))
    print("부칙 Parent 수:", len(supplementary_parents))

    print("\n===== 세분화 단위별 Child 수 =====")

    for granularity, count in sorted(granularity_counts.items()):
        print(granularity or "(미지정)", ":", count)

    print("\n===== 주요 조문 =====")
    print("제2조 Child 수:", article_counts["제2조"])
    print("제34조 Child 수:", article_counts["제34조"])

    print("\n===== 검증 =====")

    failed = False

    for label, passed in results:
        status = "PASS" if passed else "FAIL"

        if not passed:
            failed = True

        print(f"{status}: {label}")

    if main_mismatch:
        print("FAIL 상세 (본문 원문 불일치):", main_mismatch)

    if supplementary_mismatch:
        print("FAIL 상세 (부칙 원문 불일치):", supplementary_mismatch)

    if invalid_parent_ref:
        print("FAIL 상세 (parent_id 오류):", invalid_parent_ref)

    if empty_text:
        print("FAIL 상세 (빈 텍스트):", empty_text)

    if failed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
