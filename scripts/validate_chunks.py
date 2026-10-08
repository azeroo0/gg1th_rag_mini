import json
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks.json"

# 정의 전용 Child 추가 이전(205 Child) 스냅샷.
# 기존 Child의 ID/text/source_text가 보존되었는지 비교하는 기준으로 사용한다.
V1_JSON_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks_v1.json"

EXPECTED_V1_CHILD_COUNT = 205
EXPECTED_CHILD_COUNT = 206
EXPECTED_PARENT_COUNT = 49

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
    article2_parent_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제2조"
    article2_item4_id = f"{article2_parent_id}-p1-i4"
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

    # 4-1. 전체 Parent/Child 개수
    check(
        f"전체 Parent {EXPECTED_PARENT_COUNT}개",
        len(parents) == EXPECTED_PARENT_COUNT,
    )
    check(
        f"전체 Child {EXPECTED_CHILD_COUNT}개",
        len(children) == EXPECTED_CHILD_COUNT,
    )

    supplementary_children = [
        c for c in children if c["document_type"] == "supplementary"
    ]
    check("부칙 Child 8개 유지", len(supplementary_children) == 8)

    # 4-2. 제2조제4호 정의 전용 Child 검사
    child_map = {c["id"]: c for c in children}
    definition_id = f"{article2_item4_id}-definition"
    definition_child = child_map.get(definition_id)

    definition_only_children = [
        c for c in children if c.get("chunk_role") == "definition_only"
    ]

    check("제2조제4호 정의 전용 Child 존재", definition_child is not None)
    check("정의 전용 Child 정확히 1개", len(definition_only_children) == 1)

    if definition_child is not None:
        item4_child = child_map[article2_item4_id]

        check(
            "정의 전용 Child chunk_role 정확",
            definition_child.get("chunk_role") == "definition_only",
        )
        check(
            "정의 전용 source_text가 제2조제4호 원문의 일부",
            definition_child["source_text"] in item4_child["source_text"],
        )
        check(
            "정의 전용 source_text에 가목 이하 열거 목록 미포함",
            "\n" not in definition_child["source_text"],
        )
        check(
            "정의 전용 Child granularity=item",
            definition_child.get("granularity") == "item",
        )
        check(
            "정의 전용 Child parent_id가 제2조 Parent",
            definition_child.get("parent_id") == article2_parent_id,
        )

    # 4-3. v1 스냅샷 대비 기존 Child 보존 검사
    v1_children = None

    if V1_JSON_PATH.exists():
        with open(V1_JSON_PATH, "r", encoding="utf-8") as file:
            v1_children = json.load(file)["children"]

    if v1_children is None:
        check(
            f"v1 스냅샷 존재 ({V1_JSON_PATH.name})",
            False,
        )
    else:
        v1_map = {c["id"]: c for c in v1_children}

        check(
            f"v1 스냅샷 Child {EXPECTED_V1_CHILD_COUNT}개",
            len(v1_children) == EXPECTED_V1_CHILD_COUNT,
        )
        check(
            "기존 Child ID 전체 유지",
            set(v1_map).issubset(set(child_map)),
        )

        text_changed = [
            cid for cid in v1_map
            if cid in child_map
            and (
                v1_map[cid]["text"] != child_map[cid]["text"]
                or v1_map[cid]["source_text"] != child_map[cid]["source_text"]
            )
        ]

        check("기존 Child text/source_text 변경 없음", len(text_changed) == 0)

        added_ids = sorted(set(child_map) - set(v1_map))

        check(
            "신규 Child는 정의 전용 Child 1개뿐",
            added_ids == [definition_id],
        )

        if text_changed:
            print("FAIL 상세 (기존 Child 변경):", text_changed)

        if added_ids != [definition_id]:
            print("FAIL 상세 (예상 밖 신규 Child):", added_ids)

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

    role_counts = Counter(
        child.get("chunk_role", "") for child in children
    )

    print("\n===== chunk_role 분포 =====")

    for role, count in sorted(role_counts.items()):
        print(role or "(미지정)", ":", count)

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
