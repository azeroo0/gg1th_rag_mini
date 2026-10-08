import json
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks.json"

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    parents = data["parents"]
    children = data["children"]

    parent_ids = [parent["id"] for parent in parents]
    child_ids = [child["id"] for child in children]
    parent_id_set = set(parent_ids)

    errors = []

    if len(parent_ids) != len(set(parent_ids)):
        errors.append("Parent ID 중복 발견")

    if len(child_ids) != len(set(child_ids)):
        errors.append("Child ID 중복 발견")

    for child in children:
        if child["parent_id"] not in parent_id_set:
            errors.append(f"Parent 참조 오류: {child['id']}")

        if not child["text"].strip():
            errors.append(f"빈 텍스트 발견: {child['id']}")

    article_counts = Counter(
        child["article"] for child in children
    )

    print("===== 청킹 검증 결과 =====")
    print("Parent 수:", len(parents))
    print("Child 수:", len(children))
    print("전체 조문 수:", len(set(parent["article"] for parent in parents)))

    print("\n===== 주요 조문 =====")
    print("제2조 Child 수:", article_counts["제2조"])
    print("제34조 Child 수:", article_counts["제34조"])

    print("\n===== 검증 =====")

    if errors:
        for error in errors:
            print("FAIL:", error)
        raise SystemExit(1)

    print("PASS: ID 및 Parent 참조 검증 완료")
    print("PASS: 빈 Child 텍스트 없음")

    print("\n참고: 원문 누락 및 부칙 포함 여부는 별도 검증 필요")

if __name__ == "__main__":
    main()