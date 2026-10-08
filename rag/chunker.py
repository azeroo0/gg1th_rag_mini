import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
XML_PATH = BASE_DIR / "data" / "ai_basic_law" / "ai_basic_law.xml"
OUTPUT_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks.json"

LAW_NUMBER = "21311"
LAW_NAME = "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법"
LAW_EFFECTIVE_DATE = "20260721"

# 항이 길고 호마다 서로 다른 의무/조건을 규정하여, 항 전체 Child만으로는
# 검색 정확도가 떨어지는 경우에 한해 호 단위 세부 Child를 추가 생성한다.
# key: (조문 ID, 항 위치 인덱스 1부터 시작)
ITEM_SPLIT_RULES = {
    ("제34조", 1): (
        "사업자가 이행해야 하는 조치를 6개 호로 열거하며, "
        "각 호가 위험관리ㆍ설명ㆍ이용자보호ㆍ관리감독ㆍ문서화 등 "
        "서로 다른 의무를 규정하므로 호 단위 세부 Child를 추가한다."
    ),
}

# 호 내용이 길고 목마다 서로 다른 적용 범위를 규정하여 목 단위
# 세부 Child가 필요한 경우에 한해 추가 생성한다.
# key: (조문 ID, 호번호 문자열에서 마지막 '.' 제거)
SUBITEM_SPLIT_RULES = {
    ("제2조", "4"): (
        "고영향 인공지능의 정의가 가목부터 카목까지 11개의 서로 독립적인 "
        "적용 영역을 열거하므로 목 단위 세부 Child를 추가한다."
    ),
}

def get_text(element, tag):
    return (element.findtext(tag) or "").strip()

def parse_article(article):
    number = get_text(article, "조문번호")
    branch = get_text(article, "조문가지번호")
    title = get_text(article, "조문제목")
    effective_date = get_text(article, "조문시행일자")

    article_id = f"제{number}조"

    if branch and branch != "0":
        article_id += f"의{branch}"

    result = {
        "article": article_id,
        "title": title,
        "effective_date": effective_date,
        "content": get_text(article, "조문내용"),
        "paragraphs": [],
    }

    for paragraph in article.findall("항"):
        paragraph_data = {
            "number": get_text(paragraph, "항번호"),
            "content": get_text(paragraph, "항내용"),
            "items": [],
        }

        for item in paragraph.findall("호"):
            item_data = {
                "number": get_text(item, "호번호"),
                "content": get_text(item, "호내용"),
                "subitems": [],
            }

            for subitem in item.findall("목"):
                item_data["subitems"].append({
                    "number": get_text(subitem, "목번호"),
                    "content": get_text(subitem, "목내용"),
                })

            paragraph_data["items"].append(item_data)

        result["paragraphs"].append(paragraph_data)

    return result

def build_paragraph_text(paragraph):
    parts = [paragraph["content"]]

    for item in paragraph["items"]:
        parts.append(item["content"])

        for subitem in item["subitems"]:
            parts.append(subitem["content"])

    return "\n".join(part for part in parts if part)

def build_item_children(paragraph, p_index, parent_id, header, common_metadata):
    """
    항 전체 Child는 유지한 채, 호 단위 세부 Child를 추가로 생성한다.
    text에는 상위 항의 조건을 포함하고, source_text는 호(및 목) 원문만 사용한다.
    """

    paragraph_condition = paragraph["content"]
    children = []

    for i_index, item in enumerate(paragraph["items"], 1):
        item_parts = [item["content"]]

        for subitem in item["subitems"]:
            item_parts.append(subitem["content"])

        source_text = "\n".join(part for part in item_parts if part)

        text_parts = [header]

        if paragraph_condition:
            text_parts.append(paragraph_condition)

        text_parts.append(source_text)

        children.append({
            "id": f"{parent_id}-p{p_index}-i{i_index}",
            "parent_id": parent_id,
            "chunk_type": "child",
            **common_metadata,
            "paragraph": paragraph["number"],
            "item": item["number"],
            "subitem": "",
            "granularity": "item",
            "text": "\n".join(text_parts),
            "source_text": source_text,
        })

    return children

def build_subitem_children(item, item_id, parent_id, header, common_metadata, paragraph_number):
    """
    호 전체 Child는 유지한 채, 목 단위 세부 Child를 추가로 생성한다.
    text에는 조문 제목과 호(상위) 정의를 함께 제공하고,
    source_text는 목 원문만 사용한다.
    """

    item_definition = item["content"]
    children = []

    for subitem in item["subitems"]:
        label = subitem["number"].rstrip(".")
        source_text = subitem["content"]

        text_parts = [header]

        if item_definition:
            text_parts.append(item_definition)

        text_parts.append(source_text)

        children.append({
            "id": f"{item_id}-{label}",
            "parent_id": parent_id,
            "chunk_type": "child",
            **common_metadata,
            "paragraph": paragraph_number,
            "item": item["number"],
            "subitem": subitem["number"],
            "granularity": "subitem",
            "text": "\n".join(part for part in text_parts if part),
            "source_text": source_text,
        })

    return children

def build_article_chunks(parsed, chapter="", section=""):
    article_id = parsed["article"]
    title = parsed["title"]
    effective_date = LAW_EFFECTIVE_DATE

    parent_id = f"{LAW_NUMBER}-{effective_date}-{article_id}"
    header = f"{article_id}({title})" if title else article_id

    parent_parts = []
    children = []

    common_metadata = {
        "document_type": "main",
        "law_name": LAW_NAME,
        "law_number": LAW_NUMBER,
        "effective_date": effective_date,
        "chapter": chapter,
        "section": section,
        "article": article_id,
        "article_title": title,
    }

    if parsed["paragraphs"]:
        for p_index, paragraph in enumerate(parsed["paragraphs"], 1):
            full_text = build_paragraph_text(paragraph)
            parent_parts.append(full_text)

            if article_id == "제2조" and paragraph["items"]:
                # 제2조는 항번호/항내용이 없는 정의 조항이므로
                # 호 단위 Child를 기본 단위로 유지한다. (기존 동작 유지)
                for i_index, item in enumerate(paragraph["items"], 1):
                    item_parts = [
                        paragraph["content"],
                        item["content"],
                    ]

                    for subitem in item["subitems"]:
                        item_parts.append(subitem["content"])

                    child_text = "\n".join(
                        part for part in item_parts if part
                    )

                    item_id = f"{parent_id}-p{p_index}-i{i_index}"

                    children.append({
                        "id": item_id,
                        "parent_id": parent_id,
                        "chunk_type": "child",
                        **common_metadata,
                        "paragraph": paragraph["number"],
                        "item": item["number"],
                        "subitem": "",
                        "granularity": "item",
                        "text": f"{header}\n{child_text}",
                        "source_text": child_text,
                    })

                    subitem_key = (article_id, item["number"].rstrip("."))

                    if SUBITEM_SPLIT_RULES.get(subitem_key) and item["subitems"]:
                        children.extend(
                            build_subitem_children(
                                item=item,
                                item_id=item_id,
                                parent_id=parent_id,
                                header=header,
                                common_metadata=common_metadata,
                                paragraph_number=paragraph["number"],
                            )
                        )

            else:
                children.append({
                    "id": f"{parent_id}-p{p_index}",
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    **common_metadata,
                    "paragraph": paragraph["number"],
                    "item": "",
                    "subitem": "",
                    "granularity": "paragraph",
                    "text": f"{header}\n{full_text}",
                    "source_text": full_text,
                })

                item_key = (article_id, p_index)

                if ITEM_SPLIT_RULES.get(item_key) and paragraph["items"]:
                    children.extend(
                        build_item_children(
                            paragraph=paragraph,
                            p_index=p_index,
                            parent_id=parent_id,
                            header=header,
                            common_metadata=common_metadata,
                        )
                    )

    else:
        parent_parts.append(parsed["content"])

        children.append({
            "id": f"{parent_id}-full",
            "parent_id": parent_id,
            "chunk_type": "child",
            **common_metadata,
            "paragraph": "",
            "item": "",
            "subitem": "",
            "granularity": "article",
            "text": parsed["content"],
            "source_text": parsed["content"],
        })

    parent = {
        "id": parent_id,
        "chunk_type": "parent",
        **common_metadata,
        "paragraph": "",
        "item": "",
        "subitem": "",
        "granularity": "article",
        "text": f"{header}\n" + "\n".join(parent_parts),
    }

    return parent, children

def split_supplementary(content):
    """
    부칙의 조문, 생략 안내, 조문 번호가 없는 내용을 구분한다.
    원문의 순서와 텍스트를 보존한다.
    """

    pattern = (
        r"(?m)^(?:"
        r"제\d+조(?:의\d+)?(?:\([^)]+\))?"
        r"|제\d+조부터\s*제\d+조까지\s*생략"
        r")(?=\s|$)"
        r"|제\d+조(?:의\d+)?\s+생략"
    )

    matches = list(re.finditer(pattern, content))

    if not matches:
        return [{
            "article": "부칙",
            "text": content.strip(),
            "content_type": "general",
        }]

    chunks = []
    prefix = content[:matches[0].start()].strip()

    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(content)
        )

        chunk_text = content[start:end].strip()
        matched_heading = match.group()

        is_omission = bool(
            re.match(
                r"제\d+조부터\s*제\d+조까지\s*생략",
                matched_heading,
            )
        ) or bool(
            re.fullmatch(
                r"제\d+조(?:의\d+)?\s*생략",
                matched_heading,
            )
        )

        if is_omission:
            article_id = matched_heading
            content_type = "omission"
        else:
            article_match = re.match(
                r"제\d+조(?:의\d+)?",
                matched_heading,
            )
            article_id = article_match.group()
            content_type = "article"

        if index == 0 and prefix:
            chunk_text = f"{prefix}\n\n{chunk_text}"

        chunks.append({
            "article": article_id,
            "text": chunk_text,
            "content_type": content_type,
        })

    return chunks

def parse_supplementary(root):
    """
    부칙단위별 Parent를 생성하고
    부칙 내부 조문별 Child를 생성한다.
    """

    parents = []
    children = []

    for supplement in root.findall("./부칙/부칙단위"):
        promulgation_number = get_text(
            supplement, "부칙공포번호"
        )
        promulgation_date = get_text(
            supplement, "부칙공포일자"
        )
        content = get_text(supplement, "부칙내용")

        if not content:
            continue

        supplement_key = supplement.get("부칙키", "")

        parent_id = (
            f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}"
            f"-supplementary-{promulgation_number}"
            f"-{promulgation_date}"
        )

        header = (
            f"{LAW_NAME} 부칙 "
            f"제{promulgation_number}호 "
            f"({promulgation_date})"
        )

        common_metadata = {
            "document_type": "supplementary",
            "law_name": LAW_NAME,
            "law_number": LAW_NUMBER,
            "effective_date": LAW_EFFECTIVE_DATE,
            "promulgation_number": promulgation_number,
            "promulgation_date": promulgation_date,
            "supplementary_key": supplement_key,
            "chapter": "",
            "section": "",
            "article_title": "",
        }

        parent = {
            "id": parent_id,
            "chunk_type": "parent",
            **common_metadata,
            "article": "부칙",
            "paragraph": "",
            "item": "",
            "subitem": "",
            "granularity": "supplementary",
            "text": content,
        }

        parents.append(parent)

        supplementary_chunks = split_supplementary(content)

        for index, chunk in enumerate(
            supplementary_chunks, start=1
        ):
            child = {
                "id": f"{parent_id}-c{index}",
                "parent_id": parent_id,
                "chunk_type": "child",
                "content_type": chunk["content_type"],
                **common_metadata,
                "article": chunk["article"],
                "paragraph": "",
                "item": "",
                "subitem": "",
                "granularity": "supplementary",
                "text": f"{header}\n{chunk['text']}",
                "source_text": chunk["text"],
            }

            children.append(child)

    return parents, children

def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

def get_main_base_children(article_id, parent_id, related_children):
    """
    항목 추가 이전부터 존재하던 '기본 단위' Child만 골라낸다.
    (제2조는 호 단위, 그 외 조문은 항 단위가 기본 단위이며,
    항이 없는 조문은 -full 하나가 기본 단위다.)
    이 기본 단위 Child들만 모으면 조문 원문과 공백만 다르게 일치해야 한다.
    """

    full_id = f"{parent_id}-full"

    if any(child["id"] == full_id for child in related_children):
        return [c for c in related_children if c["id"] == full_id]

    if article_id == "제2조":
        pattern = re.compile(r"^" + re.escape(parent_id) + r"-p\d+-i\d+$")
    else:
        pattern = re.compile(r"^" + re.escape(parent_id) + r"-p\d+$")

    return [c for c in related_children if pattern.match(c["id"])]

def validate_chunks(result):
    """
    ID 중복, Parent 참조, 빈 텍스트, Parent/Child 개수,
    제2조/제34조 세부 Child 존재 여부, 법률 원문 보존(본문+부칙),
    메타데이터 일관성을 검사한다.
    """

    parents = result["parents"]
    children = result["children"]

    parent_ids = [parent["id"] for parent in parents]
    child_ids = [child["id"] for child in children]

    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("Parent ID 중복이 발견되었습니다.")

    if len(child_ids) != len(set(child_ids)):
        raise ValueError("Child ID 중복이 발견되었습니다.")

    parent_map = {
        parent["id"]: parent for parent in parents
    }

    for child in children:
        if child["parent_id"] not in parent_map:
            raise ValueError(
                f"Parent 참조 오류: {child['id']}"
            )

        if not child["text"].strip():
            raise ValueError(
                f"빈 Child 텍스트: {child['id']}"
            )

    # 본문/부칙 Parent 개수 검사
    main_parents = [p for p in parents if p["document_type"] == "main"]
    supplementary_parents = [
        p for p in parents if p["document_type"] == "supplementary"
    ]

    if len(main_parents) != 46:
        raise ValueError(
            f"본문 Parent 개수 불일치: {len(main_parents)} (기대값 46)"
        )

    if len(supplementary_parents) != 3:
        raise ValueError(
            f"부칙 Parent 개수 불일치: {len(supplementary_parents)} (기대값 3)"
        )

    # 제2조제4호 / 제34조제1항 필수 세부 Child 검사
    article2_parent_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제2조"
    article2_item4_id = f"{article2_parent_id}-p1-i4"
    article34_parent_id = f"{LAW_NUMBER}-{LAW_EFFECTIVE_DATE}-제34조"
    article34_paragraph1_id = f"{article34_parent_id}-p1"

    if article2_item4_id not in child_ids:
        raise ValueError("제2조제4호 전체 Child가 존재하지 않습니다.")

    subitem_labels = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카"]
    expected_subitem_ids = [
        f"{article2_item4_id}-{label}" for label in subitem_labels
    ]

    missing_subitems = [
        cid for cid in expected_subitem_ids if cid not in child_ids
    ]

    if missing_subitems:
        raise ValueError(
            f"제2조제4호 가목~카목 세부 Child 누락: {missing_subitems}"
        )

    if article34_paragraph1_id not in child_ids:
        raise ValueError("제34조제1항 전체 Child가 존재하지 않습니다.")

    expected_item_ids = [
        f"{article34_parent_id}-p1-i{i}" for i in range(1, 7)
    ]

    missing_items = [
        cid for cid in expected_item_ids if cid not in child_ids
    ]

    if missing_items:
        raise ValueError(
            f"제34조제1항 제1호~제6호 세부 Child 누락: {missing_items}"
        )

    # 본문 조문 원문 보존 검사 (기본 단위 Child 재조합 == Parent 원문)
    for parent in main_parents:
        related_children = [
            child for child in children if child["parent_id"] == parent["id"]
        ]

        base_children = get_main_base_children(
            parent["article"], parent["id"], related_children
        )

        if not base_children:
            raise ValueError(f"본문 기본 단위 Child 누락: {parent['id']}")

        reconstructed = "\n".join(
            child["source_text"] for child in base_children
        )

        header = (
            f"{parent['article']}({parent['article_title']})"
            if parent["article_title"]
            else parent["article"]
        )

        body = parent["text"]

        if body.startswith(header + "\n"):
            body = body[len(header) + 1:]

        if normalize_whitespace(reconstructed) != normalize_whitespace(body):
            raise ValueError(f"본문 원문 불일치: {parent['id']}")

    # 부칙 Child의 원문 연결 상태 검사
    for parent in supplementary_parents:
        related_children = [
            child
            for child in children
            if child["parent_id"] == parent["id"]
        ]

        if not related_children:
            raise ValueError(
                f"부칙 Child 누락: {parent['id']}"
            )

        reconstructed = "\n".join(
            child["source_text"]
            for child in related_children
        )

        if normalize_whitespace(reconstructed) != normalize_whitespace(parent["text"]):
            raise ValueError(
                f"부칙 원문 불일치: {parent['id']}"
            )

    # 동일 법령 버전의 메타데이터 일관성 검사
    law_names = {c["law_name"] for c in parents + children}
    law_numbers = {c["law_number"] for c in parents + children}
    main_effective_dates = {
        c["effective_date"] for c in parents + children if c["document_type"] == "main"
    }

    if len(law_names) != 1:
        raise ValueError(f"law_name 불일치: {law_names}")

    if len(law_numbers) != 1:
        raise ValueError(f"law_number 불일치: {law_numbers}")

    if len(main_effective_dates) != 1:
        raise ValueError(f"본문 effective_date 불일치: {main_effective_dates}")

    return True

def parse_law():
    root = ET.parse(XML_PATH).getroot()

    all_parents = []
    all_children = []

    current_chapter = ""
    current_section = ""

    # 본문 조문 처리
    for article in root.findall("./조문/조문단위"):
        article_type = get_text(article, "조문여부")
        content = get_text(article, "조문내용")

        if article_type != "조문":
            chapter_match = re.search(
                r"제\d+장\s+.+", content
            )
            section_match = re.search(
                r"제\d+절\s+.+", content
            )

            if chapter_match:
                current_chapter = chapter_match.group()
                current_section = ""

            elif section_match:
                current_section = section_match.group()

            continue

        parsed = parse_article(article)

        parent, children = build_article_chunks(
            parsed,
            chapter=current_chapter,
            section=current_section,
        )

        all_parents.append(parent)
        all_children.extend(children)

    # 부칙 처리
    supplementary_parents, supplementary_children = (
        parse_supplementary(root)
    )

    all_parents.extend(supplementary_parents)
    all_children.extend(supplementary_children)

    # 별표 및 서식 존재 여부 확인
    appendix_count = len(root.findall(".//별표단위"))
    form_count = len(root.findall(".//서식단위"))

    result = {
        "metadata": {
            "law_name": LAW_NAME,
            "law_number": LAW_NUMBER,
            "effective_date": LAW_EFFECTIVE_DATE,
            "chunking_strategy": "hierarchical_parent_child",
            "source_types": [
                "main",
                "supplementary",
            ],
            "appendix_count": appendix_count,
            "form_count": form_count,
        },
        "parents": all_parents,
        "children": all_children,
    }

    validate_chunks(result)

    return result

BACKUP_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks.backup.json"

def backup_existing_output():
    """
    기존 결과 JSON을 덮어쓰기 전에 백업한다.
    """

    if OUTPUT_PATH.exists():
        BACKUP_PATH.write_text(
            OUTPUT_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

def main():
    result = parse_law()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    backup_existing_output()

    OUTPUT_PATH.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    parents = result["parents"]
    children = result["children"]

    main_parents = [
        p for p in parents
        if p["document_type"] == "main"
    ]

    main_children = [
        c for c in children
        if c["document_type"] == "main"
    ]

    supplementary_parents = [
        p for p in parents
        if p["document_type"] == "supplementary"
    ]

    supplementary_children = [
        c for c in children
        if c["document_type"] == "supplementary"
    ]

    print("===== 전체 법률 청킹 결과 =====")
    print("본문 Parent:", len(main_parents))
    print("본문 Child:", len(main_children))
    print("부칙 Parent:", len(supplementary_parents))
    print("부칙 Child:", len(supplementary_children))
    print("전체 Parent:", len(parents))
    print("전체 Child:", len(children))

    print("\n===== 별표 및 서식 =====")
    print("별표단위:", result["metadata"]["appendix_count"])
    print("서식단위:", result["metadata"]["form_count"])

    print("\n===== 부칙 청킹 상세 =====")

    for child in supplementary_children:
        print(
            child["id"],
            "|",
            child["article"],
            "|",
            len(child["text"]),
            "chars",
        )

    granularity_counts = Counter(
        child["granularity"] for child in main_children
    )

    print("\n===== 본문 세분화 단위 =====")

    for granularity, count in sorted(granularity_counts.items()):
        print(granularity, ":", count)

    print("\n===== 세부 Child 추가 규칙 =====")

    for (article_id, p_index), reason in ITEM_SPLIT_RULES.items():
        print(f"{article_id} 제{p_index}항 (호 단위 추가):", reason)

    for (article_id, item_number), reason in SUBITEM_SPLIT_RULES.items():
        print(f"{article_id} 제{item_number}호 (목 단위 추가):", reason)

    print("\n===== 검증 결과 =====")
    print("ID 및 Parent 참조 검증 통과")
    print("Parent/Child 개수 검증 통과")
    print("제2조제4호/제34조제1항 세부 Child 검증 통과")
    print("본문 원문 보존 검증 통과")
    print("부칙 원문 보존 검증 통과")
    print("메타데이터 일관성 검증 통과")
    print("저장 경로:", OUTPUT_PATH)

if __name__ == "__main__":
    main()