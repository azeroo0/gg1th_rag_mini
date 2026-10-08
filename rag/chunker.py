import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
XML_PATH = BASE_DIR / "data" / "ai_basic_law" / "ai_basic_law.xml"
OUTPUT_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks.json"

LAW_NUMBER = "21311"
LAW_NAME = "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법"
LAW_EFFECTIVE_DATE = "20260721"

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

                    children.append({
                        "id": f"{parent_id}-p{p_index}-i{i_index}",
                        "parent_id": parent_id,
                        "chunk_type": "child",
                        **common_metadata,
                        "paragraph": paragraph["number"],
                        "item": item["number"],
                        "text": f"{header}\n{child_text}",
                        "source_text": child_text,
                    })

            else:
                children.append({
                    "id": f"{parent_id}-p{p_index}",
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    **common_metadata,
                    "paragraph": paragraph["number"],
                    "item": "",
                    "text": f"{header}\n{full_text}",
                    "source_text": full_text,
                })

    else:
        parent_parts.append(parsed["content"])

        children.append({
            "id": f"{parent_id}-full",
            "parent_id": parent_id,
            "chunk_type": "child",
            **common_metadata,
            "paragraph": "",
            "item": "",
            "text": parsed["content"],
            "source_text": parsed["content"],
        })

    parent = {
        "id": parent_id,
        "chunk_type": "parent",
        **common_metadata,
        "text": f"{header}\n" + "\n".join(parent_parts),
    }

    return parent, children

def split_supplementary(content):
    """
    부칙 내용을 조문 단위로 분리한다.

    조문 제목이 없는 부칙은 전체 내용을 하나의 청크로 유지한다.
    첫 조문 앞의 부칙 제목도 첫 번째 청크에 포함한다.
    """

    pattern = (
        r"(?m)^제(\d+)조"
        r"(?:의(\d+))?"
        r"(?:\(([^)\n]+)\))?"
        r"(?=\s|$)"
    )

    matches = list(re.finditer(pattern, content))

    if not matches:
        return [{
            "article": "부칙",
            "text": content.strip(),
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

        article_text = content[start:end].strip()

        if index == 0 and prefix:
            article_text = f"{prefix}\n\n{article_text}"

        article_id = f"제{match.group(1)}조"

        if match.group(2):
            article_id += f"의{match.group(2)}"

        chunks.append({
            "article": article_id,
            "text": article_text,
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
                **common_metadata,
                "article": chunk["article"],
                "paragraph": "",
                "item": "",
                "text": f"{header}\n{chunk['text']}",
                "source_text": chunk["text"],
            }

            children.append(child)

    return parents, children

def validate_chunks(result):
    """
    ID 중복, Parent 참조, 빈 텍스트를 검사한다.
    부칙 Child의 원문 연결 상태도 검사한다.
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

    for parent in parents:
        if parent["document_type"] != "supplementary":
            continue

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

        def normalize(text):
            return re.sub(r"\s+", " ", text).strip()

        if normalize(reconstructed) != normalize(parent["text"]):
            raise ValueError(
                f"부칙 원문 불일치: {parent['id']}"
            )

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

def main():
    result = parse_law()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print("\n===== 검증 결과 =====")
    print("ID 및 Parent 참조 검증 통과")
    print("부칙 원문 보존 검증 통과")
    print("저장 경로:", OUTPUT_PATH)

if __name__ == "__main__":
    main()