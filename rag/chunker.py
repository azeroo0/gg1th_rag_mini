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

    if parsed["paragraphs"]:
        for p_index, paragraph in enumerate(parsed["paragraphs"], 1):
            full_text = build_paragraph_text(paragraph)
            parent_parts.append(full_text)

            # 정의 조항은 호 단위로 세분화
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
                        "law_name": LAW_NAME,
                        "law_number": LAW_NUMBER,
                        "effective_date": effective_date,
                        "chapter": chapter,
                        "section": section,
                        "article": article_id,
                        "article_title": title,
                        "paragraph": paragraph["number"],
                        "item": item["number"],
                        "text": f"{header}\n{child_text}",
                    })

            else:
                children.append({
                    "id": f"{parent_id}-p{p_index}",
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "law_name": LAW_NAME,
                    "law_number": LAW_NUMBER,
                    "effective_date": effective_date,
                    "chapter": chapter,
                    "section": section,
                    "article": article_id,
                    "article_title": title,
                    "paragraph": paragraph["number"],
                    "item": "",
                    "text": f"{header}\n{full_text}",
                })

    else:
        parent_parts.append(parsed["content"])

        children.append({
            "id": f"{parent_id}-full",
            "parent_id": parent_id,
            "chunk_type": "child",
            "law_name": LAW_NAME,
            "law_number": LAW_NUMBER,
            "effective_date": effective_date,
            "chapter": chapter,
            "section": section,
            "article": article_id,
            "article_title": title,
            "paragraph": "",
            "item": "",
            "text": parsed["content"],
        })

    parent = {
        "id": parent_id,
        "chunk_type": "parent",
        "law_name": LAW_NAME,
        "law_number": LAW_NUMBER,
        "effective_date": effective_date,
        "chapter": chapter,
        "section": section,
        "article": article_id,
        "article_title": title,
        "text": f"{header}\n" + "\n".join(parent_parts),
    }

    return parent, children

def parse_law():
    root = ET.parse(XML_PATH).getroot()

    all_parents = []
    all_children = []

    current_chapter = ""
    current_section = ""

    for article in root.findall(".//조문단위"):
        article_type = get_text(article, "조문여부")
        content = get_text(article, "조문내용")

        if article_type != "조문":
            chapter_match = re.search(r"제\d+장\s+.+", content)
            section_match = re.search(r"제\d+절\s+.+", content)

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

    return {
        "metadata": {
            "law_name": LAW_NAME,
            "law_number": LAW_NUMBER,
            "effective_date": LAW_EFFECTIVE_DATE,
            "chunking_strategy": "hierarchical_parent_child",
        },
        "parents": all_parents,
        "children": all_children,
    }

def main():
    result = parse_law()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("전체 법률 청킹 완료")
    print("Parent Chunk 수:", len(result["parents"]))
    print("Child Chunk 수:", len(result["children"]))
    print("저장 위치:", OUTPUT_PATH)

    print("\n===== Child Chunk 예시 =====")

    for child in result["children"][:3]:
        print("\nID:", child["id"])
        print("조문:", child["article"])
        print("내용:", child["text"][:200])

if __name__ == "__main__":
    main()