import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from rag.chunker import parse_article

XML_PATH = BASE_DIR / "data" / "ai_basic_law" / "ai_basic_law.xml"
OUTPUT_PATH = BASE_DIR / "data" / "ai_basic_law" / "chunk_test.json"

def build_chunks(article):
    parsed = parse_article(article)

    article_id = parsed["article"]
    title = parsed["title"]
    effective_date = parsed["effective_date"]

    parent_id = f"{effective_date}-{article_id}"

    header = f"{article_id}({title})"
    parent_parts = [header]
    children = []

    for index, paragraph in enumerate(parsed["paragraphs"], start=1):
        paragraph_text = paragraph["content"]

        item_texts = []

        for item in paragraph["items"]:
            item_texts.append(item["content"])

            for subitem in item["subitems"]:
                item_texts.append(subitem["content"])

        full_text = "\n".join(
            part for part in [paragraph_text, *item_texts] if part
        )

        parent_parts.append(full_text)

        child = {
            "id": f"{parent_id}-p{index}",
            "parent_id": parent_id,
            "chunk_type": "child",
            "article": article_id,
            "article_title": title,
            "paragraph": paragraph["number"],
            "effective_date": effective_date,
            "text": f"{header}\n{full_text}",
        }

        children.append(child)

    parent = {
        "id": parent_id,
        "chunk_type": "parent",
        "article": article_id,
        "article_title": title,
        "effective_date": effective_date,
        "text": "\n".join(parent_parts),
    }

    return {
        "parent": parent,
        "children": children,
    }

def main():
    root = ET.parse(XML_PATH).getroot()

    for article in root.findall(".//조문단위"):
        if (
            article.findtext("조문번호") == "34"
            and article.findtext("조문여부") == "조문"
        ):
            result = build_chunks(article)

            OUTPUT_PATH.write_text(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print("청킹 완료")
            print("Parent ID:", result["parent"]["id"])
            print("Parent 수:", 1)
            print("Child 수:", len(result["children"]))
            print("저장 경로:", OUTPUT_PATH)

            for child in result["children"]:
                print("\nChild ID:", child["id"])
                print("조항:", child["article"], child["paragraph"])
                print("텍스트 길이:", len(child["text"]))

            return

    raise ValueError("제34조를 찾지 못했습니다.")

if __name__ == "__main__":
    main()