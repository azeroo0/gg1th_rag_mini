from pathlib import Path
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent.parent
XML_PATH = BASE_DIR / "data" / "ai_basic_law" / "ai_basic_law.xml"

def parse_article(article):
    number = article.findtext("조문번호", default="")
    branch = article.findtext("조문가지번호", default="")
    title = article.findtext("조문제목", default="")
    effective_date = article.findtext("조문시행일자", default="")

    article_id = f"제{number}조"

    if branch and branch != "0":
        article_id += f"의{branch}"

    result = {
        "article": article_id,
        "title": title,
        "effective_date": effective_date,
        "paragraphs": [],
    }

    for paragraph in article.findall("항"):
        paragraph_data = {
            "number": paragraph.findtext("항번호", default=""),
            "content": paragraph.findtext("항내용", default="").strip(),
            "items": [],
        }

        for item in paragraph.findall("호"):
            item_data = {
                "number": item.findtext("호번호", default=""),
                "content": item.findtext("호내용", default="").strip(),
                "subitems": [],
            }

            for subitem in item.findall("목"):
                item_data["subitems"].append({
                    "number": subitem.findtext("목번호", default=""),
                    "content": subitem.findtext(
                        "목내용", default=""
                    ).strip(),
                })

            paragraph_data["items"].append(item_data)

        result["paragraphs"].append(paragraph_data)

    return result

def main():
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    for article in root.findall(".//조문단위"):
        number = article.findtext("조문번호", default="")
        article_type = article.findtext("조문여부", default="")

        if number == "34" and article_type == "조문":
            parsed = parse_article(article)

            print("조문:", parsed["article"])
            print("제목:", parsed["title"])
            print("시행일:", parsed["effective_date"])

            for paragraph in parsed["paragraphs"]:
                print("\n항:", paragraph["number"])
                print("내용:", paragraph["content"])

                for item in paragraph["items"]:
                    print("  호:", item["number"], item["content"])

                    for subitem in item["subitems"]:
                        print(
                            "    목:",
                            subitem["number"],
                            subitem["content"],
                        )

if __name__ == "__main__":
    main()