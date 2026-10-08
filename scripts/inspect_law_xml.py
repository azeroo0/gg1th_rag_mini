from pathlib import Path
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent.parent
XML_PATH = BASE_DIR / "data" / "ai_basic_law" / "ai_basic_law.xml"

tree = ET.parse(XML_PATH)
root = tree.getroot()

articles = root.findall(".//조문단위")

print(f"전체 조문단위 수: {len(articles)}")

print("\n===== 조문 목록 =====")

for article in articles:
    number = article.findtext("조문번호", default="")
    branch = article.findtext("조문가지번호", default="")
    title = article.findtext("조문제목", default="")
    content = article.findtext("조문내용", default="").strip()
    article_type = article.findtext("조문여부", default="")

    article_id = f"제{number}조"
    if branch and branch != "0":
        article_id += f"의{branch}"

    print(
        f"{article_id} | "
        f"유형: {article_type} | "
        f"제목: {title} | "
        f"내용: {content[:80]}"
    )

print("\n===== 제2조 / 제34조 상세 구조 =====")

for article in articles:
    number = article.findtext("조문번호", default="")
    branch = article.findtext("조문가지번호", default="")

    if number in ("2", "34") and branch in ("", "0"):
        print(f"\n--- 제{number}조 ---")
        print(ET.tostring(
            article,
            encoding="unicode"
        )[:6000])