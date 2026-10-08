from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

BASE_DIR = Path(__file__).resolve().parent.parent
XML_PATH = BASE_DIR / "data" / "ai_basic_law" / "ai_basic_law.xml"

root = ET.parse(XML_PATH).getroot()

print("===== XML 최상위 구조 =====")
for child in root:
    print(child.tag)

print("\n===== 부칙 / 별표 / 서식 관련 태그 =====")

keywords = ("부칙", "별표", "서식", "별지")
counts = Counter()

for element in root.iter():
    if any(keyword in element.tag for keyword in keywords):
        counts[element.tag] += 1

for tag, count in sorted(counts.items()):
    print(f"{tag}: {count}")

print("\n===== 관련 XML 샘플 =====")

for keyword in keywords:
    matches = [
        element
        for element in root.iter()
        if keyword in element.tag
    ]

    print(f"\n[{keyword}] 발견된 요소: {len(matches)}")

    if matches:
        sample = ET.tostring(
            matches[0],
            encoding="unicode"
        )
        print(sample[:3000])
    else:
        print("관련 태그 없음")