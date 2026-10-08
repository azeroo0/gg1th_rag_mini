import os
from pathlib import Path
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent

# .env 로드
load_dotenv(BASE_DIR / ".env")

API_OC = os.getenv("LAW_API_OC")

if not API_OC:
    raise ValueError("LAW_API_OC가 설정되지 않았습니다.")

# 법령 본문 조회 API
URL = "https://www.law.go.kr/DRF/lawService.do"

# STEP 1에서 확인한 정확한 법령 버전
params = {
    "OC": API_OC,
    "target": "eflaw",
    "MST": "282791",
    "efYd": "20260721",
    "type": "XML",
}

# 저장 위치
SAVE_DIR = BASE_DIR / "data" / "ai_basic_law"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PATH = SAVE_DIR / "ai_basic_law.xml"

try:
    response = requests.get(
        URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    # XML 파싱 검증
    root = ET.fromstring(response.content)

    # 오류 응답이 아니라 실제 법령 데이터인지 확인
    if root.find(".//조문단위") is None:
        raise ValueError(
            "응답에서 조문단위를 찾지 못했습니다. "
            f"응답 시작: {response.text[:500]}"
        )

    # XML 파일 저장
    SAVE_PATH.write_bytes(response.content)

    print("법령 본문 다운로드 성공")
    print("HTTP 상태:", response.status_code)
    print("XML Root:", root.tag)
    print("저장 위치:", SAVE_PATH)

    # 조문 개수 확인
    articles = root.findall(".//조문단위")
    print("조문단위 개수:", len(articles))

    # 샘플 조문 출력
    if articles:
        first = articles[0]
        print("\n===== 첫 번째 조문 =====")
        print(ET.tostring(first, encoding="unicode")[:1000])

except (requests.RequestException, ET.ParseError, ValueError) as e:
    print("법령 본문 조회 실패:", e)