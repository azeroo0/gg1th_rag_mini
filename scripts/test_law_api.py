import os
import requests
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
load_dotenv()

API_OC = os.getenv("LAW_API_OC")

if not API_OC:
    raise ValueError("LAW_API_OC가 .env에 설정되지 않았습니다.")

# 국가법령정보센터 시행일 기준 법령 검색 API
url = "https://www.law.go.kr/DRF/lawSearch.do"

params = {
    "OC": API_OC,
    "target": "eflaw",
    "type": "XML",
    "search": 1,
    "query": "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법",
    "display": 20,
}

try:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    print("HTTP 상태 코드:", response.status_code)
    print("응답 Content-Type:", response.headers.get("Content-Type"))
    print("\n===== API 응답 =====")
    print(response.text[:5000])

except requests.RequestException as e:
    print("API 요청 실패:", e)