"""
실험 결과 파일 저장 공통 유틸리티.

기존 결과 파일을 덮어쓰지 않도록, 같은 경로에 파일이 이미 있으면
파일명에 실행 타임스탬프를 붙여 새 파일로 저장한다.
"""

import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "eval" / "results"

def now_timestamp():
    return datetime.now().strftime("%Y%m%dT%H%M%S")

def save_json(path: Path, payload: dict, timestamp: str) -> Path:
    """
    payload를 path에 저장한다.
    path가 이미 있으면 '<stem>_<timestamp><suffix>'로 저장한다.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    target = path

    if target.exists():
        target = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return target

def save_text(path: Path, text: str, timestamp: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    target = path

    if target.exists():
        target = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

    target.write_text(text, encoding="utf-8")

    return target
