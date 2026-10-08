# eval/results

Dense Retrieval 실험 결과를 실험별로 분리해 저장한다.
기존 결과 파일은 덮어쓰지 않고, 같은 경로에 파일이 있으면
파일명에 실행 타임스탬프를 붙여 새 파일로 저장한다.
(`eval/result_io.py`)

## 디렉터리

```
dense_v1/
  diagnostic_results.json   정의 질문 5개 진단 결과 (beopjeong_law_v1)
  retrieval_results.json    Golden Set 검색 결과 (파일 배치 후 생성)
  metrics.json              Golden Set 평가 지표 (파일 배치 후 생성)

dense_v2/
  diagnostic_results.json   정의 질문 5개 진단 결과 (beopjeong_law_v2)
  retrieval_results.json    Golden Set 검색 결과 (파일 배치 후 생성)
  metrics.json              Golden Set 평가 지표 (파일 배치 후 생성)

comparisons/
  dense_v1_vs_v2.json       진단 질문 v1/v2 비교
  dense_v1_vs_v2.md         진단 질문 v1/v2 비교 요약
  golden_set_dense_v1_vs_v2.json   Golden Set v1/v2 비교 (파일 배치 후 생성)
  golden_set_dense_v1_vs_v2.md     Golden Set v1/v2 비교 요약 (파일 배치 후 생성)

smoke_fixture/              평가 코드 동작 확인용 임시 결과 (성능 근거 아님)
```

## 생성 방법

```
python scripts/compare_dense_versions.py   # 진단 질문 5개 v1/v2 비교
python eval/evaluate.py                    # Golden Set v1/v2 평가
```

Golden Set 파일이 없으면 `eval/evaluate.py`는 임의로 파일을 만들지 않고
필요한 경로를 안내한 뒤 종료한다.
