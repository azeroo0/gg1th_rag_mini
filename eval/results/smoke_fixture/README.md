# smoke_fixture

이 디렉터리의 결과는 팀 Golden Test Set 결과가 아니다.

`eval/golden_set_v2.jsonl` / `eval/golden_set.jsonl`이 아직 저장소에 없어서,
`eval/evaluate.py`의 동작만 확인하기 위해 임시 질문 7개로 만든 파일이다.
질문과 정답 조문은 검증용으로 급히 만든 것이며 팀이 합의한 정답이 아니다.
지표 값을 성능 근거로 인용하지 말 것.

임시 파일 경로는 각 결과의 `golden_set_path` 필드에 기록되어 있다.
팀 Golden Set을 배치한 뒤 `python eval/evaluate.py`를 실행하면
`eval/results/dense_v1/`, `eval/results/dense_v2/`,
`eval/results/comparisons/`에 정식 결과가 생성된다.
