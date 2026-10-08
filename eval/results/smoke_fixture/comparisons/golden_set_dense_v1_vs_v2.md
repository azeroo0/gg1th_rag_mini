# Golden Set 평가: Dense v1 vs Dense v2

- 실행 시각: 20261008T151256
- Golden Set: /tmp/claude-1000/-mnt-c-Users-Admin-gg1th-rag-mini/610d9f42-ce6f-4245-a754-44cc264c14a7/scratchpad/smoke_golden_set.jsonl
- Embedding 모델: text-embedding-3-small
- Top-K: 5
- 전체 질문: 7
- answerable 질문: 6
- answerable=false 질문: 1 (일반 정답 검색 평가에서 제외)

## 지표 비교

| 평가 수준 | 지표 | v1 | v2 | 변화 |
| --- | --- | --- | --- | --- |
| article_level | hit_at_1 | 0.3333 | 0.8333 | +0.5 |
| article_level | hit_at_3 | 0.8333 | 0.8333 | +0.0 |
| article_level | hit_at_5 | 0.8333 | 0.8333 | +0.0 |
| article_level | recall_at_5 | 0.75 | 0.75 | +0.0 |
| article_level | mrr | 0.5556 | 0.8333 | +0.2777 |
| provision_level | hit_at_1 | 0.3333 | 0.6667 | +0.3334 |
| provision_level | hit_at_3 | 0.6667 | 0.6667 | +0.0 |
| provision_level | hit_at_5 | 0.8333 | 0.8333 | +0.0 |
| provision_level | recall_at_5 | 0.75 | 0.75 | +0.0 |
| provision_level | mrr | 0.5139 | 0.7 | +0.1861 |
| precise_diagnosis | hit_at_1 | 0.1667 | 0.5 | +0.3333 |
| precise_diagnosis | hit_at_3 | 0.1667 | 0.5 | +0.3333 |
| precise_diagnosis | hit_at_5 | 0.5 | 0.8333 | +0.3333 |
| precise_diagnosis | recall_at_5 | 0.4167 | 0.75 | +0.3333 |
| precise_diagnosis | mrr | 0.2417 | 0.5667 | +0.325 |

article_level은 조문(제N조) 단위 적중률이다. 세부 조항 적중률이 아니다.
precise_diagnosis는 정답이 항ㆍ호ㆍ목까지 지정된 질문만
조문 경로 완전 일치로 평가한 결과다.

## 질문별 변화 (provision 수준)

- 개선: 2건
  - S1: 2 -> 1
  - S2: 3 -> 1
- 악화: 1건
  - S5: 4 -> 5
- 변화 없음: 3건
