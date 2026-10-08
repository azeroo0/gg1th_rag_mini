# Golden Set 평가: Dense v1 vs Dense v2

- 실행 시각: 20261008T155724
- Golden Set: /mnt/c/Users/Admin/gg1th_rag_mini/eval/golden_set_v2.jsonl
- Embedding 모델: text-embedding-3-small
- Top-K: 5
- 전체 질문: 45
- answerable 질문: 36
- answerable=false 질문: 9 (일반 정답 검색 평가에서 제외)

## 지표 비교

| 평가 수준 | 지표 | v1 | v2 | 변화 |
| --- | --- | --- | --- | --- |
| article_level | hit_at_1 | 0.6667 | 0.6944 | +0.0277 |
| article_level | hit_at_3 | 0.8611 | 0.8611 | +0.0 |
| article_level | hit_at_5 | 0.8611 | 0.8611 | +0.0 |
| article_level | recall_at_5 | 0.8009 | 0.8009 | +0.0 |
| article_level | mrr | 0.75 | 0.7639 | +0.0139 |
| provision_level | hit_at_1 | 0.6667 | 0.6944 | +0.0277 |
| provision_level | hit_at_3 | 0.7778 | 0.7778 | +0.0 |
| provision_level | hit_at_5 | 0.8611 | 0.8611 | +0.0 |
| provision_level | recall_at_5 | 0.8009 | 0.8009 | +0.0 |
| provision_level | mrr | 0.7338 | 0.7463 | +0.0125 |
| precise_diagnosis | hit_at_1 | - | - | - |
| precise_diagnosis | hit_at_3 | - | - | - |
| precise_diagnosis | hit_at_5 | - | - | - |
| precise_diagnosis | recall_at_5 | - | - | - |
| precise_diagnosis | mrr | - | - | - |

article_level은 조문(제N조) 단위 적중률이다. 세부 조항 적중률이 아니다.
precise_diagnosis는 정답이 항ㆍ호ㆍ목까지 지정된 질문만
조문 경로 완전 일치로 평가한 결과다.

## 질문별 변화 (provision 수준)

- 개선: 1건
  - q01: 2 -> 1
- 악화: 1건
  - q12: 4 -> 5
- 변화 없음: 34건
