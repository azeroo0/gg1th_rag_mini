# Golden Set 평가: Hybrid (Dense + BM25 + RRF)

- 실행 시각: 20261008T163556
- 컬렉션: beopjeong_law_v2
- Embedding 모델: text-embedding-3-small
- Dense Top-K: 20 / BM25 Top-K: 20 / RRF k: 60 / 최종 Top-K: 5
- 전체 질문: 45
- answerable 질문: 36
- answerable=false 질문: 9 (일반 정답 검색 평가에서 제외)

## 검색 평가 지표

| 평가 수준 | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| article_level | 0.7778 | 0.9722 | 0.9722 | 0.9213 | 0.875 |
| provision_level | 0.7778 | 0.9444 | 0.9722 | 0.9213 | 0.8681 |
| precise_diagnosis | None | None | None | None | None |

article_level은 조문(제N조) 단위 적중률이다. 세부 조항 적중률이 아니다.
precise_diagnosis는 정답이 항ㆍ호ㆍ목까지 지정된 질문만
조문 경로 완전 일치로 평가한 결과다.

## Dense v2 vs Hybrid 순위 비교 (5개 지정 질문)

### 고영향 인공지능이란 무엇인가요?

Dense v2 Top-5:
  1. 제2조제4호(정의)
  2. 제34조제1항제2호
  3. 제2조제4호바목
  4. 제34조제1항제1호
  5. 제34조제1항제4호

Hybrid Top-5 (dense_rank/bm25_rank):
  1. 제2조제4호(정의) (dense=1, bm25=2)
  2. 제35조제1항 (dense=7, bm25=1)
  3. 제34조제1항제4호 (dense=5, bm25=3)
  4. 제34조제1항제1호 (dense=4, bm25=9)
  5. 제34조제1항제2호 (dense=2, bm25=12)
