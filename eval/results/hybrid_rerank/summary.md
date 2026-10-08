# Golden Set 평가: Hybrid + Qwen Reranking (C+D)

- 실행 시각: 20261008T170807
- 컬렉션: beopjeong_law_v2
- Embedding 모델: text-embedding-3-small
- Reranker 모델: Qwen/Qwen3-Reranker-0.6B
- Reranker 디바이스: cuda
- Reranker 로딩 시간: 5.2초
- Reranker 추론 시간(전체): 9.0초 / 처리 쌍 수: 400
- Hybrid 후보 Top-K: 8 / 최종 평가 Top-K: 5
- 전체 질문: 45

## 검색 평가 지표 (Reranking 후)

| 평가 수준 | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| article_level | 0.8056 | 1.0 | 1.0 | 0.9491 | 0.8981 |
| provision_level | 0.8056 | 1.0 | 1.0 | 0.9491 | 0.8981 |
| precise_diagnosis | None | None | None | None | None |

## Reranking 전후 순위 변화 (provision 수준, answerable 질문만)

- 개선: 4건
  - q07: 4 -> 2
  - q10: 2 -> 1
  - q13: 2 -> 1
  - q17: 8 -> 1
- 악화: 3건
  - q15: 2 -> 3
  - q16: 1 -> 2
  - q19: 1 -> 2
- 변화 없음: 29건

## Hybrid vs Reranking 순위 비교 (5개 지정 질문)

### 고영향 인공지능이란 무엇인가요?

Hybrid(Reranking 전) Top-5:
  1. 제2조제4호(정의)
  2. 제35조제1항
  3. 제34조제1항제4호
  4. 제34조제1항제1호
  5. 제34조제1항제2호

Reranking 후 Top-5:
  1. 제2조제4호(정의) (score=1.0, hybrid_rank_before=1)
  2. 제35조제1항 (score=1.0, hybrid_rank_before=2)
  3. 제33조제1항 (score=1.0, hybrid_rank_before=6)
  4. 제34조제1항제4호 (score=0.9883, hybrid_rank_before=3)
  5. 제34조제1항제6호 (score=0.9688, hybrid_rank_before=8)

### 생성형 인공지능이란 무엇인가요?

Hybrid(Reranking 전) Top-5:
  1. 제31조제2항
  2. 제2조제5호
  3. 제31조제3항
  4. 제31조제1항
  5. 제2조제6호

Reranking 후 Top-5:
  1. 제31조제2항 (score=1.0, hybrid_rank_before=1)
  2. 제2조제5호 (score=1.0, hybrid_rank_before=2)
  3. 제31조제1항 (score=0.9883, hybrid_rank_before=4)
  4. 제31조제3항 (score=0.4531, hybrid_rank_before=3)
  5. 제2조제6호 (score=0.4531, hybrid_rank_before=5)

### 채용 심사에 AI를 사용하면 고영향 인공지능인가요?

Hybrid(Reranking 전) Top-5:
  1. 제34조제1항제2호
  2. 제2조제4호(정의)
  3. 제2조제4호사목
  4. 제34조제1항
  5. 제34조제1항제4호

Reranking 후 Top-5:
  1. 제33조제1항 (score=0.9961, hybrid_rank_before=6)
  2. 제35조제1항 (score=0.9961, hybrid_rank_before=8)
  3. 제34조제1항 (score=0.9688, hybrid_rank_before=4)
  4. 제2조제4호(정의) (score=0.9414, hybrid_rank_before=2)
  5. 제34조제1항제4호 (score=0.8945, hybrid_rank_before=5)

### 고영향 인공지능 사업자의 의무는 무엇인가요?

Hybrid(Reranking 전) Top-5:
  1. 제34조제1항제4호
  2. 제34조제1항제1호
  3. 제34조제1항제3호
  4. 제34조제1항제6호
  5. 제31조제1항

Reranking 후 Top-5:
  1. 제31조제1항 (score=1.0, hybrid_rank_before=5)
  2. 제35조제1항 (score=1.0, hybrid_rank_before=6)
  3. 제34조제1항제4호 (score=0.9922, hybrid_rank_before=1)
  4. 제34조제1항제6호 (score=0.9648, hybrid_rank_before=4)
  5. 제34조제1항제2호 (score=0.0117, hybrid_rank_before=8)

### AI로 생성한 콘텐츠는 표시해야 하나요?

Hybrid(Reranking 전) Top-5:
  1. 제31조제2항
  2. 제2조제5호
  3. 제31조제3항
  4. 제16조제2항
  5. 제2조제12호

Reranking 후 Top-5:
  1. 제31조제3항 (score=1.0, hybrid_rank_before=3)
  2. 제31조제2항 (score=0.9961, hybrid_rank_before=1)
  3. 제2조제5호 (score=0.042, hybrid_rank_before=2)
  4. 제6조제2항 (score=0.0293, hybrid_rank_before=6)
  5. 제16조제2항 (score=0.0038, hybrid_rank_before=4)
