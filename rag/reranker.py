"""
Qwen3-Reranker-0.6B를 이용한 Cross-Encoder 방식 Reranking.

모델: Qwen/Qwen3-Reranker-0.6B
공식 사용 방법(Qwen3-Reranker 모델 카드 기준):
- 일반 생성형 Qwen 모델이 아니라 질문-문서 쌍의 관련성을 판단하도록
  학습된 전용 Reranker다.
- 입력은 "<Instruct>...\n<Query>...\n<Document>..." 형식의 프롬프트로
  구성하고, 모델이 다음 토큰으로 "yes"/"no"를 생성할 확률을 비교해
  관련성 점수를 계산한다(생성형 디코딩이 아니라 다음 토큰 logits만 사용).
- score = sigmoid(logit("yes") - logit("no"))
  이 값은 "정답일 확률"이 아니라 "yes가 no보다 얼마나 더 가능성이
  높은가"를 0~1로 정규화한 상대 점수이며, 재정렬(ranking) 목적에만
  사용한다.

CrossEncoder(sentence-transformers) 호환성:
- Qwen3-Reranker는 causal LM 구조이고 표준 CrossEncoder(BERT 계열
  시퀀스 분류 head)와 다른 입출력 방식을 쓰므로, sentence-transformers의
  CrossEncoder로 그대로 불러올 수 없다. 이 모듈은 transformers의
  AutoModelForCausalLM + AutoTokenizer를 모델 카드 방식대로 직접 사용한다.

모델 로딩은 지연 로딩(lazy loading)으로 구현한다. Hybrid 모드(C)에서는
이 모듈을 import하더라도 get_reranker()를 호출하지 않으면 모델이
로딩되지 않는다.
"""

import time

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

DEFAULT_MAX_LENGTH = 4096
DEFAULT_BATCH_SIZE = 4

PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class RerankerUnavailableError(RuntimeError):
    """torch/transformers 미설치 또는 모델 로딩 실패 시 발생한다."""


class QwenReranker:
    """
    Qwen3-Reranker-0.6B를 지연 로딩하여 질문-문서 쌍의 관련성을 재계산한다.

    인스턴스를 만드는 시점에는 모델을 로딩하지 않고, 처음 score()/rerank()를
    호출할 때 로딩한다. GPU가 있으면 GPU를 쓰고, 없으면 CPU로 진행한다.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.requested_device = device

        self._tokenizer = None
        self._model = None
        self._device = None
        self._token_true_id = None
        self._token_false_id = None

        self.stats = {
            "loaded": False,
            "load_seconds": None,
            "device": None,
            "inference_calls": 0,
            "inference_seconds": 0.0,
            "pairs_scored": 0,
        }

    def is_loaded(self):
        return self._model is not None

    def _load(self):
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RerankerUnavailableError(
                "torch/transformers가 설치되지 않아 Qwen Reranker를 "
                f"로딩할 수 없습니다: {error}"
            ) from error

        started = time.monotonic()

        device = self.requested_device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left"
            )
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            model = model.to(device)
            model.eval()
        except Exception as error:
            raise RerankerUnavailableError(
                f"Qwen Reranker 모델 로딩에 실패했습니다: {error}"
            ) from error

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._torch = torch

        self._token_true_id = tokenizer.convert_tokens_to_ids("yes")
        self._token_false_id = tokenizer.convert_tokens_to_ids("no")

        self.stats["loaded"] = True
        self.stats["load_seconds"] = round(time.monotonic() - started, 2)
        self.stats["device"] = device

    def _format_pair(self, instruction, query, document):
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _score_batch(self, pairs):
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        texts = [PREFIX + pair + SUFFIX for pair in pairs]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            true_logits = logits[:, self._token_true_id]
            false_logits = logits[:, self._token_false_id]
            scores = torch.sigmoid(true_logits - false_logits)

        return scores.float().cpu().tolist()

    def score(self, query, documents, instruction=DEFAULT_INSTRUCTION):
        """
        질문과 문서 목록 각각에 대한 관련성 점수를 계산한다.

        반환 점수는 sigmoid(logit(yes) - logit(no))이며, 실제 정답일
        확률이 아니라 재정렬 목적의 상대 점수다.
        """

        self._load()

        pairs = [self._format_pair(instruction, query, doc) for doc in documents]

        scores = []

        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]

            started = time.monotonic()
            batch_scores = self._score_batch(batch)
            elapsed = time.monotonic() - started

            self.stats["inference_calls"] += 1
            self.stats["inference_seconds"] += elapsed
            self.stats["pairs_scored"] += len(batch)

            scores.extend(batch_scores)

        return scores

    def rerank(self, query, candidates, text_field="text", instruction=DEFAULT_INSTRUCTION):
        """
        후보 목록(dict 리스트)을 관련성 점수로 재정렬한다.

        candidates는 Hybrid Search 결과 형식(dict with text_field, id 등)을
        그대로 받아, rerank_score/rerank_rank/original_rank를 추가해 반환한다.
        """

        if not candidates:
            return []

        documents = [candidate.get(text_field, "") for candidate in candidates]
        scores = self.score(query, documents, instruction=instruction)

        reranked = []

        for original_rank, (candidate, score) in enumerate(
            zip(candidates, scores), start=1
        ):
            entry = dict(candidate)
            entry["rerank_score"] = float(score)
            entry["original_rank"] = candidate.get("rank", original_rank)
            reranked.append(entry)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

        for rank, entry in enumerate(reranked, start=1):
            entry["rerank_rank"] = rank

        return reranked


_reranker_instance = None


def get_reranker(**kwargs):
    """
    프로세스 내에서 QwenReranker 인스턴스를 재사용한다(모델은 첫 사용 시 로딩).
    """

    global _reranker_instance

    if _reranker_instance is None:
        _reranker_instance = QwenReranker(**kwargs)

    return _reranker_instance


def is_reranker_available():
    """
    torch/transformers 설치 여부만 가볍게 확인한다(모델 다운로드는 하지 않음).
    """

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False

    return True


if __name__ == "__main__":
    reranker = get_reranker()

    query = "고영향 인공지능이란 무엇인가요?"
    documents = [
        "4. \"고영향 인공지능\"이란 사람의 생명, 신체의 안전 및 기본권에 중대한 "
        "영향을 미치거나 위험을 초래할 우려가 있는 인공지능시스템이다.",
        "제1조(목적) 이 법은 인공지능의 건전한 발전과 신뢰 기반 조성에 필요한 "
        "기본적인 사항을 규정한다.",
    ]

    scores = reranker.score(query, documents)

    for doc, score in zip(documents, scores):
        print(round(score, 4), "|", doc[:40])

    print("\n통계:", reranker.stats)
