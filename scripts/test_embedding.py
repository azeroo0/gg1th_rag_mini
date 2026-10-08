from common.ai_model import get_embedding_model
from common.config import EMBEDDING_MODEL

def main():
    embeddings = get_embedding_model()

    text = "고영향 인공지능이란 무엇인가요?"

    print("Embedding 모델:", EMBEDDING_MODEL)
    print("테스트 문장:", text)

    vector = embeddings.embed_query(text)

    print("Embedding 생성 성공")
    print("Vector 차원:", len(vector))
    print("Vector 타입:", type(vector).__name__)

if __name__ == "__main__":
    main()