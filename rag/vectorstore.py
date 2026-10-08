import json
import uuid
from pathlib import Path

from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from common.ai_model import get_embedding_model
from common.qdrant import get_qdrant_client

BASE_DIR = Path(__file__).resolve().parent.parent
# v1 컬렉션은 정의 전용 Child 추가 이전 스냅샷(205 Child)으로 구성되므로
# 재현성을 위해 legal_chunks_v1.json을 사용한다.
# v2 컬렉션 구축은 scripts/build_dense_v2.py가 담당한다.
CHUNKS_PATH = (
    BASE_DIR / "data" / "ai_basic_law" / "legal_chunks_v1.json"
)

COLLECTION_NAME = "beopjeong_law_v1"
EXPECTED_CHILD_COUNT = 205
VECTOR_SIZE = 1536
BATCH_SIZE = 20

def load_chunks():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    children = data["children"]

    if len(children) != EXPECTED_CHILD_COUNT:
        raise ValueError(
            f"예상 Child 수: {EXPECTED_CHILD_COUNT}, "
            f"실제 Child 수: {len(children)}"
        )

    return children

def create_collection(client):
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        existing_count = info.points_count

        if existing_count:
            raise RuntimeError(
                f"기존 컬렉션에 {existing_count}개 Point가 있습니다. "
                "기존 데이터 보호를 위해 저장을 중단합니다."
            )

        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print("컬렉션 생성:", COLLECTION_NAME)

def save_chunks(client, embeddings, children):
    for start in range(0, len(children), BATCH_SIZE):
        batch = children[start:start + BATCH_SIZE]

        texts = [chunk["text"] for chunk in batch]
        vectors = embeddings.embed_documents(texts)

        if len(vectors) != len(batch):
            raise ValueError("Embedding 개수가 Chunk 개수와 다릅니다.")

        points = []

        for chunk, vector in zip(batch, vectors):
            if len(vector) != VECTOR_SIZE:
                raise ValueError(
                    f"Embedding 차원 오류: {len(vector)}"
                )

            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])
            )

            payload = dict(chunk)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(
            f"저장 완료: {start + len(batch)}/{len(children)}"
        )

def main():
    children = load_chunks()

    client = get_qdrant_client()
    embeddings = get_embedding_model()

    create_collection(client)
    save_chunks(client, embeddings, children)

    info = client.get_collection(COLLECTION_NAME)

    print("\n===== 저장 결과 =====")
    print("컬렉션:", COLLECTION_NAME)
    print("저장된 Point:", info.points_count)
    print("Embedding 차원:", VECTOR_SIZE)

if __name__ == "__main__":
    main()