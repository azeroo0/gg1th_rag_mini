"""
beopjeong_law_v2 컬렉션을 구축한다.

v1(beopjeong_law_v1)의 기존 벡터와 Payload를 그대로 복사하고,
새로 추가된 정의 전용 Child 1개만 추가 임베딩한다.
기존 205개 벡터를 동일하게 유지해야 정의 전용 Child 추가 효과만
분리해서 비교할 수 있다.

v1 컬렉션은 읽기만 하며 수정하지 않는다.
v2 컬렉션이 이미 존재하면 덮어쓰지 않고 작업을 중단한다.
"""

import json
import sys
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from common.ai_model import get_embedding_model
from common.config import EMBEDDING_MODEL
from common.qdrant import get_qdrant_client
from rag.vectorstore import VECTOR_SIZE

V1_COLLECTION = "beopjeong_law_v1"
V2_COLLECTION = "beopjeong_law_v2"

V2_CHUNKS_PATH = BASE_DIR / "data" / "ai_basic_law" / "legal_chunks_v2.json"

EXPECTED_V1_POINTS = 205
EXPECTED_V2_POINTS = 206

SCROLL_BATCH = 100

def to_point_id(chunk_id):
    """
    rag/vectorstore.py와 동일한 결정적 Point ID 규칙을 사용한다.
    """

    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

def load_v2_children():
    with open(V2_CHUNKS_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["children"]

def read_all_points(client, collection_name):
    """
    컬렉션의 모든 Point를 벡터와 Payload를 포함해 읽는다.
    """

    points = []
    offset = None

    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        points.extend(batch)

        if offset is None:
            break

    return points

def check_preconditions(client):
    if not client.collection_exists(V1_COLLECTION):
        raise RuntimeError(
            f"기준 컬렉션이 없습니다: {V1_COLLECTION}"
        )

    if client.collection_exists(V2_COLLECTION):
        info = client.get_collection(V2_COLLECTION)

        raise RuntimeError(
            f"{V2_COLLECTION} 컬렉션이 이미 존재합니다. "
            f"(Point {info.points_count}개) "
            "기존 데이터 보호를 위해 삭제나 덮어쓰기를 하지 않고 중단합니다. "
            "재구축이 필요하면 컬렉션 이름을 바꾸거나 직접 삭제한 뒤 다시 실행하십시오."
        )

    v1_info = client.get_collection(V1_COLLECTION)

    if v1_info.points_count != EXPECTED_V1_POINTS:
        raise RuntimeError(
            f"{V1_COLLECTION} Point 수가 예상과 다릅니다: "
            f"{v1_info.points_count} (기대값 {EXPECTED_V1_POINTS})"
        )

    v1_vectors = v1_info.config.params.vectors

    if v1_vectors.size != VECTOR_SIZE:
        raise RuntimeError(
            f"v1 벡터 차원 불일치: {v1_vectors.size} (기대값 {VECTOR_SIZE})"
        )

    return v1_vectors

def copy_points(client, source_points):
    """
    v1에서 읽은 Point를 ID, 벡터, Payload 그대로 v2에 복사한다.
    """

    for start in range(0, len(source_points), SCROLL_BATCH):
        batch = source_points[start:start + SCROLL_BATCH]

        points = [
            PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
            )
            for point in batch
        ]

        client.upsert(
            collection_name=V2_COLLECTION,
            points=points,
            wait=True,
        )

        print(f"복사 완료: {start + len(batch)}/{len(source_points)}")

def add_new_children(client, embeddings, new_children):
    texts = [chunk["text"] for chunk in new_children]
    vectors = embeddings.embed_documents(texts)

    if len(vectors) != len(new_children):
        raise RuntimeError("Embedding 개수가 Chunk 개수와 다릅니다.")

    points = []

    for chunk, vector in zip(new_children, vectors):
        if len(vector) != VECTOR_SIZE:
            raise RuntimeError(f"Embedding 차원 오류: {len(vector)}")

        points.append(
            PointStruct(
                id=to_point_id(chunk["id"]),
                vector=vector,
                payload=dict(chunk),
            )
        )

    client.upsert(
        collection_name=V2_COLLECTION,
        points=points,
        wait=True,
    )

    for chunk in new_children:
        print("추가 임베딩:", chunk["id"])

def verify(client, source_points, new_children):
    """
    기존 Point의 ID, 벡터, Payload가 그대로 복사되었는지와
    신규 Child가 추가되었는지 검증한다.
    """

    v2_points = read_all_points(client, V2_COLLECTION)
    v2_map = {str(point.id): point for point in v2_points}

    checks = []

    def check(label, condition):
        checks.append((label, condition))

    check(
        f"v1 Point {EXPECTED_V1_POINTS}개 전체 읽기",
        len(source_points) == EXPECTED_V1_POINTS,
    )
    check(
        f"v2 Point {EXPECTED_V2_POINTS}개",
        len(v2_points) == EXPECTED_V2_POINTS,
    )

    missing_ids = []
    vector_mismatch = []
    payload_mismatch = []

    for point in source_points:
        copied = v2_map.get(str(point.id))

        if copied is None:
            missing_ids.append(str(point.id))
            continue

        if list(copied.vector) != list(point.vector):
            vector_mismatch.append(str(point.id))

        if copied.payload != point.payload:
            payload_mismatch.append(str(point.id))

    check("기존 Point ID 전체 복사", len(missing_ids) == 0)
    check("기존 벡터 동일", len(vector_mismatch) == 0)
    check("기존 Payload 동일", len(payload_mismatch) == 0)

    new_ids = [to_point_id(chunk["id"]) for chunk in new_children]
    source_ids = {str(point.id) for point in source_points}

    check(
        "신규 Child Point 추가",
        all(new_id in v2_map for new_id in new_ids),
    )
    check(
        "신규 Point ID가 기존 ID와 충돌하지 않음",
        all(new_id not in source_ids for new_id in new_ids),
    )

    v2_info = client.get_collection(V2_COLLECTION)

    check(
        "v2 벡터 차원 일치",
        v2_info.config.params.vectors.size == VECTOR_SIZE,
    )

    v1_info = client.get_collection(V1_COLLECTION)

    check(
        f"v1 Point {EXPECTED_V1_POINTS}개 유지(비파괴 확인)",
        v1_info.points_count == EXPECTED_V1_POINTS,
    )

    print("\n===== v2 구축 검증 =====")

    failed = False

    for label, passed in checks:
        if not passed:
            failed = True

        print(f"{'PASS' if passed else 'FAIL'}: {label}")

    if missing_ids:
        print("FAIL 상세 (미복사 ID):", missing_ids[:10])

    if vector_mismatch:
        print("FAIL 상세 (벡터 불일치):", vector_mismatch[:10])

    if payload_mismatch:
        print("FAIL 상세 (Payload 불일치):", payload_mismatch[:10])

    return not failed

def main():
    client = get_qdrant_client()

    v1_vectors = check_preconditions(client)

    children = load_v2_children()
    child_map = {chunk["id"]: chunk for chunk in children}

    print("v2 청크 파일 Child 수:", len(children))

    source_points = read_all_points(client, V1_COLLECTION)

    print(f"{V1_COLLECTION}에서 읽은 Point:", len(source_points))

    if len(source_points) != EXPECTED_V1_POINTS:
        raise RuntimeError(
            f"v1 Point를 모두 읽지 못했습니다: {len(source_points)}"
        )

    copied_chunk_ids = {
        (point.payload or {}).get("id", "") for point in source_points
    }

    new_children = [
        chunk for chunk in children
        if chunk["id"] not in copied_chunk_ids
    ]

    print("추가 임베딩 대상 Child:", [c["id"] for c in new_children])

    missing_in_file = sorted(copied_chunk_ids - set(child_map))

    if missing_in_file:
        raise RuntimeError(
            f"v1에만 존재하고 v2 청크 파일에 없는 Child: {missing_in_file}"
        )

    if len(children) - len(source_points) != len(new_children):
        raise RuntimeError("신규 Child 수 계산이 맞지 않습니다.")

    client.create_collection(
        collection_name=V2_COLLECTION,
        vectors_config=VectorParams(
            size=v1_vectors.size,
            distance=v1_vectors.distance,
        ),
    )

    print("컬렉션 생성:", V2_COLLECTION)

    copy_points(client, source_points)

    if new_children:
        embeddings = get_embedding_model()
        add_new_children(client, embeddings, new_children)

    passed = verify(client, source_points, new_children)

    info = client.get_collection(V2_COLLECTION)

    print("\n===== 구축 결과 =====")
    print("컬렉션:", V2_COLLECTION)
    print("Point 수:", info.points_count)
    print("Embedding 모델:", EMBEDDING_MODEL)
    print("Embedding 차원:", info.config.params.vectors.size)
    print("Distance:", info.config.params.vectors.distance)

    if not passed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
