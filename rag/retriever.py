from common.ai_model import get_embedding_model
from common.qdrant import get_qdrant_client

COLLECTION_NAME = "beopjeong_law_v1"

def dense_search(question: str, top_k: int = 5):
    embeddings = get_embedding_model()
    client = get_qdrant_client()

    query_vector = embeddings.embed_query(question)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )

    results = []

    for point in response.points:
        payload = point.payload or {}

        results.append({
            "score": float(point.score),
            "id": payload.get("id", ""),
            "parent_id": payload.get("parent_id", ""),
            "article": payload.get("article", ""),
            "article_title": payload.get("article_title", ""),
            "paragraph": payload.get("paragraph", ""),
            "item": payload.get("item", ""),
            "subitem": payload.get("subitem", ""),
            "granularity": payload.get("granularity", ""),
            "document_type": payload.get("document_type", ""),
            "text": payload.get("text", ""),
            "source_text": payload.get("source_text", ""),
        })

    return results

if __name__ == "__main__":
    question = "고영향 인공지능이란 무엇인가요?"

    print("질문:", question)

    results = dense_search(question, top_k=5)

    for index, result in enumerate(results, start=1):
        print(f"\n[{index}]")
        print("Score:", round(result["score"], 4))
        print("Article:", result["article"])
        print("Paragraph:", result["paragraph"])
        print("Item:", result["item"])
        print("Subitem:", result["subitem"])
        print("Granularity:", result["granularity"])
        print("Content:", result["source_text"][:300])