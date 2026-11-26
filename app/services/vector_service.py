
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from uuid import uuid4


# Create a persistent local Qdrant database
client = QdrantClient(path="qdrant_storage")

COLLECTION_NAME = "document_chunks"

VECTOR_SIZE = 384


# Create collection if it does not already exist
if not client.collection_exists(COLLECTION_NAME):

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )


def store_embeddings(
    chunks: list[str],
    embeddings: list[list[float]],
    filename: str
) -> int:

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings must have equal lengths.")
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=filename)
                    )
                ]
            )
        )
    )
    points = []

    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

        point = PointStruct(
            id=str(uuid4()),
            vector=embedding,
            payload={
                "filename": filename,
                "chunk_index": index,
                "text": chunk
            }
        )

        points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return len(points)


def search_similar_chunks(
    query_embedding: list[float],
    limit: int = 3
) -> list[dict]:

    # Search for similar vectors in Qdrant
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
        with_payload=True
    )

    results = []

    # Extract matching text and metadata
    for point in response.points:

        results.append({
            "text": point.payload.get("text", ""),
            "filename": point.payload.get("filename", ""),
            "chunk_index": point.payload.get("chunk_index", -1),
            "score": point.score
        })

    return results

