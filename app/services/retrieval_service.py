
from app.services.embedding_service import generate_embeddings
from app.services.vector_service import search_similar_chunks


def retrieve_relevant_chunks(
    question: str,
    limit: int = 3
) -> list[dict]:

    # Generate embedding for the user's question
    question_embeddings = generate_embeddings([question])

    # Get the first (and only) question embedding
    query_embedding = question_embeddings[0]

    # Search Qdrant
    results = search_similar_chunks(
        query_embedding=query_embedding,
        limit=limit
    )

    return results
