
from sentence_transformers import SentenceTransformer


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    """
    Convert text chunks into numerical embeddings.

    Args:
        chunks: List of text chunks.

    Returns:
        List of numerical embedding vectors.
    """

    if not chunks:
        raise ValueError("No text chunks provided.")

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    return embeddings.tolist()

