
from app.services.embedding_service import generate_embeddings


sample_chunks = [
    "Python is a programming language.",
    "Machine learning helps computers learn from data.",
    "FastAPI is used to create APIs."
]

embeddings = generate_embeddings(sample_chunks)

print("Total embeddings:", len(embeddings))

print("Dimensions:", len(embeddings[0]))

print("First embedding:")
print(embeddings[0])
