
from app.services.retrieval_service import retrieve_relevant_chunks


question = "Which programming languages does Pratik know?"

results = retrieve_relevant_chunks(question, limit=3)

print("\nUser Question:", question)

print("\nRetrieved Results:")

for index, result in enumerate(results, start=1):

    print(f"\nResult {index}")
    print("Filename:", result["filename"])
    print("Chunk Index:", result["chunk_index"])
    print("Similarity Score:", result["score"])
    print("Text:", result["text"])
