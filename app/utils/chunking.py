
from langchain_text_splitters import RecursiveCharacterTextSplitter


def fixed_size_chunking(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


def recursive_chunking(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    return chunks


def chunk_text(
    text: str,
    strategy: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    if strategy == "fixed":
        return fixed_size_chunking(
            text, chunk_size, chunk_overlap
        )

    elif strategy == "recursive":
        return recursive_chunking(
            text, chunk_size, chunk_overlap
        )

    else:
        raise ValueError(
            "Invalid strategy. Choose fixed or recursive."
        )
