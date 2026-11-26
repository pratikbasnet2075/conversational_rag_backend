
from pathlib import Path
import fitz


def extract_text(file_path: str) -> str:
    """
    Extract text from a PDF or TXT file.

    Args:
        file_path: Path of the uploaded document.

    Returns:
        Extracted text as a string.
    """

    path = Path(file_path)
    file_extension = path.suffix.lower()

    if file_extension == ".txt":
        text = path.read_text(encoding="utf-8")

    elif file_extension == ".pdf":
        text = ""

        with fitz.open(file_path) as pdf_document:
            for page in pdf_document:
                text += page.get_text() + "\n"

    else:
        raise ValueError("Only PDF and TXT files are supported.")

    if not text.strip():
        raise ValueError("No readable text found in the document.")

    return text
