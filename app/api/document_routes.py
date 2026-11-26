
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pathlib import Path
import shutil

from app.utils.text_extractor import extract_text
from app.utils.chunking import chunk_text
from app.services.embedding_service import generate_embeddings
from app.services.vector_service import store_embeddings

from fastapi import Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import Document

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("fixed"),
    db: Session = Depends(get_db)
):
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are allowed."
        )

    file_path = UPLOAD_DIR / Path(file.filename).name

    try:
        # Save uploaded document
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract document text
        extracted_text = extract_text(str(file_path))

        # Divide text into chunks
        chunks = chunk_text(
            text=extracted_text,
            strategy=chunking_strategy,
            chunk_size=500,
            chunk_overlap=50
        )

        # Generate embeddings
        embeddings = generate_embeddings(chunks)

        # Store embeddings in Qdrant
        total_stored = store_embeddings(
            chunks=chunks,
            embeddings=embeddings,
            filename=file.filename
        )

        # Create a document record
        document = Document(
            filename=file.filename,
            chunking_strategy=chunking_strategy,
            text_length=len(extracted_text),
            total_chunks=len(chunks),
            total_embeddings=len(embeddings)
        )

        db.add(document)
        db.commit()
        db.refresh(document)

        return {
            "message": "Document uploaded successfully",
            "document_id": document.id,
            "filename": file.filename,
            "chunking_strategy": chunking_strategy,
            "text_length": len(extracted_text),
            "total_chunks": len(chunks),
            "total_embeddings": len(embeddings),
            "stored_vectors": total_stored

        }

    except ValueError as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Error processing document."
        ) from error

    finally:
        await file.close()
