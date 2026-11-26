import logging
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from groq import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    GroqError,
    RateLimitError,
)
from pydantic import BaseModel, StringConstraints
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.services.booking_service import handle_booking
from app.services.chat_memory import load_history, redis_client, save_turn
from app.services.llm_service import generate_answer, rewrite_question
from app.services.retrieval_service import retrieve_relevant_chunks


router = APIRouter()
logger = logging.getLogger("uvicorn.error")


class ChatRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    session_id: UUID | None = None


class ChatSource(BaseModel):
    filename: str
    chunk_index: int


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    sources: list[ChatSource]


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Annotated[Session, Depends(get_db)]) -> ChatResponse:
    session_id = request.session_id or uuid4()
    try:
        with redis_client() as memory:
            booking_reply = handle_booking(memory, db, session_id, request.question)
            if booking_reply is not None:
                return ChatResponse(session_id=session_id, answer=booking_reply, sources=[])
            history = load_history(memory, session_id)
            query = rewrite_question(request.question, history) if history else request.question
            results = retrieve_relevant_chunks(query, limit=3)
            generated = generate_answer(request.question, [chunk["text"] for chunk in results], history)
            save_turn(memory, session_id, request.question, generated.answer)
    except RedisError as error:
        logger.warning("Chat memory failed: %s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Chat memory is unavailable.") from error
    except SQLAlchemyError as error:
        db.rollback()
        logger.warning("Booking persistence failed: %s", type(error).__name__)
        raise HTTPException(status_code=503, detail="Booking storage is unavailable.") from error
    except AuthenticationError as error:
        raise HTTPException(status_code=502, detail="Groq rejected the API key.") from error
    except RateLimitError as error:
        raise HTTPException(status_code=503, detail="Groq rate limit reached. Try again shortly.") from error
    except APIConnectionError as error:
        raise HTTPException(status_code=502, detail="Could not connect to Groq.") from error
    except APIStatusError as error:
        code = error.code
        if code not in ("invalid_request_error", "json_validate_failed", "model_not_found", "model_decommissioned"):
            code = None
        logger.warning("Groq rejected the request with HTTP %s (code: %s)", error.status_code, code)
        detail = f"Groq returned HTTP {error.status_code}"
        if code:
            detail += f" ({code})"
        raise HTTPException(
            status_code=502, detail=f"{detail}."
        ) from error
    except GroqError as error:
        logger.warning("Groq client failed: %s", type(error).__name__)
        raise HTTPException(
            status_code=503, detail="Could not initialize Groq. Check GROQ_API_KEY in the project .env file."
        ) from error
    except ValueError as error:
        logger.warning("Groq answer validation failed: %s", type(error).__name__)
        raise HTTPException(status_code=502, detail="Groq returned an invalid answer.") from error

    sources = [
        ChatSource.model_validate(results[source_id - 1])
        for source_id in dict.fromkeys(generated.source_ids)
    ]
    return ChatResponse(session_id=session_id, answer=generated.answer, sources=sources)
