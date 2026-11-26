
from fastapi import FastAPI


from app.api.chat_routes import router as chat_router
from app.api.document_routes import router as document_router
from app.database.database import Base, engine
from app.database import models


# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Conversational RAG Backend",
    description="AI-powered document question answering system",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to Conversational RAG Backend"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "Conversational RAG"
    }


app.include_router(
    document_router,
    prefix="/api/v1",
    tags=["Document Ingestion"]
)

app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
