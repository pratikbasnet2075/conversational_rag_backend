# Conversational RAG Backend

A backend application built with FastAPI as part of the Palm Mind AI technical assessment. It supports document ingestion, retrieval-augmented question answering, multi-turn conversations, and conversational interview booking.

The RAG pipeline is implemented directly without using prebuilt retrieval chains.

## Features

- PDF and TXT document ingestion
- Selectable fixed-size and recursive text chunking
- Semantic search using Qdrant
- Context-aware question answering with source references
- Multi-turn conversations using Redis
- LLM-powered interview booking with input validation and confirmation
- Persistent document metadata and interview records using SQLite

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | Qdrant |
| LLM Provider | Groq |
| Conversation Memory | Redis |
| Database | SQLite |
| Language | Python |

## Architecture

### Document Ingestion

```text
Document Upload
      ↓
Text Extraction
      ↓
Text Chunking (Fixed / Recursive)
      ↓
Embedding Generation
      ↓
Qdrant Vector Storage + SQLite Metadata
```

### Conversational RAG

```text
User Query
      ↓
Redis Conversation History
      ↓
Semantic Retrieval from Qdrant
      ↓
Context and LLM Processing
      ↓
Answer with Source References
```

The conversational endpoint also supports interview booking by collecting candidate details across multiple messages and persisting the booking after confirmation.

## Getting Started

### Prerequisites

- Python 3.10+
- Groq API key
- Running Redis instance

### Installation

Clone the repository and create a virtual environment.

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
REDIS_URL=redis://localhost:6379/0
```

A hosted Redis instance can also be used by providing its connection URL.

### Running the Application

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

API documentation is available at:

http://localhost:8000/docs

The embedding model is downloaded automatically on the first run if it is not already cached.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/documents/upload` | Upload and index documents |
| POST | `/api/v1/chat` | Conversational RAG and interview booking |
| GET | `/health` | Application health check |

### 1. Document Ingestion

Upload a PDF or TXT document using multipart form data.

**Parameters**

| Parameter | Description |
|---|---|
| `file` | PDF or UTF-8 TXT document |
| `chunking_strategy` | `fixed` or `recursive` |

A resume is included as an example document to demonstrate the retrieval pipeline.

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@examples/resume.pdf" \
  -F "chunking_strategy=recursive"
```

The endpoint extracts text, generates chunks and embeddings, stores vectors in Qdrant, and saves document metadata in SQLite.

### 2. Conversational RAG

**POST** `/api/v1/chat`

After uploading the resume, send a question:

```json
{
  "question": "What technical skills does the candidate have?"
}
```

The response contains the generated answer, source references, and a session ID.

Use the returned session ID to ask follow-up questions:

```json
{
  "question": "What projects has the candidate worked on?",
  "session_id": "<returned_session_id>"
}
```

```json
{
  "question": "Which of those projects involve AI/ML ?",
  "session_id": "<returned_session_id>"
}
```

Conversation history is maintained in Redis, allowing the assistant to understand references to previous messages.

Each session retains the last 10 messages and expires after 24 hours of inactivity.

### 3. Interview Booking

Interview booking is handled through the same conversational endpoint.

An example conversation:

| User | Assistant |
|---|---|
| I'd like to book an interview. | Asks for the candidate's details. |
| My name is Ram Prasad Sharma. | Requests the remaining information. |
| My email is ram@example.com. | Requests the preferred date and time. |
| December 10, 2026, at 11 AM. | Presents a booking summary for confirmation. |
| Confirm booking. | Saves the booking and returns a booking ID. |

The same session ID must be used throughout the conversation.

Booking details are extracted using the LLM, validated by the backend, and stored in SQLite after confirmation.

The default timezone is `Asia/Kathmandu`. Pending bookings can be discarded by sending `cancel booking`.

This implementation records interview bookings locally rather than integrating with an external calendar service.

## Data Persistence

| Storage | Purpose |
|---|---|
| Qdrant | Document embeddings and searchable text chunks |
| SQLite | Document metadata and confirmed interview bookings |
| Redis | Conversation history and booking sessions |
| Local filesystem | Uploaded documents |

Qdrant runs locally without requiring a separate vector database server.

## Implementation Notes

- The retrieval and generation pipeline is implemented without RetrievalQAChain.
- Both document chunking strategies are selectable at ingestion time.
- The same embedding model is used for document indexing and query retrieval.
- Interview bookings require confirmation before being persisted.
- PDF extraction requires a readable text layer; scanned documents requiring OCR are not currently supported.
- No frontend is required. All functionality is accessible through REST APIs and Swagger documentation.

## Project Structure

```text
app/
├── api/          # Document ingestion and conversational endpoints
├── database/     # SQLite configuration and database models
├── schemas/      # Request and response schemas
├── services/     # Embeddings, retrieval, LLM, memory and booking
├── utils/        # Text extraction and chunking
└── main.py       # FastAPI application

examples/         # Sample document for testing
tests/            # Embedding and retrieval tests
```

The application separates API routing, business logic, database operations, and document processing into independent modules.

## Author

Pratik Basnet

Developed as part of the Palm Mind AI technical assessment.