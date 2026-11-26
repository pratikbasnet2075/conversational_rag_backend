
from sqlalchemy import Column, Date, Integer, String, DateTime, Time
from sqlalchemy.sql import func

from app.database.database import Base


class InterviewBooking(Base):
    __tablename__ = "interview_bookings"

    id = Column(String(36), primary_key=True)
    session_id = Column(String(36), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(254), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(Time, nullable=False)
    timezone = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Document(Base):

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String, nullable=False)

    chunking_strategy = Column(String, nullable=False)

    text_length = Column(Integer, nullable=False)

    total_chunks = Column(Integer, nullable=False)

    total_embeddings = Column(Integer, nullable=False)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now()
    )
