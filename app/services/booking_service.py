import json
from datetime import date, datetime, time
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from email_validator import EmailNotValidError, validate_email
from pydantic import BaseModel, ConfigDict
from redis import Redis
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from app.database.models import InterviewBooking
from app.services.chat_memory import SESSION_TTL
from app.services.llm_service import MODEL, groq_client


BOOKING_ZONE = ZoneInfo("Asia/Kathmandu")
FIELDS = ("name", "email", "date", "time")
CONFIRMATION = "confirm booking"
CANCELLATION = "cancel booking"


class BookingDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None
    email: str | None
    date: str | None
    time: str | None


class BookingExtraction(BookingDetails):
    is_booking: bool


class BookingDraft(BaseModel):
    id: UUID
    details: BookingDetails
    awaiting_confirmation: bool = False


def extract_booking(question: str, draft: BookingDraft | None) -> BookingExtraction:
    with groq_client() as client:
        completion = client.chat.completions.create(
            model=MODEL,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "booking_extraction",
                    "strict": True,
                    "schema": BookingExtraction.model_json_schema(),
                },
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Identify an intent to schedule an interview or provide details for "
                        "an existing interview draft. General questions about interviews are "
                        "not booking requests. Extract only new or corrected details from "
                        "the current message. Use null for absent or ambiguous fields. "
                        "Use ISO YYYY-MM-DD and 24-hour HH:MM in the supplied timezone. "
                        "Convert explicitly specified other timezones when unambiguous. "
                        "Do not invent any details or claim a booking is confirmed."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "message": question,
                        "today": datetime.now(BOOKING_ZONE).date().isoformat(),
                        "timezone": BOOKING_ZONE.key,
                        "draft": draft.details.model_dump() if draft else None,
                    }),
                },
            ],
        )
    content = completion.choices[0].message.content if completion.choices else None
    if not content:
        raise ValueError("Groq returned no booking extraction.")
    return BookingExtraction.model_validate_json(content)


def validate_details(details: BookingDetails) -> list[str]:
    issues = []
    if not details.name or len(details.name) > 200:
        issues.append("your name")
    try:
        details.email = validate_email(
            details.email or "", check_deliverability=False
        ).normalized
    except EmailNotValidError:
        issues.append("a valid email address")

    booking_date = None
    booking_time = None
    now = datetime.now(BOOKING_ZONE)
    try:
        booking_date = date.fromisoformat(details.date or "")
        if booking_date < now.date():
            raise ValueError
        details.date = booking_date.isoformat()
    except ValueError:
        issues.append("a valid date today or later")
    try:
        booking_time = datetime.strptime(details.time or "", "%H:%M").time()
        details.time = booking_time.strftime("%H:%M")
    except ValueError:
        issues.append("a valid time")
    if not issues and booking_date and booking_time:
        if datetime.combine(booking_date, booking_time, BOOKING_ZONE) <= now:
            issues.append("a future date and time")
    return issues


def handle_booking(memory: Redis, db: Session, session_id: UUID, question: str) -> str | None:
    key = f"booking:{session_id}"
    stored = memory.get(key)
    draft = BookingDraft.model_validate_json(stored) if stored else None
    command = question.casefold()

    if command == CANCELLATION:
        return "Booking draft cancelled." if memory.delete(key) else "There is no booking draft to cancel."
    if command == CONFIRMATION:
        if not draft or not draft.awaiting_confirmation:
            return "There is no booking draft ready to confirm."
        details = draft.details
        if issues := validate_details(details):
            return "Before I can book the interview, please provide " + ", ".join(issues) + "."
        statement = insert(InterviewBooking).values(
            id=str(draft.id),
            session_id=str(session_id),
            name=details.name,
            email=details.email,
            date=date.fromisoformat(details.date),
            time=time.fromisoformat(details.time),
            timezone=BOOKING_ZONE.key,
        ).on_conflict_do_nothing(index_elements=["id"])
        db.execute(statement)
        db.commit()
        memory.delete(key)
        return f"Interview booked for {details.date} at {details.time} ({BOOKING_ZONE.key}). Booking ID: {draft.id}."

    extracted = extract_booking(question, draft)
    if not extracted.is_booking:
        return None
    values = draft.details.model_dump() if draft else dict.fromkeys(FIELDS)
    for field in FIELDS:
        value = getattr(extracted, field)
        if value and value.strip():
            values[field] = value.strip()
    details = BookingDetails.model_validate(values)
    issues = validate_details(details)
    draft = BookingDraft(
        id=draft.id if draft else uuid4(),
        details=details,
        awaiting_confirmation=not issues,
    )
    memory.set(key, draft.model_dump_json(), ex=SESSION_TTL)
    if issues:
        return "To book the interview, please provide " + ", ".join(issues) + "."
    return (
        f"Interview for {details.name} ({details.email}) on {details.date} "
        f"at {details.time} ({BOOKING_ZONE.key}). "
        "Reply 'confirm booking' to save it, or send corrections."
    )
