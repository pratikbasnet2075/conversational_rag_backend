import json
import os
from pathlib import Path
from typing import Literal, TypedDict
from uuid import UUID

from dotenv import dotenv_values
from redis import Redis


ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
HISTORY_LIMIT = 10
SESSION_TTL = 24 * 60 * 60


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def redis_client() -> Redis:
    url = os.environ.get("REDIS_URL") or dotenv_values(ENV_FILE).get("REDIS_URL")
    return Redis.from_url(
        url or "redis://localhost:6379/0",
        decode_responses=True,
        socket_connect_timeout=3,
        socket_timeout=3,
    )


def load_history(client: Redis, session_id: UUID) -> list[ChatMessage]:
    messages = client.lrange(f"chat:{session_id}", -HISTORY_LIMIT, -1)
    return [json.loads(message) for message in messages]


def save_turn(client: Redis, session_id: UUID, question: str, answer: str) -> None:
    key = f"chat:{session_id}"
    with client.pipeline(transaction=True) as pipeline:
        pipeline.rpush(
            key,
            json.dumps({"role": "user", "content": question}),
            json.dumps({"role": "assistant", "content": answer}),
        )
        pipeline.ltrim(key, -HISTORY_LIMIT, -1)
        pipeline.expire(key, SESSION_TTL)
        pipeline.execute()
