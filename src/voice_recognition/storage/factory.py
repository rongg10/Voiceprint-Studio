from __future__ import annotations

from pathlib import Path

from voice_recognition.core.models import RecognitionScope

from .base import SpeakerRepository
from .session_repository import SessionSpeakerRepository
from .sqlite_repository import SQLiteSpeakerRepository


def build_repository(
    scope: RecognitionScope,
    db_path: str | Path = Path("data") / "speakers.db",
) -> SpeakerRepository:
    if scope == RecognitionScope.GLOBAL:
        return SQLiteSpeakerRepository(db_path=db_path)
    if scope == RecognitionScope.SESSION:
        return SessionSpeakerRepository()
    raise ValueError(f"Unsupported scope: {scope}")
