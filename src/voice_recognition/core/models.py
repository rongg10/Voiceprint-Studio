from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


Vector = tuple[float, ...]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RecognitionScope(str, Enum):
    GLOBAL = "global"
    SESSION = "session"


class SignalType(str, Enum):
    SPEECH = "speech"
    NOISE = "noise"
    SILENCE = "silence"


class EventType(str, Enum):
    MATCH = "match"
    NEW_SPEAKER = "new_speaker"
    UNKNOWN_SPEECH = "unknown_speech"
    NOISE = "noise"
    SILENCE = "silence"


@dataclass(slots=True)
class Speaker:
    id: int | None
    name: str
    centroid: Vector
    prototypes: tuple[Vector, ...] = field(default_factory=tuple)
    sample_count: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def all_prototypes(self) -> tuple[Vector, ...]:
        if self.prototypes:
            return self.prototypes
        return (self.centroid,)


@dataclass(slots=True)
class FrameInference:
    signal: SignalType
    embedding: Vector | None = None
    source: str = "unknown"
    ts: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class RecognitionEvent:
    event_type: EventType
    scope: RecognitionScope
    ts: str = field(default_factory=utc_now_iso)
    speaker_id: int | None = None
    speaker_name: str | None = None
    score: float | None = None
    confidence: float | None = None
    source: str = "unknown"
    details: str | None = None
