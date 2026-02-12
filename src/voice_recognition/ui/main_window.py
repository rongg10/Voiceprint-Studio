from __future__ import annotations

from dataclasses import dataclass

from voice_recognition.core.models import RecognitionEvent, Speaker


@dataclass(slots=True)
class SpeakerListItem:
    id: int | None
    name: str
    sample_count: int
    is_highlighted: bool = False
    confidence: float | None = None


class UIAdapter:
    """
    UI abstraction for later desktop integration.
    A concrete PySide6 implementation can replace this without touching the engine.
    """

    def update_speakers(self, speakers: list[Speaker], active_speaker_id: int | None) -> None:
        raise NotImplementedError

    def append_event(self, event: RecognitionEvent) -> None:
        raise NotImplementedError
