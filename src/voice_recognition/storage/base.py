from __future__ import annotations

from abc import ABC, abstractmethod

from voice_recognition.core.models import RecognitionEvent, RecognitionScope, Speaker


class SpeakerRepository(ABC):
    def __init__(self, scope: RecognitionScope) -> None:
        self.scope = scope

    @abstractmethod
    def list_speakers(self) -> list[Speaker]:
        raise NotImplementedError

    @abstractmethod
    def create_speaker(self, speaker: Speaker) -> Speaker:
        raise NotImplementedError

    @abstractmethod
    def update_speaker(self, speaker: Speaker) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_event(self, event: RecognitionEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def list_events(self, limit: int = 100) -> list[RecognitionEvent]:
        raise NotImplementedError

    @abstractmethod
    def next_newcomer_name(self, prefix: str = "新人") -> str:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
