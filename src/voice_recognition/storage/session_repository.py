from __future__ import annotations

from voice_recognition.core.models import RecognitionEvent, RecognitionScope, Speaker, utc_now_iso
from voice_recognition.core.vector_ops import newcomer_index

from .base import SpeakerRepository


class SessionSpeakerRepository(SpeakerRepository):
    """Ephemeral in-memory repository for session scope."""

    def __init__(self) -> None:
        super().__init__(RecognitionScope.SESSION)
        self._speakers: dict[int, Speaker] = {}
        self._events: list[RecognitionEvent] = []
        self._next_id = 1

    def list_speakers(self) -> list[Speaker]:
        return [self._clone_speaker(speaker) for _, speaker in sorted(self._speakers.items())]

    def create_speaker(self, speaker: Speaker) -> Speaker:
        now = utc_now_iso()
        created = Speaker(
            id=self._next_id,
            name=speaker.name,
            centroid=tuple(speaker.centroid),
            prototypes=tuple(tuple(vector) for vector in speaker.all_prototypes()),
            sample_count=speaker.sample_count,
            created_at=now,
            updated_at=now,
        )
        self._speakers[self._next_id] = created
        self._next_id += 1
        return self._clone_speaker(created)

    def update_speaker(self, speaker: Speaker) -> None:
        if speaker.id is None or speaker.id not in self._speakers:
            raise ValueError(f"Speaker id={speaker.id} does not exist.")
        current = self._speakers[speaker.id]
        current.name = speaker.name
        current.centroid = tuple(speaker.centroid)
        current.prototypes = tuple(tuple(vector) for vector in speaker.all_prototypes())
        current.sample_count = speaker.sample_count
        current.updated_at = speaker.updated_at

    def save_event(self, event: RecognitionEvent) -> None:
        self._events.append(event)

    def list_events(self, limit: int = 100) -> list[RecognitionEvent]:
        if limit <= 0:
            return []
        return list(self._events[-limit:])

    def next_newcomer_name(self, prefix: str = "新人") -> str:
        max_index = 0
        for speaker in self._speakers.values():
            index = newcomer_index(speaker.name, prefix=prefix)
            if index is not None and index > max_index:
                max_index = index
        return f"{prefix}{max_index + 1}"

    def close(self) -> None:
        self._speakers.clear()
        self._events.clear()

    @staticmethod
    def _clone_speaker(speaker: Speaker) -> Speaker:
        return Speaker(
            id=speaker.id,
            name=speaker.name,
            centroid=tuple(speaker.centroid),
            prototypes=tuple(tuple(vector) for vector in speaker.all_prototypes()),
            sample_count=speaker.sample_count,
            created_at=speaker.created_at,
            updated_at=speaker.updated_at,
        )
