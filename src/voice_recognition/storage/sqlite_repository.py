from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from voice_recognition.core.models import (
    EventType,
    RecognitionEvent,
    RecognitionScope,
    Speaker,
    utc_now_iso,
)
from voice_recognition.core.vector_ops import newcomer_index

from .base import SpeakerRepository


class SQLiteSpeakerRepository(SpeakerRepository):
    """Persistent repository for global scope."""

    def __init__(self, db_path: str | Path) -> None:
        super().__init__(RecognitionScope.GLOBAL)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            with self._connection:
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS speakers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        sample_count INTEGER NOT NULL,
                        centroid_json TEXT NOT NULL,
                        prototypes_json TEXT NULL
                    )
                    """
                )
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        speaker_id INTEGER NULL,
                        speaker_name TEXT NULL,
                        score REAL NULL,
                        confidence REAL NULL,
                        source TEXT NOT NULL,
                        details TEXT NULL
                    )
                    """
                )
                self._ensure_speaker_columns()

    def _ensure_speaker_columns(self) -> None:
        columns = self._connection.execute("PRAGMA table_info(speakers)").fetchall()
        names = {str(row["name"]) for row in columns}
        if "prototypes_json" not in names:
            self._connection.execute("ALTER TABLE speakers ADD COLUMN prototypes_json TEXT NULL")

    def list_speakers(self) -> list[Speaker]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, name, created_at, updated_at, sample_count, centroid_json, prototypes_json
                FROM speakers
                ORDER BY id ASC
                """
            ).fetchall()
            return [self._row_to_speaker(row) for row in rows]

    def create_speaker(self, speaker: Speaker) -> Speaker:
        now = utc_now_iso()
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO speakers (name, created_at, updated_at, sample_count, centroid_json, prototypes_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    speaker.name,
                    now,
                    now,
                    speaker.sample_count,
                    json.dumps(list(speaker.centroid)),
                    json.dumps([list(vector) for vector in speaker.all_prototypes()]),
                ),
            )
            self._connection.commit()
            speaker_id = int(cursor.lastrowid)
        return Speaker(
            id=speaker_id,
            name=speaker.name,
            centroid=tuple(speaker.centroid),
            prototypes=tuple(tuple(vector) for vector in speaker.all_prototypes()),
            sample_count=speaker.sample_count,
            created_at=now,
            updated_at=now,
        )

    def update_speaker(self, speaker: Speaker) -> None:
        if speaker.id is None:
            raise ValueError("Cannot update speaker without id.")
        with self._lock:
            self._connection.execute(
                """
                UPDATE speakers
                SET name = ?, updated_at = ?, sample_count = ?, centroid_json = ?, prototypes_json = ?
                WHERE id = ?
                """,
                (
                    speaker.name,
                    speaker.updated_at,
                    speaker.sample_count,
                    json.dumps(list(speaker.centroid)),
                    json.dumps([list(vector) for vector in speaker.all_prototypes()]),
                    speaker.id,
                ),
            )
            self._connection.commit()

    def save_event(self, event: RecognitionEvent) -> None:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO events (
                    ts, event_type, scope, speaker_id, speaker_name, score,
                    confidence, source, details
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.ts,
                    event.event_type.value,
                    event.scope.value,
                    event.speaker_id,
                    event.speaker_name,
                    event.score,
                    event.confidence,
                    event.source,
                    event.details,
                ),
            )
            self._connection.commit()

    def list_events(self, limit: int = 100) -> list[RecognitionEvent]:
        if limit <= 0:
            return []
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT ts, event_type, scope, speaker_id, speaker_name, score, confidence, source, details
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        events: list[RecognitionEvent] = []
        for row in rows:
            event_type_value = row["event_type"]
            scope_value = row["scope"]
            events.append(
                RecognitionEvent(
                    event_type=EventType(event_type_value),
                    scope=RecognitionScope(scope_value),
                    ts=row["ts"],
                    speaker_id=row["speaker_id"],
                    speaker_name=row["speaker_name"],
                    score=row["score"],
                    confidence=row["confidence"],
                    source=row["source"],
                    details=row["details"],
                )
            )
        return list(reversed(events))

    def next_newcomer_name(self, prefix: str = "新人") -> str:
        with self._lock:
            rows = self._connection.execute("SELECT name FROM speakers").fetchall()
        max_index = 0
        for row in rows:
            index = newcomer_index(row["name"], prefix=prefix)
            if index is not None and index > max_index:
                max_index = index
        return f"{prefix}{max_index + 1}"

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _row_to_speaker(row: sqlite3.Row) -> Speaker:
        centroid_list = json.loads(row["centroid_json"])
        raw_prototypes = row["prototypes_json"]
        prototypes: tuple[tuple[float, ...], ...] = ()
        if raw_prototypes:
            parsed = json.loads(raw_prototypes)
            if isinstance(parsed, list):
                normalized: list[tuple[float, ...]] = []
                for item in parsed:
                    if isinstance(item, list):
                        normalized.append(tuple(float(value) for value in item))
                prototypes = tuple(normalized)
        return Speaker(
            id=row["id"],
            name=row["name"],
            centroid=tuple(float(value) for value in centroid_list),
            prototypes=prototypes,
            sample_count=row["sample_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
