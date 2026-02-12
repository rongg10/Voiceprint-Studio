from __future__ import annotations

import math

from voice_recognition.core.models import (
    EventType,
    FrameInference,
    RecognitionEvent,
    RecognitionScope,
    Speaker,
    SignalType,
    utc_now_iso,
)
from voice_recognition.storage.base import SpeakerRepository

from .enrollment import AutoEnrollmentManager
from .matcher import SpeakerMatcher


class DiarizationEngine:
    """
    Online diarization-style clustering:
    - Match known clusters with lower thresholds.
    - If no confident match, still allow enrollment (no ambiguous-high-score block).
    """

    def __init__(
        self,
        repository: SpeakerRepository,
        scope: RecognitionScope,
        matcher: SpeakerMatcher | None = None,
        enrollment: AutoEnrollmentManager | None = None,
        centroid_update_threshold: float = 0.80,
        centroid_adapt_rate: float = 0.22,
        prototype_novelty_threshold: float = 0.88,
        max_prototypes_per_speaker: int = 6,
        soft_match_threshold: float | None = None,
        enrollment_reset_non_speech_frames: int = 10,
    ) -> None:
        self.repository = repository
        self.scope = scope
        self.matcher = matcher or SpeakerMatcher()
        self.enrollment = enrollment or AutoEnrollmentManager()
        self.centroid_update_threshold = centroid_update_threshold
        self.centroid_adapt_rate = max(0.02, min(0.8, centroid_adapt_rate))
        self.prototype_novelty_threshold = prototype_novelty_threshold
        self.max_prototypes_per_speaker = max(1, int(max_prototypes_per_speaker))
        default_soft = max(
            0.50,
            min(self.matcher.config.match_threshold - 0.10, self.matcher.config.match_threshold - 0.02),
        )
        self.soft_match_threshold = (
            default_soft if soft_match_threshold is None else max(0.30, float(soft_match_threshold))
        )
        self.enrollment_reset_non_speech_frames = max(1, int(enrollment_reset_non_speech_frames))
        self._non_speech_frames = 0

    def process(self, frame: FrameInference) -> RecognitionEvent:
        if frame.signal == SignalType.NOISE:
            self._non_speech_frames += 1
            details = None
            if self._non_speech_frames >= self.enrollment_reset_non_speech_frames:
                self.enrollment.reset(reason="reset_on_noise")
                details = "enrollment_reset_on_noise"
            event = RecognitionEvent(
                event_type=EventType.NOISE,
                scope=self.scope,
                source=frame.source,
                details=details,
            )
            self.repository.save_event(event)
            return event

        if frame.signal == SignalType.SILENCE:
            self._non_speech_frames += 1
            details = None
            if self._non_speech_frames >= self.enrollment_reset_non_speech_frames:
                self.enrollment.reset(reason="reset_on_silence")
                details = "enrollment_reset_on_silence"
            event = RecognitionEvent(
                event_type=EventType.SILENCE,
                scope=self.scope,
                source=frame.source,
                details=details,
            )
            self.repository.save_event(event)
            return event

        if frame.embedding is None:
            raise ValueError("Speech frame must include embedding.")
        self._non_speech_frames = 0

        speakers = self.repository.list_speakers()
        if not speakers:
            created = self.enrollment.consider(
                embedding=frame.embedding,
                best_known_score=0.0,
                repository=self.repository,
                known_speaker_count=0,
            )
            if created is not None:
                event = RecognitionEvent(
                    event_type=EventType.NEW_SPEAKER,
                    scope=self.scope,
                    source=frame.source,
                    speaker_id=created.id,
                    speaker_name=created.name,
                    details="auto_enrolled",
                )
                self.repository.save_event(event)
                return event
            event = RecognitionEvent(
                event_type=EventType.UNKNOWN_SPEECH,
                scope=self.scope,
                source=frame.source,
                details=self._merge_details("no_speakers_yet", self.enrollment.last_reason),
            )
            self.repository.save_event(event)
            return event

        decision = self.matcher.match(frame.embedding, speakers)
        ambiguous = (
            decision.second_best_score is not None
            and (decision.score - decision.second_best_score) < self.matcher.config.min_margin
        )
        skipped_detail = (
            f"skipped_dim_mismatch={decision.skipped_speakers}"
            if decision.skipped_speakers > 0
            else None
        )
        if decision.speaker is not None:
            updated = decision.speaker
            if decision.score >= self.centroid_update_threshold:
                updated = self._update_speaker(decision.speaker, frame.embedding)
            event = RecognitionEvent(
                event_type=EventType.MATCH,
                scope=self.scope,
                source=frame.source,
                speaker_id=updated.id,
                speaker_name=updated.name,
                score=round(decision.score, 6),
                confidence=round(decision.confidence, 2),
                details=self._merge_details(
                    self._scoring_details(decision),
                    (
                        f"second_best={decision.second_best_score:.4f}"
                        if decision.second_best_score is not None
                        else None
                    ),
                    skipped_detail,
                ),
            )
            self.enrollment.reset(reason="matched_known_speaker")
            self.repository.save_event(event)
            return event

        if decision.top_speaker is not None and decision.score >= self.soft_match_threshold and not ambiguous:
            updated = decision.top_speaker
            soft_update_threshold = max(self.soft_match_threshold + 0.05, self.centroid_update_threshold - 0.06)
            if decision.score >= soft_update_threshold:
                updated = self._update_speaker(decision.top_speaker, frame.embedding)
            event = RecognitionEvent(
                event_type=EventType.MATCH,
                scope=self.scope,
                source=frame.source,
                speaker_id=updated.id,
                speaker_name=updated.name,
                score=round(decision.score, 6),
                confidence=round(decision.confidence, 2),
                details=self._merge_details("soft_match", self._scoring_details(decision), skipped_detail),
            )
            self.enrollment.reset(reason="soft_matched_known_speaker")
            self.repository.save_event(event)
            return event

        created = self.enrollment.consider(
            embedding=frame.embedding,
            best_known_score=decision.score,
            repository=self.repository,
            known_speaker_count=len(speakers),
        )
        if created is not None:
            event = RecognitionEvent(
                event_type=EventType.NEW_SPEAKER,
                scope=self.scope,
                source=frame.source,
                speaker_id=created.id,
                speaker_name=created.name,
                score=round(decision.score, 6),
                confidence=round(decision.confidence, 2),
                details=self._merge_details("auto_enrolled", skipped_detail),
            )
            self.repository.save_event(event)
            return event

        event = RecognitionEvent(
            event_type=EventType.UNKNOWN_SPEECH,
            scope=self.scope,
            source=frame.source,
            score=round(decision.score, 6),
            confidence=round(decision.confidence, 2),
            details=self._merge_details(
                self._scoring_details(decision),
                self.enrollment.last_reason,
                skipped_detail,
            ),
        )
        self.repository.save_event(event)
        return event

    def _update_speaker(self, speaker: Speaker, embedding: tuple[float, ...]) -> Speaker:
        if not embedding:
            return speaker
        normalized_embedding = self._normalize_vector(embedding)
        old_count = max(0, speaker.sample_count)
        total_count = old_count + 1
        if len(speaker.centroid) == len(normalized_embedding):
            centroid = self._blend_vectors(
                left=speaker.centroid,
                right=normalized_embedding,
                ratio=self.centroid_adapt_rate,
            )
        else:
            centroid = normalized_embedding
        prototypes = [
            self._normalize_vector(vector)
            for vector in speaker.all_prototypes()
            if len(vector) == len(normalized_embedding)
        ]
        if not prototypes:
            prototypes = [centroid]
        else:
            best_index = 0
            best_score = -1.0
            for index, vector in enumerate(prototypes):
                score = sum(a * b for a, b in zip(vector, normalized_embedding))
                if score > best_score:
                    best_score = score
                    best_index = index
            if (
                best_score < self.prototype_novelty_threshold
                and len(prototypes) < self.max_prototypes_per_speaker
            ):
                prototypes.append(normalized_embedding)
            else:
                prototypes[best_index] = self._blend_vectors(
                    left=prototypes[best_index],
                    right=normalized_embedding,
                    ratio=self.centroid_adapt_rate,
                )
        speaker.centroid = centroid
        speaker.prototypes = tuple(prototypes)
        speaker.sample_count = total_count
        speaker.updated_at = utc_now_iso()
        self.repository.update_speaker(speaker)
        return speaker

    @staticmethod
    def _normalize_vector(vector: tuple[float, ...]) -> tuple[float, ...]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return tuple(0.0 for _ in vector)
        return tuple(float(value / norm) for value in vector)

    def _blend_vectors(
        self,
        left: tuple[float, ...],
        right: tuple[float, ...],
        ratio: float,
    ) -> tuple[float, ...]:
        blended = tuple(
            float((1.0 - ratio) * left[index] + ratio * right[index])
            for index in range(len(right))
        )
        return self._normalize_vector(blended)

    @staticmethod
    def _merge_details(*parts: str | None) -> str | None:
        merged = [part for part in parts if part]
        if not merged:
            return None
        return "; ".join(merged)

    @staticmethod
    def _scoring_details(decision) -> str | None:
        if decision.raw_score is None and decision.normalized_score is None:
            return None
        parts: list[str] = []
        if decision.raw_score is not None:
            parts.append(f"raw={decision.raw_score:.4f}")
        if decision.normalized_score is not None:
            parts.append(f"norm={decision.normalized_score:.4f}")
        return " ".join(parts) if parts else None
