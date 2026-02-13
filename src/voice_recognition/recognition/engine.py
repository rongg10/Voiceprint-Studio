from __future__ import annotations

import math
import time

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


class RecognitionEngine:
    def __init__(
        self,
        repository: SpeakerRepository,
        scope: RecognitionScope,
        matcher: SpeakerMatcher | None = None,
        enrollment: AutoEnrollmentManager | None = None,
        centroid_update_threshold: float = 0.88,
        centroid_adapt_rate: float = 0.18,
        prototype_novelty_threshold: float = 0.90,
        max_prototypes_per_speaker: int = 8,
        soft_match_threshold: float | None = None,
        enrollment_reset_non_speech_frames: int = 12,
        min_seconds_between_new_speakers: float = 0.0,
        recent_match_seconds: float = 0.0,
        recent_match_threshold: float = 0.0,
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
            0.55,
            min(self.matcher.config.match_threshold - 0.12, self.matcher.config.match_threshold - 0.01),
        )
        self.soft_match_threshold = (
            default_soft if soft_match_threshold is None else max(0.30, float(soft_match_threshold))
        )
        self.enrollment_reset_non_speech_frames = max(1, int(enrollment_reset_non_speech_frames))
        self._non_speech_frames = 0
        self.min_seconds_between_new_speakers = max(0.0, float(min_seconds_between_new_speakers))
        self.recent_match_seconds = max(0.0, float(recent_match_seconds))
        self.recent_match_threshold = max(0.0, float(recent_match_threshold))
        self._last_enroll_ts = 0.0
        self._last_match_ts = 0.0
        self._last_match_speaker_id: int | None = None

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
        known_count = len(speakers)
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
        score_margin = self._score_margin(decision.score, decision.second_best_score)
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
            self._last_match_ts = time.monotonic()
            self._last_match_speaker_id = updated.id
            self.repository.save_event(event)
            return event

        soft_score_threshold = self._effective_soft_score_threshold(known_speaker_count=known_count)
        soft_confidence_floor = self._effective_soft_confidence_floor(known_speaker_count=known_count)
        soft_margin_floor = self._effective_soft_margin_floor(known_speaker_count=known_count)
        if (
            decision.top_speaker is not None
            and decision.score >= soft_score_threshold
            and decision.confidence >= soft_confidence_floor
            and score_margin >= soft_margin_floor
            and not ambiguous
        ):
            matched = decision.top_speaker
            event = RecognitionEvent(
                event_type=EventType.MATCH,
                scope=self.scope,
                source=frame.source,
                speaker_id=matched.id,
                speaker_name=matched.name,
                score=round(decision.score, 6),
                confidence=round(decision.confidence, 2),
                details=self._merge_details("soft_match", self._scoring_details(decision), skipped_detail),
            )
            self.enrollment.reset(reason="soft_matched_known_speaker")
            self._last_match_ts = time.monotonic()
            self._last_match_speaker_id = matched.id
            self.repository.save_event(event)
            return event

        # High score but still no match usually means close/ambiguous known speakers.
        # Do not auto-enroll this region to avoid duplicate identities.
        if decision.score >= self.matcher.config.match_threshold:
            self.enrollment.reset(reason="ambiguous_high_score")
            event = RecognitionEvent(
                event_type=EventType.UNKNOWN_SPEECH,
                scope=self.scope,
                source=frame.source,
                score=round(decision.score, 6),
                confidence=round(decision.confidence, 2),
                details=self._merge_details(
                    self._scoring_details(decision),
                    (
                        f"ambiguous_high_score second_best={decision.second_best_score:.4f}"
                        if decision.second_best_score is not None
                        else "ambiguous_high_score"
                    ),
                    skipped_detail,
                ),
            )
            self.repository.save_event(event)
            return event

        if self.recent_match_seconds > 0 and self._last_match_speaker_id is not None:
            elapsed = time.monotonic() - self._last_match_ts
            if elapsed <= self.recent_match_seconds:
                last_speaker = next(
                    (speaker for speaker in speakers if speaker.id == self._last_match_speaker_id),
                    None,
                )
                if last_speaker is not None:
                    local_best = self.matcher._best_score_for_speaker(frame.embedding, last_speaker)
                    if local_best is not None:
                        recent_score, _ = local_best
                        recent_confidence = self.matcher._to_confidence(recent_score)
                        threshold = self.recent_match_threshold or max(0.0, self.soft_match_threshold - 0.10)
                        threshold = max(
                            threshold,
                            self._effective_soft_score_threshold(known_speaker_count=known_count) - 0.03,
                        )
                        confidence_floor = max(
                            0.0,
                            self._effective_soft_confidence_floor(known_speaker_count=known_count) - 5.0,
                        )
                        allow_recent_match = recent_score >= threshold and recent_confidence >= confidence_floor
                        if (
                            allow_recent_match
                            and decision.top_speaker is not None
                            and decision.top_speaker.id != last_speaker.id
                        ):
                            competing = self.matcher._best_score_for_speaker(frame.embedding, decision.top_speaker)
                            if competing is not None:
                                competing_score, _ = competing
                                if (competing_score - recent_score) >= max(self.matcher.config.min_margin, 0.04):
                                    allow_recent_match = False
                        if allow_recent_match:
                            event = RecognitionEvent(
                                event_type=EventType.MATCH,
                                scope=self.scope,
                                source=frame.source,
                                speaker_id=last_speaker.id,
                                speaker_name=last_speaker.name,
                                score=round(recent_score, 6),
                                confidence=round(recent_confidence, 2),
                                details=self._merge_details("recent_match", skipped_detail),
                            )
                            self.enrollment.reset(reason="recent_matched_known_speaker")
                            self._last_match_ts = time.monotonic()
                            self._last_match_speaker_id = last_speaker.id
                            self.repository.save_event(event)
                            return event

        if self.min_seconds_between_new_speakers > 0:
            elapsed = time.monotonic() - self._last_enroll_ts
            if elapsed < self.min_seconds_between_new_speakers:
                event = RecognitionEvent(
                    event_type=EventType.UNKNOWN_SPEECH,
                    scope=self.scope,
                    source=frame.source,
                    score=round(decision.score, 6),
                    confidence=round(decision.confidence, 2),
                    details=self._merge_details(
                        self._scoring_details(decision),
                        f"enroll_cooldown={self.min_seconds_between_new_speakers - elapsed:.1f}s",
                        skipped_detail,
                    ),
                )
                self.repository.save_event(event)
                return event

        created = self.enrollment.consider(
            embedding=frame.embedding,
            best_known_score=decision.score,
            repository=self.repository,
            known_speaker_count=known_count,
        )
        if created is not None:
            self._last_enroll_ts = time.monotonic()
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
    def _score_margin(score: float, second_best_score: float | None) -> float:
        if second_best_score is None:
            return 1.0
        return score - second_best_score

    def _effective_soft_score_threshold(self, known_speaker_count: int) -> float:
        gap = 0.12
        if known_speaker_count >= 12:
            gap = 0.05
        elif known_speaker_count >= 6:
            gap = 0.08
        return max(self.soft_match_threshold, self.matcher.config.match_threshold - gap)

    @staticmethod
    def _effective_soft_confidence_floor(known_speaker_count: int) -> float:
        if known_speaker_count >= 12:
            return 30.0
        if known_speaker_count >= 6:
            return 20.0
        if known_speaker_count >= 3:
            return 10.0
        return 0.0

    def _effective_soft_margin_floor(self, known_speaker_count: int) -> float:
        floor = self.matcher.config.min_margin
        if known_speaker_count >= 12:
            return max(floor, 0.08)
        if known_speaker_count >= 6:
            return max(floor, 0.06)
        return floor

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
