from __future__ import annotations

import math
from dataclasses import dataclass

from voice_recognition.core.models import Speaker, Vector
from voice_recognition.core.vector_ops import cosine_similarity


@dataclass(slots=True)
class MatchDecision:
    speaker: Speaker | None
    top_speaker: Speaker | None
    score: float
    confidence: float
    second_best_score: float | None = None
    skipped_speakers: int = 0
    raw_score: float | None = None
    normalized_score: float | None = None


@dataclass(slots=True)
class MatcherConfig:
    match_threshold: float = 0.82
    confidence_low: float = 0.50
    confidence_high: float = 0.92
    min_margin: float = 0.06
    # cosine | asnorm
    score_backend: str = "cosine"
    asnorm_top_k: int = 30
    asnorm_min_cohort: int = 8
    # Final score uses a weighted blend between raw cosine and normalized score.
    asnorm_blend: float = 0.35
    # Logistic calibration for normalized score.
    calibration_scale: float = 4.0
    calibration_bias: float = -2.0


class SpeakerMatcher:
    def __init__(self, config: MatcherConfig | None = None) -> None:
        self.config = config or MatcherConfig()
        backend = self.config.score_backend.strip().lower()
        if backend not in {"cosine", "asnorm"}:
            raise ValueError(f"Unsupported matcher score backend: {self.config.score_backend}")
        self.config.score_backend = backend

    def match(self, embedding: Vector, speakers: list[Speaker]) -> MatchDecision:
        if not speakers:
            return MatchDecision(
                speaker=None,
                top_speaker=None,
                score=0.0,
                confidence=0.0,
                second_best_score=None,
                skipped_speakers=0,
            )
        best_speaker: Speaker | None = None
        best_score = -1.0
        best_raw_score: float | None = None
        best_normalized_score: float | None = None
        second_best = -1.0
        skipped = 0
        for speaker in speakers:
            local_best = self._best_score_for_speaker(embedding=embedding, speaker=speaker)
            if local_best is None:
                skipped += 1
                continue
            raw_score, best_prototype = local_best
            local_score = raw_score
            normalized_score: float | None = None
            if self.config.score_backend == "asnorm":
                cohort = self._cohort_vectors(
                    speakers=speakers,
                    exclude_speaker_id=speaker.id,
                    dim=len(embedding),
                )
                normalized_score = self._asnorm_score(
                    query=embedding,
                    target=best_prototype,
                    raw_score=raw_score,
                    cohort=cohort,
                )
                if normalized_score is not None:
                    local_score = self._blend_score(
                        raw_score=raw_score,
                        normalized_score=normalized_score,
                    )
            if local_score > best_score:
                second_best = best_score
                best_score = local_score
                best_speaker = speaker
                best_raw_score = raw_score
                best_normalized_score = normalized_score
            elif local_score > second_best:
                second_best = local_score
        if best_speaker is None:
            return MatchDecision(
                speaker=None,
                top_speaker=None,
                score=0.0,
                confidence=0.0,
                second_best_score=None,
                skipped_speakers=skipped,
            )
        confidence = self._to_confidence(best_score)
        second_best_score = second_best if second_best >= 0 else None
        if best_score >= self.config.match_threshold:
            if second_best_score is not None and (best_score - second_best_score) < self.config.min_margin:
                return MatchDecision(
                    speaker=None,
                    top_speaker=best_speaker,
                    score=best_score,
                    confidence=confidence,
                    second_best_score=second_best_score,
                    skipped_speakers=skipped,
                    raw_score=best_raw_score,
                    normalized_score=best_normalized_score,
                )
            return MatchDecision(
                speaker=best_speaker,
                top_speaker=best_speaker,
                score=best_score,
                confidence=confidence,
                second_best_score=second_best_score,
                skipped_speakers=skipped,
                raw_score=best_raw_score,
                normalized_score=best_normalized_score,
            )
        return MatchDecision(
            speaker=None,
            top_speaker=best_speaker,
            score=best_score,
            confidence=confidence,
            second_best_score=second_best_score,
            skipped_speakers=skipped,
            raw_score=best_raw_score,
            normalized_score=best_normalized_score,
        )

    def _to_confidence(self, score: float) -> float:
        low = self.config.confidence_low
        high = self.config.confidence_high
        if high <= low:
            return 0.0
        scaled = (score - low) / (high - low)
        clipped = max(0.0, min(1.0, scaled))
        curved = clipped**1.8
        return curved * 100.0

    @staticmethod
    def _best_score_for_speaker(embedding: Vector, speaker: Speaker) -> tuple[float, Vector] | None:
        scores: list[tuple[float, Vector]] = []
        for prototype in speaker.all_prototypes():
            if len(prototype) != len(embedding):
                continue
            scores.append((cosine_similarity(embedding, prototype), prototype))
        if not scores:
            return None
        scores.sort(key=lambda item: item[0], reverse=True)
        return scores[0]

    @staticmethod
    def _cohort_vectors(
        speakers: list[Speaker],
        exclude_speaker_id: int | None,
        dim: int,
    ) -> list[Vector]:
        cohort: list[Vector] = []
        for speaker in speakers:
            if exclude_speaker_id is not None and speaker.id == exclude_speaker_id:
                continue
            for prototype in speaker.all_prototypes():
                if len(prototype) == dim:
                    cohort.append(prototype)
        return cohort

    def _asnorm_score(
        self,
        query: Vector,
        target: Vector,
        raw_score: float,
        cohort: list[Vector],
    ) -> float | None:
        top_k = max(1, int(self.config.asnorm_top_k))
        min_cohort = max(2, int(self.config.asnorm_min_cohort))
        if len(cohort) < min_cohort:
            return None
        q_scores = sorted((cosine_similarity(query, vec) for vec in cohort), reverse=True)[:top_k]
        t_scores = sorted((cosine_similarity(target, vec) for vec in cohort), reverse=True)[:top_k]
        if len(q_scores) < min_cohort or len(t_scores) < min_cohort:
            return None
        q_mean, q_std = self._mean_std(q_scores)
        t_mean, t_std = self._mean_std(t_scores)
        z = 0.5 * (((raw_score - q_mean) / q_std) + ((raw_score - t_mean) / t_std))
        return self._sigmoid(self.config.calibration_scale * z + self.config.calibration_bias)

    def _blend_score(self, raw_score: float, normalized_score: float) -> float:
        blend = max(0.0, min(1.0, float(self.config.asnorm_blend)))
        return float((1.0 - blend) * raw_score + blend * normalized_score)

    @staticmethod
    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 1.0
        mean = sum(values) / float(len(values))
        variance = sum((value - mean) ** 2 for value in values) / float(len(values))
        std = max(variance**0.5, 1e-6)
        return mean, std

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 35.0:
            return 1.0
        if value <= -35.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-value))
