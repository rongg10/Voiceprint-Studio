from __future__ import annotations

from dataclasses import dataclass

from voice_recognition.core.models import Speaker, Vector
from voice_recognition.core.vector_ops import average_vectors, cosine_similarity
from voice_recognition.storage.base import SpeakerRepository


@dataclass(slots=True)
class EnrollmentConfig:
    newcomer_prefix: str = "新人"
    min_segments: int = 3
    max_buffer_segments: int = 20
    min_cluster_similarity: float = 0.76
    # If unknown speech is too similar to an existing speaker (above this score),
    # do not auto-enroll to avoid duplicate identities.
    new_speaker_threshold: float = 0.72
    safety_similarity_threshold: float = 0.80
    cooldown_segments: int = 6
    min_pairwise_similarity: float = 0.62
    max_pairwise_std: float = 0.18
    # Bootstrap phase (disabled when <= 0).
    # Used when the library has very few speakers to avoid "unknown forever".
    bootstrap_until_speakers: int = 0
    bootstrap_min_segments: int | None = None
    bootstrap_min_cluster_similarity: float | None = None
    bootstrap_min_pairwise_similarity: float | None = None
    bootstrap_max_pairwise_std: float | None = None
    bootstrap_new_speaker_threshold: float | None = None
    bootstrap_safety_similarity_threshold: float | None = None


class AutoEnrollmentManager:
    """Buffers unknown speech embeddings and enrolls stable new speakers."""

    def __init__(self, config: EnrollmentConfig | None = None) -> None:
        self.config = config or EnrollmentConfig()
        self._buffer: list[Vector] = []
        self._score_buffer: list[float] = []
        self._cooldown_left = 0
        self._cooldown_anchor: Vector | None = None
        self._last_reason: str | None = None

    @property
    def last_reason(self) -> str | None:
        return self._last_reason

    def consider(
        self,
        embedding: Vector,
        best_known_score: float,
        repository: SpeakerRepository,
        known_speaker_count: int | None = None,
    ) -> Speaker | None:
        self._last_reason = None
        if known_speaker_count is None:
            known_speaker_count = len(repository.list_speakers())
        use_bootstrap = bool(self.config.bootstrap_until_speakers) and (
            known_speaker_count <= int(self.config.bootstrap_until_speakers)
        )
        min_segments = self._bootstrap_int(self.config.min_segments, self.config.bootstrap_min_segments, use_bootstrap)
        min_cluster_similarity = self._bootstrap_float(
            self.config.min_cluster_similarity,
            self.config.bootstrap_min_cluster_similarity,
            use_bootstrap,
        )
        min_pairwise_similarity = self._bootstrap_float(
            self.config.min_pairwise_similarity,
            self.config.bootstrap_min_pairwise_similarity,
            use_bootstrap,
        )
        max_pairwise_std = self._bootstrap_float(
            self.config.max_pairwise_std,
            self.config.bootstrap_max_pairwise_std,
            use_bootstrap,
        )
        new_speaker_threshold = self._bootstrap_float(
            self.config.new_speaker_threshold,
            self.config.bootstrap_new_speaker_threshold,
            use_bootstrap,
        )
        safety_similarity_threshold = self._bootstrap_float(
            self.config.safety_similarity_threshold,
            self.config.bootstrap_safety_similarity_threshold,
            use_bootstrap,
        )
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            if self._cooldown_anchor is not None and len(self._cooldown_anchor) == len(embedding):
                cooldown_threshold = max(
                    new_speaker_threshold,
                    safety_similarity_threshold - 0.02,
                )
                cooldown_score = cosine_similarity(embedding, self._cooldown_anchor)
                if cooldown_score >= cooldown_threshold:
                    self._last_reason = self._format_reason(
                        f"cooldown={self._cooldown_left + 1} sim={cooldown_score:.4f}",
                        use_bootstrap,
                    )
                    return None
            if self._cooldown_left <= 0:
                self._cooldown_anchor = None

        if best_known_score >= new_speaker_threshold:
            self.reset(
                reason=self._format_reason(f"near_known={best_known_score:.4f}", use_bootstrap),
            )
            return None

        self._buffer.append(embedding)
        self._score_buffer.append(float(best_known_score))
        max_keep = max(min_segments, self.config.max_buffer_segments)
        if len(self._buffer) > max_keep:
            self._buffer = self._buffer[-max_keep:]
            self._score_buffer = self._score_buffer[-max_keep:]
        if len(self._buffer) < min_segments:
            self._last_reason = self._format_reason(
                f"buffer={len(self._buffer)}/{min_segments}",
                use_bootstrap,
            )
            return None

        window = self._buffer[-min_segments:]
        score_window = self._score_buffer[-min_segments:]
        if score_window and max(score_window) >= safety_similarity_threshold:
            self._last_reason = self._format_reason(
                f"near_known_window={max(score_window):.4f}",
                use_bootstrap,
            )
            return None

        pairwise_mean, pairwise_std = self._pairwise_stats(window)
        if pairwise_mean < min_pairwise_similarity:
            self._last_reason = self._format_reason(
                f"pairwise_mean={pairwise_mean:.4f}",
                use_bootstrap,
            )
            return None
        if pairwise_std > max_pairwise_std:
            self._last_reason = self._format_reason(
                f"pairwise_std={pairwise_std:.4f}",
                use_bootstrap,
            )
            return None

        centroid = average_vectors(window)
        stability = min(cosine_similarity(vector, centroid) for vector in window)
        if stability < min_cluster_similarity:
            self._last_reason = self._format_reason(
                f"cluster_stability={stability:.4f}",
                use_bootstrap,
            )
            return None

        if self._is_close_to_known(
            centroid=centroid,
            repository=repository,
            safety_similarity_threshold=safety_similarity_threshold,
        ):
            self.reset(reason=self._format_reason("centroid_near_known", use_bootstrap))
            return None

        newcomer_name = repository.next_newcomer_name(prefix=self.config.newcomer_prefix)
        speaker = Speaker(
            id=None,
            name=newcomer_name,
            centroid=centroid,
            prototypes=(centroid,),
            sample_count=len(window),
        )
        created = repository.create_speaker(speaker)
        self.reset(reason=self._format_reason("auto_enrolled", use_bootstrap))
        self._cooldown_left = max(0, int(self.config.cooldown_segments))
        self._cooldown_anchor = created.centroid
        return created

    def reset(self, reason: str | None = None) -> None:
        self._buffer.clear()
        self._score_buffer.clear()
        if reason is not None:
            self._last_reason = reason

    def _is_close_to_known(
        self,
        centroid: Vector,
        repository: SpeakerRepository,
        safety_similarity_threshold: float,
    ) -> bool:
        for speaker in repository.list_speakers():
            for prototype in speaker.all_prototypes():
                if len(prototype) != len(centroid):
                    continue
                if cosine_similarity(centroid, prototype) >= safety_similarity_threshold:
                    return True
        return False

    @staticmethod
    def _pairwise_stats(vectors: list[Vector]) -> tuple[float, float]:
        if len(vectors) < 2:
            return 1.0, 0.0
        similarities: list[float] = []
        for left_index in range(len(vectors)):
            for right_index in range(left_index + 1, len(vectors)):
                left = vectors[left_index]
                right = vectors[right_index]
                if len(left) != len(right):
                    continue
                similarities.append(cosine_similarity(left, right))
        if not similarities:
            return 0.0, 1.0
        mean = sum(similarities) / float(len(similarities))
        variance = sum((value - mean) ** 2 for value in similarities) / float(len(similarities))
        return mean, variance**0.5

    @staticmethod
    def _bootstrap_int(base: int, override: int | None, enabled: bool) -> int:
        if enabled and override is not None:
            return int(override)
        return int(base)

    @staticmethod
    def _bootstrap_float(base: float, override: float | None, enabled: bool) -> float:
        if enabled and override is not None:
            return float(override)
        return float(base)

    @staticmethod
    def _format_reason(reason: str, use_bootstrap: bool) -> str:
        if not use_bootstrap:
            return reason
        return f"bootstrap:{reason}"
