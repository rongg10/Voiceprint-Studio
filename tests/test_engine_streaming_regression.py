from __future__ import annotations

import math
import random
import unittest

from voice_recognition.core.models import EventType, FrameInference, RecognitionScope, SignalType
from voice_recognition.recognition.enrollment import AutoEnrollmentManager, EnrollmentConfig
from voice_recognition.recognition.engine import RecognitionEngine
from voice_recognition.recognition.matcher import MatcherConfig, SpeakerMatcher
from voice_recognition.storage.session_repository import SessionSpeakerRepository


def _normalize(v: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm <= 0:
        return v
    return tuple(x / norm for x in v)


class EngineStreamingRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = SessionSpeakerRepository()
        self.engine = RecognitionEngine(
            repository=self.repo,
            scope=RecognitionScope.SESSION,
            matcher=SpeakerMatcher(
                MatcherConfig(
                    match_threshold=0.82,
                    confidence_low=0.50,
                    confidence_high=0.92,
                    min_margin=0.06,
                )
            ),
            enrollment=AutoEnrollmentManager(
                EnrollmentConfig(
                    newcomer_prefix="新人",
                    min_segments=4,
                    max_buffer_segments=20,
                    min_cluster_similarity=0.80,
                    new_speaker_threshold=0.76,
                    safety_similarity_threshold=0.80,
                    cooldown_segments=6,
                )
            ),
            soft_match_threshold=0.68,
            enrollment_reset_non_speech_frames=4,
        )

    def tearDown(self) -> None:
        self.repo.close()

    def test_single_speaker_long_stream_does_not_duplicate_enrollment(self) -> None:
        rng = random.Random(7)
        base = _normalize((1.0, 0.2, 0.1))
        events = []
        for idx in range(60):
            if idx % 5 == 4:
                events.append(self.engine.process(FrameInference(signal=SignalType.SILENCE, source="stream.wav")))
                continue
            jitter = (
                base[0] + rng.uniform(-0.06, 0.06),
                base[1] + rng.uniform(-0.06, 0.06),
                base[2] + rng.uniform(-0.04, 0.04),
            )
            emb = _normalize(jitter)
            events.append(
                self.engine.process(
                    FrameInference(signal=SignalType.SPEECH, embedding=emb, source="stream.wav")
                )
            )

        new_count = sum(1 for event in events if event.event_type == EventType.NEW_SPEAKER)
        match_count = sum(1 for event in events if event.event_type == EventType.MATCH)
        self.assertLessEqual(new_count, 1, "single speaker stream should not create many new IDs")
        self.assertGreater(match_count, 10, "single speaker stream should be matched repeatedly")

    def test_two_speakers_stream_should_stay_near_two_identities(self) -> None:
        rng = random.Random(11)
        speaker_a = _normalize((1.0, 0.2, 0.1))
        speaker_b = _normalize((0.1, 1.0, 0.2))

        events = []
        for idx in range(120):
            if idx % 11 == 10:
                events.append(self.engine.process(FrameInference(signal=SignalType.NOISE, source="mix.wav")))
                continue
            base = speaker_a if (idx // 20) % 2 == 0 else speaker_b
            jitter = (
                base[0] + rng.uniform(-0.07, 0.07),
                base[1] + rng.uniform(-0.07, 0.07),
                base[2] + rng.uniform(-0.05, 0.05),
            )
            emb = _normalize(jitter)
            events.append(
                self.engine.process(FrameInference(signal=SignalType.SPEECH, embedding=emb, source="mix.wav"))
            )

        new_count = sum(1 for event in events if event.event_type == EventType.NEW_SPEAKER)
        self.assertLessEqual(new_count, 3, "two-speaker stream should not explode into many IDs")

    def test_repeated_short_bursts_across_pause_can_still_enroll(self) -> None:
        repo = SessionSpeakerRepository()
        try:
            engine = RecognitionEngine(
                repository=repo,
                scope=RecognitionScope.SESSION,
                matcher=SpeakerMatcher(
                    MatcherConfig(
                        match_threshold=0.82,
                        confidence_low=0.50,
                        confidence_high=0.92,
                        min_margin=0.06,
                    )
                ),
                enrollment=AutoEnrollmentManager(
                    EnrollmentConfig(
                        newcomer_prefix="新人",
                        min_segments=4,
                        max_buffer_segments=20,
                        min_cluster_similarity=0.80,
                        new_speaker_threshold=0.76,
                        safety_similarity_threshold=0.80,
                        cooldown_segments=6,
                    )
                ),
                soft_match_threshold=0.68,
                enrollment_reset_non_speech_frames=12,
            )
            emb = _normalize((0.2, 1.0, 0.1))
            events: list[EventType] = []
            for _ in range(2):
                events.append(
                    engine.process(
                        FrameInference(signal=SignalType.SPEECH, embedding=emb, source="short_clip.wav")
                    ).event_type
                )
            for _ in range(8):
                events.append(
                    engine.process(FrameInference(signal=SignalType.SILENCE, source="gap.wav")).event_type
                )
            for _ in range(2):
                events.append(
                    engine.process(
                        FrameInference(signal=SignalType.SPEECH, embedding=emb, source="short_clip.wav")
                    ).event_type
                )

            self.assertIn(EventType.NEW_SPEAKER, events)
        finally:
            repo.close()


if __name__ == "__main__":
    unittest.main()
