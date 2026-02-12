from __future__ import annotations

import math
import unittest

from voice_recognition.core.models import EventType, FrameInference, RecognitionScope, SignalType
from voice_recognition.recognition.enrollment import AutoEnrollmentManager, EnrollmentConfig
from voice_recognition.recognition.engine import RecognitionEngine
from voice_recognition.recognition.matcher import MatcherConfig, SpeakerMatcher
from voice_recognition.storage.session_repository import SessionSpeakerRepository


class RecognitionEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = SessionSpeakerRepository()
        self.engine = RecognitionEngine(
            repository=self.repository,
            scope=RecognitionScope.SESSION,
            matcher=SpeakerMatcher(
                MatcherConfig(match_threshold=0.90, confidence_low=0.70, confidence_high=0.98)
            ),
            enrollment=AutoEnrollmentManager(
                EnrollmentConfig(
                    newcomer_prefix="新人",
                    min_segments=2,
                    min_cluster_similarity=0.95,
                    new_speaker_threshold=0.85,
                    safety_similarity_threshold=0.92,
                )
            ),
        )

    def tearDown(self) -> None:
        self.repository.close()

    def test_noise_frame_is_marked_as_noise(self) -> None:
        event = self.engine.process(FrameInference(signal=SignalType.NOISE, source="noise.mp3"))
        self.assertEqual(EventType.NOISE, event.event_type)
        self.assertEqual([], self.repository.list_speakers())

    def test_auto_enrollment_then_match(self) -> None:
        first = self.engine.process(
            FrameInference(signal=SignalType.SPEECH, embedding=(0.99, 0.01, 0.0), source="a1.mp3")
        )
        second = self.engine.process(
            FrameInference(signal=SignalType.SPEECH, embedding=(0.98, 0.02, 0.01), source="a1.mp3")
        )
        self.assertEqual(EventType.UNKNOWN_SPEECH, first.event_type)
        self.assertEqual(EventType.NEW_SPEAKER, second.event_type)
        self.assertEqual("新人1", second.speaker_name)
        self.assertEqual(1, len(self.repository.list_speakers()))

        third = self.engine.process(
            FrameInference(signal=SignalType.SPEECH, embedding=(0.97, 0.03, 0.01), source="a2.mp3")
        )
        self.assertEqual(EventType.MATCH, third.event_type)
        self.assertEqual("新人1", third.speaker_name)
        self.assertIsNotNone(third.confidence)

    def test_mid_similarity_unknown_can_still_enroll(self) -> None:
        # Seed one known speaker manually.
        from voice_recognition.core.models import Speaker

        self.repository.create_speaker(
            Speaker(id=None, name="新人1", centroid=(1.0, 0.0, 0.0), sample_count=6)
        )
        # Build an unknown vector with cosine ~0.95 to known (below match=0.90 in this test's matcher
        # would match). So create a dedicated engine for this scenario with higher match threshold.
        engine = RecognitionEngine(
            repository=self.repository,
            scope=RecognitionScope.SESSION,
            matcher=SpeakerMatcher(
                MatcherConfig(match_threshold=0.965, confidence_low=0.70, confidence_high=0.995)
            ),
            enrollment=AutoEnrollmentManager(
                EnrollmentConfig(
                    newcomer_prefix="新人",
                    min_segments=2,
                    min_cluster_similarity=0.95,
                    new_speaker_threshold=0.975,
                    safety_similarity_threshold=0.97,
                )
            ),
            soft_match_threshold=0.96,
        )
        x = math.sqrt(max(0.0, 1.0 - 0.95 * 0.95))
        emb = (0.95, x, 0.0)
        first = engine.process(FrameInference(signal=SignalType.SPEECH, embedding=emb, source="u1.mp3"))
        second = engine.process(FrameInference(signal=SignalType.SPEECH, embedding=emb, source="u2.mp3"))
        self.assertEqual(EventType.UNKNOWN_SPEECH, first.event_type)
        self.assertEqual(EventType.NEW_SPEAKER, second.event_type)
        self.assertEqual("新人2", second.speaker_name)

    def test_ambiguous_high_score_should_not_auto_enroll(self) -> None:
        from voice_recognition.core.models import Speaker

        self.repository.create_speaker(
            Speaker(id=None, name="新人1", centroid=(1.0, 0.0, 0.0), sample_count=8)
        )
        self.repository.create_speaker(
            Speaker(id=None, name="新人2", centroid=(0.9798, 0.2, 0.0), sample_count=8)
        )
        engine = RecognitionEngine(
            repository=self.repository,
            scope=RecognitionScope.SESSION,
            matcher=SpeakerMatcher(
                MatcherConfig(
                    match_threshold=0.965,
                    confidence_low=0.70,
                    confidence_high=0.995,
                    min_margin=0.015,
                )
            ),
            enrollment=AutoEnrollmentManager(
                EnrollmentConfig(
                    newcomer_prefix="新人",
                    min_segments=2,
                    min_cluster_similarity=0.95,
                    new_speaker_threshold=0.975,
                    safety_similarity_threshold=0.97,
                )
            ),
        )
        emb = (0.99499, 0.1, 0.0)
        event1 = engine.process(FrameInference(signal=SignalType.SPEECH, embedding=emb, source="a.mp3"))
        event2 = engine.process(FrameInference(signal=SignalType.SPEECH, embedding=emb, source="b.mp3"))
        self.assertEqual(EventType.UNKNOWN_SPEECH, event1.event_type)
        self.assertEqual(EventType.UNKNOWN_SPEECH, event2.event_type)
        self.assertIn("ambiguous_high_score", event1.details or "")
        self.assertEqual(2, len(self.repository.list_speakers()))

    def test_dimension_mismatch_is_skipped_instead_of_crash(self) -> None:
        from voice_recognition.core.models import Speaker

        self.repository.create_speaker(
            Speaker(
                id=None,
                name="旧维度",
                centroid=(1.0, 0.0, 0.0),
                prototypes=((1.0, 0.0, 0.0),),
                sample_count=8,
            )
        )
        event = self.engine.process(
            FrameInference(signal=SignalType.SPEECH, embedding=(0.8, 0.2), source="dim_shift.mp3")
        )
        self.assertEqual(EventType.UNKNOWN_SPEECH, event.event_type)
        self.assertIn("skipped_dim_mismatch=1", event.details or "")

    def test_short_silence_does_not_break_enrollment_buffer(self) -> None:
        engine = RecognitionEngine(
            repository=self.repository,
            scope=RecognitionScope.SESSION,
            matcher=SpeakerMatcher(
                MatcherConfig(match_threshold=0.95, confidence_low=0.70, confidence_high=0.995)
            ),
            enrollment=AutoEnrollmentManager(
                EnrollmentConfig(
                    newcomer_prefix="新人",
                    min_segments=3,
                    min_cluster_similarity=0.92,
                    new_speaker_threshold=0.90,
                    safety_similarity_threshold=0.95,
                )
            ),
            enrollment_reset_non_speech_frames=4,
            soft_match_threshold=0.93,
        )
        self.assertEqual(
            EventType.UNKNOWN_SPEECH,
            engine.process(
                FrameInference(signal=SignalType.SPEECH, embedding=(0.99, 0.01, 0.0), source="s1.mp3")
            ).event_type,
        )
        self.assertEqual(
            EventType.SILENCE,
            engine.process(FrameInference(signal=SignalType.SILENCE, source="gap.mp3")).event_type,
        )
        self.assertEqual(
            EventType.UNKNOWN_SPEECH,
            engine.process(
                FrameInference(signal=SignalType.SPEECH, embedding=(0.98, 0.02, 0.0), source="s2.mp3")
            ).event_type,
        )
        third = engine.process(
            FrameInference(signal=SignalType.SPEECH, embedding=(0.97, 0.03, 0.0), source="s3.mp3")
        )
        self.assertEqual(EventType.NEW_SPEAKER, third.event_type)

    def test_soft_match_can_recover_low_score_same_speaker(self) -> None:
        from voice_recognition.core.models import Speaker

        self.repository.create_speaker(
            Speaker(id=None, name="新人1", centroid=(1.0, 0.0, 0.0), sample_count=8)
        )
        engine = RecognitionEngine(
            repository=self.repository,
            scope=RecognitionScope.SESSION,
            matcher=SpeakerMatcher(
                MatcherConfig(match_threshold=0.90, confidence_low=0.70, confidence_high=0.995)
            ),
            enrollment=AutoEnrollmentManager(
                EnrollmentConfig(
                    newcomer_prefix="新人",
                    min_segments=2,
                    min_cluster_similarity=0.92,
                    new_speaker_threshold=0.88,
                    safety_similarity_threshold=0.92,
                )
            ),
            soft_match_threshold=0.75,
        )
        # cosine ~= 0.78: below hard match threshold but above soft match threshold.
        emb = (0.78, math.sqrt(max(0.0, 1.0 - 0.78 * 0.78)), 0.0)
        event = engine.process(FrameInference(signal=SignalType.SPEECH, embedding=emb, source="soft.mp3"))
        self.assertEqual(EventType.MATCH, event.event_type)
        self.assertEqual("新人1", event.speaker_name)
        self.assertIn("soft_match", event.details or "")


if __name__ == "__main__":
    unittest.main()
