from __future__ import annotations

import unittest

from voice_recognition.core.models import Speaker
from voice_recognition.recognition.matcher import MatcherConfig, SpeakerMatcher


class MatcherBehaviorTests(unittest.TestCase):
    def test_confidence_not_saturated_for_mid_high_score(self) -> None:
        matcher = SpeakerMatcher(
            MatcherConfig(match_threshold=0.95, confidence_low=0.86, confidence_high=0.995)
        )
        # Construct a score around 0.96.
        query = (1.0, 0.0)
        speaker = Speaker(id=1, name="新人1", centroid=(0.96, 0.28), sample_count=3)
        decision = matcher.match(query, [speaker])
        self.assertIsNotNone(decision.speaker)
        self.assertGreater(decision.score, 0.95)
        self.assertLess(decision.confidence, 100.0)

    def test_margin_rule_rejects_ambiguous_match(self) -> None:
        matcher = SpeakerMatcher(
            MatcherConfig(match_threshold=0.95, confidence_low=0.86, confidence_high=0.995, min_margin=0.02)
        )
        query = (1.0, 0.0)
        # Two close speakers around similar angle should be treated as ambiguous.
        a = Speaker(id=1, name="新人1", centroid=(0.97, 0.24), sample_count=5)
        b = Speaker(id=2, name="新人2", centroid=(0.968, 0.251), sample_count=4)
        decision = matcher.match(query, [a, b])
        self.assertIsNone(decision.speaker)
        self.assertIsNotNone(decision.second_best_score)

    def test_multi_prototype_uses_best_profile(self) -> None:
        matcher = SpeakerMatcher(
            MatcherConfig(match_threshold=0.90, confidence_low=0.60, confidence_high=0.98, min_margin=0.02)
        )
        query = (1.0, 0.0)
        speaker = Speaker(
            id=1,
            name="新人1",
            centroid=(0.2, 0.98),
            prototypes=((0.2, 0.98), (0.995, 0.1)),
            sample_count=10,
        )
        decision = matcher.match(query, [speaker])
        self.assertIsNotNone(decision.speaker)
        self.assertGreater(decision.score, 0.95)

    def test_dimension_mismatch_is_skipped(self) -> None:
        matcher = SpeakerMatcher(
            MatcherConfig(match_threshold=0.90, confidence_low=0.60, confidence_high=0.98, min_margin=0.02)
        )
        query = (1.0, 0.0)
        bad = Speaker(id=1, name="坏向量", centroid=(1.0, 0.0, 0.0), sample_count=2)
        good = Speaker(id=2, name="好向量", centroid=(0.99, 0.12), sample_count=2)
        decision = matcher.match(query, [bad, good])
        self.assertIsNotNone(decision.speaker)
        self.assertEqual("好向量", decision.speaker.name if decision.speaker else None)
        self.assertEqual(1, decision.skipped_speakers)

    def test_asnorm_backend_emits_normalized_score_when_cohort_available(self) -> None:
        matcher = SpeakerMatcher(
            MatcherConfig(
                match_threshold=0.70,
                confidence_low=0.40,
                confidence_high=0.90,
                score_backend="asnorm",
                asnorm_top_k=3,
                asnorm_min_cohort=2,
                asnorm_blend=0.5,
            )
        )
        query = (1.0, 0.0)
        target = Speaker(id=1, name="目标", centroid=(0.97, 0.24), sample_count=3)
        cohort_a = Speaker(id=2, name="cohort_a", centroid=(0.4, 0.9), sample_count=3)
        cohort_b = Speaker(id=3, name="cohort_b", centroid=(0.2, 0.98), sample_count=3)
        decision = matcher.match(query, [target, cohort_a, cohort_b])
        self.assertIsNotNone(decision.top_speaker)
        self.assertIsNotNone(decision.normalized_score)
        assert decision.normalized_score is not None
        self.assertGreaterEqual(decision.normalized_score, 0.0)
        self.assertLessEqual(decision.normalized_score, 1.0)

    def test_asnorm_backend_falls_back_to_raw_when_cohort_is_small(self) -> None:
        matcher = SpeakerMatcher(
            MatcherConfig(
                match_threshold=0.70,
                confidence_low=0.40,
                confidence_high=0.90,
                score_backend="asnorm",
                asnorm_top_k=2,
                asnorm_min_cohort=5,
            )
        )
        query = (1.0, 0.0)
        target = Speaker(id=1, name="目标", centroid=(0.99, 0.1), sample_count=3)
        decision = matcher.match(query, [target])
        self.assertIsNotNone(decision.top_speaker)
        self.assertIsNone(decision.normalized_score)
        self.assertIsNotNone(decision.raw_score)


if __name__ == "__main__":
    unittest.main()
