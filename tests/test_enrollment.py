from __future__ import annotations

import unittest

from voice_recognition.recognition.enrollment import AutoEnrollmentManager, EnrollmentConfig
from voice_recognition.storage.session_repository import SessionSpeakerRepository


class EnrollmentTests(unittest.TestCase):
    def test_sliding_window_can_enroll_after_unstable_prefix(self) -> None:
        repo = SessionSpeakerRepository()
        try:
            mgr = AutoEnrollmentManager(
                EnrollmentConfig(
                    min_segments=3,
                    max_buffer_segments=8,
                    min_cluster_similarity=0.90,
                    new_speaker_threshold=0.70,
                )
            )
            # Unstable first samples.
            self.assertIsNone(mgr.consider((1.0, 0.0, 0.0), best_known_score=0.0, repository=repo))
            self.assertIsNone(mgr.consider((0.0, 1.0, 0.0), best_known_score=0.0, repository=repo))
            self.assertIsNone(mgr.consider((1.0, 0.0, 0.0), best_known_score=0.0, repository=repo))
            # Then stable cluster from a new speaker.
            self.assertIsNone(mgr.consider((0.90, 0.10, 0.0), best_known_score=0.0, repository=repo))
            created = mgr.consider((0.89, 0.11, 0.0), best_known_score=0.0, repository=repo)
            if created is None:
                created = mgr.consider((0.91, 0.09, 0.0), best_known_score=0.0, repository=repo)
            self.assertIsNotNone(created)
            assert created is not None
            self.assertEqual("新人1", created.name)
            self.assertEqual(3, created.sample_count)
        finally:
            repo.close()

    def test_cooldown_prevents_immediate_duplicate_enrollment(self) -> None:
        repo = SessionSpeakerRepository()
        try:
            mgr = AutoEnrollmentManager(
                EnrollmentConfig(
                    min_segments=2,
                    max_buffer_segments=8,
                    min_cluster_similarity=0.90,
                    new_speaker_threshold=0.70,
                    safety_similarity_threshold=0.95,
                    cooldown_segments=3,
                )
            )
            self.assertIsNone(mgr.consider((0.92, 0.08, 0.0), best_known_score=0.0, repository=repo))
            created = mgr.consider((0.91, 0.09, 0.0), best_known_score=0.0, repository=repo)
            self.assertIsNotNone(created)
            assert created is not None

            for _ in range(3):
                self.assertIsNone(
                    mgr.consider((0.91, 0.09, 0.0), best_known_score=0.0, repository=repo)
                )
            self.assertEqual(1, len(repo.list_speakers()))
        finally:
            repo.close()

    def test_cooldown_allows_different_speaker_while_blocking_duplicates(self) -> None:
        repo = SessionSpeakerRepository()
        try:
            mgr = AutoEnrollmentManager(
                EnrollmentConfig(
                    min_segments=2,
                    max_buffer_segments=8,
                    min_cluster_similarity=0.85,
                    min_pairwise_similarity=0.80,
                    max_pairwise_std=0.20,
                    new_speaker_threshold=0.70,
                    safety_similarity_threshold=0.90,
                    cooldown_segments=4,
                )
            )
            self.assertIsNone(mgr.consider((0.92, 0.08, 0.0), best_known_score=0.0, repository=repo))
            created = mgr.consider((0.91, 0.09, 0.0), best_known_score=0.0, repository=repo)
            self.assertIsNotNone(created)
            assert created is not None
            # A clearly different voice should still be enroll-able during cooldown.
            self.assertIsNone(mgr.consider((0.10, 0.90, 0.0), best_known_score=0.0, repository=repo))
            created_2 = mgr.consider((0.09, 0.91, 0.0), best_known_score=0.0, repository=repo)
            self.assertIsNotNone(created_2)
            self.assertEqual(2, len(repo.list_speakers()))
        finally:
            repo.close()

    def test_pairwise_variance_guard_blocks_unstable_cluster(self) -> None:
        repo = SessionSpeakerRepository()
        try:
            mgr = AutoEnrollmentManager(
                EnrollmentConfig(
                    min_segments=4,
                    max_buffer_segments=8,
                    min_cluster_similarity=0.70,
                    min_pairwise_similarity=0.60,
                    max_pairwise_std=0.05,
                    new_speaker_threshold=0.70,
                )
            )
            # Two distinct patterns interleaved: centroid similarity can look acceptable,
            # but pairwise variance is high and should block auto enrollment.
            stream = [
                (0.98, 0.02, 0.0),
                (0.60, 0.80, 0.0),
                (0.97, 0.03, 0.0),
                (0.58, 0.82, 0.0),
                (0.99, 0.01, 0.0),
                (0.57, 0.83, 0.0),
            ]
            created = None
            for emb in stream:
                created = mgr.consider(emb, best_known_score=0.0, repository=repo) or created
            self.assertIsNone(created)
            self.assertEqual(0, len(repo.list_speakers()))
        finally:
            repo.close()


if __name__ == "__main__":
    unittest.main()
