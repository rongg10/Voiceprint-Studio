from __future__ import annotations

import unittest
from pathlib import Path

from voice_recognition.evaluation.offline_eval import (
    FileEmbedding,
    collect_pair_scores,
    compute_closed_set_top1,
    compute_verification_metrics,
    fit_platt_scaler,
)
from voice_recognition.recognition.matcher import MatcherConfig, SpeakerMatcher


class OfflineEvalMetricsTests(unittest.TestCase):
    def test_verification_metrics_for_separable_embeddings(self) -> None:
        items = [
            FileEmbedding(path=Path("a1.wav"), label="a", embedding=(1.0, 0.0)),
            FileEmbedding(path=Path("a2.wav"), label="a", embedding=(0.99, 0.1)),
            FileEmbedding(path=Path("b1.wav"), label="b", embedding=(0.0, 1.0)),
            FileEmbedding(path=Path("b2.wav"), label="b", embedding=(0.1, 0.99)),
        ]
        metrics = compute_verification_metrics(items)
        self.assertEqual(2, int(metrics["target_trials"]))
        self.assertEqual(4, int(metrics["nontarget_trials"]))
        self.assertIsNotNone(metrics["eer"])
        assert metrics["eer"] is not None
        self.assertLess(float(metrics["eer"]), 0.20)

    def test_fit_platt_scaler_returns_finite_params(self) -> None:
        items = [
            FileEmbedding(path=Path("a1.wav"), label="a", embedding=(1.0, 0.0)),
            FileEmbedding(path=Path("a2.wav"), label="a", embedding=(0.99, 0.1)),
            FileEmbedding(path=Path("b1.wav"), label="b", embedding=(0.0, 1.0)),
            FileEmbedding(path=Path("b2.wav"), label="b", embedding=(0.1, 0.99)),
        ]
        pairs = collect_pair_scores(items)
        scores = [score for score, _ in pairs]
        labels = [label for _, label in pairs]
        scale, bias = fit_platt_scaler(scores=scores, labels=labels, steps=200)
        self.assertTrue(abs(scale) < 100)
        self.assertTrue(abs(bias) < 100)

    def test_closed_set_top1_metric_shape(self) -> None:
        items = [
            FileEmbedding(path=Path("a1.wav"), label="a", embedding=(1.0, 0.0)),
            FileEmbedding(path=Path("a2.wav"), label="a", embedding=(0.99, 0.1)),
            FileEmbedding(path=Path("b1.wav"), label="b", embedding=(0.0, 1.0)),
            FileEmbedding(path=Path("b2.wav"), label="b", embedding=(0.1, 0.99)),
        ]
        matcher = SpeakerMatcher(MatcherConfig(match_threshold=0.5))
        metrics = compute_closed_set_top1(file_embs=items, matcher=matcher)
        self.assertIn("top1", metrics)
        self.assertIsNotNone(metrics["top1"])


if __name__ == "__main__":
    unittest.main()
