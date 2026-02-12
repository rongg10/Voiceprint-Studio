from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from voice_recognition.audio import AudioSource
from voice_recognition.core.models import RecognitionScope
from voice_recognition.live_service import LiveConfig, LiveRecognitionService
from voice_recognition.storage.session_repository import SessionSpeakerRepository


class LiveServiceTests(unittest.TestCase):
    def test_mix_to_mono_uses_all_channels(self) -> None:
        left = np.zeros((8,), dtype=np.float32)
        right = np.full((8,), 0.02, dtype=np.float32)
        stereo = np.column_stack([left, right])
        mono = LiveRecognitionService._mix_to_mono(stereo)
        self.assertEqual((8,), mono.shape)
        self.assertTrue(np.all(mono > 0.0))

    def test_build_runtime_components_uses_neural_backend_defaults(self) -> None:
        service = LiveRecognitionService()
        repo = SessionSpeakerRepository()
        captured = {}

        class _DummyProcessor:
            def __init__(self, config):
                captured["config"] = config

        config = LiveConfig(
            scope=RecognitionScope.SESSION,
            source=AudioSource.MICROPHONE,
            device_index=0,
            backend="neural",
            vad_min_samples_for_embedding=1000,  # should be clamped up
        )
        with patch("voice_recognition.live_service.SignalProcessor", new=_DummyProcessor):
            _, engine = service._build_runtime_components(config=config, repository=repo)
        proc_config = captured["config"]
        self.assertEqual("neural", proc_config.backend)
        self.assertGreaterEqual(proc_config.vad_min_samples_for_embedding, 9600)
        self.assertGreaterEqual(engine.matcher.config.match_threshold, 0.80)
        self.assertGreaterEqual(engine.centroid_update_threshold, engine.matcher.config.match_threshold)
        self.assertGreaterEqual(engine.enrollment.config.min_segments, 3)
        self.assertGreaterEqual(engine.enrollment_reset_non_speech_frames, 10)
        repo.close()

    def test_build_runtime_components_clamps_invalid_threshold_order(self) -> None:
        service = LiveRecognitionService()
        repo = SessionSpeakerRepository()
        captured = {}

        class _DummyProcessor:
            def __init__(self, config):
                captured["config"] = config

        config = LiveConfig(
            scope=RecognitionScope.SESSION,
            source=AudioSource.SYSTEM,
            device_index=0,
            match_threshold=0.70,
            new_speaker_threshold=0.95,
        )
        with patch("voice_recognition.live_service.SignalProcessor", new=_DummyProcessor):
            _, engine = service._build_runtime_components(config=config, repository=repo)
        self.assertLess(engine.enrollment.config.new_speaker_threshold, engine.matcher.config.match_threshold)
        repo.close()

    def test_run_reports_error_when_device_initialization_fails(self) -> None:
        service = LiveRecognitionService()
        config = LiveConfig(
            scope=RecognitionScope.SESSION,
            source=AudioSource.MICROPHONE,
            device_index=999,
        )
        repo = SessionSpeakerRepository()
        with (
            patch("voice_recognition.live_service.build_repository", return_value=repo),
            patch(
                "voice_recognition.live_service.AudioInputManager.get_device",
                side_effect=ValueError("Invalid device index: 999"),
            ),
        ):
            service._running = True
            service._run(config)
        update = service.updates.get_nowait()
        self.assertEqual("error", str(update.event.details))
        self.assertIn("Invalid device index", str(update.error))
        self.assertFalse(service.running)


if __name__ == "__main__":
    unittest.main()
