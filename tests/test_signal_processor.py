from __future__ import annotations

import unittest

import numpy as np

from voice_recognition.audio.signal_processor import SignalProcessor
from voice_recognition.core.models import SignalType


class SignalProcessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = SignalProcessor()
        self.sample_rate = 16000

    def test_silence_detection(self) -> None:
        chunk = np.zeros(self.sample_rate, dtype=np.float32)
        frame = self.processor.infer(chunk, source="unit-test")
        self.assertEqual(SignalType.SILENCE, frame.signal)
        self.assertIsNone(frame.embedding)

    def test_noise_detection(self) -> None:
        rng = np.random.default_rng(42)
        chunk = rng.normal(0.0, 0.2, self.sample_rate).astype(np.float32)
        self.processor.infer(np.zeros(self.sample_rate, dtype=np.float32), source="unit-test")
        self.processor.infer(np.zeros(self.sample_rate, dtype=np.float32), source="unit-test")
        frame = self.processor.infer(chunk, source="unit-test")
        self.assertEqual(SignalType.NOISE, frame.signal)
        self.assertIsNone(frame.embedding)

    def test_speech_like_signal_returns_embedding(self) -> None:
        t = np.arange(self.sample_rate, dtype=np.float32) / float(self.sample_rate)
        signal = 0.5 * np.sin(2 * np.pi * 180 * t) + 0.35 * np.sin(2 * np.pi * 360 * t)
        envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)
        chunk = (signal * envelope).astype(np.float32)
        self.processor.infer(np.zeros(self.sample_rate, dtype=np.float32), source="unit-test")
        self.processor.infer(np.zeros(self.sample_rate, dtype=np.float32), source="unit-test")
        frame = self.processor.infer(chunk, source="unit-test")
        self.assertEqual(SignalType.SPEECH, frame.signal)
        self.assertIsNotNone(frame.embedding)
        self.assertGreater(len(frame.embedding or ()), 10)

    def test_low_level_background_is_silence(self) -> None:
        rng = np.random.default_rng(7)
        low_noise = rng.normal(0.0, 0.002, self.sample_rate).astype(np.float32)
        # Warm-up calibration windows.
        self.processor.infer(low_noise, source="system")
        self.processor.infer(low_noise, source="system")
        frame = self.processor.infer(low_noise, source="system")
        self.assertEqual(SignalType.SILENCE, frame.signal)

    def test_system_mode_detects_non_silent_audio(self) -> None:
        t = np.arange(self.sample_rate, dtype=np.float32) / float(self.sample_rate)
        low_speech = (
            0.06 * np.sin(2 * np.pi * 180 * t)
            + 0.04 * np.sin(2 * np.pi * 360 * t)
            + 0.01 * np.sin(2 * np.pi * 720 * t)
        ).astype(np.float32)
        self.processor.infer(np.zeros(self.sample_rate, dtype=np.float32), source="system")
        self.processor.infer(np.zeros(self.sample_rate, dtype=np.float32), source="system")
        frame = self.processor.infer(low_speech, source="system")
        self.assertIn(frame.signal, {SignalType.SPEECH, SignalType.NOISE})

    def test_music_like_tone_is_noise(self) -> None:
        t = np.arange(self.sample_rate, dtype=np.float32) / float(self.sample_rate)
        music = (
            0.08 * np.sin(2 * np.pi * 440 * t)
            + 0.08 * np.sin(2 * np.pi * 880 * t)
            + 0.08 * np.sin(2 * np.pi * 1760 * t)
        ).astype(np.float32)
        self.processor.infer(np.zeros(self.sample_rate, dtype=np.float32), source="system")
        self.processor.infer(np.zeros(self.sample_rate, dtype=np.float32), source="system")
        frame = self.processor.infer(music, source="system")
        self.assertEqual(SignalType.NOISE, frame.signal)

    def test_podcast_like_mix_is_speech(self) -> None:
        t = np.arange(self.sample_rate, dtype=np.float32) / float(self.sample_rate)
        speech = (
            0.05 * np.sin(2 * np.pi * 170 * t)
            + 0.03 * np.sin(2 * np.pi * 340 * t)
        ) * (0.35 + 0.65 * np.maximum(0.0, np.sin(2 * np.pi * 3.5 * t)))
        music = 0.02 * np.sin(2 * np.pi * 440 * t) + 0.015 * np.sin(2 * np.pi * 880 * t)
        chunk = (speech + music).astype(np.float32)
        self.processor.infer(np.zeros(self.sample_rate, dtype=np.float32), source="system")
        self.processor.infer(np.zeros(self.sample_rate, dtype=np.float32), source="system")
        frame = self.processor.infer(chunk, source="system")
        self.assertEqual(SignalType.SPEECH, frame.signal)

    def test_calibration_does_not_swallow_immediate_speech(self) -> None:
        t = np.arange(self.sample_rate, dtype=np.float32) / float(self.sample_rate)
        speech = (
            0.05 * np.sin(2 * np.pi * 190 * t)
            + 0.04 * np.sin(2 * np.pi * 360 * t)
        ).astype(np.float32)
        # First chunk is speech: should not be forced to warmup-silence.
        frame = self.processor.infer(speech, source="microphone")
        self.assertIn(frame.signal, {SignalType.SPEECH, SignalType.NOISE})


if __name__ == "__main__":
    unittest.main()
