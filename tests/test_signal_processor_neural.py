from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from voice_recognition.audio.signal_processor import SignalProcessor, SignalProcessorConfig
from voice_recognition.core.models import SignalType


class SignalProcessorNeuralTests(unittest.TestCase):
    def test_neural_backend_path_emits_speech_embedding(self) -> None:
        class _DummyNeuralBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def extract_speech(self, signal: np.ndarray):
                return 0.35, signal

            def embedding(self, speech_wave: np.ndarray):
                return (1.0, 0.0, 0.0)

            def mark_non_speech(self) -> None:
                return

        with patch("voice_recognition.audio.signal_processor._NeuralSpeechBackend", new=_DummyNeuralBackend):
            processor = SignalProcessor(
                SignalProcessorConfig(
                    backend="neural",
                    sample_rate=16000,
                    vad_speech_ratio_threshold=0.12,
                    silence_rms_threshold=0.0001,
                    silence_peak_threshold=0.0002,
                )
            )
            chunk = np.full((16000,), 0.02, dtype=np.float32)
            frame = processor.infer(chunk, source="microphone")

        self.assertEqual(SignalType.SPEECH, frame.signal)
        self.assertEqual((1.0, 0.0, 0.0), frame.embedding)

    def test_neural_backend_path_marks_low_speech_ratio_as_noise(self) -> None:
        class _DummyNeuralBackend:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def extract_speech(self, signal: np.ndarray):
                return 0.05, signal

            def embedding(self, speech_wave: np.ndarray):
                return (1.0, 0.0, 0.0)

            def mark_non_speech(self) -> None:
                return

        with patch("voice_recognition.audio.signal_processor._NeuralSpeechBackend", new=_DummyNeuralBackend):
            processor = SignalProcessor(
                SignalProcessorConfig(
                    backend="neural",
                    sample_rate=16000,
                    vad_speech_ratio_threshold=0.12,
                    silence_rms_threshold=0.0001,
                    silence_peak_threshold=0.0002,
                )
            )
            chunk = np.full((16000,), 0.02, dtype=np.float32)
            frame = processor.infer(chunk, source="microphone")

        self.assertEqual(SignalType.NOISE, frame.signal)
        self.assertIsNone(frame.embedding)


if __name__ == "__main__":
    unittest.main()
