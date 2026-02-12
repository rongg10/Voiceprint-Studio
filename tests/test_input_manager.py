from __future__ import annotations

import unittest
from unittest.mock import patch

from voice_recognition.audio.input_manager import AudioInputManager, AudioSource


class _DummySD:
    @staticmethod
    def query_devices():
        return [
            {"name": "MacBook Pro Microphone", "max_input_channels": 2},
            {"name": "USB Mic", "max_input_channels": 1},
            {"name": "Built-in Output", "max_input_channels": 0},
        ]


class InputManagerTests(unittest.TestCase):
    def test_system_mode_does_not_fallback_to_microphone(self) -> None:
        with patch("voice_recognition.audio.input_manager.sd", new=_DummySD()):
            devices = AudioInputManager.list_capture_devices(AudioSource.SYSTEM)
        self.assertEqual([], devices)

    def test_microphone_mode_lists_input_devices(self) -> None:
        with patch("voice_recognition.audio.input_manager.sd", new=_DummySD()):
            devices = AudioInputManager.list_capture_devices(AudioSource.MICROPHONE)
        self.assertGreaterEqual(len(devices), 1)
        self.assertTrue(any("Mic" in device.name for device in devices))

    def test_probe_device_uses_stream_data(self) -> None:
        class _Stream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self, size):
                import numpy as np

                # First channel silent, second channel active: probe should still
                # detect signal after channel mix-down.
                data = np.column_stack(
                    [
                        np.zeros((size,), dtype=np.float32),
                        np.full((size,), 0.02, dtype=np.float32),
                    ]
                )
                return data, False

        class _ProbeSD(_DummySD):
            InputStream = _Stream

        with patch("voice_recognition.audio.input_manager.sd", new=_ProbeSD()):
            metrics = AudioInputManager.probe_device(device_index=0, duration_seconds=0.1, channels=2)
        self.assertGreater(float(metrics["rms"]), 0.0)
        self.assertGreater(float(metrics["peak"]), 0.0)


if __name__ == "__main__":
    unittest.main()
