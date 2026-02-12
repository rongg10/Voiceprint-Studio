from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from voice_recognition.audio import AudioSource
from voice_recognition.audio.loopback_setup import LoopbackSetupResult
from voice_recognition.core.models import EventType, RecognitionEvent, RecognitionScope, Speaker
from voice_recognition.live_service import LiveUpdate
from voice_recognition.web_controller import WebController


class WebControllerTests(unittest.TestCase):
    def test_match_event_highlights_active_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            update = LiveUpdate(
                event=RecognitionEvent(
                    event_type=EventType.MATCH,
                    scope=RecognitionScope.SESSION,
                    speaker_id=1,
                    speaker_name="新人1",
                    score=0.93,
                    confidence=89.2,
                    source=AudioSource.MICROPHONE.value,
                ),
                speakers=[
                    Speaker(id=1, name="新人1", centroid=(0.1, 0.2), sample_count=4),
                    Speaker(id=2, name="新人2", centroid=(0.3, 0.4), sample_count=3),
                ],
            )
            controller._apply_update(update)
            snapshot = controller.snapshot()
            self.assertEqual("Matched Speaker", snapshot["status"])
            self.assertEqual("新人1", snapshot["currentSpeaker"])
            self.assertEqual("89.20%", snapshot["confidence"])
            active_rows = [row for row in snapshot["speakers"] if row["active"]]
            self.assertEqual(1, len(active_rows))
            self.assertEqual(1, active_rows[0]["id"])

    def test_noise_event_sets_background_noise_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            update = LiveUpdate(
                event=RecognitionEvent(
                    event_type=EventType.NOISE,
                    scope=RecognitionScope.SESSION,
                    source=AudioSource.SYSTEM.value,
                ),
                speakers=[],
            )
            controller._apply_update(update)
            snapshot = controller.snapshot()
            self.assertEqual("Background Noise", snapshot["status"])
            self.assertEqual("-", snapshot["currentSpeaker"])
            self.assertEqual("-", snapshot["confidence"])

    def test_setup_loopback_updates_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            fake = LoopbackSetupResult(
                ok=True,
                configured=False,
                platform="macos",
                message="setup done",
                steps=["step1"],
                logs=["log1"],
            )
            with patch.object(controller.loopback_setup, "run", return_value=fake):
                result = controller.setup_loopback()
            snapshot = controller.snapshot()
            self.assertTrue(bool(result["ok"]))
            self.assertIn("setup done", " ".join(snapshot["events"]))
            self.assertIn("setup done", str(snapshot.get("setupNotice")))

    def test_setup_loopback_async_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            fake = LoopbackSetupResult(
                ok=True,
                configured=True,
                platform="macos",
                message="configured",
                steps=[],
                logs=[],
            )
            with patch.object(controller.loopback_setup, "run", return_value=fake):
                started = controller.start_setup_loopback()
                self.assertTrue(bool(started["running"]))
                for _ in range(30):
                    status = controller.setup_loopback_status()
                    if not bool(status["running"]):
                        break
                    time.sleep(0.01)
                self.assertFalse(bool(status["running"]))
                self.assertTrue(bool((status["result"] or {}).get("ok")))
                snapshot = controller.snapshot()
                self.assertFalse(bool(snapshot.get("setupRunning")))

    def test_diagnostics_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            with (
                patch.object(controller, "list_devices", side_effect=[[], []]),
            ):
                info = controller.diagnostics()
            self.assertIn("microphoneCount", info)
            self.assertIn("systemCount", info)
            self.assertIn("blackholeDriverPresent", info)

    def test_probe_device_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            with patch(
                "voice_recognition.web_controller.AudioInputManager.probe_device",
                return_value={"rms": 0.01, "peak": 0.04},
            ):
                result = controller.probe_device(source=AudioSource.MICROPHONE, device_index=0)
            self.assertTrue(bool(result["ok"]))
            self.assertIn("hasSignal", result)

    def test_probe_device_system_uses_stereo_channels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            with patch(
                "voice_recognition.web_controller.AudioInputManager.probe_device",
                return_value={"rms": 0.002, "peak": 0.01},
            ) as mocked_probe:
                result = controller.probe_device(source=AudioSource.SYSTEM, device_index=3)
            self.assertTrue(bool(result["ok"]))
            mocked_probe.assert_called_once_with(
                device_index=3,
                sample_rate=None,
                channels=2,
                duration_seconds=1.0,
            )

    def test_auto_select_device_for_system(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            with patch.object(
                controller,
                "list_devices",
                return_value=[{"index": 9, "name": "BlackHole 2ch", "channels": 2}],
            ):
                result = controller.auto_select_device(source=AudioSource.SYSTEM)
            self.assertTrue(bool(result["ok"]))
            self.assertEqual(9, int(result["device"]["index"]))

    def test_auto_select_device_for_microphone_by_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            with (
                patch.object(
                    controller,
                    "list_devices",
                    return_value=[
                        {"index": 1, "name": "Mic A", "channels": 1},
                        {"index": 2, "name": "Mic B", "channels": 1},
                    ],
                ),
                patch.object(
                    controller,
                    "probe_device",
                    side_effect=[
                        {"ok": True, "rms": 0.002, "peak": 0.01},
                        {"ok": True, "rms": 0.010, "peak": 0.03},
                    ],
                ),
            ):
                result = controller.auto_select_device(source=AudioSource.MICROPHONE)
            self.assertTrue(bool(result["ok"]))
            self.assertEqual(2, int(result["device"]["index"]))

    def test_request_system_reboot_macos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            with (
                patch("voice_recognition.web_controller.platform.system", return_value="Darwin"),
                patch("voice_recognition.web_controller.subprocess.run"),
            ):
                result = controller.request_system_reboot()
            self.assertTrue(bool(result["ok"]))

    def test_start_applies_embedding_override_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            captured = {}

            def _fake_start(config):
                captured["config"] = config

            with patch.object(controller.service, "start", side_effect=_fake_start):
                controller.start(
                    scope=RecognitionScope.SESSION,
                    source=AudioSource.MICROPHONE,
                    device_index=2,
                    tuning={
                        "matchThreshold": 0.99,
                        "newSpeakerThreshold": 0.99,
                        "minMargin": 0.99,
                        "minSegments": 9,
                        "minClusterSimilarity": 0.99,
                        "embeddingModels": [
                            "hf:microsoft/wavlm-base-plus-sv",
                            "speechbrain/spkrec-ecapa-voxceleb",
                        ],
                    },
                )
            config = captured["config"]
            self.assertAlmostEqual(0.80, float(config.match_threshold), places=6)
            self.assertAlmostEqual(0.62, float(config.new_speaker_threshold), places=6)
            self.assertAlmostEqual(0.05, float(config.min_match_margin), places=6)
            self.assertEqual(5, int(config.enrollment_min_segments))
            self.assertAlmostEqual(0.82, float(config.enrollment_min_cluster_similarity), places=6)
            self.assertEqual(
                (
                    "hf:microsoft/wavlm-base-plus-sv",
                    "speechbrain/spkrec-ecapa-voxceleb",
                ),
                tuple(config.embedding_models),
            )

    def test_start_rejects_stale_device_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller = WebController(db_path=Path(directory) / "speakers.db", sample_rate=16000)
            with patch.object(
                controller,
                "list_devices",
                return_value=[{"index": 1, "name": "Mic A", "channels": 1, "sampleRate": 16000}],
            ):
                with self.assertRaisesRegex(RuntimeError, "输入设备已失效"):
                    controller.start(
                        scope=RecognitionScope.SESSION,
                        source=AudioSource.MICROPHONE,
                        device_index=5,
                    )


if __name__ == "__main__":
    unittest.main()
