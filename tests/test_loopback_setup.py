from __future__ import annotations

import unittest
from unittest.mock import patch

from voice_recognition.audio.loopback_setup import LoopbackSetupAssistant


class LoopbackSetupTests(unittest.TestCase):
    def test_macos_without_brew_returns_actionable_instructions(self) -> None:
        assistant = LoopbackSetupAssistant()
        with (
            patch("voice_recognition.audio.loopback_setup.platform.system", return_value="Darwin"),
            patch("voice_recognition.audio.loopback_setup.shutil.which", return_value=None),
            patch.object(LoopbackSetupAssistant, "_has_system_loopback", return_value=False),
        ):
            result = assistant.run()
        self.assertFalse(result.configured)
        self.assertFalse(result.ok)
        self.assertIn("Homebrew", result.message)
        self.assertGreaterEqual(len(result.steps), 1)

    def test_existing_loopback_short_circuit(self) -> None:
        assistant = LoopbackSetupAssistant()
        with (
            patch("voice_recognition.audio.loopback_setup.platform.system", return_value="Darwin"),
            patch.object(LoopbackSetupAssistant, "_has_system_loopback", return_value=True),
        ):
            result = assistant.run()
        self.assertTrue(result.ok)
        self.assertTrue(result.configured)

    def test_macos_not_visible_after_install_returns_not_configured(self) -> None:
        assistant = LoopbackSetupAssistant()
        with (
            patch("voice_recognition.audio.loopback_setup.platform.system", return_value="Darwin"),
            patch("voice_recognition.audio.loopback_setup.shutil.which", return_value="/opt/homebrew/bin/brew"),
            patch.object(LoopbackSetupAssistant, "_has_system_loopback", side_effect=[False, False, False]),
            patch.object(LoopbackSetupAssistant, "_run_cmd", return_value={"ok": True, "logs": []}),
        ):
            result = assistant.run()
        self.assertFalse(result.configured)
        self.assertFalse(result.ok)
        self.assertTrue(result.requires_reboot)
        self.assertIn("重启", result.message)

    def test_unsupported_platform_returns_manual_steps(self) -> None:
        assistant = LoopbackSetupAssistant()
        with patch("voice_recognition.audio.loopback_setup.platform.system", return_value="Linux"):
            result = assistant.run()
        self.assertFalse(result.ok)
        self.assertFalse(result.configured)
        self.assertGreaterEqual(len(result.steps), 1)


if __name__ == "__main__":
    unittest.main()
