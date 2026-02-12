from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .input_manager import AudioInputManager, AudioSource


@dataclass(slots=True)
class LoopbackSetupResult:
    ok: bool
    configured: bool
    platform: str
    message: str
    requires_reboot: bool = False
    steps: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "configured": self.configured,
            "platform": self.platform,
            "message": self.message,
            "requiresReboot": self.requires_reboot,
            "steps": list(self.steps),
            "logs": list(self.logs),
        }


class LoopbackSetupAssistant:
    """Best-effort one-click loopback setup helper."""

    def run(self) -> LoopbackSetupResult:
        system = platform.system().lower()
        if system == "darwin":
            return self._run_macos()
        if system == "windows":
            return self._run_windows()
        return LoopbackSetupResult(
            ok=False,
            configured=False,
            platform=system,
            message="当前系统暂不支持自动安装回环设备，请手动安装。",
            requires_reboot=False,
            steps=[
                "安装一个虚拟回环输入设备（例如 BlackHole、VB-Cable）。",
                "重启本程序并在 system 模式下刷新设备列表。",
            ],
        )

    def _run_macos(self) -> LoopbackSetupResult:
        logs: list[str] = []
        if self._has_system_loopback():
            return LoopbackSetupResult(
                ok=True,
                configured=True,
                platform="macos",
                message="已检测到回环输入设备，可直接使用 system 模式。",
                requires_reboot=False,
                logs=logs,
            )

        brew = shutil.which("brew")
        if not brew:
            return LoopbackSetupResult(
                ok=False,
                configured=False,
                platform="macos",
                message="未检测到 Homebrew，无法自动安装 BlackHole。",
                requires_reboot=False,
                steps=[
                    "先安装 Homebrew: https://brew.sh/",
                    "安装后再次点击“一键配置回环设备”。",
                ],
                logs=logs,
            )

        list_result = self._run_cmd([brew, "list", "--cask", "blackhole-2ch"], timeout=120)
        logs.extend(list_result["logs"])
        if not list_result["ok"]:
            install_result = self._run_cmd([brew, "install", "--cask", "blackhole-2ch"], timeout=240)
            logs.extend(install_result["logs"])
            if not install_result["ok"]:
                return LoopbackSetupResult(
                    ok=False,
                    configured=False,
                    platform="macos",
                    message="BlackHole 自动安装失败。",
                    requires_reboot=False,
                    steps=[
                        "手动执行: brew install --cask blackhole-2ch",
                        "安装完成后重启本程序并刷新设备。",
                    ],
                    logs=logs,
                )
        plugin_exists = any(
            path.exists()
            for path in [
                Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver"),
                Path("/Library/Audio/Plug-Ins/HAL/BlackHole16ch.driver"),
            ]
        )
        # Avoid reinstall in background because it may require interactive sudo
        # password and appears stuck in GUI.
        if not plugin_exists and not self._has_system_loopback():
            logs.append("driver missing after install; skipping background reinstall (interactive sudo required).")

        open_result = self._run_cmd(["open", "-a", "Audio MIDI Setup"], timeout=20)
        logs.extend(open_result["logs"])

        configured = self._has_system_loopback()
        logs.append(f"driver_present={plugin_exists}")
        logs.append(f"loopback_visible={configured}")
        requires_reboot = plugin_exists and not configured
        steps = [
            "在“音频 MIDI 设置”中确认 BlackHole 2ch 已出现。",
            "需要边听边录时，创建“多输出设备”(耳机 + BlackHole)。",
            "在本程序选择 system 模式并刷新设备列表，选择 BlackHole 输入。",
        ]
        if requires_reboot:
            steps.extend(
                [
                    "必须先重启电脑（CoreAudio 未加载新驱动时不会出现 BlackHole）。",
                    "在 macOS“系统设置 -> 隐私与安全性”确认没有被拦截的音频驱动安装。",
                ]
            )
        return LoopbackSetupResult(
            ok=configured,
            configured=configured,
            platform="macos",
            message=(
                "已检测到 BlackHole，可继续配置路由。"
                if configured
                else (
                    "BlackHole 已安装但系统未加载。请先重启电脑。"
                    if requires_reboot
                    else "已执行安装流程，但系统仍未识别 BlackHole。"
                )
            ),
            requires_reboot=requires_reboot,
            steps=steps,
            logs=logs,
        )

    def _run_windows(self) -> LoopbackSetupResult:
        logs: list[str] = []
        if self._has_system_loopback():
            return LoopbackSetupResult(
                ok=True,
                configured=True,
                platform="windows",
                message="已检测到系统回环输入设备，可直接使用 system 模式。",
                requires_reboot=False,
                logs=logs,
            )

        # Best effort: open Windows sound settings for enabling Stereo Mix.
        open_settings = self._run_cmd(
            ["cmd", "/c", "start", "ms-settings:sound"],
            timeout=20,
            shell=False,
        )
        logs.extend(open_settings["logs"])
        return LoopbackSetupResult(
            ok=True,
            configured=False,
            platform="windows",
            message="已打开 Windows 声音设置，请启用 Stereo Mix 或安装 VB-Cable。",
            requires_reboot=False,
            steps=[
                "在“声音设置 -> 输入设备”启用 Stereo Mix（若硬件支持）。",
                "若无 Stereo Mix，请安装 VB-Cable 后重启本程序。",
                "回到本程序点“刷新设备”，在 system 模式选择回环输入设备。",
            ],
            logs=logs,
        )

    def _has_system_loopback(self) -> bool:
        try:
            devices = AudioInputManager.list_capture_devices(AudioSource.SYSTEM)
        except Exception:
            return False
        return len(devices) > 0

    def _run_cmd(
        self,
        cmd: list[str],
        timeout: int,
        shell: bool = False,
    ) -> dict[str, object]:
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=shell,
            )
        except Exception as exc:
            return {"ok": False, "logs": [f"$ {' '.join(cmd)}", f"failed: {exc}"]}

        logs = [f"$ {' '.join(cmd)}", f"exit={completed.returncode}"]
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        if stdout:
            logs.append(f"stdout: {stdout[:300]}")
        if stderr:
            logs.append(f"stderr: {stderr[:300]}")
        return {"ok": completed.returncode == 0, "logs": logs}
