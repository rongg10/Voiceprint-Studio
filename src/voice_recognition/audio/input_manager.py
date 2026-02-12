from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - handled at runtime on missing backend
    sd = None


class AudioSource(str, Enum):
    MICROPHONE = "microphone"
    SYSTEM = "system"
    FILE = "file"


@dataclass(slots=True)
class AudioInputConfig:
    source: AudioSource
    file_path: Path | None = None
    sample_rate: int = 16000
    channels: int = 1
    block_size: int = 1024
    device: int | None = None


SYSTEM_DEVICE_KEYWORDS = (
    "blackhole",
    "loopback",
    "stereo mix",
    "what u hear",
    "monitor",
    "vb-audio",
    "virtual",
)


@dataclass(slots=True)
class CaptureDevice:
    index: int
    name: str
    max_input_channels: int
    default_sample_rate: int


class AudioInputManager:
    """Helpers for discovering capture devices and validating configuration."""

    @staticmethod
    def ensure_backend() -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice is not available. Install dependencies first: pip install -r requirements.txt"
            )

    @staticmethod
    def list_capture_devices(source: AudioSource) -> list[CaptureDevice]:
        AudioInputManager.ensure_backend()
        devices = sd.query_devices()
        capture_devices: list[CaptureDevice] = []
        for index, device in enumerate(devices):
            if int(device["max_input_channels"]) <= 0:
                continue
            capture_devices.append(
                CaptureDevice(
                    index=index,
                    name=str(device["name"]),
                    max_input_channels=int(device["max_input_channels"]),
                    default_sample_rate=int(float(device.get("default_samplerate", 16000))),
                )
            )
        if source == AudioSource.MICROPHONE:
            return AudioInputManager._filter_microphone_devices(capture_devices)
        if source == AudioSource.SYSTEM:
            return AudioInputManager._filter_system_devices(capture_devices)
        return capture_devices

    @staticmethod
    def default_device_for(source: AudioSource) -> CaptureDevice | None:
        devices = AudioInputManager.list_capture_devices(source)
        return devices[0] if devices else None

    @staticmethod
    def get_device(index: int) -> CaptureDevice:
        AudioInputManager.ensure_backend()
        devices = sd.query_devices()
        if index < 0 or index >= len(devices):
            raise ValueError(f"Invalid device index: {index}")
        device = devices[index]
        if int(device["max_input_channels"]) <= 0:
            raise ValueError(f"Device index {index} has no input channels.")
        return CaptureDevice(
            index=index,
            name=str(device["name"]),
            max_input_channels=int(device["max_input_channels"]),
            default_sample_rate=int(float(device.get("default_samplerate", 16000))),
        )

    @staticmethod
    def probe_device(
        device_index: int,
        sample_rate: int | None = None,
        channels: int = 1,
        duration_seconds: float = 1.0,
        block_size: int = 1024,
    ) -> dict[str, float]:
        AudioInputManager.ensure_backend()
        dev = AudioInputManager.get_device(device_index)
        sr = int(sample_rate or dev.default_sample_rate or 16000)
        ch = max(1, min(channels, dev.max_input_channels))
        total = int(sr * duration_seconds)
        if total <= 0:
            raise ValueError("duration_seconds must be positive.")
        chunks: list[np.ndarray] = []
        remaining = total
        with sd.InputStream(
            samplerate=sr,
            channels=ch,
            blocksize=block_size,
            device=device_index,
            dtype="float32",
        ) as stream:
            while remaining > 0:
                size = min(block_size, remaining)
                data, overflowed = stream.read(size)
                if overflowed:
                    # keep data but report overflow in metrics
                    pass
                mono = np.mean(data, axis=1) if data.ndim > 1 else data
                chunks.append(mono.astype(np.float32, copy=False))
                remaining -= size
        signal = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
        if signal.size == 0:
            return {"rms": 0.0, "peak": 0.0}
        rms = float(np.sqrt(np.mean(np.square(signal))) + 1e-12)
        peak = float(np.max(np.abs(signal)) + 1e-12)
        return {"rms": rms, "peak": peak}

    @staticmethod
    def _filter_microphone_devices(devices: list[CaptureDevice]) -> list[CaptureDevice]:
        filtered = []
        for device in devices:
            lowered = device.name.lower()
            if any(keyword in lowered for keyword in SYSTEM_DEVICE_KEYWORDS):
                continue
            filtered.append(device)
        candidates = filtered or devices
        return sorted(candidates, key=AudioInputManager._microphone_priority)

    @staticmethod
    def _filter_system_devices(devices: list[CaptureDevice]) -> list[CaptureDevice]:
        filtered = []
        for device in devices:
            lowered = device.name.lower()
            if any(keyword in lowered for keyword in SYSTEM_DEVICE_KEYWORDS):
                filtered.append(device)
        # Do not fall back to generic microphone devices. If empty, user likely
        # does not have a loopback input configured for system capture.
        return filtered

    @staticmethod
    def _microphone_priority(device: CaptureDevice) -> tuple[int, str]:
        name = device.name.lower()
        # Prefer wired/built-in microphones over Bluetooth headset defaults,
        # which often have extremely low gain profiles for speech capture.
        if any(token in name for token in ["airpods", "bluetooth", "hands-free"]):
            return (3, name)
        if "macbook" in name or "built-in" in name:
            return (0, name)
        if "usb" in name:
            return (1, name)
        return (2, name)
