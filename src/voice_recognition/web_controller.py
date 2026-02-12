from __future__ import annotations

import json
import os
import platform
import queue
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from voice_recognition.audio import AudioInputManager, AudioSource, LoopbackSetupAssistant
from voice_recognition.core.models import EventType, RecognitionEvent, RecognitionScope, Speaker
from voice_recognition.live_service import LiveConfig, LiveRecognitionService, LiveUpdate


@dataclass(slots=True)
class WebState:
    running: bool = False
    status: str = "Idle"
    current_speaker: str = "-"
    confidence: str = "-"
    active_speaker_id: int | None = None
    active_confidence: float | None = None
    scope: str = RecognitionScope.GLOBAL.value
    source: str = AudioSource.MICROPHONE.value
    device_index: int | None = None
    error: str | None = None
    setup_notice: str | None = None
    setup_running: bool = False
    tuning: dict[str, object] = field(default_factory=dict)
    speakers: list[dict[str, object]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


class WebController:
    def __init__(self, db_path: Path, sample_rate: int) -> None:
        self.db_path = db_path
        self.sample_rate = sample_rate
        self.service = LiveRecognitionService()
        self.loopback_setup = LoopbackSetupAssistant()
        self._lock = threading.Lock()
        self._state = WebState()
        self._max_events = 120
        self._setup_running = False
        self._setup_result: dict[str, object] | None = None

    def list_devices(self, source: AudioSource) -> list[dict[str, object]]:
        devices = AudioInputManager.list_capture_devices(source=source)
        return [
            {
                "index": device.index,
                "name": device.name,
                "channels": device.max_input_channels,
                "sampleRate": device.default_sample_rate,
            }
            for device in devices
        ]

    def diagnostics(self) -> dict[str, object]:
        from pathlib import Path

        mic_devices = self.list_devices(AudioSource.MICROPHONE)
        sys_devices = self.list_devices(AudioSource.SYSTEM)
        return {
            "microphoneCount": len(mic_devices),
            "systemCount": len(sys_devices),
            "microphoneDevices": mic_devices,
            "systemDevices": sys_devices,
            "blackholeDriverPresent": any(
                p.exists()
                for p in [
                    Path("/Library/Audio/Plug-Ins/HAL/BlackHole2ch.driver"),
                    Path("/Library/Audio/Plug-Ins/HAL/BlackHole16ch.driver"),
                ]
            ),
        }

    def probe_device(self, source: AudioSource, device_index: int) -> dict[str, object]:
        try:
            metrics = AudioInputManager.probe_device(
                device_index=device_index,
                sample_rate=None,
                channels=2 if source == AudioSource.SYSTEM else 1,
                duration_seconds=1.0,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        rms = float(metrics["rms"])
        peak = float(metrics["peak"])
        if source == AudioSource.SYSTEM:
            expected_rms = 0.0009
        else:
            expected_rms = 0.0015
        has_signal = rms >= expected_rms or peak >= (expected_rms * 4.0)
        return {
            "ok": True,
            "rms": rms,
            "peak": peak,
            "hasSignal": has_signal,
            "expectedRms": expected_rms,
        }

    def auto_select_device(self, source: AudioSource) -> dict[str, object]:
        devices = self.list_devices(source)
        if not devices:
            return {"ok": False, "error": "当前模式没有可用输入设备。"}
        if source == AudioSource.SYSTEM:
            chosen = devices[0]
            return {"ok": True, "device": chosen, "reason": "system 模式优先选择第一个回环设备。"}

        best: dict[str, object] | None = None
        best_score = -1.0
        probe_logs: list[dict[str, object]] = []
        for device in devices:
            result = self.probe_device(source=source, device_index=int(device["index"]))
            probe_logs.append({"device": device, "probe": result})
            if not bool(result.get("ok")):
                continue
            rms = float(result.get("rms", 0.0))
            peak = float(result.get("peak", 0.0))
            name = str(device.get("name", "")).lower()
            priority = 1.0
            if any(token in name for token in ["airpods", "bluetooth", "hands-free"]):
                priority = 0.35
            elif any(token in name for token in ["macbook", "built-in"]):
                priority = 1.1
            score = (rms * 0.75 + peak * 0.25) * priority
            if score > best_score:
                best_score = score
                best = device
        if best is None:
            return {"ok": False, "error": "设备探测失败，请手动选择设备。", "probes": probe_logs}
        return {"ok": True, "device": best, "probes": probe_logs}

    def request_system_reboot(self) -> dict[str, object]:
        system = platform.system().lower()
        if system == "darwin":
            cmd = ["osascript", "-e", 'tell app "System Events" to restart']
        elif system == "windows":
            cmd = ["shutdown", "/r", "/t", "5"]
        else:
            return {"ok": False, "error": f"当前系统不支持程序内重启: {system}"}
        try:
            subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=15)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "message": "已发送重启请求，请保存其他工作。"}

    def start(
        self,
        scope: RecognitionScope,
        source: AudioSource,
        device_index: int,
        tuning: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            if self.service.running:
                raise RuntimeError("Service is already running.")
            self._validate_device_for_source(source=source, device_index=device_index)
            tuning = tuning or {}
            file_tuning = self._load_tuning_file()
            mode = "recognition"
            embedding_models = tuning.get("embeddingModels")
            models: tuple[str, ...] = (
                "hf:microsoft/wavlm-base-plus-sv",
                "speechbrain/spkrec-ecapa-voxceleb",
            )
            if isinstance(embedding_models, str):
                parts = [item.strip() for item in embedding_models.split(",")]
                models = tuple([item for item in parts if item]) or models
            elif isinstance(embedding_models, (list, tuple)):
                models = tuple(str(item).strip() for item in embedding_models if str(item).strip()) or models
            config = LiveConfig(
                scope=RecognitionScope.GLOBAL,
                source=source,
                device_index=device_index,
                mode=mode,
                sample_rate=self.sample_rate,
                channels=2 if source == AudioSource.SYSTEM else 1,
                db_path=self.db_path,
                embedding_models=models,
                embedding_fusion=str(tuning.get("embeddingFusion", "average")),
                embedding_window_seconds=self._coerce_float(
                    tuning.get("embeddingWindowSeconds"),
                    default=2.4,
                ),
                match_threshold=self._coerce_float(
                    self._tuning_value(file_tuning, "matchThreshold", "match_threshold"),
                    default=0.80,
                ),
                new_speaker_threshold=self._coerce_float(
                    self._tuning_value(file_tuning, "newSpeakerThreshold", "new_speaker_threshold"),
                    default=0.62,
                ),
                min_match_margin=self._coerce_float(
                    self._tuning_value(file_tuning, "minMatchMargin", "minMargin", "min_match_margin"),
                    default=0.05,
                ),
                score_backend="asnorm",
                asnorm_top_k=self._coerce_int(
                    self._tuning_value(file_tuning, "asnormTopK", "asnorm_top_k"),
                    default=12,
                ),
                asnorm_min_cohort=self._coerce_int(
                    self._tuning_value(file_tuning, "asnormMinCohort", "asnorm_min_cohort"),
                    default=2,
                ),
                asnorm_blend=self._coerce_float(
                    self._tuning_value(file_tuning, "asnormBlend", "asnorm_blend"),
                    default=0.35,
                ),
                score_calibration_scale=self._coerce_float(
                    self._tuning_value(file_tuning, "scoreCalibrationScale", "calibrationScale", "calibration_scale"),
                    default=4.0,
                ),
                score_calibration_bias=self._coerce_float(
                    self._tuning_value(file_tuning, "scoreCalibrationBias", "calibrationBias", "calibration_bias"),
                    default=-2.0,
                ),
                enrollment_min_segments=self._coerce_int(
                    self._tuning_value(file_tuning, "enrollmentMinSegments", "minSegments", "min_segments"),
                    default=5,
                ),
                enrollment_min_cluster_similarity=self._coerce_float(
                    self._tuning_value(
                        file_tuning,
                        "enrollmentMinClusterSimilarity",
                        "minClusterSimilarity",
                        "min_cluster_similarity",
                    ),
                    default=0.82,
                ),
                enrollment_min_pairwise_similarity=self._coerce_float(
                    self._tuning_value(
                        file_tuning,
                        "enrollmentMinPairwiseSimilarity",
                        "minPairwiseSimilarity",
                        "min_pairwise_similarity",
                    ),
                    default=0.70,
                ),
                enrollment_max_pairwise_std=self._coerce_float(
                    self._tuning_value(
                        file_tuning,
                        "enrollmentMaxPairwiseStd",
                        "maxPairwiseStd",
                        "max_pairwise_std",
                    ),
                    default=0.12,
                ),
                enrollment_cooldown_segments=self._coerce_int(
                    self._tuning_value(
                        file_tuning,
                        "enrollmentCooldownSegments",
                        "cooldownSegments",
                        "cooldown_segments",
                    ),
                    default=10,
                ),
            )
            self.service.start(config)
            self._state.running = True
            self._state.status = "Starting"
            self._state.scope = RecognitionScope.GLOBAL.value
            self._state.source = source.value
            self._state.device_index = device_index
            self._state.error = None
            self._state.current_speaker = "-"
            self._state.confidence = "-"
            self._state.active_speaker_id = None
            self._state.active_confidence = None
            self._state.tuning = {
                "embeddingModels": list(config.embedding_models),
                "embeddingFusion": config.embedding_fusion,
                "embeddingWindowSeconds": config.embedding_window_seconds,
            }
            if file_tuning:
                self._append_event(f"tuning loaded: {(self.db_path.parent / 'tuning.json').name}")
            self._append_event("service starting")

    def _validate_device_for_source(self, source: AudioSource, device_index: int) -> None:
        devices = self.list_devices(source)
        if not devices:
            raise RuntimeError("当前模式没有可用输入设备。请先刷新设备并检查系统输入配置。")
        valid_indexes = {int(device["index"]) for device in devices}
        if int(device_index) not in valid_indexes:
            raise RuntimeError("所选输入设备已失效。请点击“刷新设备”后重新选择。")

    def stop(self) -> None:
        with self._lock:
            if not self.service.running and not self._state.running:
                return
            self.service.stop()
            self._state.running = False
            self._state.status = "Stopped"
            self._append_event("service stopped")

    def reset_library(self, scope: RecognitionScope) -> None:
        with self._lock:
            if self.service.running:
                raise RuntimeError("Stop service before resetting library.")
            if self.db_path.exists():
                os.remove(self.db_path)
            self._state.speakers = []
            self._state.current_speaker = "-"
            self._state.confidence = "-"
            self._state.active_speaker_id = None
            self._state.active_confidence = None
            self._state.error = None
            self._state.status = "Library Reset"
            self._append_event(f"library reset ({scope.value})")

    def snapshot(self) -> dict[str, object]:
        self._drain_updates()
        with self._lock:
            return {
                "running": self._state.running,
                "status": self._state.status,
                "currentSpeaker": self._state.current_speaker,
                "confidence": self._state.confidence,
                "activeSpeakerId": self._state.active_speaker_id,
                "scope": self._state.scope,
                "source": self._state.source,
                "deviceIndex": self._state.device_index,
                "error": self._state.error,
                "setupNotice": self._state.setup_notice,
                "setupRunning": self._state.setup_running,
                "tuning": dict(self._state.tuning),
                "speakers": list(self._state.speakers),
                "events": list(self._state.events),
            }

    def setup_loopback(self) -> dict[str, object]:
        result = self.loopback_setup.run()
        with self._lock:
            self._append_event(f"loopback setup: {result.message}")
            for log in result.logs:
                self._append_event(f"loopback log: {log}")
            for step in result.steps:
                self._append_event(f"loopback step: {step}")
            notice = result.message
            if result.steps:
                notice = f"{notice} 下一步：{'；'.join(result.steps)}"
            self._state.setup_notice = notice
            if result.ok:
                self._state.error = None
            else:
                self._state.error = result.message
        return result.to_dict()

    def start_setup_loopback(self) -> dict[str, object]:
        with self._lock:
            if self._setup_running:
                return {"started": False, "running": True, "message": "配置任务已在执行中。"}
            self._setup_running = True
            self._state.setup_running = True
            self._setup_result = None
            self._state.setup_notice = "回环设备配置进行中，请稍候..."
            self._append_event("loopback setup started")
        thread = threading.Thread(target=self._run_setup_loopback_task, daemon=True)
        thread.start()
        return {"started": True, "running": True, "message": "已启动回环配置任务。"}

    def setup_loopback_status(self) -> dict[str, object]:
        with self._lock:
            return {
                "running": self._setup_running,
                "result": self._setup_result,
            }

    def _run_setup_loopback_task(self) -> None:
        try:
            result = self.setup_loopback()
        except Exception as exc:
            result = {
                "ok": False,
                "configured": False,
                "platform": "unknown",
                "message": str(exc),
                "requiresReboot": False,
                "steps": [],
                "logs": [],
            }
            with self._lock:
                self._append_event(f"loopback setup error: {exc}")
                self._state.error = str(exc)
        finally:
            with self._lock:
                self._setup_running = False
                self._state.setup_running = False
                self._setup_result = result

    def shutdown(self) -> None:
        self.stop()

    def _drain_updates(self) -> None:
        while True:
            try:
                update = self.service.updates.get_nowait()
            except queue.Empty:
                break
            self._apply_update(update)
        with self._lock:
            self._state.running = self.service.running

    def _apply_update(self, update: LiveUpdate) -> None:
        with self._lock:
            event = update.event
            if update.error:
                self._state.error = update.error
                self._state.status = "Error"
                self._append_event(f"error: {update.error}")

            if event.event_type == EventType.MATCH:
                self._state.status = "Matched Speaker"
                self._state.current_speaker = event.speaker_name or "-"
                self._state.confidence = f"{event.confidence:.2f}%" if event.confidence is not None else "-"
                self._state.active_speaker_id = event.speaker_id
                self._state.active_confidence = event.confidence
            elif event.event_type == EventType.NEW_SPEAKER:
                self._state.status = "New Speaker Registered"
                self._state.current_speaker = event.speaker_name or "-"
                self._state.confidence = f"{event.confidence:.2f}%" if event.confidence is not None else "-"
                self._state.active_speaker_id = event.speaker_id
                self._state.active_confidence = event.confidence
            elif event.event_type == EventType.UNKNOWN_SPEECH:
                self._state.status = "Human Speech (Unknown)"
                self._state.current_speaker = "-"
                self._state.confidence = "-"
                self._state.active_speaker_id = None
                self._state.active_confidence = None
            elif event.event_type == EventType.NOISE:
                if event.details != "error":
                    self._state.status = "Background Noise"
                    self._state.current_speaker = "-"
                    self._state.confidence = "-"
                    self._state.active_speaker_id = None
                    self._state.active_confidence = None
            elif event.event_type == EventType.SILENCE:
                self._state.status = "Silence"
                if event.details != "service_started":
                    self._state.current_speaker = "-"
                    self._state.confidence = "-"
                self._state.active_speaker_id = None
                self._state.active_confidence = None

            self._state.speakers = self._speaker_rows(update.speakers)
            self._append_event(self._event_line(event))

    def _speaker_rows(self, speakers: list[Speaker]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for speaker in speakers:
            is_active = speaker.id == self._state.active_speaker_id
            confidence = (
                f"{self._state.active_confidence:.2f}%"
                if is_active and self._state.active_confidence is not None
                else "-"
            )
            rows.append(
                {
                    "id": speaker.id,
                    "name": speaker.name,
                    "samples": speaker.sample_count,
                    "confidence": confidence,
                    "active": is_active,
                }
            )
        return rows

    def _event_line(self, event: RecognitionEvent) -> str:
        score = f"{event.score:.4f}" if event.score is not None else "-"
        confidence = f"{event.confidence:.2f}%" if event.confidence is not None else "-"
        detail = f" [{event.details}]" if event.details else ""
        return (
            f"{event.event_type.value:<14} speaker={event.speaker_name or '-':<8} "
            f"score={score:<7} confidence={confidence:<8} source={event.source}{detail}"
        )

    def _append_event(self, line: str) -> None:
        self._state.events.append(line)
        if len(self._state.events) > self._max_events:
            self._state.events = self._state.events[-self._max_events :]

    @staticmethod
    def _coerce_float(value: object, default: float) -> float:
        if value is None:
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        if value is None:
            return int(default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _load_tuning_file(self) -> dict[str, object]:
        path = self.db_path.parent / "tuning.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        tuning = payload.get("tuning")
        if isinstance(tuning, dict):
            return dict(tuning)
        return dict(payload)

    @staticmethod
    def _tuning_value(mapping: dict[str, object], *keys: str) -> object | None:
        for key in keys:
            if key in mapping and mapping[key] is not None:
                return mapping[key]
        return None
