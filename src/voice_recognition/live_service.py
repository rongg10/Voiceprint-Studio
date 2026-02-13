from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voice_recognition.audio import AudioInputManager, AudioSource, SignalProcessor, SignalProcessorConfig
from voice_recognition.core.models import EventType, RecognitionEvent, RecognitionScope, Speaker
from voice_recognition.recognition import (
    AutoEnrollmentManager,
    DiarizationEngine,
    EnrollmentConfig,
    MatcherConfig,
    RecognitionEngine,
    SpeakerMatcher,
)
from voice_recognition.storage.base import SpeakerRepository
from voice_recognition.storage.factory import build_repository

try:
    import sounddevice as sd
except Exception:  # pragma: no cover - runtime environment dependent
    sd = None


@dataclass(slots=True)
class LiveConfig:
    scope: RecognitionScope
    source: AudioSource
    device_index: int
    mode: str = "recognition"  # recognition | diarization
    sample_rate: int = 16000  # target processing sample rate
    channels: int = 1
    block_size: int = 1024
    chunk_seconds: float = 1.0
    hop_seconds: float = 0.5
    db_path: Path = Path("data") / "speakers.db"
    backend: str = "neural"
    neural_device: str = "auto"
    embedding_models: tuple[str, ...] = (
        "hf:microsoft/wavlm-base-plus-sv",
        "speechbrain/spkrec-ecapa-voxceleb",
    )
    embedding_fusion: str = "average"
    embedding_window_seconds: float | None = 2.4
    match_threshold: float = 0.82
    new_speaker_threshold: float = 0.62
    min_match_margin: float = 0.05
    score_backend: str = "asnorm"
    asnorm_top_k: int = 12
    asnorm_min_cohort: int = 2
    asnorm_blend: float = 0.35
    score_calibration_scale: float = 4.0
    score_calibration_bias: float = -2.0
    confidence_low: float = 0.50
    confidence_high: float = 0.92
    soft_match_threshold: float = 0.68
    centroid_update_threshold: float = 0.88
    centroid_adapt_rate: float = 0.18
    prototype_novelty_threshold: float = 0.90
    max_prototypes_per_speaker: int = 8
    enrollment_min_segments: int = 5
    enrollment_min_cluster_similarity: float = 0.82
    enrollment_min_pairwise_similarity: float = 0.70
    enrollment_max_pairwise_std: float = 0.12
    enrollment_cooldown_segments: int = 10
    enrollment_reset_non_speech_frames: int = 12
    vad_threshold: float = 0.38
    vad_speech_ratio_threshold: float = 0.12
    vad_min_speech_ms: int = 130
    vad_min_samples_for_embedding: int = 9000
    speaker_change_similarity_threshold: float = 0.58
    instant_embedding_min_samples: int = 4800


@dataclass(slots=True)
class LiveUpdate:
    event: RecognitionEvent
    speakers: list[Speaker]
    error: str | None = None


class LiveRecognitionService:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._updates: queue.Queue[LiveUpdate] = queue.Queue()
        self._running = False

    @property
    def updates(self) -> queue.Queue[LiveUpdate]:
        return self._updates

    @property
    def running(self) -> bool:
        return self._running

    def start(self, config: LiveConfig) -> None:
        if self._running:
            raise RuntimeError("Service is already running.")
        if sd is None:
            raise RuntimeError(
                "sounddevice is missing. Run: pip install -r requirements.txt"
            )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(config,), daemon=True)
        self._thread.start()
        self._running = True

    def stop(self, timeout_seconds: float = 2.0) -> None:
        if not self._running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
        self._running = False
        self._thread = None

    def _run(self, config: LiveConfig) -> None:
        repository: SpeakerRepository | None = None
        try:
            repository = build_repository(scope=config.scope, db_path=config.db_path)
            capture_device = AudioInputManager.get_device(config.device_index)
            capture_sample_rate = int(capture_device.default_sample_rate or config.sample_rate)
            desired_channels = max(1, config.channels)
            if config.source == AudioSource.SYSTEM:
                desired_channels = max(2, desired_channels)
            capture_channels = max(1, min(desired_channels, capture_device.max_input_channels))
            processor, engine = self._build_runtime_components(config=config, repository=repository)
            self._emit_last_state(repository, scope=config.scope, source=config.source.value)
            self._capture_loop(
                config=config,
                processor=processor,
                engine=engine,
                repository=repository,
                capture_sample_rate=capture_sample_rate,
                capture_channels=capture_channels,
            )
        except Exception as exc:
            speakers: list[Speaker] = []
            if repository is not None:
                try:
                    speakers = repository.list_speakers()
                except Exception:
                    speakers = []
            self._emit_error(
                str(exc),
                scope=config.scope,
                source=config.source.value,
                speakers=speakers,
            )
        finally:
            if repository is not None:
                repository.close()
            self._running = False

    def _build_runtime_components(
        self,
        config: LiveConfig,
        repository: SpeakerRepository,
    ) -> tuple[SignalProcessor, RecognitionEngine | DiarizationEngine]:
        is_system = config.source == AudioSource.SYSTEM
        mode = str(config.mode or "recognition").strip().lower()
        is_diarization = mode == "diarization"
        has_hf_model = any(
            str(model).strip().lower().startswith(("hf:", "huggingface:", "transformers:", "wavlm:"))
            for model in config.embedding_models
        )
        match_threshold = self._clamp(config.match_threshold, 0.50, 0.995)
        min_threshold_gap = 0.08
        new_speaker_upper = match_threshold - min_threshold_gap
        if new_speaker_upper < 0.40:
            new_speaker_upper = match_threshold - 0.02
        new_speaker_threshold = self._clamp(config.new_speaker_threshold, 0.40, new_speaker_upper)
        min_match_margin = self._clamp(config.min_match_margin, 0.01, 0.20)
        confidence_low = self._clamp(config.confidence_low, 0.0, 0.98)
        confidence_high = self._clamp(config.confidence_high, confidence_low + 0.01, 1.0)
        soft_match_threshold = self._clamp(
            config.soft_match_threshold,
            0.40,
            match_threshold - 0.01,
        )
        centroid_update_threshold = self._clamp(
            max(config.centroid_update_threshold, match_threshold + 0.01),
            match_threshold + 0.01,
            0.995,
        )
        if is_diarization:
            match_threshold = self._clamp(match_threshold - 0.08, 0.55, 0.92)
            new_speaker_threshold = self._clamp(new_speaker_threshold - 0.08, 0.40, match_threshold - 0.02)
            min_match_margin = self._clamp(min_match_margin - 0.02, 0.01, 0.12)
            soft_match_threshold = self._clamp(soft_match_threshold - 0.06, 0.40, match_threshold - 0.01)
            centroid_update_threshold = self._clamp(
                max(match_threshold + 0.01, centroid_update_threshold - 0.06),
                match_threshold + 0.01,
                0.98,
            )
        safety_similarity_threshold = self._clamp(
            max(new_speaker_threshold + 0.02, match_threshold - 0.04),
            0.45,
            match_threshold - 0.01,
        )
        min_segments = max(3, int(config.enrollment_min_segments))
        min_cluster_similarity = self._clamp(config.enrollment_min_cluster_similarity, 0.60, 0.99)
        if is_diarization:
            min_segments = max(2, min_segments - 1)
            min_cluster_similarity = self._clamp(min_cluster_similarity - 0.06, 0.55, 0.92)
        cooldown_segments = max(0, int(config.enrollment_cooldown_segments))
        non_speech_reset_frames = max(
            10 if not is_system else 14,
            int(config.enrollment_reset_non_speech_frames),
        )
        bootstrap_until = 0 if not is_diarization else 1
        bootstrap_min_segments = max(2, min_segments - 1)
        bootstrap_min_cluster_similarity = self._clamp(min_cluster_similarity - 0.06, 0.55, min_cluster_similarity)
        base_pairwise_similarity = self._clamp(config.enrollment_min_pairwise_similarity, 0.40, 0.99)
        base_pairwise_std = self._clamp(config.enrollment_max_pairwise_std, 0.01, 0.40)
        if is_diarization:
            base_pairwise_similarity = self._clamp(base_pairwise_similarity - 0.06, 0.40, 0.90)
            base_pairwise_std = self._clamp(base_pairwise_std + 0.05, 0.01, 0.40)
        bootstrap_min_pairwise_similarity = self._clamp(
            base_pairwise_similarity - 0.08,
            0.40,
            base_pairwise_similarity,
        )
        bootstrap_max_pairwise_std = self._clamp(
            base_pairwise_std + 0.06,
            base_pairwise_std,
            0.40,
        )
        bootstrap_new_speaker_threshold = self._clamp(
            new_speaker_threshold - 0.10,
            0.40,
            new_speaker_threshold,
        )
        bootstrap_safety_similarity_threshold = self._clamp(
            max(bootstrap_new_speaker_threshold + 0.02, safety_similarity_threshold - 0.08),
            0.45,
            safety_similarity_threshold,
        )

        embedding_window_seconds = config.embedding_window_seconds
        if has_hf_model and embedding_window_seconds is not None and embedding_window_seconds < 3.0:
            embedding_window_seconds = 3.2 if is_system else 2.8

        processor = SignalProcessor(
            SignalProcessorConfig(
                sample_rate=config.sample_rate,
                backend=config.backend,
                model_cache_dir=str((config.db_path.parent / "models").resolve()),
                neural_device=str(config.neural_device),
                embedding_models=tuple(config.embedding_models),
                embedding_fusion=str(config.embedding_fusion),
                embedding_window_seconds=embedding_window_seconds,
                vad_threshold=self._clamp(
                    config.vad_threshold,
                    0.05,
                    0.95,
                ),
                vad_speech_ratio_threshold=self._clamp(
                    max(config.vad_speech_ratio_threshold, 0.14 if is_system else 0.12),
                    0.01,
                    0.95,
                ),
                vad_min_speech_ms=max(config.vad_min_speech_ms, 150 if is_system else 130),
                vad_min_samples_for_embedding=max(
                    config.vad_min_samples_for_embedding,
                    12000 if is_system else 9600,
                ),
                speaker_change_similarity_threshold=self._clamp(
                    config.speaker_change_similarity_threshold,
                    0.10,
                    0.95,
                ),
                instant_embedding_min_samples=max(
                    1600,
                    int(config.instant_embedding_min_samples),
                ),
                silence_rms_threshold=0.0009 if is_system else 0.0008,
                silence_peak_threshold=0.0028 if is_system else 0.003,
                speech_vote_threshold=5,
                min_speech_rms_ratio=1.25 if is_system else 1.1,
                startup_calibration_chunks=3 if is_system else 2,
                calibration_max_rms=0.008 if is_system else 0.006,
                calibration_max_peak=0.045 if is_system else 0.02,
                noise_floor_multiplier=1.35 if is_system else 1.2,
                min_pitch_strength=0.20 if is_system else 0.24,
                min_voiced_ratio=0.10 if is_system else 0.13,
            )
        )
        score_backend = "cosine" if is_diarization else config.score_backend
        matcher = SpeakerMatcher(
            MatcherConfig(
                match_threshold=match_threshold,
                confidence_low=confidence_low,
                confidence_high=confidence_high,
                min_margin=min_match_margin,
                score_backend=score_backend,
                asnorm_top_k=max(2, int(config.asnorm_top_k)),
                asnorm_min_cohort=max(2, int(config.asnorm_min_cohort)),
                asnorm_blend=self._clamp(config.asnorm_blend, 0.0, 1.0),
                calibration_scale=self._clamp(config.score_calibration_scale, 0.1, 12.0),
                calibration_bias=self._clamp(config.score_calibration_bias, -8.0, 8.0),
            )
        )
        enrollment = AutoEnrollmentManager(
            EnrollmentConfig(
                newcomer_prefix="说话人" if is_diarization else "新人",
                new_speaker_threshold=new_speaker_threshold,
                min_cluster_similarity=min_cluster_similarity,
                min_segments=min_segments,
                max_buffer_segments=max(24, min_segments * 4),
                safety_similarity_threshold=safety_similarity_threshold,
                cooldown_segments=cooldown_segments,
                min_pairwise_similarity=base_pairwise_similarity,
                max_pairwise_std=base_pairwise_std,
                bootstrap_until_speakers=bootstrap_until,
                bootstrap_min_segments=bootstrap_min_segments,
                bootstrap_min_cluster_similarity=bootstrap_min_cluster_similarity,
                bootstrap_min_pairwise_similarity=bootstrap_min_pairwise_similarity,
                bootstrap_max_pairwise_std=bootstrap_max_pairwise_std,
                bootstrap_new_speaker_threshold=bootstrap_new_speaker_threshold,
                bootstrap_safety_similarity_threshold=bootstrap_safety_similarity_threshold,
            )
        )
        if is_diarization:
            engine = DiarizationEngine(
                repository=repository,
                scope=config.scope,
                matcher=matcher,
                enrollment=enrollment,
                centroid_update_threshold=centroid_update_threshold,
                centroid_adapt_rate=self._clamp(config.centroid_adapt_rate, 0.04, 0.50),
                prototype_novelty_threshold=self._clamp(
                    config.prototype_novelty_threshold,
                    0.55,
                    0.95,
                ),
                max_prototypes_per_speaker=max(2, int(config.max_prototypes_per_speaker)),
                soft_match_threshold=soft_match_threshold,
                enrollment_reset_non_speech_frames=non_speech_reset_frames,
            )
        else:
            engine = RecognitionEngine(
                repository=repository,
                scope=config.scope,
                matcher=matcher,
                enrollment=enrollment,
                centroid_update_threshold=centroid_update_threshold,
                centroid_adapt_rate=self._clamp(config.centroid_adapt_rate, 0.03, 0.45),
                prototype_novelty_threshold=self._clamp(
                    config.prototype_novelty_threshold,
                    0.60,
                    0.99,
                ),
                max_prototypes_per_speaker=max(2, int(config.max_prototypes_per_speaker)),
                soft_match_threshold=soft_match_threshold,
                enrollment_reset_non_speech_frames=non_speech_reset_frames,
                min_seconds_between_new_speakers=8.0,
                recent_match_seconds=3.0,
                recent_match_threshold=max(0.58, soft_match_threshold - 0.10),
            )
        return processor, engine

    def _capture_loop(
        self,
        config: LiveConfig,
        processor: SignalProcessor,
        engine: RecognitionEngine | DiarizationEngine,
        repository: SpeakerRepository,
        capture_sample_rate: int,
        capture_channels: int,
    ) -> None:
        chunk_seconds = float(config.chunk_seconds)
        hop_seconds = float(config.hop_seconds)
        if config.source == AudioSource.SYSTEM and any(
            str(model).strip().lower().startswith(("hf:", "huggingface:", "transformers:", "wavlm:"))
            for model in config.embedding_models
        ):
            chunk_seconds = max(chunk_seconds, 1.4)
            hop_seconds = max(hop_seconds, 1.0)
        chunk_samples = int(capture_sample_rate * chunk_seconds)
        hop_samples = int(capture_sample_rate * hop_seconds)
        if chunk_samples <= 0 or hop_samples <= 0:
            raise ValueError("Chunk and hop duration must be positive.")

        buffer = np.zeros((0,), dtype=np.float32)
        chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=6)
        last_overflow_warn = 0.0

        def consumer() -> None:
            while not self._stop_event.is_set():
                try:
                    item = chunk_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if item is None:
                    break
                chunk = item
                if capture_sample_rate != config.sample_rate:
                    chunk = self._resample_audio(
                        chunk=chunk,
                        src_rate=capture_sample_rate,
                        dst_rate=config.sample_rate,
                    )
                frame = processor.infer(chunk=chunk, source=config.source.value)
                event = engine.process(frame)
                speakers = repository.list_speakers()
                self._updates.put(LiveUpdate(event=event, speakers=speakers))
            return

        worker = threading.Thread(target=consumer, daemon=True)
        worker.start()

        block_size = int(config.block_size)
        if config.source == AudioSource.SYSTEM:
            block_size = max(block_size, 2048)
        stream = sd.InputStream(
            samplerate=capture_sample_rate,
            channels=capture_channels,
            blocksize=block_size,
            device=config.device_index,
            dtype="float32",
            latency="high" if config.source == AudioSource.SYSTEM else "low",
        )
        with stream:
            while not self._stop_event.is_set():
                audio_block, overflowed = stream.read(block_size)
                if overflowed:
                    now = time.monotonic()
                    if now - last_overflow_warn > 2.5:
                        self._emit_error(
                            "Audio buffer overflow detected; consider lowering sample rate/block size.",
                            scope=config.scope,
                            source=config.source.value,
                            speakers=repository.list_speakers(),
                        )
                        last_overflow_warn = now
                    buffer = np.zeros((0,), dtype=np.float32)
                    continue
                mono = self._mix_to_mono(audio_block)
                buffer = np.concatenate([buffer, mono.astype(np.float32, copy=False)])

                while len(buffer) >= chunk_samples:
                    chunk = buffer[:chunk_samples]
                    buffer = buffer[hop_samples:]
                    if chunk_queue.full():
                        try:
                            chunk_queue.get_nowait()
                        except queue.Empty:
                            pass
                    try:
                        chunk_queue.put_nowait(chunk)
                    except queue.Full:
                        pass
        try:
            chunk_queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=1.0)

    @staticmethod
    def _resample_audio(chunk: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("Sample rates must be positive.")
        if src_rate == dst_rate:
            return chunk.astype(np.float32, copy=False)
        src_len = len(chunk)
        if src_len <= 1:
            return chunk.astype(np.float32, copy=False)
        dst_len = int(round(src_len * (float(dst_rate) / float(src_rate))))
        if dst_len <= 1:
            dst_len = 2
        src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        out = np.interp(dst_x, src_x, chunk).astype(np.float32)
        return out

    @staticmethod
    def _mix_to_mono(audio_block: np.ndarray) -> np.ndarray:
        if audio_block.ndim <= 1:
            return audio_block.astype(np.float32, copy=False)
        return np.mean(audio_block, axis=1).astype(np.float32, copy=False)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        if lower > upper:
            lower, upper = upper, lower
        return max(lower, min(upper, float(value)))

    def _emit_last_state(self, repository, scope: RecognitionScope, source: str) -> None:
        speakers = repository.list_speakers()
        event = RecognitionEvent(
            event_type=EventType.SILENCE,
            scope=scope,
            source=source,
            details="service_started",
        )
        self._updates.put(LiveUpdate(event=event, speakers=speakers))

    def _emit_error(
        self,
        message: str,
        scope: RecognitionScope,
        source: str,
        speakers: list[Speaker] | None = None,
    ) -> None:
        event = RecognitionEvent(
            event_type=EventType.NOISE,
            scope=scope,
            source=source,
            details="error",
        )
        self._updates.put(LiveUpdate(event=event, speakers=speakers or [], error=message))
