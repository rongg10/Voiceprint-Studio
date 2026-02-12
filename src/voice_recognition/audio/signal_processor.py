from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

from voice_recognition.core.models import FrameInference, SignalType

# SpeechBrain 1.0.3 emits a deprecated CUDA AMP warning even on CPU inference paths.
# Suppress this noisy warning to keep runtime logs actionable.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.custom_fwd.*deprecated.*",
    category=FutureWarning,
)


@dataclass(slots=True)
class SignalProcessorConfig:
    sample_rate: int = 16000
    backend: str = "dsp"  # dsp | neural | auto
    model_cache_dir: str = "data/models"
    neural_device: str = "auto"
    embedding_models: tuple[str, ...] = (
        "hf:microsoft/wavlm-base-plus-sv",
        "speechbrain/spkrec-ecapa-voxceleb",
    )
    embedding_fusion: str = "average"  # average | concat
    embedding_window_seconds: float | None = None
    vad_threshold: float = 0.42
    vad_speech_ratio_threshold: float = 0.18
    vad_min_speech_ms: int = 120
    vad_min_samples_for_embedding: int = 2400
    speaker_change_similarity_threshold: float = 0.58
    instant_embedding_min_samples: int = 4800
    silence_rms_threshold: float = 0.008
    silence_peak_threshold: float = 0.015
    speech_vote_threshold: int = 5
    min_speech_rms_ratio: float = 1.6
    min_pitch_strength: float = 0.22
    min_voiced_ratio: float = 0.10
    noise_floor_alpha: float = 0.92
    noise_floor_multiplier: float = 1.6
    startup_calibration_chunks: int = 2
    calibration_max_rms: float = 0.02
    calibration_max_peak: float = 0.08
    noise_floor_update_max_ratio: float = 2.5
    n_fft: int = 512
    n_mels: int = 24
    n_mfcc: int = 13
    pre_emphasis: float = 0.97
    frame_ms: float = 25.0
    hop_ms: float = 10.0


class SignalProcessor:
    """
    Lightweight speech/noise decision + MFCC-based embedding extractor.
    This is a demo-grade implementation optimized for no external ML model dependencies.
    """

    def __init__(self, config: SignalProcessorConfig | None = None) -> None:
        self.config = config or SignalProcessorConfig()
        self._noise_floor = self.config.silence_rms_threshold
        self._warmup_left = self.config.startup_calibration_chunks
        self._neural_backend: _NeuralSpeechBackend | None = None
        backend = self.config.backend.strip().lower()
        if backend not in {"dsp", "neural", "auto"}:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
        if backend in {"neural", "auto"}:
            try:
                self._neural_backend = _NeuralSpeechBackend(
                    sample_rate=self.config.sample_rate,
                    cache_dir=Path(self.config.model_cache_dir),
                    device=self.config.neural_device,
                    embedding_models=self.config.embedding_models,
                    embedding_fusion=self.config.embedding_fusion,
                    embedding_window_seconds=self.config.embedding_window_seconds,
                    vad_threshold=self.config.vad_threshold,
                    vad_min_speech_ms=self.config.vad_min_speech_ms,
                    vad_min_samples_for_embedding=self.config.vad_min_samples_for_embedding,
                    speaker_change_similarity_threshold=self.config.speaker_change_similarity_threshold,
                    instant_embedding_min_samples=self.config.instant_embedding_min_samples,
                )
            except Exception:
                if backend == "neural":
                    raise

    def infer(self, chunk: np.ndarray, source: str) -> FrameInference:
        if self._neural_backend is not None:
            return self._infer_neural(chunk=chunk, source=source)
        return self._infer_dsp(chunk=chunk, source=source)

    def _infer_neural(self, chunk: np.ndarray, source: str) -> FrameInference:
        signal_raw = self._to_mono(chunk)
        raw_rms = float(np.sqrt(np.mean(np.square(signal_raw))) + 1e-12)
        raw_peak = float(np.max(np.abs(signal_raw)) + 1e-12)
        silence_threshold = self._silence_threshold(source=source)
        if raw_peak < self.config.silence_peak_threshold or raw_rms < silence_threshold:
            self._update_noise_floor(raw_rms)
            self._neural_backend.mark_non_speech()
            return FrameInference(signal=SignalType.SILENCE, source=source)

        assert self._neural_backend is not None
        speech_ratio, speech_audio = self._neural_backend.extract_speech(signal_raw)
        if speech_ratio < self.config.vad_speech_ratio_threshold or speech_audio.size == 0:
            self._update_noise_floor(raw_rms)
            self._neural_backend.mark_non_speech()
            return FrameInference(signal=SignalType.NOISE, source=source)
        embedding = self._neural_backend.embedding(speech_audio)
        if embedding is None:
            self._update_noise_floor(raw_rms)
            return FrameInference(signal=SignalType.NOISE, source=source)
        return FrameInference(signal=SignalType.SPEECH, embedding=embedding, source=source)

    def _infer_dsp(self, chunk: np.ndarray, source: str) -> FrameInference:
        signal_raw = self._to_mono(chunk)
        raw_rms = float(np.sqrt(np.mean(np.square(signal_raw))) + 1e-12)
        raw_peak = float(np.max(np.abs(signal_raw)) + 1e-12)

        if self._warmup_left > 0:
            if (
                raw_rms <= self.config.calibration_max_rms
                and raw_peak <= self.config.calibration_max_peak
            ):
                self._update_noise_floor(raw_rms)
                self._warmup_left -= 1
                return FrameInference(signal=SignalType.SILENCE, source=source)
            # If input is already active speech/audio, skip calibration to avoid
            # poisoning the noise floor.
            self._warmup_left = 0

        silence_threshold = self._silence_threshold(source=source)

        if raw_peak < self.config.silence_peak_threshold or raw_rms < silence_threshold:
            self._update_noise_floor(raw_rms)
            return FrameInference(signal=SignalType.SILENCE, source=source)

        signal = self._normalize(signal_raw)
        features = self._acoustic_features(signal)
        pitch = self._pitch_features(signal)
        speech_votes = self._speech_votes(
            feats=features,
            pitch=pitch,
            raw_rms=raw_rms,
            silence_threshold=silence_threshold,
        )
        if speech_votes < self.config.speech_vote_threshold:
            self._update_noise_floor(raw_rms)
            return FrameInference(signal=SignalType.NOISE, source=source)

        embedding = self._embedding(
            signal=signal,
            feats=features,
            pitch=pitch,
            raw_rms=raw_rms,
            raw_peak=raw_peak,
            silence_threshold=silence_threshold,
        )
        return FrameInference(signal=SignalType.SPEECH, embedding=embedding, source=source)

    def _to_mono(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.ndim > 1:
            chunk = np.mean(chunk, axis=1)
        return chunk.astype(np.float32, copy=False)

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(signal)) + 1e-12)
        if peak == 0.0:
            return signal
        return signal / peak

    def _acoustic_features(self, signal: np.ndarray) -> dict[str, float]:
        spectrum = np.abs(np.fft.rfft(signal, n=self.config.n_fft))
        power = np.square(spectrum) + 1e-12
        frequencies = np.fft.rfftfreq(self.config.n_fft, d=1.0 / self.config.sample_rate)

        centroid = float(np.sum(frequencies * spectrum) / np.sum(spectrum))
        spread = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * spectrum) / np.sum(spectrum)))
        flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(signal))).astype(np.float32)))

        cumulative = np.cumsum(power)
        rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
        rolloff = float(frequencies[min(rolloff_idx, len(frequencies) - 1)])
        total_power = float(np.sum(power) + 1e-12)
        speech_band = float(
            np.sum(power[(frequencies >= 200.0) & (frequencies <= 3800.0)]) / total_power
        )
        high_band = float(np.sum(power[frequencies >= 4000.0]) / total_power)

        frame_len = int(self.config.sample_rate * 0.020)
        frame_step = int(self.config.sample_rate * 0.010)
        env_frames = self._frame_signal(signal, frame_len=max(1, frame_len), frame_step=max(1, frame_step))
        envelope = np.sqrt(np.mean(np.square(env_frames), axis=1) + 1e-12)
        modulation_depth = float(np.std(envelope) / (np.mean(envelope) + 1e-6))
        modulation_diff = float(np.mean(np.abs(np.diff(envelope))) if len(envelope) > 1 else 0.0)

        return {
            "centroid": centroid,
            "spread": spread,
            "flatness": flatness,
            "zcr": zcr,
            "rolloff": rolloff,
            "speech_band_ratio": speech_band,
            "high_band_ratio": high_band,
            "mod_depth": modulation_depth,
            "mod_diff": modulation_diff,
        }

    def _speech_votes(
        self,
        feats: dict[str, float],
        pitch: np.ndarray,
        raw_rms: float,
        silence_threshold: float,
    ) -> int:
        votes = 0
        if 120.0 <= feats["centroid"] <= 3600.0:
            votes += 1
        if 250.0 <= feats["spread"] <= 4200.0:
            votes += 1
        if 0.02 <= feats["zcr"] <= 0.28:
            votes += 1
        if feats["flatness"] <= 0.65:
            votes += 1
        if 400.0 <= feats["rolloff"] <= 5200.0:
            votes += 1
        if 0.35 <= feats["speech_band_ratio"] <= 0.96:
            votes += 1
        if feats["high_band_ratio"] <= 0.40:
            votes += 1
        if feats["mod_depth"] >= 0.08:
            votes += 1
        if feats["mod_diff"] >= 0.008:
            votes += 1
        pitch_strength = float(pitch[2]) if len(pitch) >= 3 else 0.0
        voiced_ratio = float(pitch[3]) if len(pitch) >= 4 else 0.0
        pitch_std = float(pitch[1]) if len(pitch) >= 2 else 0.0
        if pitch_strength >= self.config.min_pitch_strength:
            votes += 1
        if voiced_ratio >= self.config.min_voiced_ratio:
            votes += 1
        rms_ratio = raw_rms / max(silence_threshold, 1e-6)
        if rms_ratio >= self.config.min_speech_rms_ratio:
            votes += 1
        if (
            feats["mod_depth"] < 0.03
            and pitch_strength > 0.45
            and voiced_ratio > 0.75
            and pitch_std < 0.03
        ):
            # Stable harmonic tones are usually music/background, not conversational speech.
            votes -= 3
        if (
            feats["mod_depth"] < 0.02
            and feats["mod_diff"] < 0.006
            and pitch_strength > 0.50
            and voiced_ratio > 0.85
        ):
            votes -= 4
        if feats["speech_band_ratio"] > 0.97 and pitch_std < 0.02:
            votes -= 2
        if feats["flatness"] > 0.82:
            votes -= 2
        if feats["spread"] < 180.0:
            votes -= 1
        if feats["speech_band_ratio"] < 0.22 or feats["high_band_ratio"] > 0.62:
            votes -= 2
        return votes

    def _embedding(
        self,
        signal: np.ndarray,
        feats: dict[str, float],
        pitch: np.ndarray,
        raw_rms: float,
        raw_peak: float,
        silence_threshold: float,
    ) -> tuple[float, ...]:
        mfcc = self._mfcc(signal)
        mfcc_mean = np.mean(mfcc, axis=0)
        mfcc_std = np.std(mfcc, axis=0)
        mfcc_skew = np.mean(
            np.power((mfcc - mfcc_mean) / (mfcc_std + 1e-6), 3.0),
            axis=0,
        )

        log_mel = self._log_mel(signal)
        mel_mean = np.mean(log_mel, axis=0)
        mel_std = np.std(log_mel, axis=0)
        rms_ratio = min(raw_rms / max(silence_threshold, 1e-6), 8.0) / 8.0

        summary = np.array(
            [
                float(raw_rms * 20.0),
                float(raw_peak),
                float(rms_ratio),
                feats["centroid"] / 8000.0,
                feats["spread"] / 8000.0,
                feats["flatness"],
                feats["zcr"],
                feats["rolloff"] / 8000.0,
            ],
            dtype=np.float32,
        )
        vector = np.concatenate(
            [
                mfcc_mean,
                mfcc_std,
                mfcc_skew,
                mel_mean,
                mel_std,
                summary,
                pitch,
            ]
        ).astype(np.float32)
        vector = (vector - np.mean(vector)) / (np.std(vector) + 1e-6)
        norm = float(np.linalg.norm(vector) + 1e-12)
        return tuple(float(value) for value in (vector / norm))

    def _silence_threshold(self, source: str) -> float:
        threshold = max(
            self.config.silence_rms_threshold,
            self._noise_floor * self.config.noise_floor_multiplier,
        )
        if source == "system":
            threshold = max(threshold, self.config.silence_rms_threshold * 1.15)
        return threshold

    def _update_noise_floor(self, raw_rms: float) -> None:
        if raw_rms > self._noise_floor * self.config.noise_floor_update_max_ratio:
            return
        alpha = self.config.noise_floor_alpha
        self._noise_floor = alpha * self._noise_floor + (1.0 - alpha) * raw_rms

    def _mfcc(self, signal: np.ndarray) -> np.ndarray:
        log_mel = self._log_mel(signal)
        mfcc = np.dot(log_mel, self._dct_matrix.T)
        return mfcc[:, : self.config.n_mfcc]

    def _log_mel(self, signal: np.ndarray) -> np.ndarray:
        emphasized = np.append(signal[0], signal[1:] - self.config.pre_emphasis * signal[:-1])
        frame_len = int(self.config.sample_rate * (self.config.frame_ms / 1000.0))
        frame_step = int(self.config.sample_rate * (self.config.hop_ms / 1000.0))
        if frame_len <= 0 or frame_step <= 0:
            raise ValueError("Invalid frame or hop configuration.")

        frames = self._frame_signal(emphasized, frame_len=frame_len, frame_step=frame_step)
        frames *= np.hamming(frame_len)

        magnitude = np.absolute(np.fft.rfft(frames, self.config.n_fft))
        power = (1.0 / self.config.n_fft) * np.square(magnitude)

        mel_energies = np.dot(power, self._mel_filterbank.T)
        mel_energies = np.where(mel_energies <= 1e-12, 1e-12, mel_energies)
        return np.log(mel_energies)

    def _pitch_features(self, signal: np.ndarray) -> np.ndarray:
        frame_len = int(self.config.sample_rate * 0.030)
        frame_step = int(self.config.sample_rate * 0.010)
        frames = self._frame_signal(signal, frame_len=frame_len, frame_step=frame_step)
        energies = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
        voiced_mask = energies > max(self.config.silence_rms_threshold * 2.0, 0.015)
        voiced = frames[voiced_mask]
        if len(voiced) == 0:
            return np.zeros((4,), dtype=np.float32)

        min_lag = int(self.config.sample_rate / 420.0)
        max_lag = int(self.config.sample_rate / 70.0)
        pitches: list[float] = []
        strengths: list[float] = []
        for frame in voiced[:160]:
            frame = frame - np.mean(frame)
            auto = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
            if max_lag >= len(auto):
                continue
            region = auto[min_lag:max_lag]
            if region.size == 0:
                continue
            lag = int(np.argmax(region)) + min_lag
            peak = float(auto[lag])
            base = float(auto[0]) + 1e-12
            strength = peak / base
            if strength < 0.20:
                continue
            pitches.append(float(self.config.sample_rate / lag))
            strengths.append(strength)

        if not pitches:
            return np.zeros((4,), dtype=np.float32)

        pitch_arr = np.array(pitches, dtype=np.float32)
        strength_arr = np.array(strengths, dtype=np.float32)
        return np.array(
            [
                float(np.median(pitch_arr) / 320.0),
                float(np.std(pitch_arr) / 120.0),
                float(np.mean(strength_arr)),
                float(len(pitch_arr) / max(len(voiced), 1)),
            ],
            dtype=np.float32,
        )

    def _frame_signal(self, signal: np.ndarray, frame_len: int, frame_step: int) -> np.ndarray:
        if len(signal) < frame_len:
            pad_width = frame_len - len(signal)
            signal = np.pad(signal, (0, pad_width), mode="constant")

        num_frames = 1 + int((len(signal) - frame_len) / frame_step)
        expected_len = frame_len + (num_frames - 1) * frame_step
        if len(signal) < expected_len:
            signal = np.pad(signal, (0, expected_len - len(signal)), mode="constant")

        indices = (
            np.tile(np.arange(frame_len), (num_frames, 1))
            + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        )
        return signal[indices.astype(np.int32, copy=False)]

    @cached_property
    def _mel_filterbank(self) -> np.ndarray:
        low_mel = self._hz_to_mel(0.0)
        high_mel = self._hz_to_mel(self.config.sample_rate / 2.0)
        mel_points = np.linspace(low_mel, high_mel, self.config.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bins = np.floor((self.config.n_fft + 1) * hz_points / self.config.sample_rate).astype(int)
        filterbank = np.zeros((self.config.n_mels, int(self.config.n_fft / 2 + 1)), dtype=np.float32)
        for m in range(1, self.config.n_mels + 1):
            left = bins[m - 1]
            center = bins[m]
            right = bins[m + 1]
            if center <= left:
                center = left + 1
            if right <= center:
                right = center + 1
            for k in range(left, center):
                filterbank[m - 1, k] = (k - left) / float(center - left)
            for k in range(center, right):
                filterbank[m - 1, k] = (right - k) / float(right - center)
        return filterbank

    @cached_property
    def _dct_matrix(self) -> np.ndarray:
        n = self.config.n_mels
        basis = np.empty((n, n), dtype=np.float32)
        factor = np.pi / float(n)
        scale0 = np.sqrt(1.0 / n)
        scale = np.sqrt(2.0 / n)
        for k in range(n):
            alpha = scale0 if k == 0 else scale
            for i in range(n):
                basis[k, i] = alpha * np.cos((i + 0.5) * k * factor)
        return basis

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)


class _EmbeddingEncoder:
    def encode(self, wave: np.ndarray) -> np.ndarray | None:  # pragma: no cover - interface
        raise NotImplementedError


class _SpeechBrainEmbeddingEncoder(_EmbeddingEncoder):
    def __init__(self, source: str, savedir: Path, device: str) -> None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        run_device = self._resolve_device(device=device, torch_module=torch)
        self._device = run_device
        self._torch = torch
        self._encoder = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(savedir),
            run_opts={"device": run_device},
        )

    @staticmethod
    def _resolve_device(device: str, torch_module) -> str:
        resolved = str(device).strip().lower()
        if resolved == "auto":
            if torch_module.cuda.is_available():
                return "cuda"
            if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
                return "mps"
            return "cpu"
        if resolved == "cuda" and not torch_module.cuda.is_available():
            return "cpu"
        if resolved == "mps" and (
            not getattr(torch_module.backends, "mps", None)
            or not torch_module.backends.mps.is_available()
        ):
            return "cpu"
        return resolved

    def encode(self, wave: np.ndarray) -> np.ndarray | None:
        tensor = self._torch.from_numpy(wave.astype(np.float32, copy=False)).unsqueeze(0)
        try:
            with self._torch.no_grad():
                emb = self._encoder.encode_batch(tensor)
        except Exception:
            if self._device in {"mps", "cuda"}:
                self._device = "cpu"
                self._encoder = self._encoder.to("cpu")
                with self._torch.no_grad():
                    emb = self._encoder.encode_batch(tensor.cpu())
            else:
                raise
        arr = emb.squeeze().detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        norm = float(np.linalg.norm(arr) + 1e-12)
        if norm <= 0.0:
            return None
        return arr / norm

    def encode(self, wave: np.ndarray) -> np.ndarray | None:
        tensor = self._torch.from_numpy(wave.astype(np.float32, copy=False)).unsqueeze(0)
        with self._torch.no_grad():
            emb = self._encoder.encode_batch(tensor)
        arr = emb.squeeze().detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        norm = float(np.linalg.norm(arr) + 1e-12)
        if norm <= 0.0:
            return None
        return arr / norm


class _HFEmbeddingEncoder(_EmbeddingEncoder):
    def __init__(self, model_id: str, device: str, sample_rate: int, min_wave_seconds: float = 1.0) -> None:
        import torch
        from transformers import AutoFeatureExtractor, logging as hf_logging

        hf_logging.set_verbosity_error()

        try:
            from transformers import AutoModelForAudioXVector  # type: ignore
            model = AutoModelForAudioXVector.from_pretrained(model_id)
            self._mode = "xvector"
        except Exception:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(model_id)
            self._mode = "auto"

        self._torch = torch
        self._model = model
        self._model.eval()
        run_device = self._resolve_device(device=device, torch_module=torch)
        self._device = run_device
        self._model.to(self._device)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self._model_sample_rate = int(getattr(self._feature_extractor, "sampling_rate", sample_rate))
        self._input_sample_rate = int(sample_rate)
        self._min_samples = max(1, int(max(0.5, float(min_wave_seconds)) * self._model_sample_rate))

    def encode(self, wave: np.ndarray) -> np.ndarray | None:
        if wave.size == 0:
            return None
        if self._model_sample_rate != self._input_sample_rate:
            wave = self._resample_audio(wave, self._input_sample_rate, self._model_sample_rate)
        if wave.size < self._min_samples:
            pad_width = self._min_samples - wave.size
            wave = np.pad(wave, (0, pad_width), mode="constant")
        inputs = self._feature_extractor(
            wave.astype(np.float32, copy=False),
            sampling_rate=self._model_sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        try:
            with self._torch.no_grad():
                out = self._model(**inputs)
        except Exception:
            if self._device in {"mps", "cuda"}:
                self._device = "cpu"
                self._model = self._model.to("cpu")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with self._torch.no_grad():
                    out = self._model(**inputs)
            else:
                raise
        vec = None
        if hasattr(out, "embeddings"):
            vec = out.embeddings
        elif hasattr(out, "xvector"):
            vec = out.xvector
        elif hasattr(out, "last_hidden_state"):
            vec = out.last_hidden_state.mean(dim=1)
        if vec is None:
            return None
        arr = vec.squeeze().detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        norm = float(np.linalg.norm(arr) + 1e-12)
        if norm <= 0.0:
            return None
        return arr / norm

    @staticmethod
    def _resample_audio(wave: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate <= 0 or dst_rate <= 0:
            return wave.astype(np.float32, copy=False)
        if src_rate == dst_rate:
            return wave.astype(np.float32, copy=False)
        src_len = len(wave)
        if src_len <= 1:
            return wave.astype(np.float32, copy=False)
        dst_len = int(round(src_len * (float(dst_rate) / float(src_rate))))
        if dst_len <= 1:
            dst_len = 2
        src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        return np.interp(dst_x, src_x, wave).astype(np.float32)

    @staticmethod
    def _resolve_device(device: str, torch_module) -> str:
        resolved = str(device).strip().lower()
        if resolved == "auto":
            if torch_module.cuda.is_available():
                return "cuda"
            if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
                return "mps"
            return "cpu"
        if resolved == "cuda" and not torch_module.cuda.is_available():
            return "cpu"
        if resolved == "mps" and (
            not getattr(torch_module.backends, "mps", None)
            or not torch_module.backends.mps.is_available()
        ):
            return "cpu"
        return resolved


class _NeuralSpeechBackend:
    """Silero-VAD + neural speaker embedding backend (SpeechBrain / HF)."""

    def __init__(
        self,
        sample_rate: int,
        cache_dir: Path,
        device: str,
        embedding_models: tuple[str, ...],
        embedding_fusion: str,
        embedding_window_seconds: float | None,
        vad_threshold: float,
        vad_min_speech_ms: int,
        vad_min_samples_for_embedding: int,
        speaker_change_similarity_threshold: float,
        instant_embedding_min_samples: int,
    ) -> None:
        if sample_rate != 16000:
            raise ValueError("Neural backend currently expects 16k sample rate.")
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.vad_min_speech_ms = max(0, int(vad_min_speech_ms))
        self.vad_min_samples_for_embedding = max(400, int(vad_min_samples_for_embedding))
        self.speaker_change_similarity_threshold = max(
            0.05,
            min(0.98, float(speaker_change_similarity_threshold)),
        )
        self.instant_embedding_min_samples = max(400, int(instant_embedding_min_samples))
        if embedding_window_seconds is not None and embedding_window_seconds > 0:
            window_samples = int(float(embedding_window_seconds) * self.sample_rate)
        else:
            window_samples = int(0.80 * self.sample_rate)
        self._embedding_window_samples = max(self.vad_min_samples_for_embedding, window_samples)
        self._max_speech_cache_samples = max(self._embedding_window_samples * 4, int(8.0 * self.sample_rate))
        self._speech_cache = np.zeros((0,), dtype=np.float32)
        self._non_speech_frames = 0
        self._anchor_embedding: np.ndarray | None = None
        self._torch = torch
        self._get_speech_timestamps = get_speech_timestamps
        self._vad_model = load_silero_vad()
        self._embedding_fusion = (embedding_fusion or "average").strip().lower()
        if self._embedding_fusion not in {"average", "concat"}:
            self._embedding_fusion = "average"
        self._encoders: list[_EmbeddingEncoder] = []
        resolved_device = self._resolve_device(device=device, torch_module=torch)
        has_hf = any(str(spec).strip().lower().startswith(("hf:", "huggingface:", "transformers:", "wavlm:")) for spec in embedding_models)
        has_sb = any(
            not str(spec).strip().lower().startswith(("hf:", "huggingface:", "transformers:", "wavlm:"))
            for spec in embedding_models
        )
        sb_device = resolved_device
        hf_device = resolved_device
        if resolved_device == "mps" and has_hf and has_sb:
            # Keep WavLM on MPS, run ECAPA on CPU to reduce MPS pressure and improve stability.
            sb_device = "cpu"
        for spec in embedding_models:
            backend, model_id = self._parse_embedding_spec(str(spec))
            if backend == "speechbrain":
                safe_name = (
                    model_id.split("/", 1)[1] if model_id.startswith("speechbrain/") else model_id.replace("/", "_")
                )
                try:
                    self._encoders.append(
                        _SpeechBrainEmbeddingEncoder(
                            source=model_id,
                            savedir=cache_dir / safe_name,
                            device=sb_device,
                        )
                    )
                except Exception:
                    if sb_device != "cpu":
                        self._encoders.append(
                            _SpeechBrainEmbeddingEncoder(
                                source=model_id,
                                savedir=cache_dir / safe_name,
                                device="cpu",
                            )
                        )
                    else:
                        raise
            elif backend == "hf":
                try:
                    self._encoders.append(
                        _HFEmbeddingEncoder(model_id=model_id, device=hf_device, sample_rate=sample_rate)
                    )
                except Exception:
                    if hf_device != "cpu":
                        self._encoders.append(
                            _HFEmbeddingEncoder(model_id=model_id, device="cpu", sample_rate=sample_rate)
                        )
                    else:
                        raise
            else:
                raise ValueError(f"Unsupported embedding backend: {backend}")
        if not self._encoders:
            raise ValueError("No valid embedding models provided.")
        self._enable_instant_embedding = not any(
            isinstance(encoder, _HFEmbeddingEncoder) for encoder in self._encoders
        )

    def extract_speech(self, signal: np.ndarray) -> tuple[float, np.ndarray]:
        wave = np.clip(signal.astype(np.float32, copy=False), -1.0, 1.0)
        if wave.size == 0:
            return 0.0, np.zeros((0,), dtype=np.float32)
        tensor = self._torch.from_numpy(wave)
        timestamps = self._get_speech_timestamps(
            tensor,
            self._vad_model,
            sampling_rate=self.sample_rate,
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.vad_min_speech_ms,
            min_silence_duration_ms=80,
            speech_pad_ms=30,
            return_seconds=False,
        )
        if not timestamps:
            return 0.0, np.zeros((0,), dtype=np.float32)
        pieces: list[np.ndarray] = []
        speech_samples = 0
        for stamp in timestamps:
            start = int(stamp.get("start", 0))
            end = int(stamp.get("end", 0))
            if end <= start:
                continue
            start = max(0, min(start, wave.size))
            end = max(start, min(end, wave.size))
            if end <= start:
                continue
            pieces.append(wave[start:end])
            speech_samples += end - start
        if not pieces:
            return 0.0, np.zeros((0,), dtype=np.float32)
        merged = np.concatenate(pieces).astype(np.float32, copy=False)
        ratio = float(speech_samples / max(wave.size, 1))
        return ratio, merged

    def embedding(self, speech_wave: np.ndarray) -> tuple[float, ...] | None:
        wave = speech_wave.astype(np.float32, copy=False)
        if wave.size == 0:
            return None
        self._non_speech_frames = 0
        if (
            self._enable_instant_embedding
            and self._anchor_embedding is not None
            and wave.size >= self.instant_embedding_min_samples
        ):
            instant = self._encode_embedding(wave[-self.instant_embedding_min_samples :])
            if instant is not None:
                if self._cosine(instant, self._anchor_embedding) < self.speaker_change_similarity_threshold:
                    # Speaker changed inside a continuous stream: avoid mixing identities
                    # in the same cache window.
                    self._speech_cache = np.zeros((0,), dtype=np.float32)
        self._speech_cache = np.concatenate([self._speech_cache, wave]).astype(np.float32, copy=False)
        if self._speech_cache.size > self._max_speech_cache_samples:
            self._speech_cache = self._speech_cache[-self._max_speech_cache_samples :]
        if self._speech_cache.size < self._embedding_window_samples:
            return None
        window = self._speech_cache[-self._embedding_window_samples :]
        arr = self._encode_embedding(window)
        if arr is None:
            return None
        self._anchor_embedding = arr
        return tuple(float(x) for x in arr)

    def mark_non_speech(self) -> None:
        self._non_speech_frames += 1
        # Keep short pauses but flush cache when non-speech persists.
        if self._non_speech_frames >= 3:
            self._speech_cache = np.zeros((0,), dtype=np.float32)
            self._anchor_embedding = None

    def _encode_embedding(self, wave: np.ndarray) -> np.ndarray | None:
        embeddings: list[np.ndarray] = []
        for encoder in self._encoders:
            vec = encoder.encode(wave)
            if vec is None:
                return None
            embeddings.append(vec)
        if not embeddings:
            return None
        if self._embedding_fusion == "concat":
            fused = np.concatenate(embeddings, axis=0)
        else:
            lengths = {emb.shape[0] for emb in embeddings}
            if len(lengths) == 1:
                fused = np.mean(np.stack(embeddings, axis=0), axis=0)
            else:
                fused = np.concatenate(embeddings, axis=0)
        norm = float(np.linalg.norm(fused) + 1e-12)
        if norm <= 0.0:
            return None
        return fused / norm

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        if left.size == 0 or right.size == 0 or left.shape != right.shape:
            return 0.0
        denom = float(np.linalg.norm(left) * np.linalg.norm(right) + 1e-12)
        if denom <= 0.0:
            return 0.0
        return float(np.dot(left, right) / denom)

    @staticmethod
    def _parse_embedding_spec(spec: str) -> tuple[str, str]:
        raw = spec.strip()
        if ":" in raw:
            prefix, rest = raw.split(":", 1)
            prefix = prefix.strip().lower()
            rest = rest.strip()
            if prefix in {"speechbrain", "sb"}:
                return "speechbrain", rest
            if prefix in {"hf", "huggingface", "transformers", "wavlm"}:
                return "hf", rest
        return "speechbrain", raw

    @staticmethod
    def _resolve_device(device: str, torch_module) -> str:
        resolved = str(device).strip().lower()
        if resolved == "auto":
            if torch_module.cuda.is_available():
                return "cuda"
            if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
                return "mps"
            return "cpu"
        if resolved == "cuda" and not torch_module.cuda.is_available():
            return "cpu"
        if resolved == "mps" and (
            not getattr(torch_module.backends, "mps", None)
            or not torch_module.backends.mps.is_available()
        ):
            return "cpu"
        return resolved
