from __future__ import annotations

import argparse
import json
import math
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from voice_recognition.core.models import Speaker
from voice_recognition.audio.signal_processor import SignalProcessor, SignalProcessorConfig
from voice_recognition.core.models import EventType, RecognitionScope, SignalType
from voice_recognition.core.vector_ops import cosine_similarity
from voice_recognition.recognition.enrollment import AutoEnrollmentManager, EnrollmentConfig
from voice_recognition.recognition.engine import RecognitionEngine
from voice_recognition.recognition.matcher import MatcherConfig, SpeakerMatcher
from voice_recognition.storage.session_repository import SessionSpeakerRepository


@dataclass(slots=True)
class FileSample:
    path: Path
    label: str


@dataclass(slots=True)
class FileEmbedding:
    path: Path
    label: str
    embedding: tuple[float, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline open-set evaluation for voice_recognition.")
    parser.add_argument("--dataset", required=True, help="Dataset root, layout: dataset/<speaker_label>/*.wav")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all files")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--profile",
        choices=["balanced", "fast_enroll", "strict"],
        default="balanced",
        help="Streaming simulation profile.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional report path. If empty, print JSON only.",
    )
    parser.add_argument(
        "--score-backend",
        choices=["cosine", "asnorm"],
        default="asnorm",
        help="Scoring backend used by closed-set simulation diagnostics.",
    )
    parser.add_argument("--asnorm-top-k", type=int, default=12)
    parser.add_argument("--asnorm-min-cohort", type=int, default=2)
    parser.add_argument("--asnorm-blend", type=float, default=0.35)
    parser.add_argument("--calibration-scale", type=float, default=4.0)
    parser.add_argument("--calibration-bias", type=float, default=-2.0)
    parser.add_argument(
        "--fit-platt",
        action="store_true",
        help="Fit logistic calibration (Platt scaling) on current pair trials.",
    )
    return parser


def list_samples(dataset_root: Path) -> list[FileSample]:
    samples: list[FileSample] = []
    for speaker_dir in sorted(dataset_root.iterdir()):
        if not speaker_dir.is_dir():
            continue
        label = speaker_dir.name
        for wav_path in sorted(speaker_dir.glob("*.wav")):
            samples.append(FileSample(path=wav_path, label=label))
    return samples


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width={width} for file {path}")

    if channels > 1:
        data = data.reshape(-1, channels)
        data = np.mean(data, axis=1)
    return data.astype(np.float32, copy=False), int(sample_rate)


def resample(signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return signal.astype(np.float32, copy=False)
    if signal.size <= 1:
        return signal.astype(np.float32, copy=False)
    dst_len = int(round(signal.size * (float(dst_rate) / float(src_rate))))
    if dst_len <= 1:
        dst_len = 2
    src_x = np.linspace(0.0, 1.0, num=signal.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, signal).astype(np.float32)


def normalize(v: Iterable[float]) -> tuple[float, ...]:
    vector = tuple(float(x) for x in v)
    norm = math.sqrt(sum(x * x for x in vector))
    if norm <= 0:
        return vector
    return tuple(x / norm for x in vector)


def iter_chunks(signal: np.ndarray, sample_rate: int, chunk_seconds: float, hop_seconds: float):
    chunk = max(1, int(sample_rate * chunk_seconds))
    hop = max(1, int(sample_rate * hop_seconds))
    idx = 0
    while idx + chunk <= signal.size:
        yield signal[idx : idx + chunk]
        idx += hop


def collect_file_embeddings(
    samples: list[FileSample],
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
) -> tuple[list[FileEmbedding], dict[str, int]]:
    output: list[FileEmbedding] = []
    stats = {"files_total": len(samples), "files_with_embedding": 0, "speech_frames": 0}

    for sample in samples:
        processor = SignalProcessor(
            SignalProcessorConfig(
                sample_rate=sample_rate,
                backend="neural",
                model_cache_dir="data/models",
                vad_threshold=0.38,
                vad_speech_ratio_threshold=0.12,
                vad_min_speech_ms=130,
                vad_min_samples_for_embedding=9000,
            )
        )
        audio, sr = read_wav_mono(sample.path)
        audio = resample(audio, src_rate=sr, dst_rate=sample_rate)
        embs: list[tuple[float, ...]] = []
        for chunk in iter_chunks(audio, sample_rate=sample_rate, chunk_seconds=chunk_seconds, hop_seconds=hop_seconds):
            frame = processor.infer(chunk=chunk, source="offline")
            if frame.signal == SignalType.SPEECH and frame.embedding is not None:
                embs.append(frame.embedding)
                stats["speech_frames"] += 1
        if not embs:
            continue
        avg = normalize(np.mean(np.array(embs, dtype=np.float32), axis=0).tolist())
        output.append(FileEmbedding(path=sample.path, label=sample.label, embedding=avg))

    stats["files_with_embedding"] = len(output)
    return output, stats


def compute_verification_metrics(file_embs: list[FileEmbedding]) -> dict[str, float | int | None]:
    if len(file_embs) < 2:
        return {
            "target_trials": 0,
            "nontarget_trials": 0,
            "eer": None,
            "eer_threshold": None,
            "min_dcf_p01": None,
        }

    scores = collect_pair_scores(file_embs)

    thresholds = sorted({score for score, _ in scores})
    if not thresholds:
        return {
            "target_trials": 0,
            "nontarget_trials": 0,
            "eer": None,
            "eer_threshold": None,
            "min_dcf_p01": None,
        }

    target_total = sum(1 for _, label in scores if label == 1)
    non_total = sum(1 for _, label in scores if label == 0)

    best_eer = None
    best_eer_th = None
    best_dcf = None
    p_target = 0.01

    for th in thresholds:
        fa = 0
        miss = 0
        for score, label in scores:
            accepted = score >= th
            if label == 1 and not accepted:
                miss += 1
            elif label == 0 and accepted:
                fa += 1
        far = fa / max(non_total, 1)
        frr = miss / max(target_total, 1)
        eer_candidate = (far + frr) / 2.0
        if best_eer is None or abs(far - frr) < abs(best_eer[0] - best_eer[1]):
            best_eer = (far, frr, eer_candidate)
            best_eer_th = th

        dcf = p_target * frr + (1.0 - p_target) * far
        if best_dcf is None or dcf < best_dcf:
            best_dcf = dcf

    return {
        "target_trials": target_total,
        "nontarget_trials": non_total,
        "eer": None if best_eer is None else best_eer[2],
        "eer_threshold": best_eer_th,
        "min_dcf_p01": best_dcf,
    }


def collect_pair_scores(file_embs: list[FileEmbedding]) -> list[tuple[float, int]]:
    scores: list[tuple[float, int]] = []
    for i in range(len(file_embs)):
        for j in range(i + 1, len(file_embs)):
            left = file_embs[i]
            right = file_embs[j]
            label = 1 if left.label == right.label else 0
            score = cosine_similarity(left.embedding, right.embedding)
            scores.append((score, label))
    return scores


def fit_platt_scaler(
    scores: list[float],
    labels: list[int],
    steps: int = 1000,
    lr: float = 0.08,
    reg: float = 1e-4,
) -> tuple[float, float]:
    if not scores or len(scores) != len(labels):
        return 1.0, 0.0
    scale = 1.0
    bias = 0.0
    n = float(len(scores))
    for _ in range(max(50, steps)):
        grad_scale = 0.0
        grad_bias = 0.0
        for score, label in zip(scores, labels):
            logit = scale * score + bias
            prob = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))
            diff = prob - float(label)
            grad_scale += diff * score
            grad_bias += diff
        grad_scale = grad_scale / n + reg * scale
        grad_bias = grad_bias / n
        scale -= lr * grad_scale
        bias -= lr * grad_bias
    return float(scale), float(bias)


def compute_closed_set_top1(
    file_embs: list[FileEmbedding],
    matcher: SpeakerMatcher,
) -> dict[str, float | int | None]:
    by_label: dict[str, list[FileEmbedding]] = {}
    for item in file_embs:
        by_label.setdefault(item.label, []).append(item)
    if len(by_label) < 2:
        return {
            "labels": len(by_label),
            "probes": 0,
            "top1": None,
            "accept_rate": None,
        }
    gallery: list[Speaker] = []
    probes: list[FileEmbedding] = []
    speaker_id = 1
    for label, items in sorted(by_label.items()):
        if not items:
            continue
        gallery.append(
            Speaker(
                id=speaker_id,
                name=label,
                centroid=items[0].embedding,
                prototypes=(items[0].embedding,),
                sample_count=1,
            )
        )
        speaker_id += 1
        probes.extend(items[1:])
    if not probes or not gallery:
        return {
            "labels": len(by_label),
            "probes": 0,
            "top1": None,
            "accept_rate": None,
        }
    correct = 0
    accepted = 0
    for probe in probes:
        decision = matcher.match(probe.embedding, gallery)
        if decision.top_speaker is not None:
            accepted += 1
            if decision.top_speaker.name == probe.label:
                correct += 1
    return {
        "labels": len(by_label),
        "probes": len(probes),
        "top1": correct / float(len(probes)),
        "accept_rate": accepted / float(len(probes)),
    }


def profile_config(profile: str) -> dict[str, float | int]:
    if profile == "fast_enroll":
        return {
            "match_threshold": 0.80,
            "new_speaker_threshold": 0.72,
            "min_margin": 0.05,
            "min_segments": 3,
            "min_cluster_similarity": 0.75,
        }
    if profile == "strict":
        return {
            "match_threshold": 0.86,
            "new_speaker_threshold": 0.80,
            "min_margin": 0.07,
            "min_segments": 6,
            "min_cluster_similarity": 0.86,
        }
    return {
        "match_threshold": 0.82,
        "new_speaker_threshold": 0.76,
        "min_margin": 0.06,
        "min_segments": 4,
        "min_cluster_similarity": 0.80,
    }


def simulate_streaming(
    samples: list[FileSample],
    sample_rate: int,
    chunk_seconds: float,
    hop_seconds: float,
    profile: str,
    seed: int,
) -> dict[str, object]:
    cfg = profile_config(profile)
    repo = SessionSpeakerRepository()
    engine = RecognitionEngine(
        repository=repo,
        scope=RecognitionScope.SESSION,
        matcher=SpeakerMatcher(
            MatcherConfig(
                match_threshold=float(cfg["match_threshold"]),
                confidence_low=0.50,
                confidence_high=0.92,
                min_margin=float(cfg["min_margin"]),
            )
        ),
        enrollment=AutoEnrollmentManager(
            EnrollmentConfig(
                min_segments=int(cfg["min_segments"]),
                min_cluster_similarity=float(cfg["min_cluster_similarity"]),
                new_speaker_threshold=float(cfg["new_speaker_threshold"]),
                safety_similarity_threshold=min(float(cfg["match_threshold"]) - 0.01, 0.80),
                cooldown_segments=6,
            )
        ),
        soft_match_threshold=max(0.45, float(cfg["match_threshold"]) - 0.14),
        enrollment_reset_non_speech_frames=4,
    )

    rng = random.Random(seed)
    ordered = list(samples)
    rng.shuffle(ordered)

    label_to_speaker_ids: dict[str, set[int]] = {}
    speaker_id_to_label: dict[int, str] = {}
    unknown_files = 0
    confusion_events = 0
    total_files = 0
    files_with_speech = 0

    for sample in ordered:
        total_files += 1
        processor = SignalProcessor(
            SignalProcessorConfig(
                sample_rate=sample_rate,
                backend="neural",
                model_cache_dir="data/models",
                vad_threshold=0.38,
                vad_speech_ratio_threshold=0.12,
                vad_min_speech_ms=130,
                vad_min_samples_for_embedding=9000,
            )
        )
        audio, sr = read_wav_mono(sample.path)
        audio = resample(audio, src_rate=sr, dst_rate=sample_rate)
        events: list[tuple[EventType, int | None]] = []
        for chunk in iter_chunks(audio, sample_rate=sample_rate, chunk_seconds=chunk_seconds, hop_seconds=hop_seconds):
            frame = processor.infer(chunk=chunk, source=str(sample.path))
            event = engine.process(frame)
            if event.event_type in {EventType.MATCH, EventType.NEW_SPEAKER}:
                events.append((event.event_type, event.speaker_id))

        if events:
            files_with_speech += 1
        else:
            unknown_files += 1
            continue

        label_bucket = label_to_speaker_ids.setdefault(sample.label, set())
        for _, speaker_id in events:
            if speaker_id is None:
                continue
            known_label = speaker_id_to_label.get(speaker_id)
            if known_label is None:
                speaker_id_to_label[speaker_id] = sample.label
            elif known_label != sample.label:
                confusion_events += 1
            label_bucket.add(speaker_id)

    duplicate_labels = sum(1 for ids in label_to_speaker_ids.values() if len(ids) > 1)
    repo.close()
    return {
        "files_total": total_files,
        "files_with_detected_speaker": files_with_speech,
        "unknown_file_count": unknown_files,
        "unknown_file_rate": unknown_files / max(total_files, 1),
        "labels_with_duplicate_ids": duplicate_labels,
        "label_count": len(label_to_speaker_ids),
        "duplicate_label_rate": duplicate_labels / max(len(label_to_speaker_ids), 1),
        "confusion_events": confusion_events,
    }


def main() -> int:
    args = build_parser().parse_args()
    dataset = Path(args.dataset)
    if not dataset.exists() or not dataset.is_dir():
        raise SystemExit(f"Dataset root not found: {dataset}")

    samples = list_samples(dataset)
    if args.max_files > 0:
        samples = samples[: args.max_files]
    if not samples:
        raise SystemExit("No wav files found. Expected layout: dataset/<speaker_label>/*.wav")

    file_embs, extract_stats = collect_file_embeddings(
        samples=samples,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        hop_seconds=args.hop_seconds,
    )
    verify = compute_verification_metrics(file_embs)
    closed_set = compute_closed_set_top1(
        file_embs=file_embs,
        matcher=SpeakerMatcher(
            MatcherConfig(
                match_threshold=0.70 if args.score_backend == "asnorm" else 0.82,
                confidence_low=0.45,
                confidence_high=0.92,
                min_margin=0.04,
                score_backend=args.score_backend,
                asnorm_top_k=max(2, int(args.asnorm_top_k)),
                asnorm_min_cohort=max(2, int(args.asnorm_min_cohort)),
                asnorm_blend=max(0.0, min(1.0, float(args.asnorm_blend))),
                calibration_scale=float(args.calibration_scale),
                calibration_bias=float(args.calibration_bias),
            )
        ),
    )
    platt = None
    if args.fit_platt:
        pair_scores = collect_pair_scores(file_embs)
        if pair_scores:
            raw_scores = [item[0] for item in pair_scores]
            labels = [item[1] for item in pair_scores]
            scale, bias = fit_platt_scaler(raw_scores, labels)
            platt = {"scale": scale, "bias": bias}
    streaming = simulate_streaming(
        samples=samples,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
        hop_seconds=args.hop_seconds,
        profile=args.profile,
        seed=args.seed,
    )

    report = {
        "dataset": str(dataset.resolve()),
        "files": len(samples),
        "embedding_extraction": extract_stats,
        "verification": verify,
        "closed_set": closed_set,
        "calibration": platt,
        "streaming": streaming,
        "profile": args.profile,
        "score_backend": args.score_backend,
    }

    out = json.dumps(report, ensure_ascii=False, indent=2)
    print(out)
    if args.output_json:
        Path(args.output_json).write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
