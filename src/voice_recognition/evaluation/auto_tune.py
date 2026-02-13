from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from voice_recognition.core.vector_ops import average_vectors, cosine_similarity
from voice_recognition.evaluation.offline_eval import (
    collect_file_embeddings,
    collect_pair_scores,
    fit_platt_scaler,
    list_samples,
    resample,
)


@dataclass(slots=True)
class CalibrationStats:
    target_scores: list[float]
    non_target_scores: list[float]
    stability_scores: list[float]
    pairwise_means: list[float]
    pairwise_stds: list[float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-calibrate speaker recognition thresholds.")
    parser.add_argument("--output", default="data/tuning.json", help="Output tuning JSON path.")
    parser.add_argument("--dataset", default="openslr/librispeech_asr", help="HuggingFace dataset name.")
    parser.add_argument("--config", default="clean", help="HuggingFace dataset config.")
    parser.add_argument("--split", default="train.100", help="Dataset split.")
    parser.add_argument("--speakers", type=int, default=8, help="Number of speakers to sample.")
    parser.add_argument(
        "--samples-per-speaker",
        type=int,
        default=6,
        help="Number of utterances per speaker.",
    )
    parser.add_argument("--min-seconds", type=float, default=3.0)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-items", type=int, default=6000, help="Max items to scan in the stream.")
    parser.add_argument(
        "--output-dataset",
        default="",
        help="Optional dataset cache directory (default: data/calibration/<dataset>-<config>-<split>).",
    )
    parser.add_argument("--skip-download", action="store_true", help="Use existing dataset directory only.")
    parser.add_argument("--force", action="store_true", help="Rebuild dataset even if already present.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser


def _dataset_dir(args: argparse.Namespace) -> Path:
    if args.output_dataset:
        return Path(args.output_dataset)
    safe = args.dataset.replace("/", "_").replace(":", "_")
    return Path("data") / "calibration" / f"{safe}-{args.config}-{args.split}"


def _has_enough_data(root: Path, speakers: int, samples_per_speaker: int) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    speaker_dirs = [item for item in root.iterdir() if item.is_dir()]
    ready = 0
    for speaker_dir in speaker_dirs:
        wavs = list(speaker_dir.glob("*.wav"))
        if len(wavs) >= samples_per_speaker:
            ready += 1
    return ready >= speakers


def _pick_label(example: dict[str, object]) -> str | None:
    for key in ("speaker_id", "speaker", "speakerId", "client_id", "clientId"):
        if key in example:
            value = example.get(key)
            if value is None:
                continue
            return str(value)
    return None


def _audio_from_example(example: dict[str, object]) -> tuple[np.ndarray, int] | None:
    audio = example.get("audio")
    if audio is None:
        return None
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        array = np.asarray(audio["array"], dtype=np.float32)
        rate = int(audio["sampling_rate"])
    elif isinstance(audio, dict) and (audio.get("bytes") or audio.get("path")):
        if audio.get("bytes"):
            data, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
        else:
            data, rate = sf.read(str(audio["path"]), dtype="float32")
        array = np.asarray(data, dtype=np.float32)
        rate = int(rate)
    else:
        array = getattr(audio, "array", None)
        rate = getattr(audio, "sampling_rate", None)
        if array is None or rate is None:
            return None
        array = np.asarray(array, dtype=np.float32)
        rate = int(rate)
    if array.ndim > 1:
        array = np.mean(array, axis=1)
    return array.astype(np.float32, copy=False), rate


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0)
    data = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def prepare_dataset(
    output_dir: Path,
    dataset: str,
    config: str,
    split: str,
    speakers: int,
    samples_per_speaker: int,
    min_seconds: float,
    max_seconds: float,
    seed: int,
    max_items: int,
    sample_rate: int,
    force: bool,
    skip_download: bool,
) -> Path:
    if skip_download:
        if not _has_enough_data(output_dir, speakers, samples_per_speaker):
            raise SystemExit(f"Dataset directory missing or incomplete: {output_dir}")
        return output_dir

    if not force and _has_enough_data(output_dir, speakers, samples_per_speaker):
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    from datasets import Audio, load_dataset

    stream = load_dataset(dataset, config, split=split, streaming=True)
    if hasattr(stream, "cast_column"):
        stream = stream.cast_column("audio", Audio(decode=False))
    stream = stream.shuffle(buffer_size=1000, seed=seed)

    rng = random.Random(seed)
    counts: dict[str, int] = {}
    accepted = 0
    scanned = 0

    for example in stream:
        scanned += 1
        if max_items > 0 and scanned > max_items:
            break

        label = _pick_label(example)
        if label is None:
            continue
        label = label.strip()
        if not label:
            continue

        if label not in counts and len(counts) >= speakers:
            continue

        if counts.get(label, 0) >= samples_per_speaker:
            continue

        audio_pair = _audio_from_example(example)
        if audio_pair is None:
            continue
        audio, sr = audio_pair
        duration = audio.size / float(sr)
        if duration < min_seconds or duration > max_seconds:
            continue

        audio = resample(audio, src_rate=sr, dst_rate=sample_rate)
        filename = f"{label}_{counts.get(label, 0) + 1:02d}_{rng.randrange(100000):05d}.wav"
        _write_wav(output_dir / label / filename, audio, sample_rate)
        counts[label] = counts.get(label, 0) + 1
        accepted += 1

        if len(counts) >= speakers and all(value >= samples_per_speaker for value in counts.values()):
            break

    if not _has_enough_data(output_dir, speakers, samples_per_speaker):
        raise SystemExit(
            "Could not collect enough samples. "
            f"Speakers={len(counts)}/{speakers}, samples={counts}."
        )
    return output_dir


def _percentile(values: Iterable[float], pct: float) -> float | None:
    items = sorted(float(value) for value in values)
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    pct = max(0.0, min(100.0, float(pct))) / 100.0
    pos = (len(items) - 1) * pct
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return items[low]
    weight = pos - low
    return items[low] * (1.0 - weight) + items[high] * weight


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _pairwise_stats(embeddings: list[tuple[float, ...]]) -> tuple[float, float]:
    if len(embeddings) < 2:
        return 1.0, 0.0
    sims: list[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(cosine_similarity(embeddings[i], embeddings[j]))
    if not sims:
        return 0.0, 1.0
    mean = sum(sims) / float(len(sims))
    variance = sum((value - mean) ** 2 for value in sims) / float(len(sims))
    return mean, math.sqrt(variance)


def _collect_stats(file_embeddings) -> CalibrationStats:
    pair_scores = collect_pair_scores(file_embeddings)
    target_scores = [score for score, label in pair_scores if label == 1]
    non_target_scores = [score for score, label in pair_scores if label == 0]

    by_label: dict[str, list[tuple[float, ...]]] = {}
    for item in file_embeddings:
        by_label.setdefault(item.label, []).append(item.embedding)

    stability_scores: list[float] = []
    pairwise_means: list[float] = []
    pairwise_stds: list[float] = []

    for embeddings in by_label.values():
        if len(embeddings) < 2:
            continue
        centroid = average_vectors(embeddings)
        stability = min(cosine_similarity(vector, centroid) for vector in embeddings)
        stability_scores.append(stability)
        mean, std = _pairwise_stats(embeddings)
        pairwise_means.append(mean)
        pairwise_stds.append(std)

    return CalibrationStats(
        target_scores=target_scores,
        non_target_scores=non_target_scores,
        stability_scores=stability_scores,
        pairwise_means=pairwise_means,
        pairwise_stds=pairwise_stds,
    )


def _recommend_thresholds(stats: CalibrationStats) -> dict[str, float]:
    min_threshold_gap = 0.08
    match_threshold = _percentile(stats.non_target_scores, 99.0)
    if match_threshold is None:
        match_threshold = 0.82
    match_threshold = _clamp(match_threshold, 0.70, 0.95)

    new_speaker_threshold = _percentile(stats.target_scores, 5.0)
    if new_speaker_threshold is None:
        new_speaker_threshold = match_threshold - 0.10
    new_speaker_upper = max(0.40, match_threshold - min_threshold_gap)
    new_speaker_threshold = _clamp(new_speaker_threshold, 0.40, new_speaker_upper)

    min_cluster_similarity = _percentile(stats.stability_scores, 10.0)
    if min_cluster_similarity is None:
        min_cluster_similarity = 0.82
    min_cluster_similarity = _clamp(min_cluster_similarity, 0.65, 0.95)

    min_pairwise_similarity = _percentile(stats.pairwise_means, 10.0)
    if min_pairwise_similarity is None:
        min_pairwise_similarity = 0.70
    min_pairwise_similarity = _clamp(min_pairwise_similarity, 0.55, 0.95)

    max_pairwise_std = _percentile(stats.pairwise_stds, 90.0)
    if max_pairwise_std is None:
        max_pairwise_std = 0.12
    max_pairwise_std = _clamp(max_pairwise_std, 0.05, 0.25)

    return {
        "match_threshold": float(match_threshold),
        "new_speaker_threshold": float(new_speaker_threshold),
        "min_cluster_similarity": float(min_cluster_similarity),
        "min_pairwise_similarity": float(min_pairwise_similarity),
        "max_pairwise_std": float(max_pairwise_std),
    }


def _summary(values: Iterable[float]) -> dict[str, float | None]:
    items = list(values)
    return {
        "p01": _percentile(items, 1.0),
        "p05": _percentile(items, 5.0),
        "p10": _percentile(items, 10.0),
        "p50": _percentile(items, 50.0),
        "p90": _percentile(items, 90.0),
        "p95": _percentile(items, 95.0),
        "p99": _percentile(items, 99.0),
    }


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = _dataset_dir(args)
    dataset_dir = prepare_dataset(
        output_dir=dataset_dir,
        dataset=str(args.dataset),
        config=str(args.config),
        split=str(args.split),
        speakers=int(args.speakers),
        samples_per_speaker=int(args.samples_per_speaker),
        min_seconds=float(args.min_seconds),
        max_seconds=float(args.max_seconds),
        seed=int(args.seed),
        max_items=int(args.max_items),
        sample_rate=int(args.sample_rate),
        force=bool(args.force),
        skip_download=bool(args.skip_download),
    )

    samples = list_samples(dataset_dir)
    if not samples:
        raise SystemExit(f"No WAV files found in {dataset_dir}")
    speaker_counts: dict[str, int] = {}
    for sample in samples:
        speaker_counts[sample.label] = speaker_counts.get(sample.label, 0) + 1
    min_samples = min(speaker_counts.values()) if speaker_counts else 0
    max_samples = max(speaker_counts.values()) if speaker_counts else 0

    file_embeddings, extract_stats = collect_file_embeddings(
        samples=samples,
        sample_rate=int(args.sample_rate),
        chunk_seconds=1.0,
        hop_seconds=0.5,
    )
    if not file_embeddings:
        raise SystemExit("No embeddings extracted; check audio or models.")

    stats = _collect_stats(file_embeddings)
    thresholds = _recommend_thresholds(stats)

    pair_scores = collect_pair_scores(file_embeddings)
    if pair_scores:
        scores = [item[0] for item in pair_scores]
        labels = [item[1] for item in pair_scores]
        scale, bias = fit_platt_scaler(scores, labels)
    else:
        scale, bias = 4.0, -2.0

    tuning = {
        "matchThreshold": thresholds["match_threshold"],
        "newSpeakerThreshold": thresholds["new_speaker_threshold"],
        "minMatchMargin": 0.05,
        "scoreCalibrationScale": float(scale),
        "scoreCalibrationBias": float(bias),
        "enrollmentMinSegments": 5,
        "enrollmentMinClusterSimilarity": thresholds["min_cluster_similarity"],
        "enrollmentMinPairwiseSimilarity": thresholds["min_pairwise_similarity"],
        "enrollmentMaxPairwiseStd": thresholds["max_pairwise_std"],
        "enrollmentCooldownSegments": 10,
    }

    report = {
        "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "name": str(args.dataset),
            "config": str(args.config),
            "split": str(args.split),
            "root": str(dataset_dir.resolve()),
            "requestedSpeakers": int(args.speakers),
            "requestedSamplesPerSpeaker": int(args.samples_per_speaker),
            "speakers": int(len(speaker_counts)),
            "samplesPerSpeakerMin": int(min_samples),
            "samplesPerSpeakerMax": int(max_samples),
            "minSeconds": float(args.min_seconds),
            "maxSeconds": float(args.max_seconds),
            "sampleRate": int(args.sample_rate),
        },
        "extraction": extract_stats,
        "stats": {
            "targetScores": _summary(stats.target_scores),
            "nonTargetScores": _summary(stats.non_target_scores),
            "clusterStability": _summary(stats.stability_scores),
            "pairwiseMean": _summary(stats.pairwise_means),
            "pairwiseStd": _summary(stats.pairwise_stds),
        },
        "tuning": tuning,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
