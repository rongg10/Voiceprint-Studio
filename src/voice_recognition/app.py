from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from voice_recognition.core.models import FrameInference, RecognitionScope, SignalType
from voice_recognition.recognition.engine import RecognitionEngine
from voice_recognition.storage.factory import build_repository


def parse_scope(value: str) -> RecognitionScope:
    normalized = value.strip().lower()
    if normalized == RecognitionScope.GLOBAL.value:
        return RecognitionScope.GLOBAL
    if normalized == RecognitionScope.SESSION.value:
        return RecognitionScope.SESSION
    raise argparse.ArgumentTypeError("scope must be one of: global, session")


def load_frames(path: Path) -> list[FrameInference]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Input JSON must be a list of frames.")
    frames: list[FrameInference] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Frame #{index} must be an object.")
        signal_value = str(item.get("signal", "")).lower().strip()
        try:
            signal = SignalType(signal_value)
        except ValueError as exc:
            raise ValueError(f"Frame #{index} has invalid signal={signal_value!r}") from exc
        embedding = item.get("embedding")
        parsed_embedding = _parse_embedding(embedding, index=index)
        source = str(item.get("source", path.name))
        frames.append(FrameInference(signal=signal, embedding=parsed_embedding, source=source))
    return frames


def _parse_embedding(value: Any, index: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Frame #{index} embedding must be a list of numbers.")
    parsed: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"Frame #{index} embedding contains non-numeric values.")
        parsed.append(float(item))
    return tuple(parsed)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voice recognition demo CLI.")
    parser.add_argument(
        "--scope",
        default=RecognitionScope.GLOBAL,
        type=parse_scope,
        help="global or session",
    )
    parser.add_argument(
        "--db-path",
        default=str(Path("data") / "speakers.db"),
        help="SQLite path for global scope.",
    )
    parser.add_argument(
        "--input-json",
        default=str(Path("data") / "sample_frames.json"),
        help="Path to frame list JSON for demo playback.",
    )
    parser.add_argument(
        "--show-speakers",
        action="store_true",
        help="Print current speaker roster after processing.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    scope: RecognitionScope = args.scope
    repository = build_repository(scope=scope, db_path=args.db_path)
    engine = RecognitionEngine(repository=repository, scope=scope)
    frames = load_frames(Path(args.input_json))

    for frame in frames:
        event = engine.process(frame)
        speaker = event.speaker_name if event.speaker_name else "-"
        score = f"{event.score:.4f}" if event.score is not None else "-"
        confidence = f"{event.confidence:.2f}%" if event.confidence is not None else "-"
        print(
            f"[{event.ts}] event={event.event_type.value:<14} scope={event.scope.value:<7} "
            f"speaker={speaker:<8} score={score:<8} confidence={confidence:<8} source={event.source}"
        )

    if args.show_speakers:
        speakers = repository.list_speakers()
        print("\nRegistered speakers:")
        if not speakers:
            print("- (none)")
        for speaker in speakers:
            print(
                f"- id={speaker.id} name={speaker.name} samples={speaker.sample_count} "
                f"updated_at={speaker.updated_at}"
            )

    repository.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
