from __future__ import annotations

import argparse
import queue
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from voice_recognition.audio import AudioInputManager, AudioSource
from voice_recognition.core.models import EventType, RecognitionEvent, RecognitionScope, Speaker
from voice_recognition.live_service import LiveConfig, LiveRecognitionService


@dataclass(slots=True)
class DashboardState:
    status: str = "Idle"
    speaker: str = "-"
    confidence: str = "-"
    active_speaker_id: int | None = None
    active_confidence: float | None = None
    speakers: list[Speaker] = field(default_factory=list)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=15))
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live voiceprint console dashboard.")
    parser.add_argument(
        "--scope",
        choices=[RecognitionScope.GLOBAL.value, RecognitionScope.SESSION.value],
        default=RecognitionScope.GLOBAL.value,
    )
    parser.add_argument(
        "--source",
        choices=[AudioSource.MICROPHONE.value, AudioSource.SYSTEM.value],
        default=AudioSource.MICROPHONE.value,
    )
    parser.add_argument("--device", type=int, default=None, help="Capture device index.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--db-path", default=str(Path("data") / "speakers.db"))
    parser.add_argument(
        "--mode",
        choices=["recognition", "diarization"],
        default="recognition",
        help="recognition = 匹配入库, diarization = 聚类分离",
    )
    parser.add_argument("--match-threshold", type=float, default=0.82)
    parser.add_argument("--new-speaker-threshold", type=float, default=0.76)
    parser.add_argument("--min-margin", type=float, default=0.06)
    parser.add_argument("--min-segments", type=int, default=6)
    parser.add_argument("--min-cluster-similarity", type=float, default=0.84)
    parser.add_argument("--list-devices", action="store_true")
    return parser


def list_devices(source: AudioSource) -> list[tuple[int, str]]:
    devices = AudioInputManager.list_capture_devices(source=source)
    return [(device.index, device.name) for device in devices]


def choose_device(source: AudioSource) -> int:
    devices = list_devices(source)
    if not devices:
        raise RuntimeError("No input devices found.")
    print("\nAvailable input devices:")
    for index, name in devices:
        print(f"  {index:>3}  {name}")
    print("")
    while True:
        choice = input("Select device index: ").strip()
        if not choice:
            return devices[0][0]
        if choice.isdigit() and any(index == int(choice) for index, _ in devices):
            return int(choice)
        print("Invalid index, try again.")


def update_state(state: DashboardState, event: RecognitionEvent, speakers: list[Speaker], error: str | None) -> None:
    if error:
        state.error = error
        state.status = "Error"
        state.logs.append(f"error: {error}")
    if event.event_type == EventType.MATCH:
        state.status = "Matched Speaker"
        state.speaker = event.speaker_name or "-"
        state.confidence = f"{event.confidence:.2f}%" if event.confidence is not None else "-"
        state.active_speaker_id = event.speaker_id
        state.active_confidence = event.confidence
    elif event.event_type == EventType.NEW_SPEAKER:
        state.status = "New Speaker Registered"
        state.speaker = event.speaker_name or "-"
        state.confidence = f"{event.confidence:.2f}%" if event.confidence is not None else "-"
        state.active_speaker_id = event.speaker_id
        state.active_confidence = event.confidence
    elif event.event_type == EventType.UNKNOWN_SPEECH:
        state.status = "Human Speech (Unknown)"
        state.speaker = "-"
        state.confidence = "-"
        state.active_speaker_id = None
        state.active_confidence = None
    elif event.event_type == EventType.NOISE and event.details != "error":
        state.status = "Background Noise"
        state.speaker = "-"
        state.confidence = "-"
        state.active_speaker_id = None
        state.active_confidence = None
    elif event.event_type == EventType.SILENCE:
        state.status = "Silence"
        if event.details != "service_started":
            state.speaker = "-"
            state.confidence = "-"
        state.active_speaker_id = None
        state.active_confidence = None

    score = f"{event.score:.4f}" if event.score is not None else "-"
    conf = f"{event.confidence:.2f}%" if event.confidence is not None else "-"
    line = (
        f"{event.event_type.value:<14} speaker={event.speaker_name or '-':<8} "
        f"score={score:<7} confidence={conf:<8} source={event.source}"
    )
    if event.details:
        line += f" [{event.details}]"
    state.logs.append(line)
    state.speakers = speakers


def render(state: DashboardState, scope: RecognitionScope, source: AudioSource, device: int) -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.write("Voiceprint Recognition Demo (Console)\n")
    sys.stdout.write("=" * 78 + "\n")
    sys.stdout.write(f"Mode: {source.value}   Scope: {scope.value}   Device: {device}\n")
    sys.stdout.write(f"Status: {state.status}\n")
    sys.stdout.write(f"Current Speaker: {state.speaker}   Confidence: {state.confidence}\n")
    if state.error:
        sys.stdout.write(f"Last Error: {state.error}\n")
    sys.stdout.write("-" * 78 + "\n")
    sys.stdout.write("Registered Speakers\n")
    if not state.speakers:
        sys.stdout.write("  (none)\n")
    else:
        for speaker in state.speakers:
            marker = ">>" if speaker.id == state.active_speaker_id else "  "
            conf = "-"
            if speaker.id == state.active_speaker_id and state.active_confidence is not None:
                conf = f"{state.active_confidence:.2f}%"
            speaker_id = speaker.id if speaker.id is not None else "-"
            sys.stdout.write(
                f"{marker} id={speaker_id:<3} name={speaker.name:<8} "
                f"samples={speaker.sample_count:<3} confidence={conf}\n"
            )
    sys.stdout.write("-" * 78 + "\n")
    sys.stdout.write("Recent Events\n")
    if not state.logs:
        sys.stdout.write("  (none)\n")
    else:
        for line in state.logs:
            sys.stdout.write(f"  {line}\n")
    sys.stdout.write("-" * 78 + "\n")
    sys.stdout.write("Press Ctrl+C to stop.\n")
    sys.stdout.flush()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    source = AudioSource(args.source)
    scope = RecognitionScope(args.scope)

    if args.list_devices:
        devices = list_devices(source)
        if not devices:
            print("No input devices found.")
        for index, name in devices:
            print(f"{index}\t{name}")
        return 0

    device_index = args.device if args.device is not None else choose_device(source)
    service = LiveRecognitionService()
    state = DashboardState()

    config = LiveConfig(
        scope=scope,
        source=source,
        device_index=device_index,
        mode=args.mode,
        sample_rate=args.sample_rate,
        db_path=Path(args.db_path),
        match_threshold=args.match_threshold,
        new_speaker_threshold=args.new_speaker_threshold,
        min_match_margin=args.min_margin,
        enrollment_min_segments=args.min_segments,
        enrollment_min_cluster_similarity=args.min_cluster_similarity,
    )

    try:
        service.start(config)
        while True:
            while True:
                try:
                    update = service.updates.get_nowait()
                except queue.Empty:
                    break
                update_state(state, update.event, update.speakers, update.error)
            render(state, scope=scope, source=source, device=device_index)
            if not service.running and state.error:
                break
            time.sleep(0.12)
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
