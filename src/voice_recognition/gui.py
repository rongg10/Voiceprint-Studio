from __future__ import annotations

import argparse
import queue
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception:  # pragma: no cover - depends on local Python build
    tk = None
    messagebox = None
    ttk = None

from voice_recognition.audio import AudioInputManager, AudioSource
from voice_recognition.core.models import EventType, RecognitionEvent, RecognitionScope, Speaker
from voice_recognition.live_service import LiveConfig, LiveRecognitionService


class RecognitionApp:
    def __init__(self, root: tk.Tk, db_path: Path, sample_rate: int) -> None:
        self.root = root
        self.db_path = db_path
        self.sample_rate = sample_rate
        self.service = LiveRecognitionService()

        self.mode_var = tk.StringVar(value=AudioSource.MICROPHONE.value)
        self.scope_var = tk.StringVar(value=RecognitionScope.GLOBAL.value)
        self.device_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Idle")
        self.current_var = tk.StringVar(value="-")
        self.confidence_var = tk.StringVar(value="-")
        self.hint_var = tk.StringVar(value="")

        self.device_map: dict[str, int] = {}
        self._active_speaker_id: int | None = None
        self._active_confidence: float | None = None

        self._build_layout()
        self._refresh_devices()
        self.root.after(120, self._drain_updates)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        self.root.title("Voiceprint Recognition Demo")
        self.root.geometry("980x680")

        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Input Source").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        source_box = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=[AudioSource.MICROPHONE.value, AudioSource.SYSTEM.value],
            state="readonly",
            width=18,
        )
        source_box.grid(row=0, column=1, sticky=tk.W)
        source_box.bind("<<ComboboxSelected>>", lambda _: self._refresh_devices())

        ttk.Label(top, text="Device").grid(row=0, column=2, sticky=tk.W, padx=(16, 8))
        self.device_box = ttk.Combobox(top, textvariable=self.device_var, state="readonly", width=52)
        self.device_box.grid(row=0, column=3, sticky=tk.W)

        ttk.Button(top, text="Refresh Devices", command=self._refresh_devices).grid(
            row=0, column=4, padx=(8, 0)
        )

        ttk.Label(top, text="Library Scope").grid(row=1, column=0, sticky=tk.W, pady=(10, 0), padx=(0, 8))
        scope_box = ttk.Combobox(
            top,
            textvariable=self.scope_var,
            values=[RecognitionScope.GLOBAL.value, RecognitionScope.SESSION.value],
            state="readonly",
            width=18,
        )
        scope_box.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))

        self.start_btn = ttk.Button(top, text="Start", command=self._start)
        self.start_btn.grid(row=1, column=3, sticky=tk.W, pady=(10, 0))

        self.stop_btn = ttk.Button(top, text="Stop", command=self._stop, state=tk.DISABLED)
        self.stop_btn.grid(row=1, column=3, sticky=tk.W, padx=(74, 0), pady=(10, 0))

        ttk.Label(top, textvariable=self.hint_var, foreground="#555555").grid(
            row=2, column=0, columnspan=5, sticky=tk.W, pady=(8, 0)
        )

        status = ttk.LabelFrame(self.root, text="Live Status", padding=12)
        status.pack(fill=tk.X, padx=12, pady=(8, 0))

        ttk.Label(status, text="Current State").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Label(status, textvariable=self.status_var, foreground="#0b5cad").grid(row=0, column=1, sticky=tk.W)

        ttk.Label(status, text="Speaker").grid(row=0, column=2, sticky=tk.W, padx=(32, 8))
        ttk.Label(status, textvariable=self.current_var).grid(row=0, column=3, sticky=tk.W)

        ttk.Label(status, text="Confidence").grid(row=0, column=4, sticky=tk.W, padx=(32, 8))
        ttk.Label(status, textvariable=self.confidence_var).grid(row=0, column=5, sticky=tk.W)

        center = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        center.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        speaker_frame = ttk.LabelFrame(center, text="Registered Speakers", padding=8)
        log_frame = ttk.LabelFrame(center, text="Events", padding=8)
        center.add(speaker_frame, weight=2)
        center.add(log_frame, weight=3)

        self.tree = ttk.Treeview(
            speaker_frame,
            columns=("id", "name", "samples", "confidence"),
            show="headings",
            height=18,
        )
        self.tree.heading("id", text="ID")
        self.tree.heading("name", text="Name")
        self.tree.heading("samples", text="Samples")
        self.tree.heading("confidence", text="Confidence")
        self.tree.column("id", width=80, anchor=tk.CENTER)
        self.tree.column("name", width=120, anchor=tk.W)
        self.tree.column("samples", width=100, anchor=tk.CENTER)
        self.tree.column("confidence", width=120, anchor=tk.CENTER)
        self.tree.tag_configure("active", background="#ffefc2")
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.log_box = tk.Text(log_frame, wrap=tk.NONE, height=18, state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def _refresh_devices(self) -> None:
        source = AudioSource(self.mode_var.get())
        try:
            devices = AudioInputManager.list_capture_devices(source=source)
        except Exception as exc:
            messagebox.showerror("Audio backend error", str(exc))
            devices = []

        labels: list[str] = []
        self.device_map.clear()
        for device in devices:
            label = f"[{device.index}] {device.name}"
            labels.append(label)
            self.device_map[label] = device.index

        self.device_box["values"] = labels
        if labels:
            self.device_var.set(labels[0])
        else:
            self.device_var.set("")

        if source == AudioSource.SYSTEM:
            self.hint_var.set(
                "System mode usually requires a loopback input (BlackHole/Loopback/Stereo Mix)."
            )
        else:
            self.hint_var.set("Microphone mode listens from your selected input device.")

    def _start(self) -> None:
        if self.service.running:
            return
        selected = self.device_var.get()
        if not selected or selected not in self.device_map:
            messagebox.showwarning("No device", "Please select an input device first.")
            return

        scope = RecognitionScope(self.scope_var.get())
        source = AudioSource(self.mode_var.get())
        config = LiveConfig(
            scope=scope,
            source=source,
            device_index=self.device_map[selected],
            sample_rate=self.sample_rate,
            db_path=self.db_path,
        )
        try:
            self.service.start(config)
        except Exception as exc:
            messagebox.showerror("Start failed", str(exc))
            return

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status("Running")
        self._append_log("service started")

    def _stop(self) -> None:
        self.service.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._set_status("Stopped")
        self._append_log("service stopped")

    def _drain_updates(self) -> None:
        while True:
            try:
                update = self.service.updates.get_nowait()
            except queue.Empty:
                break

            if update.error:
                self._append_log(f"error: {update.error}")
                self._set_status("Error")

            self._render_event(update.event)
            self._render_speakers(update.speakers)

        self.root.after(120, self._drain_updates)

    def _render_event(self, event: RecognitionEvent) -> None:
        state = "Unknown"
        speaker = "-"
        confidence = "-"
        active_speaker_id: int | None = None
        active_confidence: float | None = None

        if event.event_type == EventType.MATCH:
            state = "Matched Speaker"
            speaker = event.speaker_name or "-"
            if event.confidence is not None:
                confidence = f"{event.confidence:.2f}%"
                active_confidence = event.confidence
            active_speaker_id = event.speaker_id
        elif event.event_type == EventType.NEW_SPEAKER:
            state = "New Speaker Registered"
            speaker = event.speaker_name or "-"
            if event.confidence is not None:
                confidence = f"{event.confidence:.2f}%"
                active_confidence = event.confidence
            active_speaker_id = event.speaker_id
        elif event.event_type == EventType.UNKNOWN_SPEECH:
            state = "Human Speech (Unknown)"
        elif event.event_type == EventType.NOISE:
            if event.details == "error":
                state = "Error"
            else:
                state = "Background Noise"
        elif event.event_type == EventType.SILENCE:
            state = "Silence"

        self._active_speaker_id = active_speaker_id
        self._active_confidence = active_confidence
        self.current_var.set(speaker)
        self.confidence_var.set(confidence)
        self._set_status(state)

        detail = event.details if event.details else ""
        detail_part = f" [{detail}]" if detail else ""
        score = f"{event.score:.4f}" if event.score is not None else "-"
        conf = f"{event.confidence:.2f}%" if event.confidence is not None else "-"
        self._append_log(
            f"{event.event_type.value:<14} speaker={speaker:<8} score={score:<7} "
            f"confidence={conf:<8} source={event.source}{detail_part}"
        )

    def _render_speakers(self, speakers: list[Speaker]) -> None:
        self.tree.delete(*self.tree.get_children())
        for speaker in speakers:
            is_active = speaker.id is not None and speaker.id == self._active_speaker_id
            tag = ("active",) if is_active else ()
            conf = "-"
            if is_active and self._active_confidence is not None:
                conf = f"{self._active_confidence:.2f}%"
            speaker_id = speaker.id if speaker.id is not None else "-"
            self.tree.insert(
                "",
                tk.END,
                values=(speaker_id, speaker.name, speaker.sample_count, conf),
                tags=tag,
            )

    def _set_status(self, value: str) -> None:
        self.status_var.set(value)

    def _append_log(self, line: str) -> None:
        self.log_box.config(state=tk.NORMAL)
        self.log_box.insert(tk.END, line + "\n")
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    def _on_close(self) -> None:
        self._stop()
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voiceprint recognition GUI.")
    parser.add_argument(
        "--db-path",
        default=str(Path("data") / "speakers.db"),
        help="Path to persistent SQLite database used in global scope.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Capture sample rate.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if tk is None:
        raise RuntimeError(
            "Tkinter is unavailable in this Python runtime. "
            "Use web frontend: PYTHONPATH=src python -m voice_recognition.web_frontend"
        )
    root = tk.Tk()
    RecognitionApp(root=root, db_path=Path(args.db_path), sample_rate=args.sample_rate)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
