# Speaker Recognition Demo Project

This is a locally run real-time speaker recognition project that provides browser-based operation and can be used without command-line interaction.

## Architecture Overview (Read This First)

```text
[Audio Input: Microphone / System Playback]
                |
                v
[Audio Capture and Chunking: live_service.py]
                |
                v
[Voice Activity Detection (VAD): signal_processor.py]
      | Silence          | Noise             | Human Voice
      v                  v                   v
[SILENCE event]     [NOISE event]     [Speaker Embedding Extraction: WavLM + ECAPA -> embedding]
                                                |
                                                v
                              [Similarity Matching: matcher.py (cosine + AS-Norm)]
                                    | Hit                        | Miss
                                    v                            v
                [MATCH (known speaker) and update prototype]   [Auto-enrollment decision: enrollment.py]
                                                                    | Pass        | Fail
                                                                    v             v
                                                            [Add new speaker (新人N)] [UNKNOWN_SPEECH]

All events are ultimately written to:
[SQLite: speakers / events] -> [Real-time Web UI display: web_frontend.py]
```

It is recommended to first go through the diagram above, then read "Core Capabilities" and "Project Structure" below. That will make it easier to map the implementation details.

Core capabilities:
- Real-time distinction between `silence / noise / human voice`.
- Automatically enroll stable new speakers (for example, `新人1`, `新人2`).
- Perform real-time matching for enrolled speakers and display confidence scores.
- Use `WavLM + ECAPA` fused embeddings by default.
- Use a `global` persistent database (SQLite, retained after restart) by default.
- Provide an automatic tuning script that generates `data/tuning.json` and loads it automatically at startup.

Related documents:
- `/Users/shirong/Downloads/voice_recognition/docs/requirements.md`
- `/Users/shirong/Downloads/voice_recognition/docs/design.md`

## 1. Environment Requirements

- Python `>= 3.10`
- macOS or Windows
- Internet access is required on first run to download models (they will be cached in `data/models`)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you encounter `numpy has no attribute dtypes`, make sure the dependency versions come from this repository's `requirements.txt` (`numpy>=1.26,<2.0` and `transformers==4.41.2`).

## 2. Startup

Start from the command line:

```bash
PYTHONPATH=src python -m voice_recognition.web_frontend
```

You can also double-click:
- macOS: `/Users/shirong/Downloads/voice_recognition/start_app.command`
- Windows: `/Users/shirong/Downloads/voice_recognition/start_app.bat`

Optional parameters:

```bash
PYTHONPATH=src python -m voice_recognition.web_frontend --port 8765 --db-path data/speakers.db
```

Default address after startup: `http://127.0.0.1:8765/`

## 3. Page Usage Instructions

The page has been fully localized into Chinese, and diagnostic information is shown by default.

Basic flow:
1. Select the input source: `麦克风` or `系统回放`.
2. Select the device (or keep `自动选择`).
3. Click `开始识别`.
4. During recognition, you can see: status, current speaker, confidence score, and enrolled speaker count.
5. Click `停止识别` when finished.
6. If you need to clear the database, click `清空名单`.

Additional features in system playback mode:
- `一键配置系统回放`: triggers the automatic configuration assistant.
- `重启电脑（可选）`: some macOS drivers require a restart before taking effect after installation.

## 4. Automatic Tuning (Recommended)

Automatically download public speech data, calculate thresholds, and output them to `data/tuning.json`:

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.auto_tune
```

Customize the data scale:

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.auto_tune \
  --speakers 10 \
  --samples-per-speaker 6 \
  --min-seconds 3.0 \
  --max-seconds 12.0
```

When the service starts, it will automatically read `data/tuning.json` and override the default thresholds.

## 5. Project Structure (Architecture Overview)

Data flow:
1. `web_frontend.py`: provides the web page and HTTP API.
2. `web_controller.py`: receives frontend operations, constructs `LiveConfig`, and starts the service.
3. `live_service.py`: starts the audio stream, asynchronous processing queue, and recognition engine.
4. `audio/signal_processor.py`: VAD + embedding extraction (WavLM/ECAPA fusion).
5. `recognition/engine.py`: matching, soft matching, auto-enrollment, prototype updating.
6. `recognition/matcher.py`: AS-Norm + calibrated scores + threshold decision.
7. `storage/sqlite_repository.py`: global persistent speaker database (thread-safe access).

Key directories:
- `/Users/shirong/Downloads/voice_recognition/src/voice_recognition/audio`
- `/Users/shirong/Downloads/voice_recognition/src/voice_recognition/recognition`
- `/Users/shirong/Downloads/voice_recognition/src/voice_recognition/storage`
- `/Users/shirong/Downloads/voice_recognition/src/voice_recognition/evaluation`

## 6. macOS Automation Capabilities and Boundaries

What can currently be automated:
- Automatically detect whether the loopback device is available.
- Automatically install `blackhole-2ch` when Homebrew is detected.
- Automatically open `Audio MIDI Setup` and display next-step instructions on the page.

What cannot currently be made 100% fully automatic:
- Automatically create a "Multi-Output Device (headphones + BlackHole)" and keep system routing stable over time.
- Main reason: macOS does not provide a stable public API for audio device topology management, and cross-version GUI automation is not reliable.

Conclusion:
- `BlackHole installation` can basically be automated.
- `Multi-output device creation and routing details` are still recommended for manual confirmation (but the page already provides guidance and diagnostics).

## 7. Can It Run on Windows?

Yes, but there are two cases:
- Microphone recognition: usually works directly.
- System playback recognition: requires a loopback input device such as `Stereo Mix` or `VB-Cable`.

The built-in Windows configuration assistant will open the sound settings page and guide you to enable the relevant devices.

## 8. Offline Evaluation (Optional)

If you have your own dataset (directory structure `dataset/<speaker>/*.wav`):

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.offline_eval \
  --dataset /absolute/path/to/your_dataset \
  --profile balanced
```

Optionally fit Platt calibration:

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.offline_eval \
  --dataset /absolute/path/to/your_dataset \
  --profile balanced \
  --score-backend asnorm \
  --fit-platt
```

## 9. Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 10. Frequently Asked Questions

The device list is empty in `system` mode:
- This means there is currently no available loopback input device.
- Please first run `一键配置系统回放` on the page, or manually install/enable BlackHole, Stereo Mix, or VB-Cable.

The same person is split into multiple IDs:
- Stop the service first, clear the list, then start again.
- Run automatic tuning once to generate `data/tuning.json`.
- Prefer stable input devices and avoid low-bitrate Bluetooth hands-free links.

Real-time lag or overflow:
- Prefer the default parameters and do not manually reduce the chunk size.
- In system playback mode, reduce high-load background applications.
