# Voice Recognition Demo (Button UI)

This project now provides a local frontend app (browser UI with buttons) that can:

- listen to microphone audio or system-loopback audio;
- separate silence / background noise / human speech;
- auto-enroll new speakers as `新人1`, `新人2`, ...;
- match enrolled speakers in real time and show confidence;
- support dual speaker-library scopes: `global` (persistent SQLite) and `session` (in-memory).
- run neural pipeline: `Silero VAD + ECAPA-TDNN` (SpeechBrain).
- use modern scoring pipeline in live mode: `adaptive score normalization (AS-Norm style) + logistic calibration`.

Requirement and design docs:

- `/Users/shirong/Downloads/voice_recognition/docs/requirements.md`
- `/Users/shirong/Downloads/voice_recognition/docs/design.md`

## 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

首次运行会自动下载预训练模型到 `data/models`（需要联网，约数百 MB；默认使用 WavLM + ECAPA 融合，下载更大）。
如设备支持 MPS（Apple Silicon），程序会自动使用 MPS 加速推理。
如遇 `numpy has no attribute dtypes` 错误，请确保依赖已更新到本项目 requirements 里的版本（`numpy>=1.26,<2.0` + `transformers==4.41.2`）。

## 2. Start the frontend app

```bash
PYTHONPATH=src python -m voice_recognition.web_frontend
```

Or just double click:

- macOS: `/Users/shirong/Downloads/voice_recognition/start_app.command`
- Windows: `/Users/shirong/Downloads/voice_recognition/start_app.bat`

Optional parameters:

```bash
PYTHONPATH=src python -m voice_recognition.web_frontend --port 8765 --db-path data/speakers.db
```

After startup, open `http://127.0.0.1:8765` if browser did not auto-open.

## 3. Use the button UI

1. Select `Input Source`: `microphone` or `system`.
2. Select `Device`（`system` 下请选择 `BlackHole 2ch` 这类回环输入设备）。
3. Click `刷新设备`（切换输入源后建议先刷新一次）。
4. Click `开始`.
5. Speak or play audio:
   - noise/music -> state shows `Background Noise`;
   - unknown speech -> auto-collects segments;
   - stable new speaker -> auto-registers as `新人N`;
   - known speaker -> speaker row is highlighted with confidence.
6. Click `停止`.
7. 需要清库时，先停止，再点 `清空名单`。

说明：默认启用 `AS-Norm` 评分后端；默认 embedding 为 `WavLM + ECAPA` 融合。
说明：高级 embedding 仍可通过 API 传入 `embeddingModels`（如 `hf:microsoft/wavlm-base-plus-sv` 或 `speechbrain/spkrec-ecapa-voxceleb`）与 `embeddingFusion`（`average` / `concat`）。

说明：开始后前 2~3 秒会做噪声地板校准，这段时间可能只显示 `Silence`，属于正常行为。

## 4.1 Troubleshooting: 全部识别成一个人

如果你之前已经跑过并把全局库污染了（不同人被错误并到同一ID）：

1. 先点页面 `停止`
2. 点 `清空库`（在 `global` 模式下会清空 `data/speakers.db`）
3. 重新点 `开始监听`

建议先用 `session` 模式验证，再切回 `global`。

如果 `system + session` 在静音时仍显示有人说话：

1. 确认选中的确是 loopback 输入设备，而不是带麦克风混入的复合设备。
2. 启动后静置 3 秒等校准完成，再观察状态。
3. 点 `清空库` 后重新开始，避免旧错误声纹干扰。

## 5. System audio mode notes

- macOS: usually needs a loopback virtual device (for example BlackHole/Loopback) and selecting it as input in this app.
- macOS: for hearing and capture together, set macOS Output to a Multi-Output Device that includes both `BlackHole 2ch` and your headphones.
- Windows: typically uses Stereo Mix or WASAPI loopback-compatible input.
- If no dedicated system-loopback device exists, install/configure one first.
- 如果 `system` 下设备列表为空，说明还没有可用回环输入设备；此时无法直接监听耳机播放内容。

## 6. Live console mode (optional fallback)

If you want a terminal dashboard:

```bash
PYTHONPATH=src python -m voice_recognition.live_console
```

## 7. CLI frame replay (optional)

```bash
PYTHONPATH=src python -m voice_recognition --scope global --input-json data/sample_frames.json --show-speakers
```

## 8. Run tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 9. Automatic calibration (recommended)

下载一小批公开数据并自动生成阈值（写入 `data/tuning.json`，启动服务时会自动加载）：

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.auto_tune
```

你也可以自定义采样规模：

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.auto_tune \
  --speakers 10 \
  --samples-per-speaker 6 \
  --min-seconds 3.0 \
  --max-seconds 12.0
```

## 10. Offline real-audio evaluation (optional)

准备数据目录（每个说话人一个子目录）：

```text
your_dataset/
  speaker_a/
    a1.wav
    a2.wav
  speaker_b/
    b1.wav
    b2.wav
```

运行评估：

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.offline_eval \
  --dataset /absolute/path/to/your_dataset \
  --profile balanced
```

可选：拟合分数校准参数并输出闭集 Top1 指标

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.offline_eval \
  --dataset /absolute/path/to/your_dataset \
  --profile balanced \
  --score-backend asnorm \
  --fit-platt
```

报告会输出：
- `verification`: `EER` / `min_dcf_p01`（验证任务指标）
- `closed_set`: `top1` / `accept_rate`（闭集识别质量）
- `calibration`: `scale` / `bias`（可用于在线打分校准）
- `streaming`: `unknown_file_rate` / `duplicate_label_rate`（流式识别可用性指标）
