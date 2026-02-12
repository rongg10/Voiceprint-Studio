from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from voice_recognition.audio import AudioSource
from voice_recognition.core.models import RecognitionScope
from voice_recognition.web_controller import WebController


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Voiceprint Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@700;800&family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #f7f4ef;
      --card: #ffffff;
      --border: #e7ded1;
      --text: #1b1b1b;
      --muted: #5f5b55;
      --accent: #d4492a;
      --accent-2: #1e8c7a;
      --warn: #b91c1c;
      --highlight: #fff1d6;
      --shadow: 0 18px 40px rgba(27, 24, 20, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "PingFang SC", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(circle at 6% 12%, #ffe7d7 0, transparent 45%),
        radial-gradient(circle at 96% 0%, #dff3ee 0, transparent 36%),
        radial-gradient(circle at 80% 80%, #fff2da 0, transparent 45%),
        var(--bg);
      color: var(--text);
    }
    .wrap { max-width: 1080px; margin: 24px auto; padding: 0 14px; }
    .title { font-family: "Fraunces", "PingFang SC", serif; font-size: 34px; font-weight: 800; margin: 0; letter-spacing: -0.02em; }
    .subtitle { margin: 6px 0 14px 0; color: var(--muted); font-size: 15px; }
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: var(--shadow);
      margin-bottom: 12px;
    }
    .controls {
      display: grid;
      grid-template-columns: 120px 1fr 100px 1fr 140px;
      gap: 10px;
      align-items: center;
    }
    label { color: var(--muted); font-size: 13px; }
    select, button, input[type="number"] {
      height: 42px;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0 12px;
      font-size: 15px;
      background: #fff;
    }
    input[type="number"] {
      width: 100%;
    }
    button {
      cursor: pointer;
      background: #f6f2ec;
      transition: background .15s ease, transform .15s ease;
    }
    button:hover { background: #efe7dc; transform: translateY(-1px); }
    button.primary {
      background: var(--accent);
      color: #fff;
      border: none;
    }
    button.secondary {
      background: #fff;
      border: 1px solid var(--border);
    }
    button.stop {
      background: #fee2e2;
      border-color: #fecaca;
      color: #991b1b;
    }
    .status-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 8px;
    }
    .status-card {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      background: linear-gradient(180deg, #fffdf9 0, #f8f2ea 100%);
    }
    .status-label { font-size: 12px; color: var(--muted); }
    .status-value { font-size: 20px; font-weight: 800; margin-top: 4px; }
    .hint { color: var(--muted); font-size: 13px; margin-top: 8px; }
    .error { color: var(--warn); font-size: 13px; margin-top: 6px; min-height: 18px; }
    .split {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      padding: 10px 8px;
      text-align: left;
    }
    tr.active { background: var(--highlight); }
    .log {
      background: #f6f1ea;
      color: #3b342c;
      border-radius: 12px;
      border: 1px dashed #e1d6c7;
      padding: 10px;
      min-height: 240px;
      max-height: 240px;
      overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }
    details summary {
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 6px;
    }
    @media (max-width: 980px) {
      .controls { grid-template-columns: 1fr 1fr; }
      .split { grid-template-columns: 1fr; }
      .status-grid { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">Voiceprint Studio</h1>
    <p class="subtitle">一键开始识别当前说话人，自动完成入库与高亮展示。</p>
    <div class="panel">
      <div class="controls">
        <label>输入源</label>
        <select id="source">
          <option value="microphone">microphone</option>
          <option value="system">system</option>
        </select>
        <label>设备</label>
        <select id="device"></select>
        <button id="refresh">刷新设备</button>
        <button id="start" class="primary">开始</button>
      </div>
      <div class="controls" style="grid-template-columns: 140px 140px 1fr; margin-top: 10px;">
        <button id="stop" class="stop">停止</button>
        <button id="reset" class="secondary">清空名单</button>
        <div class="hint" id="hint"></div>
      </div>
      <div class="error" id="error"></div>
      <div class="hint" id="setupNotice"></div>
    </div>

    <div class="panel status-grid">
      <div class="status-card">
        <div class="status-label">当前状态</div>
        <div class="status-value" id="status">Idle</div>
      </div>
      <div class="status-card">
        <div class="status-label">当前说话人</div>
        <div class="status-value" id="speaker">-</div>
      </div>
      <div class="status-card">
        <div class="status-label">置信度</div>
        <div class="status-value" id="confidence">-</div>
      </div>
      <div class="status-card">
        <div class="status-label">已注册人数</div>
        <div class="status-value" id="speakerCount">0</div>
      </div>
    </div>

    <div class="split">
      <div class="panel">
        <h3 style="margin: 0 0 8px 0;">入库声纹列表</h3>
        <table>
          <thead>
            <tr><th>ID</th><th>名称</th><th>样本数</th><th>置信度</th></tr>
          </thead>
          <tbody id="speakerRows"></tbody>
        </table>
      </div>
      <div class="panel">
        <details>
          <summary>诊断信息（可选）</summary>
          <div class="log" id="events"></div>
        </details>
      </div>
    </div>
  </div>

  <script>
    const sourceEl = document.getElementById('source');
    const deviceEl = document.getElementById('device');
    const statusEl = document.getElementById('status');
    const speakerEl = document.getElementById('speaker');
    const confidenceEl = document.getElementById('confidence');
    const speakerCountEl = document.getElementById('speakerCount');
    const rowsEl = document.getElementById('speakerRows');
    const eventsEl = document.getElementById('events');
    const errorEl = document.getElementById('error');
    const hintEl = document.getElementById('hint');
    const setupNoticeEl = document.getElementById('setupNotice');
    const startBtn = document.getElementById('start');
    const stopBtn = document.getElementById('stop');
    const resetBtn = document.getElementById('reset');
    const refreshBtn = document.getElementById('refresh');

    function setHint() {
      if (sourceEl.value === 'system') {
        hintEl.textContent = 'system 模式：请把系统输出切到“多输出设备(耳机+BlackHole)”，本程序选择 BlackHole 输入';
      } else {
        hintEl.textContent = 'microphone 模式直接监听麦克风输入';
      }
    }

    async function refreshDevices() {
      setHint();
      const source = sourceEl.value;
      const resp = await fetch(`/api/devices?source=${encodeURIComponent(source)}`);
      const data = await resp.json();
      deviceEl.innerHTML = '';
      const autoOpt = document.createElement('option');
      autoOpt.value = '';
      autoOpt.textContent = '自动选择';
      deviceEl.appendChild(autoOpt);
      startBtn.disabled = false;
      errorEl.textContent = '';
      for (const dev of data.devices) {
        const opt = document.createElement('option');
        opt.value = String(dev.index);
        opt.textContent = `[${dev.index}] ${dev.name}`;
        deviceEl.appendChild(opt);
      }
      if (data.devices.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '无可用输入设备';
        deviceEl.appendChild(opt);
        startBtn.disabled = true;
        if (source === 'system') {
          errorEl.textContent = '未检测到系统回环输入设备。请先安装/配置 BlackHole 或 Stereo Mix。';
        }
      } else if (source === 'microphone' && data.devices.length > 1) {
        const autoResp = await fetch('/api/auto-select-device', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({source})
        });
        const autoData = await autoResp.json();
        if (autoResp.ok && autoData.ok && autoData.device) {
          deviceEl.value = String(autoData.device.index);
        }
      }
    }

    async function start() {
      errorEl.textContent = '';
      const deviceIndex = deviceEl.value === '' ? '' : parseInt(deviceEl.value, 10);
      if (deviceEl.value !== '' && Number.isNaN(deviceIndex)) {
        errorEl.textContent = '请先选择输入设备';
        return;
      }
      const payload = {
        source: sourceEl.value,
        scope: 'global',
        device: deviceIndex,
      };
      const resp = await fetch('/api/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok) {
        errorEl.textContent = data.error || '启动失败';
      }
    }

    async function stop() {
      await fetch('/api/stop', {method: 'POST'});
    }

    async function resetLibrary() {
      errorEl.textContent = '';
      const payload = { scope: 'global' };
      const resp = await fetch('/api/reset', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok) {
        errorEl.textContent = data.error || '清空失败';
      }
    }

    function renderSpeakers(speakers) {
      rowsEl.innerHTML = '';
      for (const sp of speakers) {
        const tr = document.createElement('tr');
        if (sp.active) tr.classList.add('active');
        tr.innerHTML = `<td>${sp.id ?? '-'}</td><td>${sp.name}</td><td>${sp.samples}</td><td>${sp.confidence}</td>`;
        rowsEl.appendChild(tr);
      }
    }

    function renderEvents(lines) {
      eventsEl.textContent = lines.join('\\n');
      eventsEl.scrollTop = eventsEl.scrollHeight;
    }

    async function pollState() {
      try {
      const resp = await fetch('/api/state');
      const state = await resp.json();
        statusEl.textContent = state.status;
        speakerEl.textContent = state.currentSpeaker;
        confidenceEl.textContent = state.confidence;
      speakerCountEl.textContent = (state.speakers || []).length;
      renderSpeakers(state.speakers || []);
      renderEvents(state.events || []);
      errorEl.textContent = state.error || '';
      setupNoticeEl.textContent = state.setupNotice || '';
      startBtn.disabled = !!state.running;
      stopBtn.disabled = !state.running;
    } catch (err) {
      errorEl.textContent = '与本地服务通信失败';
    }
    }

    sourceEl.addEventListener('change', refreshDevices);
    refreshBtn.addEventListener('click', refreshDevices);
    startBtn.addEventListener('click', start);
    stopBtn.addEventListener('click', stop);
    resetBtn.addEventListener('click', resetLibrary);

    setHint();
    refreshDevices();
    pollState();
    setInterval(pollState, 500);
  </script>
</body>
</html>
"""


class FrontendServer:
    def __init__(self, host: str, port: int, db_path: Path, sample_rate: int) -> None:
        self.host = host
        self.port = port
        self.controller = WebController(db_path=db_path, sample_rate=sample_rate)
        self._httpd: ThreadingHTTPServer | None = None

    def serve(self, open_browser: bool) -> None:
        handler = self._build_handler()
        self._httpd = self._bind_server(handler)
        url = f"http://{self.host}:{self.port}/"
        print(f"Frontend started at: {url}")
        if open_browser:
            threading.Thread(target=lambda: webbrowser.open(url, new=2), daemon=True).start()
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _bind_server(self, handler: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
        last_error: Exception | None = None
        for offset in range(0, 20):
            try_port = self.port + offset
            try:
                server = ThreadingHTTPServer((self.host, try_port), handler)
                self.port = try_port
                return server
            except OSError as exc:
                last_error = exc
                continue
        if last_error is None:
            last_error = OSError("Unknown bind failure.")
        raise RuntimeError(f"Unable to bind frontend server near port {self.port}: {last_error}")

    def shutdown(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        self.controller.shutdown()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        controller = self.controller

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._respond_html(HTML)
                    return
                if parsed.path == "/api/devices":
                    query = parse_qs(parsed.query)
                    source_value = query.get("source", [AudioSource.MICROPHONE.value])[0]
                    try:
                        source = AudioSource(source_value)
                        devices = controller.list_devices(source)
                        self._respond_json({"devices": devices})
                    except Exception as exc:
                        self._respond_json({"error": str(exc), "devices": []}, status=HTTPStatus.BAD_REQUEST)
                    return
                if parsed.path == "/api/state":
                    self._respond_json(controller.snapshot())
                    return
                if parsed.path == "/api/diagnostics":
                    try:
                        self._respond_json(controller.diagnostics())
                    except Exception as exc:
                        self._respond_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                if parsed.path == "/api/setup-loopback/status":
                    self._respond_json(controller.setup_loopback_status())
                    return
                self._respond_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/api/start":
                    payload = self._read_json()
                    try:
                        source = AudioSource(str(payload.get("source", AudioSource.MICROPHONE.value)))
                        scope = RecognitionScope(str(payload.get("scope", RecognitionScope.GLOBAL.value)))
                        raw_device = payload.get("device")
                        if raw_device is None or str(raw_device).strip() == "":
                            auto = controller.auto_select_device(source=source)
                            if not bool(auto.get("ok")):
                                raise ValueError(str(auto.get("error") or "当前模式没有可用输入设备。"))
                            chosen = auto.get("device")
                            if not isinstance(chosen, dict) or "index" not in chosen:
                                raise ValueError("自动选择输入设备失败，请手动刷新设备后重试。")
                            device_index = int(chosen["index"])
                        else:
                            device_index = int(raw_device)
                        controller.start(
                            scope=scope,
                            source=source,
                            device_index=device_index,
                            tuning={
                                "scoreBackend": payload.get("scoreBackend"),
                                "asnormTopK": payload.get("asnormTopK"),
                                "asnormMinCohort": payload.get("asnormMinCohort"),
                                "asnormBlend": payload.get("asnormBlend"),
                                "calibrationScale": payload.get("calibrationScale"),
                                "calibrationBias": payload.get("calibrationBias"),
                                "matchThreshold": payload.get("matchThreshold"),
                                "newSpeakerThreshold": payload.get("newSpeakerThreshold"),
                                "minMargin": payload.get("minMargin"),
                                "minSegments": payload.get("minSegments"),
                                "minClusterSimilarity": payload.get("minClusterSimilarity"),
                            },
                        )
                        self._respond_json({"ok": True})
                    except Exception as exc:
                        self._respond_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                if parsed.path == "/api/stop":
                    controller.stop()
                    self._respond_json({"ok": True})
                    return
                if parsed.path == "/api/reset":
                    payload = self._read_json()
                    try:
                        scope = RecognitionScope(str(payload.get("scope", RecognitionScope.GLOBAL.value)))
                        controller.reset_library(scope=scope)
                        self._respond_json({"ok": True})
                    except Exception as exc:
                        self._respond_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                if parsed.path == "/api/setup-loopback":
                    try:
                        result = controller.start_setup_loopback()
                        self._respond_json(result, status=HTTPStatus.OK)
                    except Exception as exc:
                        self._respond_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                if parsed.path == "/api/auto-select-device":
                    payload = self._read_json()
                    try:
                        source = AudioSource(str(payload.get("source", AudioSource.MICROPHONE.value)))
                        result = controller.auto_select_device(source=source)
                        status = HTTPStatus.OK if bool(result.get("ok")) else HTTPStatus.BAD_REQUEST
                        self._respond_json(result, status=status)
                    except Exception as exc:
                        self._respond_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                if parsed.path == "/api/reboot":
                    try:
                        result = controller.request_system_reboot()
                        status = HTTPStatus.OK if bool(result.get("ok")) else HTTPStatus.BAD_REQUEST
                        self._respond_json(result, status=status)
                    except Exception as exc:
                        self._respond_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                if parsed.path == "/api/probe":
                    payload = self._read_json()
                    try:
                        source = AudioSource(str(payload.get("source", AudioSource.MICROPHONE.value)))
                        raw_device = payload.get("device")
                        if raw_device is None or str(raw_device).strip() == "":
                            raise ValueError("缺少 device 参数。")
                        device_index = int(raw_device)
                        result = controller.probe_device(source=source, device_index=device_index)
                        status = HTTPStatus.OK if bool(result.get("ok")) else HTTPStatus.BAD_REQUEST
                        self._respond_json(result, status=status)
                    except Exception as exc:
                        self._respond_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._respond_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

            def log_message(self, fmt: str, *args) -> None:  # noqa: D401
                return

            def _read_json(self) -> dict[str, object]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                return json.loads(raw.decode("utf-8"))

            def _respond_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _respond_html(self, html: str) -> None:
                data = html.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Voiceprint recognition web frontend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db-path", default=str(Path("data") / "speakers.db"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--no-open-browser", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server = FrontendServer(
        host=args.host,
        port=args.port,
        db_path=Path(args.db_path),
        sample_rate=args.sample_rate,
    )
    server.serve(open_browser=not args.no_open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
