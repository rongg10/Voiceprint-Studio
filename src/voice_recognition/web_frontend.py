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
  <title>声纹识别控制台</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700;800&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #f2f6fb;
      --surface: #ffffff;
      --line: #d5e0ea;
      --text: #142436;
      --muted: #5a7388;
      --brand: #1b67d5;
      --brand-hover: #1258ba;
      --warn-bg: #fff6ed;
      --warn-line: #f2d0ba;
      --danger-bg: #fff1f5;
      --danger-line: #f0c8d4;
      --ok: #1e9a62;
      --shadow: 0 12px 28px rgba(20, 36, 54, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(180deg, #edf3fa 0%, #f7fbff 100%);
      color: var(--text);
      font-family: "IBM Plex Sans", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    .container {
      max-width: 1160px;
      margin: 22px auto 30px;
      padding: 0 14px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--surface);
      box-shadow: var(--shadow);
      padding: 14px;
      margin-bottom: 12px;
    }
    .header-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
    }
    .title {
      margin: 0;
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.02em;
      font-family: "Noto Sans SC", "IBM Plex Sans", sans-serif;
    }
    .subtitle {
      margin: 6px 0 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }
    .header-actions {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .run-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid #cad8eb;
      border-radius: 999px;
      background: #f2f7fd;
      color: #2e4d69;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #93a8bd;
    }
    .dot.live {
      background: var(--ok);
      box-shadow: 0 0 0 6px rgba(30, 154, 98, 0.15);
    }
    .help-btn {
      width: 40px;
      height: 40px;
      border-radius: 999px;
      border: 1px solid #c9d7e7;
      background: #ffffff;
      color: #22445f;
      font-size: 20px;
      font-weight: 700;
      cursor: pointer;
      transition: all .12s ease;
    }
    .help-btn:hover {
      transform: translateY(-1px);
      border-color: #b8cbe0;
      background: #f5faff;
    }
    .view {
      display: none;
    }
    .view.active {
      display: block;
    }
    h2 {
      margin: 0 0 10px 0;
      font-size: 16px;
      font-weight: 800;
    }
    .field-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .field {
      display: grid;
      gap: 6px;
    }
    .field label {
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    select, button {
      height: 40px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
      color: #1b374d;
      font-size: 14px;
      font-weight: 600;
      padding: 0 10px;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
    }
    button {
      cursor: pointer;
      transition: all .12s ease;
    }
    button:hover {
      transform: translateY(-1px);
      border-color: #becddd;
    }
    .action-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(100px, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .btn-primary {
      background: var(--brand);
      border-color: var(--brand);
      color: #fff;
    }
    .btn-primary:hover {
      background: var(--brand-hover);
      border-color: var(--brand-hover);
    }
    .btn-warn {
      background: var(--warn-bg);
      border-color: var(--warn-line);
      color: #8d4b32;
    }
    .btn-danger {
      background: var(--danger-bg);
      border-color: var(--danger-line);
      color: #8f2b43;
    }
    .hint {
      margin-top: 8px;
      color: #35556d;
      font-size: 13px;
      line-height: 1.55;
    }
    .error {
      margin-top: 8px;
      min-height: 18px;
      color: #b42e46;
      font-size: 13px;
      font-weight: 600;
    }
    .hero-grid {
      display: grid;
      grid-template-columns: 1.2fr 1.2fr 0.8fr 0.8fr;
      gap: 10px;
    }
    .hero-card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fbfdff;
      padding: 12px;
    }
    .hero-label {
      font-size: 12px;
      color: var(--muted);
      font-weight: 700;
      letter-spacing: 0.01em;
    }
    .hero-value {
      margin-top: 6px;
      font-size: 20px;
      font-weight: 800;
      line-height: 1.25;
      word-break: break-word;
    }
    .hero-value.big {
      font-size: 32px;
      letter-spacing: -0.02em;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      font-size: 14px;
    }
    th, td {
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }
    th {
      background: #f2f7fd;
      color: #35566f;
      font-weight: 700;
    }
    tbody tr:last-child td { border-bottom: none; }
    tr.active { background: #fff8ea; }
    .guide-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
    }
    .guide-top-buttons {
      display: inline-flex;
      gap: 8px;
    }
    .tutorial {
      margin: 0;
      padding-left: 18px;
      line-height: 1.8;
      color: #2a475d;
      font-size: 14px;
    }
    .guide-action-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(120px, 1fr));
      gap: 8px;
    }
    .diag-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }
    .diag-item {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 9px;
      background: #f8fbff;
    }
    .diag-key {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
    }
    .diag-val {
      margin-top: 4px;
      font-size: 14px;
      font-weight: 700;
      color: #1b4662;
    }
    .log {
      margin-top: 8px;
      border: 1px solid #243b4d;
      border-radius: 10px;
      background: #101f2d;
      color: #dcecff;
      min-height: 260px;
      max-height: 260px;
      overflow: auto;
      padding: 10px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    @media (max-width: 1020px) {
      .hero-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 840px) {
      .field-grid { grid-template-columns: 1fr; }
      .action-grid { grid-template-columns: 1fr 1fr; }
      .guide-action-grid { grid-template-columns: 1fr; }
      .diag-grid { grid-template-columns: 1fr; }
      .title { font-size: 24px; }
      .header-row { flex-direction: column; }
    }
    @media (max-width: 600px) {
      .hero-grid { grid-template-columns: 1fr; }
      .guide-top { flex-direction: column; align-items: flex-start; }
      .guide-top-buttons { width: 100%; display: grid; grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <section class="panel">
      <div class="header-row">
        <div>
          <h1 class="title">声纹识别控制台</h1>
          <p class="subtitle">主界面只保留识别核心；配置、教程和诊断统一放到问号页。</p>
        </div>
        <div class="header-actions">
          <div class="run-chip"><span class="dot" id="runDot"></span><span id="runState">未采集</span></div>
          <button id="helpBtn" class="help-btn" title="配置与教程">?</button>
        </div>
      </div>
    </section>

    <section id="mainView" class="view active">
      <section class="panel">
        <h2>识别控制</h2>
        <div class="field-grid">
          <div class="field">
            <label for="source">输入源</label>
            <select id="source">
              <option value="microphone">麦克风</option>
              <option value="system">系统回放</option>
            </select>
          </div>
          <div class="field">
            <label for="device">输入设备</label>
            <select id="device"></select>
          </div>
        </div>
        <div class="action-grid">
          <button id="refresh">刷新设备</button>
          <button id="start" class="btn-primary">开始识别</button>
          <button id="stop" class="btn-warn">停止识别</button>
          <button id="reset" class="btn-danger">清空名单</button>
        </div>
        <div class="hint" id="hint"></div>
        <div class="error" id="mainError"></div>
      </section>

      <section class="panel">
        <div class="hero-grid">
          <div class="hero-card">
            <div class="hero-label">系统状态</div>
            <div class="hero-value big" id="status">待机</div>
          </div>
          <div class="hero-card">
            <div class="hero-label">当前说话人</div>
            <div class="hero-value big" id="speaker">-</div>
          </div>
          <div class="hero-card">
            <div class="hero-label">识别置信度</div>
            <div class="hero-value" id="confidence">-</div>
          </div>
          <div class="hero-card">
            <div class="hero-label">入库人数</div>
            <div class="hero-value" id="speakerCount">0</div>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>说话人列表</h2>
        <table>
          <thead>
            <tr><th>ID</th><th>名称</th><th>样本数</th><th>当前置信度</th></tr>
          </thead>
          <tbody id="speakerRows"></tbody>
        </table>
      </section>
    </section>

    <section id="guideView" class="view">
      <section class="panel">
        <div class="guide-top">
          <h2>配置与教程</h2>
          <div class="guide-top-buttons">
            <button id="enterMainBtn" class="btn-primary">进入识别界面</button>
            <button id="backBtn">返回</button>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>快速教程</h2>
        <ol class="tutorial">
          <li>先点“一键配置系统回放”，等待页面提示完成。</li>
          <li>若提示需要重启，点“重启电脑（可选）”或手动重启。</li>
          <li>回到识别界面，选择输入源和设备，再点“开始识别”。</li>
          <li>首次建议连续说几句完整话，便于稳定入库与匹配。</li>
          <li>同一环境下使用同一设备，识别稳定性更高。</li>
        </ol>
      </section>

      <section class="panel">
        <h2>系统配置</h2>
        <div class="guide-action-grid">
          <button id="setupLoopback">一键配置系统回放</button>
          <button id="reboot">重启电脑（可选）</button>
          <button id="refreshDiag">刷新诊断信息</button>
        </div>
        <div class="hint" id="setupNotice"></div>
        <div class="error" id="guideError"></div>
      </section>

      <section class="panel">
        <h2>诊断信息</h2>
        <div class="diag-grid">
          <div class="diag-item">
            <div class="diag-key">麦克风设备数量</div>
            <div class="diag-val" id="diagMicCount">-</div>
          </div>
          <div class="diag-item">
            <div class="diag-key">系统回环设备数量</div>
            <div class="diag-val" id="diagSystemCount">-</div>
          </div>
          <div class="diag-item">
            <div class="diag-key">BlackHole 驱动状态</div>
            <div class="diag-val" id="diagBlackHole">-</div>
          </div>
          <div class="diag-item">
            <div class="diag-key">当前输入源</div>
            <div class="diag-val" id="diagSource">麦克风</div>
          </div>
        </div>
        <div class="diag-item">
          <div class="diag-key">设备快照</div>
          <div class="diag-val" id="diagDevices">-</div>
        </div>
        <div class="log" id="events"></div>
      </section>
    </section>
  </div>

  <script>
    const mainViewEl = document.getElementById('mainView');
    const guideViewEl = document.getElementById('guideView');
    const helpBtn = document.getElementById('helpBtn');
    const backBtn = document.getElementById('backBtn');
    const enterMainBtn = document.getElementById('enterMainBtn');

    const sourceEl = document.getElementById('source');
    const deviceEl = document.getElementById('device');
    const statusEl = document.getElementById('status');
    const speakerEl = document.getElementById('speaker');
    const confidenceEl = document.getElementById('confidence');
    const speakerCountEl = document.getElementById('speakerCount');
    const rowsEl = document.getElementById('speakerRows');

    const runDotEl = document.getElementById('runDot');
    const runStateEl = document.getElementById('runState');
    const hintEl = document.getElementById('hint');
    const mainErrorEl = document.getElementById('mainError');

    const setupNoticeEl = document.getElementById('setupNotice');
    const guideErrorEl = document.getElementById('guideError');
    const eventsEl = document.getElementById('events');

    const diagMicCountEl = document.getElementById('diagMicCount');
    const diagSystemCountEl = document.getElementById('diagSystemCount');
    const diagBlackHoleEl = document.getElementById('diagBlackHole');
    const diagSourceEl = document.getElementById('diagSource');
    const diagDevicesEl = document.getElementById('diagDevices');

    const refreshBtn = document.getElementById('refresh');
    const startBtn = document.getElementById('start');
    const stopBtn = document.getElementById('stop');
    const resetBtn = document.getElementById('reset');

    const setupBtn = document.getElementById('setupLoopback');
    const rebootBtn = document.getElementById('reboot');
    const refreshDiagBtn = document.getElementById('refreshDiag');

    const statusMap = {
      'Idle': '待机',
      'Starting': '启动中',
      'Stopped': '已停止',
      'Library Reset': '名单已清空',
      'Matched Speaker': '已识别到已知说话人',
      'New Speaker Registered': '已自动入库新说话人',
      'Human Speech (Unknown)': '检测到人声（未识别）',
      'Background Noise': '背景噪声',
      'Silence': '静音',
      'Error': '错误',
    };

    const eventTypeMap = {
      'match': '匹配',
      'new_speaker': '新说话人',
      'unknown_speech': '未知人声',
      'noise': '噪声',
      'silence': '静音',
    };

    function sourceLabel(value) {
      return value === 'system' ? '系统回放' : '麦克风';
    }

    function localizeStatus(value) {
      if (!value) return '-';
      return statusMap[value] || value;
    }

    function localizeEventLine(line) {
      const text = String(line || '');
      if (text.includes('service starting')) return '服务启动中';
      if (text.includes('service stopped')) return '服务已停止';
      if (text.includes('tuning loaded')) return text.replace('tuning loaded', '已加载调优参数');
      if (text.includes('loopback')) {
        return text
          .replace('loopback setup started', '回环配置任务已启动')
          .replace('loopback setup error', '回环配置失败')
          .replace('loopback setup', '回环配置')
          .replace('loopback step', '配置步骤')
          .replace('loopback log', '配置日志');
      }

      let out = text;
      out = out.replace(/speaker=/g, '说话人=');
      out = out.replace(/score=/g, '分数=');
      out = out.replace(/confidence=/g, '置信度=');
      out = out.replace(/source=/g, '来源=');
      out = out.replace(/microphone/g, '麦克风');
      out = out.replace(/system/g, '系统回放');
      for (const [key, val] of Object.entries(eventTypeMap)) {
        out = out.replace(new RegExp(`^${key}\\s*`, 'i'), `${val} `);
      }
      return out;
    }

    function showView(name, updateHash = true) {
      const isGuide = name === 'guide';
      mainViewEl.classList.toggle('active', !isGuide);
      guideViewEl.classList.toggle('active', isGuide);
      if (updateHash) {
        window.location.hash = isGuide ? '#guide' : '#main';
      }
      if (isGuide) {
        refreshDiagnostics();
      }
    }

    function setHint() {
      if (sourceEl.value === 'system') {
        hintEl.textContent = '系统回放模式：建议先到问号页完成回环配置，再返回识别界面。';
      } else {
        hintEl.textContent = '麦克风模式：建议使用内置或有线麦克风，避免蓝牙免提低增益输入。';
      }
      diagSourceEl.textContent = sourceLabel(sourceEl.value);
    }

    function setRunning(running) {
      runStateEl.textContent = running ? '采集中' : '未采集';
      runDotEl.classList.toggle('live', !!running);
    }

    function setError(message) {
      const text = message || '';
      mainErrorEl.textContent = text;
      guideErrorEl.textContent = text;
    }

    async function refreshDiagnostics() {
      try {
        const resp = await fetch('/api/diagnostics');
        const data = await resp.json();
        if (!resp.ok) return;

        diagMicCountEl.textContent = String(data.microphoneCount ?? 0);
        diagSystemCountEl.textContent = String(data.systemCount ?? 0);
        diagBlackHoleEl.textContent = data.blackholeDriverPresent ? '已安装' : '未检测到';

        const micNames = (data.microphoneDevices || []).slice(0, 3).map((item) => item.name).join('、');
        const sysNames = (data.systemDevices || []).slice(0, 3).map((item) => item.name).join('、');
        const micText = micNames ? `麦克风：${micNames}` : '麦克风：无';
        const sysText = sysNames ? `系统回放：${sysNames}` : '系统回放：无';
        diagDevicesEl.textContent = `${micText} ｜ ${sysText}`;
      } catch (err) {
        diagDevicesEl.textContent = '诊断接口暂不可用';
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
      autoOpt.textContent = '自动选择（推荐）';
      deviceEl.appendChild(autoOpt);

      startBtn.disabled = false;
      setError('');

      for (const dev of (data.devices || [])) {
        const opt = document.createElement('option');
        opt.value = String(dev.index);
        opt.textContent = `[${dev.index}] ${dev.name}`;
        deviceEl.appendChild(opt);
      }

      if (!data.devices || data.devices.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = '无可用输入设备';
        deviceEl.appendChild(opt);
        startBtn.disabled = true;
        if (source === 'system') {
          setError('未检测到系统回环输入。请到问号页完成配置或手动安装 BlackHole / Stereo Mix / VB-Cable。');
        }
      } else if (source === 'microphone' && data.devices.length > 1) {
        const autoResp = await fetch('/api/auto-select-device', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({source}),
        });
        const autoData = await autoResp.json();
        if (autoResp.ok && autoData.ok && autoData.device) {
          deviceEl.value = String(autoData.device.index);
        }
      }

      await refreshDiagnostics();
    }

    async function start() {
      setError('');
      const deviceIndex = deviceEl.value === '' ? '' : parseInt(deviceEl.value, 10);
      if (deviceEl.value !== '' && Number.isNaN(deviceIndex)) {
        setError('请先选择有效输入设备。');
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
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.error || '启动失败。');
      }
    }

    async function stop() {
      await fetch('/api/stop', {method: 'POST'});
    }

    async function resetLibrary() {
      setError('');
      const resp = await fetch('/api/reset', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({scope: 'global'}),
      });
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.error || '清空名单失败。');
      }
    }

    async function setupLoopback() {
      setError('');
      setupNoticeEl.textContent = '正在执行回环设备配置，请稍候...';
      const resp = await fetch('/api/setup-loopback', {method: 'POST'});
      const data = await resp.json();
      if (!resp.ok || !data.started) {
        setError(data.error || data.message || '回环配置任务启动失败。');
      }
    }

    async function rebootSystem() {
      setError('');
      const confirmed = window.confirm('将请求系统重启。请先保存其他工作，确认继续吗？');
      if (!confirmed) return;
      const resp = await fetch('/api/reboot', {method: 'POST'});
      const data = await resp.json();
      if (!resp.ok) {
        setError(data.error || '重启请求失败。');
      } else {
        setupNoticeEl.textContent = data.message || '已发送重启请求。';
      }
    }

    function renderSpeakers(speakers) {
      rowsEl.innerHTML = '';
      if (!speakers.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td colspan="4">暂无已入库说话人</td>';
        rowsEl.appendChild(tr);
        return;
      }
      for (const sp of speakers) {
        const tr = document.createElement('tr');
        if (sp.active) tr.classList.add('active');
        tr.innerHTML = `<td>${sp.id ?? '-'}</td><td>${sp.name || '-'}</td><td>${sp.samples ?? '-'}</td><td>${sp.confidence || '-'}</td>`;
        rowsEl.appendChild(tr);
      }
    }

    function renderEvents(lines) {
      const localized = (lines || []).map((line) => localizeEventLine(line));
      eventsEl.textContent = localized.join('\\n');
      eventsEl.scrollTop = eventsEl.scrollHeight;
    }

    async function pollState() {
      try {
        const resp = await fetch('/api/state');
        const state = await resp.json();

        statusEl.textContent = localizeStatus(state.status);
        speakerEl.textContent = state.currentSpeaker || '-';
        confidenceEl.textContent = state.confidence || '-';
        speakerCountEl.textContent = String((state.speakers || []).length);
        renderSpeakers(state.speakers || []);
        renderEvents(state.events || []);

        setError(state.error || '');
        setupNoticeEl.textContent = state.setupNotice || '';
        startBtn.disabled = !!state.running;
        stopBtn.disabled = !state.running;
        setupBtn.disabled = !!state.setupRunning;
        setRunning(!!state.running);
      } catch (err) {
        setError('无法连接本地服务，请确认后端进程仍在运行。');
      }
    }

    helpBtn.addEventListener('click', () => showView('guide'));
    backBtn.addEventListener('click', () => showView('main'));
    enterMainBtn.addEventListener('click', () => showView('main'));

    sourceEl.addEventListener('change', refreshDevices);
    refreshBtn.addEventListener('click', refreshDevices);
    startBtn.addEventListener('click', start);
    stopBtn.addEventListener('click', stop);
    resetBtn.addEventListener('click', resetLibrary);
    setupBtn.addEventListener('click', setupLoopback);
    rebootBtn.addEventListener('click', rebootSystem);
    refreshDiagBtn.addEventListener('click', refreshDiagnostics);

    window.addEventListener('hashchange', () => {
      if (window.location.hash === '#guide') {
        showView('guide', false);
      } else {
        showView('main', false);
      }
    });

    setHint();
    setRunning(false);
    if (window.location.hash === '#guide') {
      showView('guide', false);
    } else {
      showView('main', false);
    }
    refreshDevices();
    pollState();
    setInterval(pollState, 500);
    setInterval(() => {
      if (guideViewEl.classList.contains('active')) {
        refreshDiagnostics();
      }
    }, 4000);
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
