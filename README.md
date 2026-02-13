# 声纹识别演示项目（中文发布版）

这是一个本地运行的实时声纹识别项目，提供浏览器页面操作，不需要命令行交互即可使用。

## 架构速览（先看这个）

```text
[音频输入: 麦克风 / 系统回放]
                |
                v
[音频采集与分块: live_service.py]
                |
                v
[语音检测(VAD): signal_processor.py]
      | 静音           | 噪声             | 人声
      v                v                  v
[SILENCE事件]     [NOISE事件]     [声纹提取: WavLM + ECAPA -> embedding]
                                            |
                                            v
                             [相似度匹配: matcher.py (cosine + AS-Norm)]
                                   | 命中                     | 未命中
                                   v                          v
                   [MATCH(已知说话人)并更新原型]      [自动入库判断: enrollment.py]
                                                                | 通过      | 不通过
                                                                v           v
                                                            [新增新人N] [UNKNOWN_SPEECH]

所有事件最终写入:
[SQLite: speakers / events] -> [Web UI 实时展示: web_frontend.py]
```

建议先按图看一遍，再往下看“核心能力”和“项目框架”，会更容易对上实现细节。

核心能力：
- 实时区分 `静音 / 噪声 / 人声`。
- 自动将稳定新说话人入库（例如 `新人1`、`新人2`）。
- 对已入库说话人进行实时匹配并显示置信度。
- 默认使用 `WavLM + ECAPA` 融合嵌入。
- 默认使用 `global` 持久化库（SQLite，重启后保留）。
- 提供自动调参脚本，生成 `data/tuning.json` 并在启动时自动加载。

相关文档：
- `/Users/shirong/Downloads/voice_recognition/docs/requirements.md`
- `/Users/shirong/Downloads/voice_recognition/docs/design.md`

## 1. 环境要求

- Python `>= 3.10`
- macOS 或 Windows
- 首次运行需要联网下载模型（会缓存到 `data/models`）

安装依赖：

```bash
python -m pip install -r requirements.txt
```

如遇 `numpy has no attribute dtypes`，请确认依赖版本来自本仓库的 `requirements.txt`（`numpy>=1.26,<2.0` 与 `transformers==4.41.2`）。

## 2. 启动方式

命令行启动：

```bash
PYTHONPATH=src python -m voice_recognition.web_frontend
```

也可以双击：
- macOS: `/Users/shirong/Downloads/voice_recognition/start_app.command`
- Windows: `/Users/shirong/Downloads/voice_recognition/start_app.bat`

可选参数：

```bash
PYTHONPATH=src python -m voice_recognition.web_frontend --port 8765 --db-path data/speakers.db
```

启动后默认地址：`http://127.0.0.1:8765/`

## 3. 页面使用说明

页面已全部中文化，默认展示诊断信息。

基础流程：
1. 选择输入源：`麦克风` 或 `系统回放`。
2. 选择设备（也可保留 `自动选择`）。
3. 点击 `开始识别`。
4. 识别过程中可看到：状态、当前说话人、置信度、入库人数。
5. 结束时点击 `停止识别`。
6. 如需清空数据库，点击 `清空名单`。

系统回放模式额外功能：
- `一键配置系统回放`：触发自动配置助手。
- `重启电脑（可选）`：某些 macOS 驱动安装后需要重启生效。

## 4. 自动调参（推荐）

自动下载公开语音数据并计算阈值，输出到 `data/tuning.json`：

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.auto_tune
```

自定义数据规模：

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.auto_tune \
  --speakers 10 \
  --samples-per-speaker 6 \
  --min-seconds 3.0 \
  --max-seconds 12.0
```

服务启动时会自动读取 `data/tuning.json`，覆盖默认阈值。

## 5. 项目框架（架构总览）

数据流：
1. `web_frontend.py`：提供网页与 HTTP API。
2. `web_controller.py`：接收前端操作，构造 `LiveConfig`，调起服务。
3. `live_service.py`：启动音频流、异步处理队列、识别引擎。
4. `audio/signal_processor.py`：VAD + 嵌入提取（WavLM/ECAPA 融合）。
5. `recognition/engine.py`：匹配、软匹配、自动入库、原型更新。
6. `recognition/matcher.py`：AS-Norm + 校准分数 + 阈值判断。
7. `storage/sqlite_repository.py`：全局持久化 speaker 库（线程安全访问）。

关键目录：
- `/Users/shirong/Downloads/voice_recognition/src/voice_recognition/audio`
- `/Users/shirong/Downloads/voice_recognition/src/voice_recognition/recognition`
- `/Users/shirong/Downloads/voice_recognition/src/voice_recognition/storage`
- `/Users/shirong/Downloads/voice_recognition/src/voice_recognition/evaluation`

## 6. macOS 自动化能力与边界

当前可自动做的事：
- 自动检测回环设备是否可用。
- 检测到 Homebrew 时自动安装 `blackhole-2ch`。
- 自动打开 `音频 MIDI 设置`，并在页面显示下一步提示。

当前不能 100% 全自动的部分：
- 自动创建“多输出设备（耳机 + BlackHole）”并保证系统长期稳定路由。
- 主要原因：macOS 对音频设备拓扑管理没有稳定公开 API，跨版本 GUI 自动化可靠性很差。

结论：
- `BlackHole 安装`基本可自动化。
- `多输出设备创建与路由细节`仍建议人工确认（但页面已给出引导和诊断）。

## 7. Windows 能否跑通

可以跑通，但分两种：
- 麦克风识别：通常可直接使用。
- 系统回放识别：需要 `Stereo Mix` 或 `VB-Cable` 这类回环输入设备。

项目内置 Windows 配置助手会打开声音设置页面，指导你启用相关设备。

## 8. 离线评估（可选）

如果你有自己的数据集（目录结构 `dataset/<speaker>/*.wav`）：

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.offline_eval \
  --dataset /absolute/path/to/your_dataset \
  --profile balanced
```

可选拟合 Platt 校准：

```bash
PYTHONPATH=src python -m voice_recognition.evaluation.offline_eval \
  --dataset /absolute/path/to/your_dataset \
  --profile balanced \
  --score-backend asnorm \
  --fit-platt
```

## 9. 测试

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 10. 常见问题

`system` 模式设备列表为空：
- 说明当前没有可用回环输入设备。
- 请先执行页面里的 `一键配置系统回放`，或手动安装/启用 BlackHole、Stereo Mix、VB-Cable。

同一人被拆成多个 ID：
- 先停止服务，清空名单，再重新开始。
- 跑一次自动调参生成 `data/tuning.json`。
- 优先使用稳定输入设备，避免蓝牙免提低码率链路。

实时卡顿或溢出：
- 优先使用默认参数，不要手动降低块大小。
- 在系统回放模式减少后台高负载应用。
