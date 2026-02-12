# 声纹识别 Demo 技术设计（v0.1）

## 1. 设计目标
- 满足 `/Users/shirong/Downloads/voice_recognition/docs/requirements.md` 的 FR/AC。
- 优先可落地与可演示，先实现稳定闭环，再迭代识别精度。

## 2. 技术栈建议
- 语言：Python 3.10+
- 实时音频采集：`sounddevice`（麦克风），系统音频按平台适配（macOS loopback / Windows WASAPI）
- 语音活动检测（VAD）：`silero-vad` 或 `webrtcvad`
- 声纹提取：`speechbrain` ECAPA-TDNN 预训练模型（embedding）
- 打分后端：`cosine` + `AS-Norm`（自适应分数归一）+ `Platt` 校准
- 持久化：`SQLite` + `numpy` 向量序列化
- UI：`PySide6`（桌面界面，支持列表高亮与状态栏）
- 测试：`pytest` + 离线 mp3 回放脚本

## 3. 总体架构
```text
Audio Source (Mic/System/MP3)
        |
   Audio Capture
        |
 Preprocess (resample, mono, normalize)
        |
  VAD + Speech/Noise Classifier
     |                 |
   speech            noise
     |                 |
Speaker Embedding      +--> UI: 背景噪声
     |
Matcher (cosine / AS-Norm + calibration)
  |             |
match         unknown
  |             |
UI高亮+置信度   Unknown Buffer/Cluster
                    |
           Auto Enrollment (新人N)
                    |
             SQLite + UI列表更新
```

## 4. 模块设计
- `src/audio/input_manager.py`
  - 设备枚举、输入源切换、流式读取。
- `src/audio/preprocess.py`
  - 重采样到 16kHz、单声道化、归一化。
- `src/vad/vad_engine.py`
  - 帧级 VAD，输出语音片段边界。
- `src/classifier/speech_noise.py`
  - 对有声段做人声/非人声判别（初期可基于 VAD + 能量/频谱特征规则，后续可替换模型）。
- `src/embedding/speaker_encoder.py`
  - 加载预训练模型并输出 embedding。
- `src/recognition/matcher.py`
  - 计算与已注册说话人的相似度，输出 Top1 与分数。
- `src/recognition/enrollment.py`
  - Unknown 缓冲与稳定性判断，自动注册 `新人N`。
- `src/storage/repository.py`
  - SQLite 读写，说话人、向量、事件日志。
- `src/ui/main_window.py`
  - 列表展示、高亮、置信度、状态栏（语音/噪声/静音）。
- `src/app.py`
  - 主循环与模块编排。

## 5. 关键算法策略

### 5.1 匹配策略
- 输入：当前片段 embedding `e_t`。
- 说话人库：每个说话人维护 centroid `c_i`（可用多样本均值）。
- 分数：
  - 原始分：`s_i = cosine(e_t, p_i)`（对每个说话人原型取最大）
  - 归一化分：`s'_i = AS-Norm(s_i, cohort)`（cohort 来自其他说话人的原型集合）
  - 最终分：`s_final = (1-α)*s_i + α*sigmoid(a*s'_i+b)`。
- 决策：
  - 若 `max(s_final) >= T_match`：命中说话人 `argmax(s_final)`。
  - 否则进入 unknown 缓冲。

### 5.2 自动注册策略（防误注册）
- unknown 连续片段累计时长 >= `D_min`（建议 2~3 秒）。
- 片段内部相似度稳定（聚类半径 < `R_max`）。
- 片段成对相似度满足稳定性约束（pairwise mean 足够高且 std 足够低）。
- 与已知说话人最大相似度仍 < `T_new`（`T_new < T_match`）。
- 满足后创建新说话人：
  - `name = 新人{N}`
  - 存储 centroid、样本数、创建时间。

### 5.3 置信度映射
- 设阈值区间 `[T_low, T_high]`，将相似度线性映射至 0~100：
  - `confidence = clip((s - T_low)/(T_high - T_low), 0, 1) * 100`
- UI 显示两位小数，便于调参。

### 5.4 背景噪声识别
- 判定逻辑：
  - 有声音但未通过人声判别 => `背景噪声`
  - 静音单独标识，不等同背景噪声
- 噪声事件写入日志，不进入注册/匹配。

### 5.5 流式片段防串人
- 在 neural backend 中维护短时 anchor embedding。
- 若当前语音片段与 anchor 相似度低于阈值，重置缓存窗口，避免把不同说话人拼到同一 embedding 中。

## 6. 数据库设计（SQLite）

### 6.1 speakers
- `id` INTEGER PK
- `name` TEXT UNIQUE（如 新人1）
- `created_at` TEXT
- `updated_at` TEXT
- `sample_count` INTEGER
- `centroid_blob` BLOB（float32 向量）

### 6.2 embeddings (可选，调试/追溯)
- `id` INTEGER PK
- `speaker_id` INTEGER FK
- `embedding_blob` BLOB
- `created_at` TEXT

### 6.3 events
- `id` INTEGER PK
- `ts` TEXT
- `event_type` TEXT（match/new_speaker/noise/silence）
- `speaker_id` INTEGER NULL
- `score` REAL NULL
- `confidence` REAL NULL
- `meta_json` TEXT NULL

## 7. UI 设计
- 顶部控制区：
  - 输入源下拉：麦克风 / 系统音频 / 文件回放
  - 开始、停止按钮
- 中部状态区：
  - 当前状态：静音 / 背景噪声 / 识别中
  - 当前命中：`新人N` + `置信度`
- 右侧列表区：
  - 所有已注册说话人
  - 当前命中高亮
  - 显示样本数与最近命中时间
- 底部日志区：
  - 滚动显示事件（时间、类型、分数）

## 8. 测试设计（含 mp3 数据）

### 8.1 数据组织建议
```text
tests/data/
  speaker_a_01.mp3
  speaker_a_02.mp3
  speaker_b_01.mp3
  noise_music_01.mp3
  noise_env_01.mp3
```

### 8.2 用例
- `TC-01` 新说话人自动注册：A 首次出现应注册 `新人1`。
- `TC-02` 已注册复现命中：A 再次出现应命中 `新人1`。
- `TC-03` 第二说话人注册：B 出现应注册 `新人2`。
- `TC-04` 背景噪声识别：音乐/环境音不应注册新人。
- `TC-05` 持久化：重启后可命中历史说话人。
- `TC-06` 长时稳定：连续 30~120 分钟运行无崩溃。

### 8.3 指标
- 说话人命中率、误报率（把陌生人识别为已知）、漏报率（已知未命中）
- 噪声误检率（噪声被当做人声）
- 平均处理延迟（秒）

## 9. AI 全自动开发与修正流程
- Step 1：读取本需求与设计文档，生成项目骨架与模块代码。
- Step 2：根据 `tests/data` 自动生成/补全测试用例与评估脚本。
- Step 3：执行测试，输出失败项与误差分析（阈值、VAD、聚类参数）。
- Step 4：自动修改代码与参数，回归测试直至通过目标阈值。
- Step 5：输出版本报告（变更摘要、指标对比、已知问题）。

## 10. 里程碑建议
- `M1（1~2天）`：最小可运行链路（采集->VAD->匹配->UI显示）。
- `M2（1~2天）`：自动注册与持久化。
- `M3（1天）`：mp3 离线评估与测试报告。
- `M4（1天）`：阈值调优与稳定性修复。

## 11. 库作用域双模式设计（global/session）
- `global`：
  - 使用 SQLite 持久化库，跨音频和重启后可继续复用。
  - 适合长期积累说话人画像。
- `session`：
  - 使用内存临时库，仅当前会话有效，关闭后清空。
  - 适合临时实验或单段音频分析，避免污染长期库。
- 两种模式复用同一识别链路，仅替换 repository 实现，降低维护复杂度。
