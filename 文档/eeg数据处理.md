# EEG 数据处理流水线

本项目包含两套独立的脑电数据处理流水线，分别服务于前端可视化和 ML 模型训练/推理。

---

## 流水线一：前端可视化流水线（实时滤波展示）

**文件**：`data_processing.py` → 被 `receiver_sender.py` 的 `build_filtered_history_payload()` 调用

**目的**：把原始信号滤波后展示在前端波形图上，让用户肉眼观察

### 流程

```
原始信号 (eeg_1, eeg_2, emg_1, emg_2, ...)
  ↓
缺失值插值 (linear interpolation)
  ↓
┌─────────────────────────────────────────────┐
│ 50Hz 陷波器 (Notch Filter)                   │
│ 系数从 C# Unity 端移植过来，8阶 IIR          │
│ bn/an 系数硬编码，与 NeuroXess SDK 完全一致   │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│ 0.5-40Hz 带通滤波器 (Bandpass)              │
│ 同样从 C# 移植的 8阶 IIR 系数 (b_main/a_main)│
│ 保留 Delta~Beta 主频段，去除基线漂移和高频噪声│
└──────────────────────┬──────────────────────┘
                       ↓
              输出 main_0.5_40Hz（干净信号）
                       ↓
         [仅 EEG 通道] 提取 4 个子频段
         ├── delta (0.35-3.5 Hz)
         ├── theta (4-7 Hz)
         ├── alpha (8-13 Hz)
         └── beta  (14-30 Hz)
```

### 特点

- 滤波器系数是**从 C# Unity 端直接移植**的，保证前后端波形一致
- 使用 `filtfilt`（零相移滤波），没有时间延迟
- 结果保留 3 位小数（与 C# 的 `OnFilterDecimal` 一致）
- **不做**伪迹回归、不做特征提取、不做异常值处理
- 所有 9 个通道都处理（eeg_1/2, emg_1/2, blink_l/r, gaze_x/y/z）

---

## 流水线二：ML 模型训练/推理流水线（离线特征提取）

**文件**：`model/preprocess.py` + `model/feature_extraction.py` + `model/model_training.py`

**目的**：从原始信号中提取数值特征，喂给机器学习模型做认知障碍分类

### 完整流程图

```
原始信号 (从 TimescaleDB 加载)
  ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: 数据清洗                                                     │
│   - 强制转 float，NaN 线性插值 + 填0                                  │
│   - handle_extreme_values: 处理电极接触不良                            │
│     EEG 阈值 ±150μV，EMG 阈值 ±300μV                                │
│     超限点 → NaN → 线性插值(limit=25点) → 前后填充 → 硬裁剪            │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: 基础滤波                                                     │
│   - 50Hz 陷波: scipy.signal.iirnotch(50, 30, 250) 动态生成            │
│   - 0.5-45Hz 带通: Butterworth 4阶                                   │
│   - 注意：上限是 45Hz（比前端的 40Hz 多 5Hz，保留 Gamma）              │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: 伪迹回归 (条件执行)                                           │
│   - 前提：原始信号 PTP ≥ 10μV                                        │
│   - 方法：以 EMG1/EMG2 为参考信号，线性回归拟合伪迹成分                │
│   - EEG_clean = EEG_filtered - w * EMG_ref                           │
│   - 若信号太弱 (<10μV) 则跳过此步，直接使用滤波后信号                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
                  输出 eeg_1_clean, eeg_2_clean
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: 滑动窗口特征提取                                             │
│   - 窗口: 2秒 (500点), 步长: 1秒 (250点), 重叠 50%                   │
│   - 每个窗口计算 42 维特征                                            │
│   - 坏窗口丢弃: PTP < 0.1μV 或 > 250μV                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           ↓
                  输出 (T, 42) 特征矩阵
                           ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: StandardScaler 标准化 (训练时 fit_transform)                  │
│ Step 6: 模型推理 (RF / XGB / EEGNet / 多模态)                        │
│ Step 7: 阈值优化 → 输出分类结果                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Step 1：数据清洗（详细）

**入口函数**：`process_pipeline(df)` in `model/preprocess.py`

#### 1.1 类型转换与缺失值处理

```python
df_numeric = df[target_cols].apply(pd.to_numeric, errors='coerce')
df.update(df_numeric.interpolate(method='linear').fillna(0))
```

- 将 `eeg_1, eeg_2, emg_1, emg_2` 四列强制转为 float
- 无法转换的值（如空字符串）变成 NaN
- NaN 先做**线性插值**（前后相邻值的线性过渡）
- 插值后仍有 NaN 的（如开头/结尾整段缺失）用 0 填充

#### 1.2 极端异常值处理 (`handle_extreme_values`)

**目的**：处理电极瞬间脱落或接触不良造成的异常尖峰

**处理流程**（逐通道）：

```
原始数据:  [..., 12.3, 287.5, -312.1, 8.7, ...]
                    ↑       ↑
               超过阈值   超过阈值
                     ↓
第1步: 超限点标记为 NaN
       [..., 12.3, NaN, NaN, 8.7, ...]
                     ↓
第2步: 线性插值 (limit=25点=0.1秒)
       如果连续 NaN ≤ 25个点 → 线性插值填补
       如果连续 NaN > 25个点 → 保留 NaN (电极掉太久，插值不可靠)
       [..., 12.3, -149.9, -149.9, 8.7, ...]  ← 假设连续<25点
                     ↓
第3步: 前后填充 (ffill + bfill)
       处理插值后残留的 NaN（如开头/结尾的大段缺失）
                     ↓
第4步: 硬裁剪 clip(-threshold, +threshold)
       确保绝对没有超出阈值的数据进入后续滤波器
```

**阈值设定**：

| 通道 | 阈值 | 理由 |
|------|------|------|
| EEG (eeg_1, eeg_2) | ±150 μV | 正常脑电信号一般在 ±50μV 以内，150μV 已是极端 |
| EMG (emg_1, emg_2) | ±300 μV | 肌电信号幅度本身比脑电大，阈值放宽 |

**为什么在滤波前做**：极端异常值（如电极瞬间脱落产生的 ±500μV 尖峰会严重扭曲滤波器的输出（IIR 滤波器有"记忆"，一个尖峰会污染后续数百毫秒的数据）。

---

### Step 2：基础滤波（详细）

**入口函数**：`apply_basic_filters(data)` in `model/preprocess.py`

#### 2.1 50Hz 陷波滤波器 (Notch Filter)

```python
notch_freq = 50.0       # 中国交流电频率
quality_factor = 30.0   # Q值，越大带宽越窄
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs=250)
data_notched = signal.filtfilt(b_notch, a_notch, data, axis=0)
```

- **目的**：去除 50Hz 工频干扰（中国交流电频率，电源线辐射到电极的噪声）
- **原理**：在 50Hz 频率处产生一个极窄的"凹陷"，几乎不影响其他频率
- **Q=30** 意味着带宽 ≈ 50/30 ≈ 1.67Hz，只消除 49.17~50.83Hz 这一小段
- **filtfilt**：零相移滤波，正向+反向各滤一次，消除时间延迟

#### 2.2 0.5-45Hz 带通滤波器 (Bandpass)

```python
lowcut = 0.5    # 下限 0.5Hz
highcut = 45.0  # 上限 45Hz
order = 4       # Butterworth 4阶
nyq = 0.5 * fs  # 奈奎斯特频率 = 125Hz
low = lowcut / nyq   # 归一化频率下限
high = highcut / nyq  # 归一化频率上限
b_band, a_band = signal.butter(order, [low, high], btype='band')
data_filtered = signal.filtfilt(b_band, a_band, data_notched, axis=0)
```

- **下限 0.5Hz**：去除直流分量和极低频的基线漂移（如呼吸、身体缓慢移动）
- **上限 45Hz**：去除高频肌电噪声和工频谐波（100Hz、150Hz 等）
- **Butterworth 4阶**：在通带内最平坦，不会引入纹波；4阶提供 -24dB/octave 的滚降
- **为什么是 45Hz 不是 40Hz**：保留 Gamma 频段 (30-45Hz) 供特征提取。Gamma 波与高级认知处理（信息整合、注意力）密切相关

**与前端流水线的关键区别**：

| 参数 | 前端 (`data_processing.py`) | ML (`model/preprocess.py`) |
|------|----------------------------|---------------------------|
| 陷波器 | 硬编码 8阶 IIR 系数 (C# 移植) | `iirnotch(50, 30, 250)` 动态生成 |
| 带通范围 | 0.5-40 Hz | 0.5-45 Hz |
| 带通阶数 | 8阶 | 4阶 Butterworth |

---

### Step 3：伪迹回归（详细）

**入口函数**：`remove_artifacts_regression(eeg_data, ref_data)` in `model/preprocess.py`

#### 3.1 为什么需要伪迹去除？

原始 EEG 信号中混杂着非脑电的干扰信号（伪迹）：
- **EMG 伪迹**：面部肌肉紧张（咬牙、皱眉）产生的肌电信号，幅度远大于脑电
- **EOG 伪迹**：眼球运动和眨眼产生的电信号（眼电信号通过头骨传导到脑电电极）

#### 3.2 线性回归法原理

假设：原始 EEG = 真实脑电 + w × 伪迹参考信号

```
EEG_raw(t) = EEG_clean(t) + w₁ × EMG1(t) + w₂ × EMG2(t) + ...
```

**步骤**：

```python
# 对每个 EEG 通道
for i in range(eeg_data.shape[1]):
    target_eeg = eeg_data[:, i]           # 待清理的 EEG 信号
    reg = LinearRegression().fit(ref_data, target_eeg)  # 用 EMG 拟合
    artifact_component = reg.predict(ref_data)           # 预测伪迹成分
    clean_eeg[:, i] = target_eeg - artifact_component    # 减去伪迹
```

1. 以 `emg_1, emg_2` 作为自变量 X，以 `eeg_1` (或 `eeg_2`) 作为因变量 Y
2. 用最小二乘法拟合权重 w：使得 `w × EMG` 最接近真实 EEG 中的伪迹部分
3. 从原始 EEG 中减去拟合出的伪迹成分

#### 3.3 条件执行：信号强度检查

```python
eeg_ptp = np.max(eeg_raw_fixed, axis=0) - np.min(eeg_raw_fixed, axis=0)
min_ptp = np.min(eeg_ptp)

if min_ptp >= 10.0:
    # 信号够强，做伪迹回归
    eeg_clean = remove_artifacts_regression(eeg_filtered, ref_filtered)
else:
    # 信号太弱，跳过（怕回归把有用信号也减掉了）
    eeg_clean = eeg_filtered
```

- **PTP (Peak-to-Peak)**：信号的最大值 - 最小值，反映信号振幅
- **阈值 10μV**：如果原始信号 PTP < 10μV，说明信号极弱（可能电极接触不好），此时做回归可能把残余的有用信号也减掉
- **实际数据观察**：很多用户的 EEG PTP 只有 0.3-0.5μV（信号极弱），这些用户会跳过伪迹回归

---

### Step 4：滑动窗口特征提取（详细）

**入口函数**：`EEGFeatureExtractor.extract_features(df_clean)` in `model/feature_extraction.py`

#### 4.1 窗口参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 窗口长度 | 2秒 (500点) | 足够捕捉 Delta (0.5Hz) 这样的低频信号（至少需要 1-2 个周期） |
| 步长 | 1秒 (250点) | 相邻窗口重叠 50%，保证特征的时间分辨率 |
| 采样率 | 250 Hz | 来自 .env 的 `EEG_ALIGN_SAMPLE_RATE_HZ` |

#### 4.2 坏窗口检测

每个窗口在提取特征前先做质量检查：

```python
for col in ['eeg_1_clean', 'eeg_2_clean']:
    ptp_amplitude = np.max(signal_data) - np.min(signal_data)
    if ptp_amplitude < 0.1 or ptp_amplitude > 250.0:
        is_bad_window = True  # 丢弃此窗口
```

| 条件 | 含义 | 原因 |
|------|------|------|
| PTP < 0.1 μV | 信号是一条直线 | 电极断开/短路，没有有效数据 |
| PTP > 250 μV | 信号振幅异常大 | 极强噪声/接触不良，数据不可信 |

#### 4.3 Welch PSD 功率谱密度估计

对每个窗口内的信号做频谱分析：

```python
freqs, psd = signal.welch(segment, fs=250, nperseg=500)
```

- **Welch 方法**：将信号分成重叠的段，对每段做 FFT，再取平均，减少频谱估计的方差
- **nperseg=500**：等于窗口长度，频率分辨率 = 250/500 = 0.5Hz
- **输出**：`freqs` (频率轴) 和 `psd` (每个频率点的功率)

#### 4.4 五个经典脑电频段

| 频段 | 频率范围 | 与认知的关系 |
|------|---------|-------------|
| **Delta (δ)** | 0.5-4 Hz | 深度睡眠、大脑基本修复。成人清醒时过高可能提示脑功能减退 |
| **Theta (θ)** | 4-8 Hz | 记忆编码、注意力调节。过高通常提示认知负荷增大或脑功能下降 |
| **Alpha (α)** | 8-13 Hz | 放松清醒状态，大脑正常工作的标志。过低提示皮层抑制功能减弱 |
| **Beta (β)** | 13-30 Hz | 专注、思考、警觉。过低提示注意力和执行功能下降 |
| **Gamma (γ)** | 30-45 Hz | 高级认知处理、信息整合。异常可能提示认知整合能力变化 |

#### 4.5 每个频段计算两个指标

**频段能量 (Band Energy)**：

```python
idx_band = np.logical_and(freqs >= low, freqs <= high)
band_energy = trapezoid(psd[idx_band], freqs[idx_band])
```

- 对该频段范围内的 PSD 曲线做梯形积分，得到该频段的总功率
- 物理含义：该频段脑电波的"总振动能量"

**微分熵 (Differential Entropy, DE)**：

```python
band_de = math.log2(band_energy + 1e-10)
```

- 对频段能量取以 2 为底的对数
- 为什么要取对数：不同人的信号强度差异很大（有人 5μV，有人 50μV），取对数后可以消除这种个体差异，只保留频段间的相对关系
- `+1e-10` 防止 log(0) 报错

#### 4.6 功率比值特征

```python
DAR   = Delta / Alpha                    # δ/α 比值
DTABR = (Delta + Theta) / (Alpha + Beta) # (δ+θ)/(α+β) 比值
BTBR  = Beta / Theta                     # β/θ 比值
Theta_Alpha = Theta / Alpha              # θ/α 比值
```

以及各频段占总能量的比例：

```python
total = Delta + Theta + Alpha + Beta + Gamma
Delta_Ratio = Delta / total   # Delta 占比
Theta_Ratio = Theta / total   # Theta 占比
...（共 5 个 Ratio）
```

**为什么需要比值特征**：
- 绝对能量值受个体差异影响（头骨厚薄、电极接触程度）
- 比值是相对指标，消除了这些干扰，更稳定

#### 4.7 EMG 特征

```python
EMG_RMS = sqrt(mean(emg_segment²))
```

- **RMS (均方根值)**：反映肌电信号的整体强度
- 面部肌肉越紧张，RMS 越高
- 高 EMG RMS 可能提示患者检查时紧张，或存在未完全去除的 EMG 伪迹

#### 4.8 游戏特征（外部数据）

除了脑电特征，还从 `game_sessions` 表加载游戏表现数据：

| 特征 | 来源 | 含义 |
|------|------|------|
| `game_hit_accuracy` | `success_count / hit_count` | 命中准确率，反映手眼协调和反应速度 |
| `game_score` | `score` 字段 | 综合得分，反映反应速度、注意力和运动协调 |

#### 4.9 最终特征矩阵

每个时间窗口产出一个 42 维特征向量：

```
EEG1 通道 (19维):
  ├── Delta_Energy, Delta_DE
  ├── Theta_Energy, Theta_DE
  ├── Alpha_Energy, Alpha_DE
  ├── Beta_Energy,  Beta_DE
  ├── Gamma_Energy, Gamma_DE
  ├── DAR, DTABR, BTBR, Theta_Alpha
  └── Delta_Ratio, Theta_Ratio, Alpha_Ratio, Beta_Ratio, Gamma_Ratio

EEG2 通道 (19维): 同上

EMG 通道 (2维):
  ├── EMG1_RMS
  └── EMG2_RMS

游戏特征 (2维):
  ├── game_hit_accuracy
  └── game_score

总计: 19 + 19 + 2 + 2 = 42 维
```

如果有 T 个时间窗口，最终输出矩阵形状为 **(T, 42)**。

---

### Step 5：StandardScaler 标准化（详细）

**入口函数**：`prepare_data(labeled_df)` in `model/model_training.py`

#### 5.1 用户级训练/测试划分

```python
# 按 user_id 分组，每个用户的所有窗口要么全在训练集，要么全在测试集
train_users, test_users = train_test_split(
    users, test_size=0.2, random_state=42, stratify=user_y)
```

- **为什么要按用户划分**：同一用户的多个窗口高度相似（来自同一人的同一段脑电），如果随机划分，同一用户的窗口可能同时出现在训练集和测试集，导致"数据泄漏"——模型记住了这个人而不是学到了通用规律
- **分层抽样**：`stratify=user_y` 保证训练集和测试集中认知障碍/正常的比例一致

#### 5.2 Z-score 标准化

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # 训练集: fit + transform
X_test_scaled = scaler.transform(X_test)          # 测试集: 只 transform
```

**公式**：`z = (x - μ) / σ`

- μ = 该特征在训练集上的均值
- σ = 该特征在训练集上的标准差
- 标准化后每个特征的均值=0，标准差=1

**为什么需要标准化**：
- 不同特征的量纲差异巨大（如 `Delta_Energy` 可能是 100，而 `Gamma_Ratio` 是 0.01）
- 树模型（RF/XGB）虽然对尺度不敏感，但标准化后特征重要性更可比
- 标准化后的数据在训练神经网络时收敛更快

**关键**：`scaler.fit()` 只在训练集上做，然后用同一个 scaler 转换测试集，避免信息泄漏。

---

### Step 6 & 7：模型推理与阈值优化（详细）

#### 6.1 支持的模型

| 模型 | 类型 | 关键参数 |
|------|------|---------|
| **RandomForest** | 集成学习 (Bagging) | 200棵树, max_depth=None, class_weight='balanced' |
| **XGBoost** | 集成学习 (Boosting) | 200棵树, max_depth=6, lr=0.1, scale_pos_weight |
| **LightGBM** | 集成学习 (Boosting) | 200棵树, is_unbalance=True |
| **EEGNet** | 深度学习 (CNN) | 专门处理原始 EEG 信号的轻量级 CNN |
| **Multimodal** | 深度学习 (Transformer+BERT) | 时序+文本多模态融合模型 |

#### 6.2 类别不平衡处理

数据集分布：认知障碍(0)=11人，正常(1)=56人，比例约 1:5

```python
# RandomForest: 内置 class_weight='balanced'
# 自动给少数类（障碍）更高的误分类惩罚

# XGBoost: scale_pos_weight = n_pos / n_neg
# 手动设置正负样本权重比

# 可选 SMOTE 过采样
# 合成新的少数类样本（在特征空间中插值）
```

#### 6.3 阈值优化

```python
# 默认阈值 0.5 不一定最优
# 在训练集上搜索 0.1~0.9，步长 0.01
# 目标：最大化障碍类(0)的 F1 分数
for thr in np.arange(0.1, 0.9, 0.01):
    y_pred = (y_prob >= thr).astype(int)
    f1_neg = f1_score(1 - y_true, 1 - y_pred)
    # 选 f1_neg 最大的阈值
```

- 默认 0.5 阈值意味着：P(正常) ≥ 0.5 → 预测正常
- 由于障碍类是少数类，默认阈值倾向于把所有人预测为正常（高召回率、低精确率）
- 阈值优化找到了一个更平衡的切分点（如 0.45），让障碍类也能被有效识别

#### 6.4 保存的产物

训练完成后保存到 `test/` 目录：

| 文件 | 内容 | 推理时需要 |
|------|------|-----------|
| `rf_model.joblib` | 训练好的随机森林模型 | 是 |
| `scaler.joblib` | StandardScaler（含训练集的 μ 和 σ） | 是 |
| `inference_meta.joblib` | 最优阈值、特征名列表、模型名 | 是 |
| `confusion_matrix.png` | 混淆矩阵图 | 否 |
| `roc_curve.png` | ROC 曲线图 | 否 |
| `feature_importance.png` | 特征重要性图 | 否 |

---

## 核心区别对比

| 维度 | 前端流水线 (`data_processing.py`) | ML 流水线 (`model/preprocess.py`) |
|------|----------------------------------|----------------------------------|
| **用途** | 波形可视化，人眼看 | 特征提取，模型分类 |
| **滤波器来源** | C# Unity 移植的硬编码系数 | scipy.signal 动态生成 |
| **50Hz 陷波** | 8阶 IIR (bn/an 系数) | `iirnotch(50, 30, 250)` |
| **带通范围** | **0.5-40 Hz** | **0.5-45 Hz** |
| **带通阶数** | 8阶 (C# 原始) | 4阶 Butterworth |
| **伪迹去除** | 无 | 线性回归法（EMG 为参考） |
| **极端值处理** | 无 | 有（电极接触不良检测+插值） |
| **子频段分离** | 有（delta/theta/alpha/beta） | 无（在特征提取阶段用 Welch PSD） |
| **特征计算** | 无 | 频段能量 + DE + 比值 + EMG RMS |
| **输出形式** | 滤波后的波形数组 | 42 维数值特征向量 |
| **处理通道** | 全部 9 个通道 | 仅 EEG1/EEG2/EMG1/EMG2 |
| **Gamma 频段** | 无（beta 到 30Hz 截止） | 有（30-45Hz） |

---

## 关键差异说明：带通上限

前端用 **40Hz** 上限，ML 用 **45Hz** 上限：

- 前端展示的波形把 40-45Hz 的 Gamma 信号过滤掉了
- ML 模型保留了这部分 Gamma 信号，并从中提取了 `Gamma_Energy` 和 `Gamma_DE` 特征
- 这是有意为之：前端需要更"干净"的波形给人看，ML 需要保留更多信息做判断

---

## 两套流水线的关系图

```
                    ┌──────────────────┐
                    │   TimescaleDB    │
                    │   (原始数据)      │
                    └───────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌──────────────────┐       ┌──────────────────┐
    │ data_processing  │       │ model/preprocess  │
    │ (前端可视化)      │       │ (ML训练/推理)      │
    │                  │       │                   │
    │ Notch + 带通     │       │ 清洗 + Notch + 带通│
    │ 仅滤波           │       │ + 伪迹回归         │
    │ 输出: 干净波形    │       │ 输出: 干净信号      │
    └────────┬─────────┘       └────────┬──────────┘
             ▼                          ▼
    ┌──────────────────┐       ┌──────────────────┐
    │ 前端 ECharts     │       │ feature_extraction│
    │ 波形展示          │       │ 滑动窗口特征提取   │
    │ + 子频段分离显示   │       │ → 42维特征向量     │
    └──────────────────┘       └────────┬──────────┘
                                        ▼
                               ┌──────────────────┐
                               │ 模型推理           │
                               │ RF/XGB/EEGNet/   │
                               │ Multimodal       │
                               └──────────────────┘
```

**一句话总结**：前端流水线是"看"，ML 流水线是"算"。前者追求与 Unity 端一致的视觉效果，后者追求信息保留和特征丰富度。
