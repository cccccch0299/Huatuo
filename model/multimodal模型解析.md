# Multimodal Model 完整流程解析

## 整体架构：两阶段多模态诊断

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: 生理语义翻译                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐       │
│  │ 特征序列      │───▶│ 模板引擎      │───▶│ Qwen3-0.5B 润色  │       │
│  │ (T, 42维)     │    │ (规则生成)     │    │ (可选，降级兜底)  │       │
│  └──────────────┘    └──────────────┘    └──────────────────┘       │
│                                                      │              │
│                                              中文临床报告 (文本)      │
└──────────────────────────────────────────────────────┼──────────────┘
                                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: 多模态联合推理                                              │
│                                                                     │
│  ┌──────────────────┐     ┌─────────────────────────┐               │
│  │ TemporalEncoder   │     │ TextEncoderWithLoRA      │               │
│  │ (时序特征→向量)    │     │ (报告文本→向量)           │               │
│  │ 42维→64维         │     │ ChineseBERT+LoRA→128维   │               │
│  │ 4层Transformer    │     │ 冻结102M + 可训练369K     │               │
│  └────────┬─────────┘     └───────────┬─────────────┘               │
│           │                           │                             │
│           └──────────┬────────────────┘                             │
│                      ▼                                              │
│           ┌──────────────────────┐                                  │
│           │ CrossAttentionFusion │                                  │
│           │ Q=text, K/V=temporal │                                  │
│           └──────────┬───────────┘                                  │
│                      ▼                                              │
│     concat[temporal_pooled(64), text_emb(128), context(128)]        │
│                      │ 320维                                        │
│                      ▼                                              │
│           ┌──────────────────┐                                      │
│           │ ClassificationHead│                                      │
│           │ 320→64→2          │                                      │
│           └──────────────────┘                                      │
│                      ▼                                              │
│               logits (B, 2) → 认知障碍/正常                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1：生理语义翻译

### 1.1 模板引擎 (PhysiologicalReportGenerator)

纯规则驱动，不依赖任何模型，输入是 `(T, 42)` 的时序特征矩阵：

| 步骤 | 做什么 | 举例 |
|------|--------|------|
| 提取频段能量 | 从42维特征中找 `EEG1_Theta_Energy` 等列 | Theta能量均值=5.67 |
| 计算趋势 | 对时间序列做线性拟合，判断上升/下降/稳定 | "呈现上升趋势" |
| 计算比值 | BTBR、DAR等，判断是否在正常范围 | "Theta/Beta比值为3.2，偏高" |
| EMG分析 | 取EMG1/EMG2的RMS均值 | "肌张力正常" |
| 异常检测 | 规则判断：Alpha<1.0 或 Theta>20.0 或 DAR>3.0 | "发现Alpha能量偏低" |
| 拼接报告 | 8-12句中文 | "在本次脑电检测中，共采集60个时间窗口..." |

### 1.2 LLM润色 (ReportPolisher)

用 Qwen3-0.5B 将模板报告润色为更自然的临床语言：
- Prompt："请将以下脑电检测数据摘要润色为专业的临床检查报告"
- 降级策略：模型不可用或润色太短（<20字）→ 直接返回模板原文

---

## Stage 2：多模态联合推理

### 2.1 TemporalEncoder（时序编码器）

```
输入: (B, T, 42)  ← 每个用户T个时间窗口，每窗口42维特征
  ↓ Linear(42→64) + 可学习位置编码
  ↓ 4层 TransformerEncoderLayer (nhead=4, dim_ff=128, GELU, dropout=0.3)
  ↓ LayerNorm
输出: (B, T, 64)
```

关键设计：
- **位置编码**：可学习的 `nn.Parameter`（非正弦），max_len=5000
- **norm_first=True**：先归一化再做attention（Pre-LN，训练更稳定）
- **padding mask**：支持变长序列，padding位置不参与attention

### 2.2 TextEncoderWithLoRA（文本编码器）

```
输入: 中文临床报告文本
  ↓ BertTokenizer (chinese-roberta-wwm-ext)
  ↓ 冻结的 BERT (12层, 768维, ~102M参数)
  │   └── 每层attention的query和value注入 LoRA (rank=4, alpha=16)
  ↓ 取CLS token → (B, 768)
  ↓ 投影层: 768→256→128 (GELU + Dropout + LayerNorm)
输出: (B, 128)
```

LoRA细节：
- **注入位置**：12层Transformer的 `attention.query` 和 `attention.value`
- **实现方式**：通过 `register_forward_hook` 在原始输出上加 `LoRA(x) = B·A·x * (alpha/rank)`
- **初始化**：A用Kaiming均匀，B用零初始化 → 训练开始时LoRA输出为0，不破坏预训练权重
- **可训练参数**：仅 LoRA(A,B) + 投影层 ≈ 369K，BERT本身完全冻结

### 2.3 CrossAttentionFusion（交叉注意力融合）

```
temporal_seq: (B, T, 64) → Linear(64→128) → K, V
text_emb:     (B, 128)   → unsqueeze(1)   → Q

MultiheadAttention(Q, K, V) → (B, 1, 128)
  ↓ 残差 + LayerNorm
  ↓ FFN(128→256→128) + 残差 + LayerNorm
输出: context (B, 128), attn_weights (B, 1, T)
```

直觉：**让文本"询问"时序数据**——"根据报告中的描述，去关注时序中哪些时间点最重要"。

### 2.4 分类头

```python
# 拼接三个向量
combined = concat[temporal_pooled(64), text_emb(128), context(128)]  # (B, 320)
# 分类
Linear(320→64) → GELU → Dropout(0.5) → Linear(64→2) → logits
```

三个向量的含义：
- `temporal_pooled`：时序特征的全局摘要（对T个时间步取mean pooling）
- `text_emb`：文本报告的语义表示
- `context`：文本引导下的时序注意力结果（"报告说Theta偏高，那就重点关注Theta高的时间段"）

---

## 数据流：从原始数据到训练样本

```
1. load_eeg_data(uid) → 原始EEG DataFrame
2. process_pipeline() → 预处理（滤波+伪迹去除）
3. EEGFeatureExtractor → 滑动窗口提取42维特征 → (T, 42)
4. 按 user_id 分组 → 每个用户一个序列
5. PhysiologicalReportGenerator → 生成中文报告
6. BertTokenizer → tokenize报告 → input_ids(128), attention_mask(128)
7. Pad/Truncate序列到固定长度 T=300
8. 训练集加数据增强（高斯噪声、随机裁剪、特征dropout）
```

---

## 训练策略

| 策略 | 具体做法 |
|------|---------|
| **用户级划分** | 按user_id分层K折，同一用户的数据不会同时出现在训练集和测试集 |
| **类别不平衡** | `WeightedRandomSampler` + `CrossEntropyLoss(weight=class_weights)` |
| **差异化学习率** | 时序/交叉/分类层 `lr=5e-4`，LoRA `lr=5e-5`（低秩适配器学慢一点） |
| **学习率调度** | `CosineAnnealingWarmRestarts(T_0=20, T_mult=2)` |
| **正则化** | Dropout(0.3/0.5) + WeightDecay(0.01) + 梯度裁剪(max_norm=1.0) + LabelSmoothing(0.1) |
| **早停** | patience=15，每10个epoch验证一次 |
| **数据增强** | 高斯噪声(5%) + 随机时间裁剪(10-20%) + 特征dropout(10%) |

---

## 参数量

| 组件 | 可训练 | 冻结 | 总计 |
|------|--------|------|------|
| TemporalEncoder | ~34K | 0 | ~34K |
| TextEncoder (BERT+LoRA) | ~300K | ~102M | ~102.3M |
| CrossAttention | ~34K | 0 | ~34K |
| Classifier | ~21K | 0 | ~21K |
| **总计** | **~369K** | **~102M** | **~102.4M** |

可训练参数仅占总参数的 0.36%，这就是 LoRA 的威力——用极少的参数适配大模型。
