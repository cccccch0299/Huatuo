# BIOT (Biosignal Transformer) 模型详细解析

## 一、模型概述

BIOT 是专为生物信号（EEG/EMG）设计的 Transformer 分类模型。它将原始时域信号通过 Patch token 化转换为序列，利用 Transformer 的自注意力机制捕获通道内和通道间的长程依赖关系，并支持自监督预训练和眼动特征混合融合。

**核心设计思想**：借鉴 ViT (Vision Transformer) 的 patch 理念，将一维生物信号切分为固定长度的 patch，每个 patch 类比 NLP 中的一个 token，从而将信号分类问题转化为序列建模问题。

---

## 二、整体架构

```
输入: (B, 4, 800)          B=batch, 4通道(2EEG+2EMG), 800点(4秒@200Hz)
    │
    ├─── PatchEmbedding ──────────────────────────────────────┐
    │    将每通道切为16个patch(每patch 50点)                     │
    │    线性投影: 50 → 128                                     │
    │    输出: (B, 64, 128)    64=4通道×16patches               │
    │                                                          │
    ├─── SignalTypeEmbedding ──────────────────────────────────┤
    │    EEG patches → +type_embed(0)                          │
    │    EMG patches → +type_embed(1)                          │
    │    让模型区分不同信号类型                                    │
    │                                                          │
    ├─── Positional Encoding ─────────────────────────────────┤
    │    可学习位置编码: (1, 65, 128)                            │
    │    65 = 64 tokens + 1 CLS                                │
    │                                                          │
    ├─── CLS Token 拼接 ──────────────────────────────────────┘
    │    在序列头部插入可学习的[CLS] token
    │    输出: (B, 65, 128)
    │
    ├─── Embedding Dropout (p=0.1)
    │
    ├─── Transformer Encoder (×6层) ──────────────────────────┐
    │    每层:                                                │
    │    ┌─ LayerNorm (Pre-Norm)                              │
    │    ├─ Multi-Head Self-Attention (8 heads)               │
    │    ├─ Residual Connection                              │
    │    ├─ LayerNorm                                        │
    │    ├─ FFN: Linear(128→512) → GELU → Dropout → Linear(512→128)
    │    └─ Residual Connection                              │
    │    输出: (B, 65, 128)                                   │
    │                                                         │
    ├─── LayerNorm ──────────────────────────────────────────┘
    │    取 CLS token 输出: (B, 128)
    │
    ├─── [可选] 眼动特征融合 (Hybrid Fusion) ─────────────────┐
    │    眼动特征 (B, 23)                                      │
    │    → Linear(23→64) → GELU → Dropout                     │
    │    → 与 CLS 拼接: (B, 128+64) = (B, 192)               │
    │                                                         │
    ├─── Classification Head ─────────────────────────────────┘
    │    Linear(192→128) → GELU → Dropout → Linear(128→2)
    │    输出: (B, 2) 二分类 logits
    │
    └─── Softmax → 概率
```

---

## 三、各组件详细解析

### 3.1 PatchEmbedding — 信号分块与投影

**文件位置**: `model/biot_model.py:30-45`

```python
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=50, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, d_model)

    def forward(self, x):
        B, C, T = x.shape                              # (B, 4, 800)
        num_patches = T // self.patch_size              # 800/50 = 16
        x = x[:, :, :num_patches * self.patch_size]     # 截断尾部不足一个patch的点
        x = x.reshape(B, C, num_patches, self.patch_size)  # (B, 4, 16, 50)
        x = x.reshape(B, C * num_patches, self.patch_size)  # (B, 64, 50)
        return self.projection(x)                       # (B, 64, 128)
```

**工作原理**：
1. 将每个通道的800个采样点切分为16个patch，每个patch包含50个点（250ms@200Hz）
2. 4个通道共产生 4×16 = 64 个 patch
3. 通过一个全连接层将每个50维 patch 投影为128维 token

**为什么 patch_size=50？**
- 200Hz 采样率下，50点 = 250ms
- 这个时间窗刚好覆盖 theta (4-8Hz) 和 alpha (8-13Hz) 脑电节律的一个完整周期
- 每个 patch 可以捕获一个局部波形特征

**输入输出**：
```
输入:  (B, 4, 800)   →  原始4通道信号
输出:  (B, 64, 128)  →  64个token，每个128维
```

---

### 3.2 SignalTypeEmbedding — 信号类型嵌入

**文件位置**: `model/biot_model.py:48-64`

```python
class SignalTypeEmbedding(nn.Module):
    def __init__(self, n_eeg_channels=2, n_emg_channels=2,
                 patches_per_channel=16, d_model=128):
        super().__init__()
        self.type_embedding = nn.Embedding(2, d_model)  # 2种类型
        self.n_eeg_patches = n_eeg_channels * patches_per_channel  # 32
        self.n_emg_patches = n_emg_channels * patches_per_channel  # 32

    def forward(self, x):
        type_ids = torch.cat([
            torch.zeros(self.n_eeg_patches),   # 前32个token → type=0 (EEG)
            torch.ones(self.n_emg_patches),     # 后32个token → type=1 (EMG)
        ]).to(x.device)
        return x + self.type_embedding(type_ids)
```

**设计动机**：
- EEG（脑电）和 EMG（肌电）是完全不同性质的信号
- EEG 反映大脑皮层神经活动，频段 0.5-45Hz，幅度较小
- EMG 反映肌肉电信号，频段更宽，幅度较大
- 类型嵌入让模型在注意力计算时能区分信号来源

**token 排列顺序**：
```
[EEG1_patch1, EEG1_patch2, ..., EEG1_patch16,   ← type=0
 EEG2_patch1, EEG2_patch2, ..., EEG2_patch16,   ← type=0
 EMG1_patch1, EMG1_patch2, ..., EMG1_patch16,   ← type=1
 EMG2_patch1, EMG2_patch2, ..., EMG2_patch16]   ← type=1
```

---

### 3.3 Positional Encoding — 位置编码

**文件位置**: `model/biot_model.py:89`

```python
self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
```

**特点**：
- **可学习**的位置编码（非固定的正弦编码）
- `max_tokens = 65`（64个信号token + 1个CLS token）
- 初始化标准差为 0.02，避免初始值过大
- Transformer 本身不具备位置感知能力，位置编码是必要的

**前向传播中的使用**（第124、132行）：
```python
# 先给信号token加位置编码（跳过CLS位）
x = x + self.pos_embed[:, 1:x.size(1)+1, :]
# 拼接CLS后，再给CLS加位置编码
x = x + self.pos_embed[:, :1, :]
```

---

### 3.4 CLS Token — 全局聚合

**文件位置**: `model/biot_model.py:90`

```python
self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
```

**作用**：
- 借鉴 BERT 的 [CLS] token 设计
- 作为一个"虚拟"token拼接在序列头部
- 通过 self-attention 与所有64个信号token交互
- 最终取 CLS 的输出作为整个序列的全局表示
- 相比对所有token做平均池化，CLS token能学到更有针对性的聚合策略

---

### 3.5 Transformer Encoder — 核心编码器

**文件位置**: `model/biot_model.py:93-98`

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128, nhead=8,
    dim_feedforward=512,    # d_model * 4
    dropout=0.1,
    activation='gelu',
    batch_first=True,
    norm_first=True,        # Pre-Norm (更稳定)
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
```

**每层结构 (Pre-Norm Transformer)**：

```
输入 x
  │
  ├─ LayerNorm(x) ─→ Multi-Head Self-Attention ─→ Dropout ─→ x + residual
  │                                                      (残差连接)
  ├─ LayerNorm(x) ─→ FFN ─→ Dropout ─→ x + residual
  │
  └─ 输出
```

**Pre-Norm vs Post-Norm**：
- **Pre-Norm**（本模型）：先归一化再计算，梯度流更稳定，训练更容易收敛
- Post-Norm：先计算再归一化，需要 warmup 等技巧

**Multi-Head Self-Attention (8 heads)**：
```
每个 head 的维度: d_model / nhead = 128 / 8 = 16

Q, K, V = Linear(x)   # (B, 65, 128) → 各(B, 65, 128)
→ 拆分为8个头: (B, 8, 65, 16)
→ Attention(Q,K,V) = softmax(QK^T/√16)V
→ 拼接: (B, 65, 128)
→ Linear 投影
```

**Feed-Forward Network**：
```
Linear(128 → 512) → GELU → Dropout(0.1) → Linear(512 → 128)
```
- 扩展比为 4x（128→512），标准 Transformer 设计
- GELU 激活函数比 ReLU 更平滑，是 Transformer 的主流选择

**为什么是6层？**
- 数据量约4765个样本，6层是防止过拟合的保守选择
- 更深的模型（12层）在小数据集上容易过拟合
- 6层已经能捕获多尺度的时间依赖关系

---

### 3.6 Classification Head — 分类头

**文件位置**: `model/biot_model.py:113-118`

**纯信号模式** (n_eye_features=0)：
```python
self.head = nn.Sequential(
    nn.Linear(128, 128),   # d_model → d_model
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(128, 2),     # d_model → n_classes
)
```

**信号+眼动融合模式** (n_eye_features=23)：
```python
# 眼动投影分支
self.eye_proj = nn.Sequential(
    nn.Linear(23, 64),     # n_eye_features → d_model//2
    nn.GELU(),
    nn.Dropout(0.1),
)

# 分类头（输入维度变大）
self.head = nn.Sequential(
    nn.Linear(192, 128),   # (128+64) → d_model
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(128, 2),
)
```

**融合方式：Hybrid Fusion（混合融合）**
```
CLS输出 (128维)  ──┐
                    ├── 拼接 (192维) → 分类头 → logits
眼动投影 (64维)  ──┘
```

相比早期融合（特征拼接后一起过 Transformer）和晚期融合（各自独立编码后融合），混合融合是一种折中方案：
- 信号分支利用 Transformer 的强大表示能力
- 眼动分支用轻量 MLP 处理结构化特征
- 两者在分类前融合，互不干扰

---

## 四、自监督预训练 (Masked Signal Modeling)

### 4.1 预训练框架

**文件位置**: `model/biot_model.py:147-212`

```python
class BIOTForPretraining(nn.Module):
    def __init__(self, biot: BIOT, mask_ratio=0.15):
        self.biot = biot                    # 骨干网络
        self.mask_ratio = mask_ratio        # 掩码比例
        self.decoder = nn.Linear(d_model, patch_size)  # 简单线性解码器
```

**预训练任务**：掩码信号建模 (Masked Signal Modeling)

灵感来自 BERT 的掩码语言模型 (MLM) 和 MAE (Masked Autoencoder)：
1. 随机掩码 15% 的 patch token
2. 用 [MASK] embedding 替换被掩码位置
3. 通过 Transformer 编码
4. 用线性解码器预测被掩码 patch 的原始信号值
5. 损失函数：MSE (均方误差)

### 4.2 预训练流程图

```
原始信号 (B, 4, 800)
    │
    ├── PatchEmbedding → tokens (B, 64, 128)
    │
    ├── 随机选择 15% 的 token (约10个)
    │   mask_indices = [3, 17, 25, 42, ...]
    │
    ├── 记录被掩码 patch 的原始值 (监督信号)
    │   masked_targets = 原始patches[mask_indices]  (B, 10, 50)
    │
    ├── 用 [MASK] embedding 替换
    │   tokens[mask_indices] = learnable_mask_embed
    │
    ├── + TypeEmbedding + PositionalEncoding + CLS
    │
    ├── Transformer Encoder (6层)
    │
    ├── 提取被掩码位置的输出
    │   encoded[mask_indices] → (B, 10, 128)
    │
    ├── 线性解码器
    │   Linear(128 → 50) → 预测值 (B, 10, 50)
    │
    └── MSE Loss(predicted, original)
```

### 4.3 预训练的价值

| 方面 | 说明 |
|------|------|
| **数据效率** | 预训练不需要标签，可以利用所有数据（含未标注数据） |
| **通用表示** | 模型学到 EEG/EMG 信号的通用模式（节律、波形、通道关系） |
| **迁移学习** | 预训练权重可迁移到不同用户、不同任务 |
| **小数据集** | 在仅4765个样本的场景下，预训练能显著缓解过拟合 |

---

## 五、数据处理流程

### 5.1 prepare_biot_data 函数

**文件位置**: `model/biot_model.py:229-416`

```
原始数据 (DataFrame)
    │
    ├── 按用户划分训练/测试集 (user-level split)
    │   防止同一用户的数据同时出现在训练和测试集中（数据泄漏）
    │
    ├── 每个用户的数据独立处理:
    │   │
    │   ├── 重采样: 250Hz → 200Hz (scipy.resample_poly)
    │   │   减少计算量，与BIOT预训练设置对齐
    │   │
    │   ├── 滑动窗口: 4秒窗口, 2秒重叠
    │   │   每窗口 800点, 步进 400点
    │   │   窗口数 ≈ (总点数 - 800) / 400 + 1
    │   │
    │   ├── 标签取窗口中心点的标签
    │   │
    │   └── 记录窗口中心的原始时间戳 (用于眼动对齐)
    │
    ├── 汇总所有用户的窗口 → X (N, 4, 800), y (N,)
    │
    ├── [可选] 眼动特征提取与对齐
    │   │
    │   ├── 按用户调用 EyeTrackingFeatureExtractor
    │   │   提取22维特征 (眨眼、注视、扫视等)
    │   │
    │   ├── 时间戳最近邻对齐 (容差2秒)
    │   │   每个信号窗口匹配最近的眼动特征
    │   │
    │   └── 无眼动数据的窗口填0, 添加 has_eye_tracking 标志
    │
    └── 返回: X_train, X_test, y_train, y_test, eye_train, eye_test
```

### 5.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 原始采样率 | 250Hz | NeuroXess 设备输出 |
| 目标采样率 | 200Hz | BIOT 输入要求 |
| 窗口长度 | 4秒 (800点) | 与BIOT预训练设置一致 |
| 窗口重叠 | 2秒 (50%重叠) | 增加样本量 |
| patch_size | 50点 (250ms) | 覆盖一个 alpha 波周期 |
| patches/通道 | 16 | 800/50 |
| 总token数 | 64 | 4通道 × 16 patches |
| 测试集比例 | 20% | 用户级别分层划分 |

---

## 六、训练策略

### 6.1 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW | 带权重衰减的Adam |
| 学习率 | 5e-4 | Transformer 标准值 |
| 权重衰减 | 0.01 | L2正则化 |
| 学习率调度 | CosineAnnealing | 余弦退火，从5e-4逐渐降到接近0 |
| Batch Size | 32 | 平衡显存和梯度稳定性 |
| Epochs | 80 | 配合早停使用 |
| 早停 Patience | 15 | 15轮无提升则停止 |
| 梯度裁剪 | max_norm=1.0 | 防止梯度爆炸 |
| Label Smoothing | 0.1 | 软化标签，防止过拟合 |

### 6.2 类别平衡

```python
class_weights = [n_pos / n_neg, 1.0]
```
- 认知障碍(0)类样本较少，赋予更高的权重
- 权重比 = 正常样本数 / 障碍样本数
- 配合 CrossEntropyLoss 使用

### 6.3 训练循环

```
for epoch in range(80):
    │
    ├── 训练阶段
    │   for batch in train_loader:
    │       forward → loss → backward → clip_grad → step
    │
    ├── 学习率调度
    │   scheduler.step()  # CosineAnnealing
    │
    ├── 每5轮验证
    │   在测试集上计算 F1
    │   如果 F1 > best_f1 → 保存权重, patience_counter=0
    │   否则 → patience_counter += 1
    │
    └── 早停检查
        if patience_counter >= 15 → 停止训练
```

---

## 七、模型参数量分析

### 7.1 纯信号模式 (n_eye_features=0)

| 组件 | 参数量 | 计算 |
|------|--------|------|
| PatchEmbedding.projection | 6,528 | 50×128 + 128 |
| SignalTypeEmbedding | 256 | 2×128 |
| Positional Encoding | 8,320 | 65×128 |
| CLS Token | 128 | 1×128 |
| Transformer (6层) | ~1,198,080 | 每层 ~199,680 |
| LayerNorm | 256 | 128×2 |
| Classification Head | 16,770 | 128×128+128 + 128×2+2 |
| **总计** | **~1,230,346** | — |

### 7.2 信号+眼动模式 (n_eye_features=23)

| 组件 | 额外参数量 | 计算 |
|------|-----------|------|
| eye_proj | 1,536 | 23×64+64 |
| head (扩大) | 24,834 | 192×128+128 + 128×2+2 |
| **总增加** | **~26,370** | — |
| **总计** | **~1,256,716** | — |

---

## 八、与其他模型的对比

| 特性 | BIOT | EEGNet | RF |
|------|------|--------|-----|
| 架构 | Transformer | CNN | 决策树集成 |
| 输入 | 4通道原始信号 | 2通道原始信号 | 提取的特征 |
| 参数量 | ~1.2M | ~2K | N/A |
| 预训练 | 支持（自监督） | 不支持 | 不支持 |
| 眼动融合 | 混合融合 | 不支持 | 早期融合 |
| 可解释性 | 注意力权重 | 有限 | 特征重要性 |
| 训练速度 | 较慢 | 中等 | 快 |
| 适合数据量 | 小-中 | 小-中 | 中-大 |

### 实验结果 (4765样本, 90用户)

| 模型 | Accuracy | F1 | AUC-ROC |
|------|----------|------|---------|
| RF + 眼动 | 0.9238 | 0.9508 | 0.9823 |
| **BIOT 纯信号** | **0.9213** | **0.9444** | **0.9735** |
| BIOT + 眼动 | 0.9014 | 0.9291 | 0.9518 |
| RF 纯信号 | 0.8967 | 0.9332 | 0.9572 |

---

## 九、设计决策总结

| 决策 | 选择 | 原因 |
|------|------|------|
| Token 化方式 | Channel-Independent Patch | 保持通道独立性，避免信号混叠 |
| Patch 大小 | 50点 (250ms) | 覆盖 theta/alpha 波周期 |
| 信号类型嵌入 | EEG/EMG 二值嵌入 | 区分不同信号模态 |
| 位置编码 | 可学习 | 比固定正弦编码更灵活 |
| Norm 策略 | Pre-Norm | 训练更稳定，无需 warmup |
| CLS Token | 可学习 | 比平均池化更能学到任务相关聚合 |
| 眼动融合 | 混合融合 (concat) | 信号走 Transformer，眼动走 MLP，各取所长 |
| 预训练策略 | 掩码信号建模 | 无需外部数据，适合小数据集场景 |
| 激活函数 | GELU | Transformer 标准选择，比 ReLU 更平滑 |
| 早停 | patience=15 | 在小数据集上防止过拟合 |
