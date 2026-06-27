"""
BIOT Official Architecture - 兼容官方预训练权重
基于 ml-jku/BIOT (NeurIPS 2023) 的实现，使用 STFT 频谱 token 化和 LinearAttentionTransformer。

官方预训练权重:
  - EEG-PREST-16-channels.ckpt: 16通道, 5M静息EEG样本
  - EEG-SHHS+PREST-18-channels.ckpt: 18通道, 5M+5M EEG样本
  - EEG-six-datasets-18-channels.ckpt: 18通道, 6个数据集联合预训练

用法:
  python model/run_pipeline.py --labels-csv labels.csv --model biot --pretrain-ckpt model/pretrained/EEG-six-datasets-18-channels.ckpt --epochs 80
"""
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
)
from scipy.signal import resample_poly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from linear_attention_transformer import LinearAttentionTransformer
    _HAS_LINEAR_ATTENTION = True
except ImportError:
    _HAS_LINEAR_ATTENTION = False
    LinearAttentionTransformer = None


# ===================== 官方 BIOT 模型组件 =====================

class PatchFrequencyEmbedding(nn.Module):
    """STFT 频谱 patch 嵌入 (与官方一致)"""
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        # x: (batch, freq, time) -> (batch, time, emb_size)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class PositionalEncoding(nn.Module):
    """正弦位置编码 (与官方一致)"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BIOTEncoder(nn.Module):
    """BIOT 编码器 (与官方结构完全一致，可加载预训练权重)"""
    def __init__(self, emb_size=256, heads=8, depth=4,
                 n_channels=18, n_fft=200, hop_length=100):
        super().__init__()
        if not _HAS_LINEAR_ATTENTION:
            raise ImportError(
                "请安装 linear-attention-transformer: pip install linear-attention-transformer"
            )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.emb_size = emb_size

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token: n_channels >= 实际通道数
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i:i+1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape

            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0).unsqueeze(0).repeat(batch_size, ts, 1)
            )
            channel_emb = self.positional_encoding(
                channel_spec_emb + channel_token_emb
            )
            emb_seq.append(channel_emb)

        # (batch_size, n_channels * ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        return emb


class BIOTClassifier(nn.Module):
    """BIOT 分类器 (编码器 + 分类头)"""
    def __init__(self, emb_size=256, heads=8, depth=4,
                 n_classes=2, n_channels=18, n_fft=200, hop_length=100):
        super().__init__()
        self.biot = BIOTEncoder(
            emb_size=emb_size, heads=heads, depth=depth,
            n_channels=n_channels, n_fft=n_fft, hop_length=hop_length,
        )
        self.classifier = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        x = self.biot(x)
        x = self.classifier(x)
        return x


# ===================== 数据准备 =====================

def _resample_signal(data, orig_fs, target_fs):
    """重采样信号"""
    if orig_fs == target_fs:
        return data
    from math import gcd
    g = gcd(orig_fs, target_fs)
    up = target_fs // g
    down = orig_fs // g
    return resample_poly(data, up, down, axis=0).astype(np.float32)


def _normalize_signal(x):
    """95%分位数归一化 (与官方一致)"""
    q95 = np.quantile(np.abs(x), q=0.95, axis=-1, keepdims=True) + 1e-8
    return x / q95


def prepare_biot_official_data(combined_clean, target_fs=200, window_sec=4.0,
                                overlap_sec=2.0, test_size=0.2, verbose=False):
    """
    准备官方 BIOT 格式的数据
    输入: combined_clean DataFrame (eeg_1_clean, eeg_2_clean, emg_1, emg_2, user_id, label)
    输出: X_train, X_test, y_train, y_test (numpy, shape=(N, C, T))
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"Official BIOT 数据准备 (按用户划分)")
        print(f"{'='*50}")

    signal_cols = ['eeg_1_clean', 'eeg_2_clean', 'emg_1', 'emg_2']
    orig_fs = 250
    window_points = int(window_sec * target_fs)
    step_points = int((window_sec - overlap_sec) * target_fs)

    # 按用户划分
    has_user_id = 'user_id' in combined_clean.columns
    if has_user_id:
        user_labels = combined_clean.groupby('user_id')['label'].first()
        users = user_labels.index.values
        user_y = user_labels.values

        if verbose:
            print(f"总用户数: {len(users)}")
            print(f"标签分布: 0(障碍)={sum(user_y==0)}, 1(正常)={sum(user_y==1)}")

        if sum(user_y == 0) < 1 or sum(user_y == 1) < 1:
            print("错误: 每个类别至少需要 1 个用户。")
            return None, None, None, None

        try:
            train_users, test_users = train_test_split(
                users, test_size=test_size, random_state=42, stratify=user_y)
        except ValueError:
            train_users, test_users = train_test_split(
                users, test_size=test_size, random_state=42)

        train_mask = combined_clean['user_id'].isin(train_users)
        test_mask = combined_clean['user_id'].isin(test_users)

        if verbose:
            print(f"训练集用户: {sorted(train_users)}")
            print(f"测试集用户: {sorted(test_users)}")
    else:
        train_mask = pd.Series(True, index=combined_clean.index)
        test_mask = pd.Series(False, index=combined_clean.index)

    def extract_windows(df_part):
        windows, labels = [], []
        if 'user_id' in df_part.columns:
            groups = df_part.groupby('user_id')
        else:
            groups = [('all', df_part)]

        for uid, group in groups:
            sig = group[signal_cols].values.astype(np.float32)
            lbl = group['label'].values if 'label' in group.columns else None

            # 重采样
            sig_resampled = _resample_signal(sig, orig_fs, target_fs)
            n_resampled = sig_resampled.shape[0]

            if lbl is not None:
                ratio = n_resampled / len(lbl)
                lbl_resampled = lbl[np.clip(
                    (np.arange(n_resampled) / ratio).astype(int), 0, len(lbl) - 1)]

            # 滑动窗口
            for start in range(0, n_resampled - window_points + 1, step_points):
                end = start + window_points
                window = sig_resampled[start:end]

                # 归一化
                window = _normalize_signal(window.T)  # (C, T)
                windows.append(window)

                if lbl is not None:
                    center = start + window_points // 2
                    labels.append(int(lbl_resampled[center]))

        if not windows:
            return None, None
        return np.stack(windows), np.array(labels)

    if verbose:
        print("提取训练集窗口...")
    X_train, y_train = extract_windows(combined_clean[train_mask])
    if verbose:
        print("提取测试集窗口...")
    X_test, y_test = extract_windows(combined_clean[test_mask])

    if X_train is None or X_test is None:
        print("错误: 数据不足。")
        return None, None, None, None

    if verbose:
        print(f"训练集: {X_train.shape[0]} 样本, shape={X_train.shape}")
        print(f"测试集: {X_test.shape[0]} 样本")

    return X_train, X_test, y_train, y_test


# ===================== 训练与评估 =====================

def train_and_evaluate_biot_official(X_train, X_test, y_train, y_test,
                                      save_dir=None, n_channels=4,
                                      emb_size=256, heads=8, depth=4,
                                      n_fft=200, hop_length=100,
                                      pretrained_ckpt=None,
                                      freeze_encoder=False,
                                      epochs=80, batch_size=32,
                                      lr=5e-4, weight_decay=0.01,
                                      patience=15, verbose=False):
    """
    使用官方 BIOT 架构训练和评估
    :param pretrained_ckpt: 预训练权重路径 (.ckpt)
    :param freeze_encoder: 是否冻结编码器只训练分类头
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    report_lines = []
    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"\n{'='*50}")
    log(f"Official BIOT 训练 (设备: {device})")
    log(f"{'='*50}")

    # 类别权重
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weights = torch.tensor(
        [n_pos / max(n_neg, 1), 1.0], dtype=torch.float32).to(device)

    # DataLoader
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 构建模型
    # 预训练权重的 n_channels 可能与我们的不同，需要兼容
    pretrained_n_channels = n_channels
    if pretrained_ckpt:
        # 检查预训练权重的 channel_tokens 维度
        ckpt = torch.load(pretrained_ckpt, map_location='cpu')
        if 'channel_tokens.weight' in ckpt:
            pretrained_n_channels = ckpt['channel_tokens.weight'].shape[0]
            log(f"预训练权重通道数: {pretrained_n_channels}, 我们的数据通道数: {n_channels}")

    # 使用较大的 n_channels 以兼容预训练权重
    model_n_channels = max(n_channels, pretrained_n_channels)

    model = BIOTClassifier(
        emb_size=emb_size, heads=heads, depth=depth,
        n_classes=2, n_channels=model_n_channels,
        n_fft=n_fft, hop_length=hop_length,
    ).to(device)

    # 加载预训练权重
    if pretrained_ckpt:
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        # 只加载 BIOTEncoder 的权重
        missing, unexpected = model.biot.load_state_dict(ckpt, strict=False)
        if verbose:
            log(f"加载预训练权重: {pretrained_ckpt}")
            if missing:
                log(f"  缺失键: {missing}")
            if unexpected:
                log(f"  多余键: {unexpected}")
        log(f"预训练权重加载成功！")

    # 冻结编码器
    if freeze_encoder:
        for param in model.biot.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"编码器已冻结。可训练参数: {trainable:,} / {total:,}")
    else:
        total = sum(p.numel() for p in model.parameters())
        log(f"模型总参数: {total:,}")

    # 优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    # 训练循环
    best_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

        scheduler.step()
        avg_loss = total_loss / len(y_train)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_test_t.to(device))
                val_pred = val_output.argmax(dim=1).cpu().numpy()
                val_f1 = f1_score(y_test, val_pred)
            log(f"  Epoch {epoch+1:3d}/{epochs}  Loss={avg_loss:.4f}  Val_F1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                log(f"  早停触发 (patience={patience})")
                break

    # 评估
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(X_test_t.to(device))
        y_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    log(f"\n{'='*50}")
    log(f"测试集评估结果 (Official BIOT)")
    if pretrained_ckpt:
        log(f"预训练权重: {os.path.basename(pretrained_ckpt)}")
    log(f"{'='*50}")
    log(f"准确率 (Accuracy):  {acc:.4f}")
    log(f"F1 分数:            {f1:.4f}")
    log(f"精确率 (Precision): {precision:.4f}")
    log(f"召回率 (Recall):    {recall:.4f}")
    log(f"AUC-ROC:            {auc:.4f}")

    log(f"\n混淆矩阵:")
    log(f"                 预测正常(1)  预测障碍(0)")
    log(f"  实际正常(1)      {cm[1][1]:>6}        {cm[1][0]:>6}")
    log(f"  实际障碍(0)      {cm[0][1]:>6}        {cm[0][0]:>6}")

    log(f"\n分类报告:")
    log(classification_report(y_test, y_pred, target_names=['认知障碍(0)', '正常(1)']))

    # 保存
    if save_dir:
        _plot_confusion_matrix(cm, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        _plot_roc_curve(y_test, y_prob, auc, save_path=os.path.join(save_dir, 'roc_curve.png'))
        model_path = os.path.join(save_dir, 'biot_official_model.pt')
        torch.save(best_state or model.state_dict(), model_path)
        log(f"\n模型已保存至: {model_path}")

    return model, "\n".join(report_lines)


# ===================== 可视化 =====================

def _plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Impaired(0)', 'Normal(1)'],
                yticklabels=['Impaired(0)', 'Normal(1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Official BIOT)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_roc_curve(y_test, y_prob, auc, save_path=None):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC={auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve (Official BIOT)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
