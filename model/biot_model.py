"""
BIOT (Biosignal Transformer): 专为生物信号设计的 Transformer 架构
支持异构通道数据 (EEG + EMG)，通过自监督预训练提升泛化能力。

输入: (B, C, T)  C=4 (2 EEG + 2 EMG), T=800 (4秒@200Hz)
Patch: 每 patch 50 点 (250ms), 每通道 16 patches, 共 64 tokens
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
)
from scipy.signal import resample_poly
from scipy import signal as sp_signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ===================== 模型组件 =====================

class PatchEmbedding(nn.Module):
    """将原始信号分块并投影为 token"""

    def __init__(self, patch_size=50, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, d_model)

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        num_patches = T // self.patch_size
        x = x[:, :, :num_patches * self.patch_size]
        x = x.reshape(B, C, num_patches, self.patch_size)
        x = x.reshape(B, C * num_patches, self.patch_size)
        return self.projection(x)  # (B, C*num_patches, d_model)


class SignalTypeEmbedding(nn.Module):
    """为 EEG 和 EMG 通道的 patches 添加类型嵌入"""

    def __init__(self, n_eeg_channels=2, n_emg_channels=2,
                 patches_per_channel=16, d_model=128):
        super().__init__()
        self.type_embedding = nn.Embedding(2, d_model)  # 0=EEG, 1=EMG
        self.n_eeg_patches = n_eeg_channels * patches_per_channel
        self.n_emg_patches = n_emg_channels * patches_per_channel

    def forward(self, x):
        # x: (B, num_tokens, d_model)
        type_ids = torch.cat([
            torch.zeros(self.n_eeg_patches, dtype=torch.long),
            torch.ones(self.n_emg_patches, dtype=torch.long),
        ]).to(x.device)
        return x + self.type_embedding(type_ids)


class BIOT(nn.Module):
    """
    Biosignal Transformer for EEG/EMG classification
    输入: (B, C, T) -> 二分类 logits
    可选: 眼动特征融合 (n_eye_features > 0)
    """

    def __init__(self, n_channels=4, patch_size=50, d_model=128,
                 nhead=8, num_layers=6, n_classes=2, dropout=0.1,
                 max_seq_len=800, n_eye_features=0):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.n_eye_features = n_eye_features
        patches_per_channel = max_seq_len // patch_size
        max_tokens = n_channels * patches_per_channel + 1  # +1 for CLS

        self.patch_embed = PatchEmbedding(patch_size, d_model)
        self.type_embed = SignalTypeEmbedding(
            n_eeg_channels=2, n_emg_channels=n_channels - 2,
            patches_per_channel=patches_per_channel, d_model=d_model,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

        # 眼动特征投影（可选）
        if n_eye_features > 0:
            self.eye_proj = nn.Sequential(
                nn.Linear(n_eye_features, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_input_dim = d_model + d_model // 2
        else:
            self.eye_proj = None
            head_input_dim = d_model

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x, eye_features=None):
        # x: (B, C, T)
        x = self.patch_embed(x)       # (B, num_tokens, d_model)
        x = self.type_embed(x)        # 信号类型嵌入
        x = x + self.pos_embed[:, 1:x.size(1) + 1, :]  # 位置编码 (跳过CLS位)

        # 添加 CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.embed_dropout(x)

        # 位置编码中 CLS 对应的部分
        x = x + self.pos_embed[:, :1, :]  # CLS 位置编码

        x = self.transformer(x)
        cls_out = self.norm(x[:, 0])   # 取 CLS token (B, d_model)

        # 混合融合：拼接眼动特征
        if self.eye_proj is not None and eye_features is not None:
            eye_emb = self.eye_proj(eye_features)  # (B, d_model//2)
            cls_out = torch.cat([cls_out, eye_emb], dim=1)  # (B, d_model+d_model//2)

        return self.head(cls_out)


# ===================== 自监督预训练 =====================

class BIOTForPretraining(nn.Module):
    """
    掩码信号建模 (Masked Signal Modeling) 预训练
    随机掩码部分 patches，让模型预测被掩码的原始信号值
    """

    def __init__(self, biot: BIOT, mask_ratio=0.15):
        super().__init__()
        self.biot = biot
        self.mask_ratio = mask_ratio
        self.patch_size = biot.patch_size
        self.d_model = biot.d_model
        # 简单的线性 decoder: d_model -> patch_size
        self.decoder = nn.Linear(biot.d_model, self.patch_size)

    def forward(self, x):
        """
        x: (B, C, T)
        返回: loss, masked_patches_pred, mask_indices
        """
        B, C, T = x.shape
        num_patches_per_ch = T // self.patch_size
        total_patches = C * num_patches_per_ch

        # 1. Patch embedding (不经过 cls token)
        tokens = self.biot.patch_embed(x)            # (B, total_patches, d_model)
        tokens = self.biot.type_embed(tokens)

        # 2. 随机选择要掩码的位置
        num_mask = max(1, int(total_patches * self.mask_ratio))
        mask_indices = torch.stack([
            torch.randperm(total_patches)[:num_mask] for _ in range(B)
        ]).to(x.device)  # (B, num_mask)

        # 3. 记录被掩码 patch 的原始值
        # 重建原始 patches
        x_patches = x.reshape(B, C, num_patches_per_ch, self.patch_size)
        x_patches = x_patches.reshape(B, total_patches, self.patch_size)
        mask_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, self.patch_size)
        masked_targets = torch.gather(x_patches, 1, mask_expanded)  # (B, num_mask, patch_size)

        # 4. 用可学习的 [MASK] embedding 替换
        mask_embed = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02).to(x.device)
        tokens_masked = tokens.clone()
        mask_idx_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        tokens_masked.scatter_(1, mask_idx_expanded, mask_embed.expand(B, num_mask, -1))

        # 5. 加入 CLS 和位置编码
        cls = self.biot.cls_token.expand(B, -1, -1)
        tokens_masked = torch.cat([cls, tokens_masked], dim=1)
        tokens_masked = tokens_masked + self.biot.pos_embed[:, :tokens_masked.size(1), :]
        tokens_masked = self.biot.embed_dropout(tokens_masked)

        # 6. Transformer forward
        encoded = self.biot.transformer(tokens_masked)
        # 取被掩码位置的输出 (跳过 CLS token，索引 +1)
        mask_positions = mask_indices + 1
        mask_pos_expanded = mask_positions.unsqueeze(-1).expand(-1, -1, self.d_model)
        masked_encoded = torch.gather(encoded, 1, mask_pos_expanded)  # (B, num_mask, d_model)

        # 7. 解码预测
        pred = self.decoder(masked_encoded)  # (B, num_mask, patch_size)

        # 8. MSE loss
        loss = nn.functional.mse_loss(pred, masked_targets)
        return loss, pred, mask_indices


# ===================== 数据准备 =====================

def _resample_signal(data: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """将信号从 orig_fs 重采样到 target_fs"""
    if orig_fs == target_fs:
        return data
    # 使用 scipy 的 resample_poly 进行有理数重采样
    from math import gcd
    g = gcd(orig_fs, target_fs)
    up = target_fs // g
    down = orig_fs // g
    return resample_poly(data, up, down, axis=0).astype(np.float32)


def prepare_biot_data(combined_clean: pd.DataFrame, patch_size=50,
                      target_fs=200, window_sec=4.0, overlap_sec=2.0,
                      test_size=0.2, combined_original=None,
                      eye_extractor=None, verbose=False):
    """
    准备 BIOT 输入数据

    :param combined_clean: 预处理后的 DataFrame，包含 eeg_1_clean, eeg_2_clean, emg_1, emg_2, user_id, label
    :param patch_size: 每个 patch 的采样点数
    :param target_fs: 目标采样率
    :param window_sec: 窗口长度 (秒)
    :param overlap_sec: 重叠长度 (秒)
    :param test_size: 测试集比例
    :param combined_original: 原始 DataFrame (用于提取眼动特征)
    :param eye_extractor: EyeTrackingFeatureExtractor 实例 (None 则不提取眼动)
    :return: (X_train, X_test, y_train, y_test, eye_train, eye_test)
             eye_train/eye_test 为 None 时忽略
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"BIOT 数据准备 (按用户划分)")
        print(f"{'='*50}")

    signal_cols = ['eeg_1_clean', 'eeg_2_clean', 'emg_1', 'emg_2']
    orig_fs = 250
    window_points = int(window_sec * target_fs)   # 800
    step_points = int((window_sec - overlap_sec) * target_fs)  # 400

    has_user_id = 'user_id' in combined_clean.columns

    # 如果 combined_clean 没有 DatetimeIndex，尝试从 combined_original 恢复
    if not isinstance(combined_clean.index, pd.DatetimeIndex) and combined_original is not None:
        if isinstance(combined_original.index, pd.DatetimeIndex) and len(combined_original) == len(combined_clean):
            combined_clean = combined_clean.copy()
            combined_clean.index = combined_original.index

    # 按用户划分训练/测试集
    if has_user_id:
        user_labels = combined_clean.groupby('user_id')['label'].first()
        users = user_labels.index.values
        user_y = user_labels.values

        if verbose:
            print(f"总用户数: {len(users)}")
            print(f"用户标签分布: 0(认知障碍)={sum(user_y==0)}, 1(正常)={sum(user_y==1)}")

        if sum(user_y == 0) < 1 or sum(user_y == 1) < 1:
            print("错误: 每个类别至少需要 1 个用户才能进行训练。")
            return None, None, None, None

        try:
            train_users, test_users = train_test_split(
                users, test_size=test_size, random_state=42, stratify=user_y)
        except ValueError:
            print("警告: 用户数太少无法分层，改用随机抽样。")
            train_users, test_users = train_test_split(
                users, test_size=test_size, random_state=42)

        train_mask = combined_clean['user_id'].isin(train_users)
        test_mask = combined_clean['user_id'].isin(test_users)

        if verbose:
            print(f"训练集用户 ({len(train_users)}): {sorted(train_users)}")
            print(f"测试集用户 ({len(test_users)}): {sorted(test_users)}")
    else:
        train_mask = pd.Series(True, index=combined_clean.index)
        test_mask = pd.Series(False, index=combined_clean.index)

    def extract_windows(df_part):
        windows = []
        labels = []
        center_timestamps = []
        # 按用户分组处理（每个用户的数据是连续的）
        if 'user_id' in df_part.columns:
            groups = df_part.groupby('user_id')
        else:
            groups = [('all', df_part)]

        for uid, group in groups:
            sig = group[signal_cols].values.astype(np.float32)
            lbl = group['label'].values if 'label' in group.columns else None
            # 保留原始时间戳用于眼动特征对齐
            group_times = group.index if isinstance(group.index, pd.DatetimeIndex) else None

            # 重采样到目标采样率
            sig_resampled = _resample_signal(sig, orig_fs, target_fs)

            # 标签也需要对应重采样后的索引
            n_resampled = sig_resampled.shape[0]
            if lbl is not None:
                ratio = n_resampled / len(lbl)
                lbl_resampled = lbl[np.clip(
                    (np.arange(n_resampled) / ratio).astype(int), 0, len(lbl) - 1)]

            # 滑动窗口切分
            for start in range(0, n_resampled - window_points + 1, step_points):
                end = start + window_points
                window = sig_resampled[start:end]  # (window_points, 4)
                windows.append(window.T)  # (4, window_points)

                if lbl is not None:
                    # 取窗口中心的标签
                    center = start + window_points // 2
                    labels.append(int(lbl_resampled[center]))

                # 记录窗口中心的原始时间戳（用于眼动特征对齐）
                if group_times is not None:
                    orig_center_idx = int((start + window_points // 2) / (n_resampled / len(group_times)))
                    orig_center_idx = min(orig_center_idx, len(group_times) - 1)
                    center_timestamps.append(group_times[orig_center_idx])

        if not windows:
            return None, None, None
        X = np.stack(windows)  # (N, 4, window_points)
        y = np.array(labels) if labels else None
        return X, y, center_timestamps if center_timestamps else None

    if verbose:
        print("正在提取训练集窗口...")
    X_train, y_train, train_timestamps = extract_windows(combined_clean[train_mask])
    if verbose:
        print("正在提取测试集窗口...")
    X_test, y_test, test_timestamps = extract_windows(combined_clean[test_mask])

    if X_train is None or X_test is None:
        print("错误: 数据提取失败，窗口数不足。")
        return None, None, None, None, None, None

    if verbose:
        print(f"\n训练集: {X_train.shape[0]} 窗口 (0={sum(y_train==0)}, 1={sum(y_train==1)})")
        print(f"测试集: {X_test.shape[0]} 窗口 (0={sum(y_test==0)}, 1={sum(y_test==1)})")
        print(f"输入形状: {X_train.shape}")  # (N, 4, 800)

    # ===== 眼动特征提取（可选） =====
    eye_train = None
    eye_test = None
    if eye_extractor is not None and combined_original is not None:
        eye_cols = ['blink_l', 'blink_r', 'gaze_x', 'gaze_y', 'gaze_z']
        has_eye = any(c in combined_original.columns for c in eye_cols)

        if has_eye and train_timestamps is not None:
            if verbose:
                print(f"\n{'='*30}")
                print("眼动特征提取 (BIOT)")
                print(f"{'='*30}")

            # 按用户提取眼动特征
            all_eye_feats = []
            if 'user_id' in combined_original.columns:
                eye_groups = combined_original.groupby('user_id')
            else:
                eye_groups = [('all', combined_original)]

            for uid, group in eye_groups:
                group_eye = group[eye_cols].dropna(how='all')
                if len(group_eye) <= eye_extractor.window_size:
                    continue
                eye_feats = eye_extractor.extract_features(group, verbose=False)
                if not eye_feats.empty:
                    eye_feats['user_id'] = uid
                    all_eye_feats.append(eye_feats)

            if all_eye_feats:
                combined_eye = pd.concat(all_eye_feats, axis=0)
                combined_eye.index.name = 'time'

                # 对齐到训练集窗口
                if train_timestamps:
                    train_idx = pd.DatetimeIndex(train_timestamps)
                    eye_aligned = combined_eye.reindex(
                        train_idx, method='nearest', tolerance=pd.Timedelta('2s'))
                    eye_train = eye_aligned.drop(columns=['user_id'], errors='ignore').fillna(0).values.astype(np.float32)
                    has_eye_flag = (~eye_aligned.isna().all(axis=1)).astype(float).values.reshape(-1, 1)
                    eye_train = np.hstack([eye_train, has_eye_flag])

                # 对齐到测试集窗口
                if test_timestamps:
                    test_idx = pd.DatetimeIndex(test_timestamps)
                    eye_aligned = combined_eye.reindex(
                        test_idx, method='nearest', tolerance=pd.Timedelta('2s'))
                    eye_test = eye_aligned.drop(columns=['user_id'], errors='ignore').fillna(0).values.astype(np.float32)
                    has_eye_flag = (~eye_aligned.isna().all(axis=1)).astype(float).values.reshape(-1, 1)
                    eye_test = np.hstack([eye_test, has_eye_flag])

                if verbose:
                    n_eye_feats = eye_train.shape[1] if eye_train is not None else 0
                    print(f"眼动特征维度: {n_eye_feats} (含 has_eye_tracking 标志)")
                    n_with = np.sum(eye_train[:, -1] > 0) if eye_train is not None else 0
                    print(f"训练集有眼动数据: {n_with}/{len(eye_train)}")
            else:
                if verbose:
                    print("无可用的眼动数据，跳过眼动特征提取。")

    return X_train, X_test, y_train, y_test, eye_train, eye_test


# ===================== 训练与评估 =====================

def train_and_evaluate_biot(X_train, X_test, y_train, y_test,
                            save_dir=None, n_channels=4,
                            patch_size=50, d_model=128, nhead=8,
                            num_layers=6, epochs=80, batch_size=32,
                            lr=5e-4, weight_decay=0.01, patience=15,
                            pretrained_model=None, eye_train=None,
                            eye_test=None, verbose=False):
    """
    训练 BIOT 模型并输出评估指标
    :param eye_train: 训练集眼动特征 (N_train, n_eye_feats) 或 None
    :param eye_test: 测试集眼动特征 (N_test, n_eye_feats) 或 None
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_timepoints = X_train.shape[2]
    n_eye_feats = eye_train.shape[1] if eye_train is not None else 0

    log(f"\n{'='*50}")
    log(f"开始训练 BIOT 二分类模型 (设备: {device})")
    if n_eye_feats > 0:
        log(f"眼动特征融合: {n_eye_feats} 维 (含 has_eye_tracking)")
    log(f"{'='*50}")

    # 类别权重
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weights = torch.tensor(
        [n_pos / max(n_neg, 1), 1.0], dtype=torch.float32).to(device)

    # 构建 DataLoader
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    if eye_train is not None:
        eye_train_t = torch.tensor(eye_train, dtype=torch.float32)
        eye_test_t = torch.tensor(eye_test, dtype=torch.float32)
        train_ds = TensorDataset(X_train_t, eye_train_t, y_train_t)
    else:
        eye_train_t = None
        eye_test_t = None
        train_ds = TensorDataset(X_train_t, y_train_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)

    # 构建模型（优先使用预训练权重）
    if pretrained_model is not None:
        model = pretrained_model.to(device)
        # 如果预训练模型不支持眼动，需要重建 head
        if n_eye_feats > 0 and getattr(model, 'n_eye_features', 0) == 0:
            model.n_eye_features = n_eye_feats
            model.eye_proj = nn.Sequential(
                nn.Linear(n_eye_feats, d_model // 2),
                nn.GELU(),
                nn.Dropout(0.1),
            ).to(device)
            model.head = nn.Sequential(
                nn.Linear(d_model + d_model // 2, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, 2),
            ).to(device)
        if verbose:
            log("使用预训练权重初始化模型。")
    else:
        model = BIOT(
            n_channels=n_channels, patch_size=patch_size, d_model=d_model,
            nhead=nhead, num_layers=num_layers, n_classes=2, dropout=0.1,
            max_seq_len=n_timepoints, n_eye_features=n_eye_feats,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"模型总参数量: {total_params:,}")
    log(f"可训练参数量: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ---------- 训练循环 ----------
    best_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if eye_train is not None:
                X_batch, eye_batch, y_batch = batch
                X_batch, eye_batch, y_batch = X_batch.to(device), eye_batch.to(device), y_batch.to(device)
            else:
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                eye_batch = None
            optimizer.zero_grad()
            output = model(X_batch, eye_features=eye_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

        scheduler.step()
        avg_loss = total_loss / len(y_train)

        # 每 5 轮打印一次
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_eye = eye_test_t.to(device) if eye_test_t is not None else None
                val_output = model(X_test_t.to(device), eye_features=val_eye)
                val_pred = val_output.argmax(dim=1).cpu().numpy()
                val_f1 = f1_score(y_test, val_pred)
            log(f"  Epoch {epoch+1:3d}/{epochs}  Loss={avg_loss:.4f}  Val_F1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= patience:
                log(f"  早停触发 (patience={patience})，停止训练。")
                break

    # ---------- 用最优权重评估 ----------
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_eye = eye_test_t.to(device) if eye_test_t is not None else None
        logits = model(X_test_t.to(device), eye_features=final_eye)
        y_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_pred = logits.argmax(dim=1).cpu().numpy()

    # ---------- 评估指标 ----------
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    log(f"\n{'='*50}")
    log(f"测试集评估结果 (BIOT)")
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

    # ---------- 保存图表 ----------
    if save_dir:
        _plot_confusion_matrix(cm, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        _plot_roc_curve(y_test, y_prob, auc, save_path=os.path.join(save_dir, 'roc_curve.png'))

    # ---------- 保存模型 ----------
    if save_dir:
        model_path = os.path.join(save_dir, 'biot_model.pt')
        torch.save(best_state or model.state_dict(), model_path)
        log(f"\n模型已保存至: {model_path}")

    report_text = "\n".join(report_lines)
    return model, report_text


# ===================== 自监督预训练 =====================

def pretrain_biot(X_train, save_dir=None, n_channels=4, patch_size=50,
                  d_model=128, nhead=8, num_layers=6, epochs=50,
                  batch_size=32, lr=1e-3, mask_ratio=0.15, verbose=False):
    """
    自监督预训练 (掩码信号建模)
    :return: 预训练好的 BIOT 模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"\n{'='*50}")
        print(f"BIOT 自监督预训练 (掩码信号建模, 设备: {device})")
        print(f"{'='*50}")

    n_timepoints = X_train.shape[2]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    train_ds = TensorDataset(X_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=False)

    # 创建模型
    biot = BIOT(
        n_channels=n_channels, patch_size=patch_size, d_model=d_model,
        nhead=nhead, num_layers=num_layers, n_classes=2, dropout=0.1,
        max_seq_len=n_timepoints,
    )
    pretrain_model = BIOTForPretraining(biot, mask_ratio=mask_ratio).to(device)

    total_params = sum(p.numel() for p in pretrain_model.parameters())
    if verbose:
        print(f"预训练模型参数量: {total_params:,}")

    optimizer = optim.AdamW(pretrain_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        pretrain_model.train()
        total_loss = 0
        for (X_batch,) in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            loss, _, _ = pretrain_model(X_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        scheduler.step()
        avg_loss = total_loss / len(X_train)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:3d}/{epochs}  Pretrain_Loss={avg_loss:.6f}")

    # 保存预训练权重
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        pretrain_path = os.path.join(save_dir, 'biot_pretrain.pt')
        torch.save(biot.state_dict(), pretrain_path)
        if verbose:
            print(f"预训练权重已保存至: {pretrain_path}")

    # 返回骨干模型 (不含预训练 decoder)
    return biot


# ===================== 可视化 =====================

def _plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Impaired(0)', 'Normal(1)'],
                yticklabels=['Impaired(0)', 'Normal(1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (BIOT)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.close()


def _plot_roc_curve(y_test, y_prob, auc, save_path=None):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve (BIOT)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.close()
