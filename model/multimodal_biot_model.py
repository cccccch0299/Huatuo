"""
Multimodal-BIOT 混合模型：用 BIOT 的 Patch-based Transformer 编码器替代多模态模型中的浅层 TemporalEncoder，
让模型直接从原始 EEG/EMG 信号中学习时序表示，并结合中文临床报告进行多模态联合推理。

架构概览:
  信号路: 原始信号 (B, 4, 800) → BIOTEncoder (PatchEmbed + TypeEmbed + 6层Transformer)
         → sequence_out (B, 65, 128) + cls_out (B, 128)
  文本路: 中文报告 → ChineseBERT+LoRA → text_emb (B, 128)
  融合:   CrossAttention(Q=text, KV=sequence) → context (B, 128)
  分类:   concat[cls_out, text_emb, context] = 384 → 64 → 2
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from biot_model import (
    PatchEmbedding, SignalTypeEmbedding, BIOT, BIOTForPretraining,
    _resample_signal, prepare_biot_data,
    _plot_confusion_matrix, _plot_roc_curve,
)
from multimodal_model import (
    PhysiologicalReportGenerator, TextEncoderWithLoRA, CrossAttentionFusion,
    LoRALayer,
)


# ===================== 模型组件 =====================

class BIOTEncoder(nn.Module):
    """
    BIOT 编码器：输出序列表示而非分类 logits。
    复用 BIOT 的 patch_embed, type_embed, pos_embed, cls_token, transformer, norm，
    去掉 head 和 eye_proj。

    输入: (B, C, T)  C=4, T=800
    输出: sequence_out (B, num_tokens+1, d_model), cls_out (B, d_model)
    """

    def __init__(self, n_channels=4, patch_size=50, d_model=128,
                 nhead=8, num_layers=6, dropout=0.1, max_seq_len=800):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
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

    def forward(self, x):
        """
        x: (B, C, T)
        返回: sequence_out (B, num_tokens+1, d_model), cls_out (B, d_model)
        """
        x = self.patch_embed(x)       # (B, num_tokens, d_model)
        x = self.type_embed(x)        # 信号类型嵌入
        x = x + self.pos_embed[:, 1:x.size(1) + 1, :]  # 位置编码 (跳过CLS位)

        # 添加 CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.embed_dropout(x)

        # CLS 位置编码
        x = x + self.pos_embed[:, :1, :]

        x = self.transformer(x)
        cls_out = self.norm(x[:, 0])       # CLS token (B, d_model)
        sequence_out = self.norm(x)        # 全序列 (B, num_tokens+1, d_model)

        return sequence_out, cls_out


class MultimodalBIOTModel(nn.Module):
    """
    Multimodal-BIOT 混合模型：
    信号路: BIOTEncoder (Patch-based Transformer) → cls_out(128) + sequence_out(65,128)
    文本路: ChineseBERT+LoRA → text_emb(128)
    融合:   CrossAttention(Q=text, KV=sequence) → context(128)
    分类:   concat[cls_out, text_emb, context] = 384 → 64 → 2
    """

    def __init__(self, n_channels=4, patch_size=50, d_model=128,
                 nhead=8, num_layers=6, text_dim=128,
                 lora_rank=4, lora_alpha=16, dropout=0.3,
                 max_seq_len=800):
        super().__init__()
        self.biot_encoder = BIOTEncoder(
            n_channels=n_channels, patch_size=patch_size, d_model=d_model,
            nhead=nhead, num_layers=num_layers, dropout=0.1,
            max_seq_len=max_seq_len,
        )
        self.text_encoder = TextEncoderWithLoRA(
            proj_dim=text_dim, lora_rank=lora_rank,
            lora_alpha=lora_alpha, dropout=dropout,
        )
        self.cross_attention = CrossAttentionFusion(
            temporal_dim=d_model, text_dim=text_dim,
            nhead=4, dropout=0.2,
        )
        # 分类头: concat[cls_out(128), text_emb(128), context(128)] = 384 → 64 → 2
        self.classifier = nn.Sequential(
            nn.Linear(d_model + text_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, signal, input_ids, attention_mask):
        """
        signal: (B, C, T) 原始信号
        input_ids: (B, seq_len) 文本 token IDs
        attention_mask: (B, seq_len) 文本注意力掩码
        返回: logits (B, 2)
        """
        # 信号编码
        sequence_out, cls_out = self.biot_encoder(signal)
        # sequence_out: (B, 65, 128), cls_out: (B, 128)

        # 文本编码
        text_emb = self.text_encoder(input_ids, attention_mask)  # (B, 128)

        # 交叉注意力
        context, _ = self.cross_attention(sequence_out, text_emb)  # (B, 128)

        # 拼接 + 分类
        combined = torch.cat([cls_out, text_emb, context], dim=-1)  # (B, 384)
        logits = self.classifier(combined)  # (B, 2)
        return logits


# ===================== 数据集 =====================

class MultimodalBIOTDataset(Dataset):
    """存储 (signal, input_ids, attention_mask, label) 的 PyTorch Dataset"""

    def __init__(self, signals, input_ids, attention_masks, labels):
        self.signals = signals              # (N, C, T) float32
        self.input_ids = input_ids          # (N, seq_len) long
        self.attention_masks = attention_masks  # (N, seq_len) long
        self.labels = labels                # (N,) long

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'signal': self.signals[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': self.labels[idx],
        }


# ===================== 数据准备 =====================

def prepare_multimodal_biot_data(combined_clean, combined_original=None,
                                  label_map=None, tokenizer=None,
                                  feature_names=None, patch_size=50,
                                  target_fs=200, window_sec=4.0,
                                  overlap_sec=2.0, test_size=0.2,
                                  verbose=False):
    """
    准备 Multimodal-BIOT 输入数据：信号路 + 文本路。

    信号路: 复用 prepare_biot_data 的逻辑 → X (N, 4, 800)
    文本路: 特征提取 → 报告生成 → tokenize → input_ids, attention_masks

    :param combined_clean: 预处理后的 DataFrame，包含 eeg_1_clean, eeg_2_clean, emg_1, emg_2, user_id, label
    :param combined_original: 原始 DataFrame (用于时间戳对齐)
    :param label_map: {user_id: label} 诊断标签映射
    :param tokenizer: BertTokenizer 实例
    :param feature_names: 特征列名列表 (用于报告生成)
    :return: (X_train, X_test, input_ids_train, input_ids_test,
              attn_train, attn_test, y_train, y_test, feature_names)
    """
    from feature_extraction import EEGFeatureExtractor

    if verbose:
        print(f"\n{'='*50}")
        print(f"Multimodal-BIOT 数据准备 (按用户划分)")
        print(f"{'='*50}")

    # ===== 1. 信号路: 复用 prepare_biot_data 的窗口提取逻辑 =====
    signal_cols = ['eeg_1_clean', 'eeg_2_clean', 'emg_1', 'emg_2']
    orig_fs = 250
    window_points = int(window_sec * target_fs)   # 800
    step_points = int((window_sec - overlap_sec) * target_fs)  # 400

    has_user_id = 'user_id' in combined_clean.columns

    # 恢复 DatetimeIndex
    if not isinstance(combined_clean.index, pd.DatetimeIndex) and combined_original is not None:
        if isinstance(combined_original.index, pd.DatetimeIndex) and len(combined_original) == len(combined_clean):
            combined_clean = combined_clean.copy()
            combined_clean.index = combined_original.index

    # ===== 2. 特征提取 (用于报告生成) =====
    if verbose:
        print("正在提取特征 (用于报告生成)...")
    extractor = EEGFeatureExtractor(fs=250, window_size_sec=2.0, overlap_sec=1.0)
    all_feat_dfs = []
    if has_user_id:
        for uid, group in combined_clean.groupby('user_id'):
            feat_df = extractor.extract_features(group.reset_index(drop=True), verbose=False)
            if not feat_df.empty:
                feat_df['user_id'] = uid
                all_feat_dfs.append(feat_df)
    else:
        feat_df = extractor.extract_features(combined_clean.reset_index(drop=True), verbose=False)
        if not feat_df.empty:
            all_feat_dfs.append(feat_df)

    if not all_feat_dfs:
        print("错误: 特征提取失败，无法生成报告。")
        return None

    combined_features = pd.concat(all_feat_dfs, axis=0)

    # 获取特征列名
    if feature_names is None:
        feature_names = [c for c in combined_features.columns if c != 'user_id']

    if verbose:
        print(f"特征维度: {len(feature_names)}")

    # 合并游戏特征 (如果有)
    if 'game_hit_accuracy' not in combined_features.columns:
        combined_features['game_hit_accuracy'] = 0.0
    if 'game_score' not in combined_features.columns:
        combined_features['game_score'] = 0.0

    # 合并 label
    if label_map and has_user_id:
        combined_features['label'] = combined_features['user_id'].map(label_map)
        combined_features = combined_features.dropna(subset=['label'])
        combined_features['label'] = combined_features['label'].astype(int)

    # 每个用户计算平均特征 (用于生成一份报告)
    user_feat_cols = [c for c in feature_names if c in combined_features.columns]
    if has_user_id and 'user_id' in combined_features.columns:
        user_mean_features = combined_features.groupby('user_id')[user_feat_cols].mean()
    else:
        user_mean_features = combined_features[user_feat_cols].mean().to_frame().T

    # 生成每个用户的报告
    report_gen = PhysiologicalReportGenerator(feature_names=user_feat_cols)
    user_reports = {}
    for uid in user_mean_features.index:
        feat_seq = user_mean_features.loc[uid].values.astype(np.float32).reshape(1, -1)
        game_acc = user_mean_features.loc[uid].get('game_hit_accuracy', None) if 'game_hit_accuracy' in user_mean_features.columns else None
        game_sc = user_mean_features.loc[uid].get('game_score', None) if 'game_score' in user_mean_features.columns else None
        report = report_gen.generate_report(feat_seq, game_accuracy=game_acc, game_score=game_sc)
        user_reports[uid] = report

    if verbose:
        print(f"已为 {len(user_reports)} 个用户生成报告。")

    # ===== 3. 用户级划分 =====
    if has_user_id:
        user_labels = combined_clean.groupby('user_id')['label'].first()
        users = user_labels.index.values
        user_y = user_labels.values

        if verbose:
            print(f"总用户数: {len(users)}")
            print(f"用户标签分布: 0(认知障碍)={sum(user_y==0)}, 1(正常)={sum(user_y==1)}")

        if sum(user_y == 0) < 1 or sum(user_y == 1) < 1:
            print("错误: 每个类别至少需要 1 个用户才能进行训练。")
            return None

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
        train_users = [0]
        test_users = []

    # ===== 4. 提取信号窗口 + tokenize 报告 =====
    def extract_windows_and_tokenize(df_part, part_user_ids):
        signals = []
        input_ids_list = []
        attn_masks_list = []
        labels = []

        if 'user_id' in df_part.columns:
            groups = df_part.groupby('user_id')
        else:
            groups = [('all', df_part)]

        for uid, group in groups:
            sig = group[signal_cols].values.astype(np.float32)
            lbl = group['label'].values if 'label' in group.columns else None

            # 重采样到目标采样率
            sig_resampled = _resample_signal(sig, orig_fs, target_fs)
            n_resampled = sig_resampled.shape[0]

            if lbl is not None:
                ratio = n_resampled / len(lbl)
                lbl_resampled = lbl[np.clip(
                    (np.arange(n_resampled) / ratio).astype(int), 0, len(lbl) - 1)]

            # 获取该用户的报告并 tokenize
            report = user_reports.get(uid, "")
            tokens = tokenizer(report, padding='max_length', truncation=True,
                               max_length=128, return_tensors='np')

            # 滑动窗口切分
            for start in range(0, n_resampled - window_points + 1, step_points):
                end = start + window_points
                window = sig_resampled[start:end]  # (window_points, 4)
                signals.append(window.T)  # (4, window_points)

                input_ids_list.append(tokens['input_ids'].squeeze(0))
                attn_masks_list.append(tokens['attention_mask'].squeeze(0))

                if lbl is not None:
                    center = start + window_points // 2
                    labels.append(int(lbl_resampled[center]))

        if not signals:
            return None, None, None, None

        X = np.stack(signals)
        ids = np.stack(input_ids_list)
        masks = np.stack(attn_masks_list)
        y = np.array(labels) if labels else None
        return X, ids, masks, y

    if verbose:
        print("正在提取训练集窗口...")
    X_train, ids_train, masks_train, y_train = extract_windows_and_tokenize(
        combined_clean[train_mask], train_users)

    if verbose:
        print("正在提取测试集窗口...")
    X_test, ids_test, masks_test, y_test = extract_windows_and_tokenize(
        combined_clean[test_mask], test_users)

    if X_train is None or X_test is None:
        print("错误: 数据提取失败，窗口数不足。")
        return None

    if verbose:
        print(f"\n训练集: {X_train.shape[0]} 窗口 (0={sum(y_train==0)}, 1={sum(y_train==1)})")
        print(f"测试集: {X_test.shape[0]} 窗口 (0={sum(y_test==0)}, 1={sum(y_test==1)})")
        print(f"信号形状: {X_train.shape}")  # (N, 4, 800)

    return (X_train, X_test, ids_train, ids_test,
            masks_train, masks_test, y_train, y_test, feature_names)


# ===================== 训练与评估 =====================

def train_and_evaluate_multimodal_biot(X_train, X_test, input_ids_train,
                                        input_ids_test, attn_train, attn_test,
                                        y_train, y_test, feature_names=None,
                                        save_dir=None, n_channels=4,
                                        patch_size=50, d_model=128, nhead=8,
                                        num_layers=6, text_dim=128,
                                        epochs=80, batch_size=8, lr=5e-4,
                                        weight_decay=0.01, patience=15,
                                        pretrained_model=None, verbose=False):
    """
    训练 Multimodal-BIOT 模型并输出评估指标。
    返回: (model, report_text)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_timepoints = X_train.shape[2]

    log(f"\n{'='*50}")
    log(f"开始训练 Multimodal-BIOT 二分类模型 (设备: {device})")
    log(f"{'='*50}")

    # 类别权重
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weights = torch.tensor(
        [n_pos / max(n_neg, 1), 1.0], dtype=torch.float32).to(device)
    log(f"类别权重: [障碍={class_weights[0]:.2f}, 正常={class_weights[1]:.2f}]")

    # 构建 Dataset 和 DataLoader
    train_dataset = MultimodalBIOTDataset(
        signals=torch.tensor(X_train, dtype=torch.float32),
        input_ids=torch.tensor(input_ids_train, dtype=torch.long),
        attention_masks=torch.tensor(attn_train, dtype=torch.long),
        labels=torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = MultimodalBIOTDataset(
        signals=torch.tensor(X_test, dtype=torch.float32),
        input_ids=torch.tensor(input_ids_test, dtype=torch.long),
        attention_masks=torch.tensor(attn_test, dtype=torch.long),
        labels=torch.tensor(y_test, dtype=torch.long),
    )

    # WeightedRandomSampler 确保每 batch 类别均衡
    sample_weights = torch.where(
        train_dataset.labels == 0,
        torch.tensor(1.0 / max(n_neg, 1)),
        torch.tensor(1.0 / max(n_pos, 1)),
    )
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, drop_last=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    # 构建模型
    model = MultimodalBIOTModel(
        n_channels=n_channels, patch_size=patch_size, d_model=d_model,
        nhead=nhead, num_layers=num_layers, text_dim=text_dim,
        lora_rank=4, lora_alpha=16, dropout=0.3,
        max_seq_len=n_timepoints,
    ).to(device)

    # 可选: 加载预训练 BIOT 权重到编码器
    if pretrained_model is not None:
        pretrained_state = pretrained_model.state_dict()
        encoder_state = {}
        for k, v in pretrained_state.items():
            # 预训练模型的 key 可能有 'biot.' 前缀 (来自 BIOTForPretraining)
            clean_key = k.replace('biot.', '') if k.startswith('biot.') else k
            if clean_key in model.biot_encoder.state_dict():
                encoder_state[clean_key] = v
        if encoder_state:
            model.biot_encoder.load_state_dict(encoder_state, strict=False)
            if verbose:
                log(f"已加载 {len(encoder_state)} 个预训练权重到 BIOTEncoder。")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"模型总参数量: {total_params:,}")
    log(f"可训练参数量: {trainable_params:,}")

    # 优化器: 不同学习率
    biot_params = list(model.biot_encoder.parameters())
    cross_params = list(model.cross_attention.parameters())
    cls_params = list(model.classifier.parameters())
    lora_params = model.text_encoder.parameters_that_require_grad()

    optimizer = optim.AdamW([
        {'params': biot_params + cross_params + cls_params, 'lr': lr, 'weight_decay': weight_decay},
        {'params': lora_params, 'lr': lr * 0.1, 'weight_decay': weight_decay},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ---------- 训练循环 ----------
    best_f1 = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            signal = batch['signal'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(signal, input_ids, attn_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # 每 5 轮验证
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    signal = batch['signal'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attn_mask = batch['attention_mask'].to(device)
                    logits = model(signal, input_ids, attn_mask)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch['label'].numpy())

            val_f1 = f1_score(all_labels, all_preds, zero_division=0)
            val_acc = accuracy_score(all_labels, all_preds)
            log(f"  Epoch {epoch+1:3d}/{epochs}  Loss={avg_loss:.4f}  Val_Acc={val_acc:.4f}  Val_F1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                log(f"  早停触发 (patience={patience})，停止训练。")
                break

    # ---------- 用最优权重评估 ----------
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            signal = batch['signal'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)

            logits = model(signal, input_ids, attn_mask)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch['label'].numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---------- 评估指标 ----------
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    log(f"\n{'='*50}")
    log(f"测试集评估结果 (Multimodal-BIOT)")
    log(f"{'='*50}")
    log(f"准确率 (Accuracy):  {acc:.4f}")
    log(f"F1 分数:            {f1:.4f}")
    log(f"精确率 (Precision): {prec:.4f}")
    log(f"召回率 (Recall):    {rec:.4f}")
    log(f"AUC-ROC:            {auc:.4f}")

    log(f"\n混淆矩阵:")
    log(f"                 预测正常(1)  预测障碍(0)")
    log(f"  实际正常(1)      {cm[1][1]:>6}        {cm[1][0]:>6}")
    log(f"  实际障碍(0)      {cm[0][1]:>6}        {cm[0][0]:>6}")

    log(f"\n分类报告:")
    log(classification_report(all_labels, all_preds,
                              target_names=['认知障碍(0)', '正常(1)'],
                              zero_division=0))

    # ---------- 保存图表 ----------
    if save_dir:
        _plot_confusion_matrix(cm, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        if len(np.unique(all_labels)) > 1:
            _plot_roc_curve(all_labels, all_probs, auc,
                            save_path=os.path.join(save_dir, 'roc_curve.png'))

    # ---------- 保存模型 ----------
    if save_dir and best_state:
        model_path = os.path.join(save_dir, 'multimodal_biot_model.pt')
        torch.save(best_state, model_path)
        log(f"\n模型已保存至: {model_path}")

    report_text = "\n".join(report_lines)
    return model, report_text
