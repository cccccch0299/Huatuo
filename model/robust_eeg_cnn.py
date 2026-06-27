"""
RobustEEGCNN: 6 层轻量 CNN，适配 Huatuo 4 通道 EEG+EMG 数据
基于 ds004504 EEG-Alzheimer-Detection 项目的 raw_eeg_cnn.py 架构
适配改动：n_channels 参数化 (19->4), num_classes 参数化 (3->2)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ===================== RobustEEGCNN 模型定义 =====================

class RobustEEGCNN(nn.Module):
    """
    6 层轻量 CNN，用于原始 EEG/EMG 信号分类。
    输入: (batch, 1, n_channels, n_timepoints)
    自动处理 3D (B,C,T) -> 4D (B,1,C,T)
    """
    def __init__(self, n_channels=4, n_timepoints=500, num_classes=2, dropout=0.5):
        super().__init__()
        self.n_channels = n_channels

        self.features = nn.Sequential(
            # Block 1: 时间特征 (1, C, T) -> (32, C, T/2)
            nn.Conv2d(1, 32, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            # Block 2: (32, C, T/2) -> (64, C, T/4)
            nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),

            # Block 3: 跨通道空间滤波 (64, C, T/4) -> (64, 1, T/4)
            nn.Conv2d(64, 64, kernel_size=(n_channels, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),

            # Block 4: (64, 1, T/4) -> (128, 1, T/8)
            nn.Conv2d(64, 128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),

            # Block 5: (128, 1, T/8) -> (128, 1, T/16)
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),

            # Block 6: (128, 1, T/16) -> (256, 1, T/32)
            nn.Conv2d(128, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # (B, C, T) -> (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


# ===================== 数据集 =====================

class RobustEEGDataset(Dataset):
    """带数据增强的 EEG/EMG 信号数据集"""
    def __init__(self, signals, labels, n_channels=4, augment=False):
        """
        signals: (N, C, T) numpy array
        labels: (N,) numpy array
        """
        self.signals = signals
        self.labels = torch.LongTensor(labels)
        self.n_channels = n_channels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()

        if self.augment:
            # 时间偏移 (30%)
            if np.random.random() < 0.3:
                shift = np.random.randint(-50, 50)
                signal = np.roll(signal, shift, axis=1)
            # 高斯噪声 (20%)
            if np.random.random() < 0.2:
                noise = np.random.normal(0, 0.01 * np.std(signal), signal.shape)
                signal = signal + noise
            # 通道 dropout (10%)
            if np.random.random() < 0.1:
                mask_ch = np.random.randint(0, self.n_channels)
                signal[mask_ch, :] = 0

        # 逐通道 z-score 标准化
        for c in range(signal.shape[0]):
            mean = signal[c].mean()
            std = signal[c].std()
            if std > 1e-6:
                signal[c] = (signal[c] - mean) / std
            else:
                signal[c] = 0

        return torch.from_numpy(signal).float(), self.labels[idx]


# ===================== 数据准备 =====================

def prepare_robust_cnn_data(labeled_df: pd.DataFrame, n_timepoints: int = 500,
                            test_size: float = 0.2, verbose: bool = False):
    """
    从预处理后的 DataFrame 准备 RobustEEGCNN 输入数据。
    提取 eeg_1_clean, eeg_2_clean, emg_1, emg_2 四列，按用户划分 train/test。
    输出: (N, 4, 500)
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"RobustEEGCNN 数据准备 (按用户划分)")
        print(f"{'='*50}")

    eeg_cols = ['eeg_1_clean', 'eeg_2_clean', 'emg_1', 'emg_2']
    n_channels = len(eeg_cols)

    # 检查列是否存在，缺失列用 0 填充
    for col in eeg_cols:
        if col not in labeled_df.columns:
            if verbose:
                print(f"警告: 缺少 {col} 列，用 0 填充。")
            labeled_df[col] = 0.0

    has_user_id = 'user_id' in labeled_df.columns

    if has_user_id:
        user_labels = labeled_df.groupby('user_id')['label'].first()
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

        train_mask = labeled_df['user_id'].isin(train_users)
        test_mask = labeled_df['user_id'].isin(test_users)
        train_df = labeled_df[train_mask]
        test_df = labeled_df[test_mask]

        if verbose:
            print(f"训练集用户 ({len(train_users)}): {sorted(train_users)}")
            print(f"测试集用户 ({len(test_users)}): {sorted(test_users)}")
    else:
        train_df, test_df = train_test_split(
            labeled_df, test_size=test_size, random_state=42, stratify=labeled_df['label'])

    def df_to_windows(df):
        X = df[eeg_cols].values.T  # (C, total_points)
        y = df['label'].values
        n_samples = X.shape[1] // n_timepoints
        X = X[:, :n_samples * n_timepoints]
        X = X.reshape(n_channels, n_samples, n_timepoints).transpose(1, 0, 2)  # (N, C, T)
        y = y[::n_timepoints][:n_samples]
        return X, y

    X_train, y_train = df_to_windows(train_df)
    X_test, y_test = df_to_windows(test_df)

    if verbose:
        print(f"\n训练集: {len(y_train)} 窗口 (0={sum(y_train==0)}, 1={sum(y_train==1)})")
        print(f"测试集: {len(y_test)} 窗口 (0={sum(y_test==0)}, 1={sum(y_test==1)})")

    return X_train, X_test, y_train, y_test


# ===================== 训练与评估 =====================

def train_and_evaluate_robust_cnn(X_train, X_test, y_train, y_test,
                                   save_dir: str = None,
                                   n_channels: int = 4, n_timepoints: int = 500,
                                   epochs: int = 100, batch_size: int = 32,
                                   lr: float = 1e-3):
    """
    训练 RobustEEGCNN 并输出完整评估指标。
    保留 ds004504 训练技巧：AMP、AdamW、ReduceLROnPlateau、早停、梯度裁剪。
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"\n{'='*50}")
    log(f"开始训练 RobustEEGCNN 二分类模型 (设备: {device})")
    log(f"{'='*50}")

    # 类别权重
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weights = torch.tensor([n_pos / max(n_neg, 1), 1.0], dtype=torch.float32).to(device)

    # 构建 DataLoader
    train_ds = RobustEEGDataset(X_train, y_train, n_channels=n_channels, augment=True)
    test_ds = RobustEEGDataset(X_test, y_test, n_channels=n_channels, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 构建模型
    model = RobustEEGCNN(n_channels=n_channels, n_timepoints=n_timepoints,
                         num_classes=2, dropout=0.5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # AMP
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数量: {total_params:,}")

    # ---------- 训练循环 ----------
    best_f1 = 0
    best_state = None
    patience_counter = 0
    patience_limit = 5

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                output = model(X_batch)
                loss = criterion(output, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y_batch)

        avg_loss = total_loss / len(y_train)

        # 每 10 轮或首轮打印
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            all_preds = []
            all_probs = []
            with torch.no_grad():
                for X_batch, _ in test_loader:
                    X_batch = X_batch.to(device)
                    with autocast(enabled=use_amp):
                        val_output = model(X_batch)
                    probs = torch.softmax(val_output, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(val_output.argmax(dim=1).cpu().numpy())

            val_f1 = f1_score(y_test, all_preds)
            scheduler.step(avg_loss)
            log(f"  Epoch {epoch+1:3d}/{epochs}  Loss={avg_loss:.4f}  Val_F1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                log(f"早停于 Epoch {epoch+1}")
                break

    # ---------- 用最优权重评估 ----------
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            with autocast(enabled=use_amp):
                logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # ---------- 评估指标 ----------
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    log(f"\n{'='*50}")
    log(f"测试集评估结果 (RobustEEGCNN)")
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
        _plot_confusion_matrix(cm, save_path=os.path.join(save_dir, 'robust_cnn_confusion_matrix.png'))
        _plot_roc_curve(y_test, y_prob, auc, save_path=os.path.join(save_dir, 'robust_cnn_roc_curve.png'))

    # ---------- 保存模型 ----------
    if save_dir:
        model_path = os.path.join(save_dir, 'robust_cnn_model.pt')
        torch.save(best_state or model.state_dict(), model_path)
        log(f"\n模型已保存至: {model_path}")

    report_text = "\n".join(report_lines)
    return model, report_text


# ===================== 可视化 =====================

def _plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Impaired(0)', 'Normal(1)'],
                yticklabels=['Impaired(0)', 'Normal(1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (RobustEEGCNN)')
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
    plt.title('ROC Curve (RobustEEGCNN)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.close()


# ===================== 形状验证 =====================

if __name__ == "__main__":
    print("="*50)
    print("RobustEEGCNN 形状验证")
    print("="*50)

    n_channels = 4
    n_timepoints = 500

    model = RobustEEGCNN(n_channels=n_channels, n_timepoints=n_timepoints, num_classes=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 测试 3D 输入 (B, C, T)
    x3d = torch.randn(2, n_channels, n_timepoints)
    out3d = model(x3d)
    print(f"3D 输入 {tuple(x3d.shape)} -> 输出 {tuple(out3d.shape)}")
    assert out3d.shape == (2, 2), f"期望 (2, 2)，实际 {out3d.shape}"

    # 测试 4D 输入 (B, 1, C, T)
    x4d = torch.randn(2, 1, n_channels, n_timepoints)
    out4d = model(x4d)
    print(f"4D 输入 {tuple(x4d.shape)} -> 输出 {tuple(out4d.shape)}")
    assert out4d.shape == (2, 2), f"期望 (2, 2)，实际 {out4d.shape}"

    print("\n所有形状验证通过!")
