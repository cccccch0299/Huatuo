"""
EEGNet: 紧凑型卷积神经网络，专为 EEG 信号分类设计
Lawhern et al., 2018 - "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
"""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ===================== EEGNet 模型定义 =====================

class EEGNet(nn.Module):
    """
    EEGNet: 紧凑型 CNN，专为多导联 EEG 时序信号设计
    输入形状: (batch, C, T)  C=通道数, T=时间采样点数
    """

    def __init__(self, n_channels=2, n_timepoints=500, n_classes=2,
                 F1=8, D=2, F2=16, dropout_rate=0.5, kernel_length=64):
        super().__init__()

        # ---- Block 1: 时间卷积 + 深度空间卷积 ----
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # ---- Block 2: 可分离卷积 ----
        self.conv3 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # depthwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),  # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # ---- 分类头 ----
        # 自动计算展平后的维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            flat_size = self._forward_features(dummy).shape[1]

        self.classifier = nn.Linear(flat_size, n_classes)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.flatten(1)

    def forward(self, x):
        # x: (batch, C, T) -> (batch, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._forward_features(x)
        return self.classifier(x)


# ===================== 数据准备 =====================

def prepare_eegnet_data(labeled_df: pd.DataFrame, n_timepoints: int = 500, test_size: float = 0.2, verbose: bool = False):
    """
    从带标签的 DataFrame 中准备 EEGNet 输入数据
    labeled_df 包含 eeg_1_clean, eeg_2_clean 列和 label, user_id 列
    按用户划分训练集/测试集
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"EEGNet 数据准备 (按用户划分)")
        print(f"{'='*50}")

    eeg_cols = ['eeg_1_clean', 'eeg_2_clean']
    n_channels = len(eeg_cols)

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
        train_df, test_df = train_test_split(labeled_df, test_size=test_size, random_state=42, stratify=labeled_df['label'])

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

def train_and_evaluate_eegnet(X_train, X_test, y_train, y_test,
                               save_dir: str = None,
                               n_channels: int = 2, n_timepoints: int = 500,
                               epochs: int = 100, batch_size: int = 32, lr: float = 1e-3):
    """
    训练 EEGNet 并输出评估指标
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"\n{'='*50}")
    log(f"开始训练 EEGNet 二分类模型 (设备: {device})")
    log(f"{'='*50}")

    # 类别权重
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weights = torch.tensor([n_pos / max(n_neg, 1), 1.0], dtype=torch.float32).to(device)

    # 构建 DataLoader
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    # 构建模型
    model = EEGNet(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数量: {total_params:,}")

    # ---------- 训练循环 ----------
    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)

        scheduler.step()
        avg_loss = total_loss / len(y_train)

        # 每 10 轮打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_test_t.to(device))
                val_pred = val_output.argmax(dim=1).cpu().numpy()
                val_f1 = f1_score(y_test, val_pred)
            log(f"  Epoch {epoch+1:3d}/{epochs}  Loss={avg_loss:.4f}  Val_F1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ---------- 用最优权重评估 ----------
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(X_test_t.to(device))
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
    log(f"测试集评估结果 (EEGNet)")
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
        model_path = os.path.join(save_dir, 'eegnet_model.pt')
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
    plt.title('Confusion Matrix (EEGNet)')
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
    plt.title('ROC Curve (EEGNet)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.close()
