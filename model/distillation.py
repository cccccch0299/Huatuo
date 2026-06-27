"""
Knowledge Distillation: RobustEEGCNN Teacher → BIOT Student

流程:
1. Teacher: RobustEEGCNN(4ch, 500pt, 2cls) 在 Huatuo 上训练
   (可选用 ds004504 预训练 backbone 初始化)
2. Student: BIOT(4ch, 500pt, 2cls) 在 Huatuo 上训练，用 teacher 的 soft logit 蒸馏

两者处理相同的 500 点窗口数据，teacher 实时生成 soft label 指导 student。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
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

from robust_eeg_cnn import RobustEEGCNN, RobustEEGDataset
from biot_model import BIOT

# ds004504 路径
DS004504_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'other_dataset', 'ds004504',
                            'EEG-Alzheimer-Detection')
DS004504_CACHE = os.path.join(DS004504_DIR, 'data', 'raw_eeg_5s_clean.npz')


# ===================== ds004504 Teacher 预训练 =====================

def train_teacher_on_ds004504(epochs=15, batch_size=64, lr=1e-3,
                               save_path=None, verbose=False):
    """
    在 ds004504 上预训练 teacher (19ch, 1280pt, 3cls)，
    提取 backbone 权重用于后续迁移。
    """
    if not os.path.exists(DS004504_CACHE):
        print(f"错误: ds004504 缓存不存在: {DS004504_CACHE}")
        return None

    data = np.load(DS004504_CACHE, allow_pickle=True)
    signals = data['signals']  # (13919, 19, 1280)
    labels_raw = data['labels']
    subject_ids = data['subject_ids']
    data.close()

    label_to_3cls = {'C': 0, 'A': 1, 'F': 2}
    labels = np.array([label_to_3cls[l] for l in labels_raw])

    if verbose:
        from collections import Counter
        print(f"\n{'='*50}")
        print(f"ds004504 Teacher 预训练 (19ch, 1280pt, 3cls)")
        print(f"{'='*50}")
        print(f"数据: {signals.shape}, 标签: {Counter(labels_raw)}")

    sys.path.insert(0, os.path.join(DS004504_DIR, 'src'))
    from training_utils import subject_level_split
    train_idx, val_idx, _, _ = subject_level_split(
        subject_ids, labels, test_size=0.2, seed=42)

    X_train, X_val = signals[train_idx], signals[val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]

    train_ds = _DS004504Dataset(X_train, y_train, augment=True)
    val_ds = _DS004504Dataset(X_val, y_val, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RobustEEGCNN(n_channels=19, n_timepoints=1280,
                         num_classes=3, dropout=0.5).to(device)

    from collections import Counter
    counts = Counter(y_train)
    weights = torch.FloatTensor(
        [len(y_train) / (3 * counts[i]) for i in range(3)]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    if verbose:
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    best_val_acc = 0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                output = model(X_batch)
                loss = criterion(output, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y_batch)
            train_correct += (output.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += (output.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        val_acc = val_correct / val_total
        scheduler.step(val_loss / val_total)

        if verbose:
            print(f"  Epoch {epoch+1:2d}/{epochs}  "
                  f"Train={train_correct/train_total:.4f}  Val={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= 5:
            if verbose:
                print(f"  早停于 Epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state or model.state_dict(), save_path)
        if verbose:
            print(f"ds004504 Teacher 已保存: {save_path}")

    if verbose:
        print(f"ds004504 Teacher 最佳验证准确率: {best_val_acc:.4f}")

    model.eval()
    return model


class _DS004504Dataset(Dataset):
    def __init__(self, signals, labels, augment=False):
        self.signals = signals
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx].copy()
        if self.augment:
            if np.random.random() < 0.3:
                signal = np.roll(signal, np.random.randint(-50, 50), axis=1)
            if np.random.random() < 0.2:
                signal += np.random.normal(0, 0.01 * np.std(signal), signal.shape)
            if np.random.random() < 0.1:
                signal[np.random.randint(0, signal.shape[0]), :] = 0
        for c in range(signal.shape[0]):
            m, s = signal[c].mean(), signal[c].std()
            signal[c] = (signal[c] - m) / s if s > 1e-6 else 0
        return torch.from_numpy(signal).float(), self.labels[idx]


# ===================== 权重迁移 =====================

def transfer_backbone_weights(src_model, dst_model):
    """
    从 ds004504 teacher (19ch, 3cls) 迁移 backbone 权重到
    Huatuo teacher (4ch, 2cls)。

    迁移 Block 1,2,4,5,6 的 Conv+BN，跳过 Block 3 (空间滤波) 和 classifier。
    """
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    transferred = 0

    for name, param in src_state.items():
        # 跳过 Block 3 (features.6-9) 和 classifier
        if any(f'features.{i}' in name for i in [6, 7, 8, 9]) or \
           'classifier' in name:
            continue
        if name in dst_state and param.shape == dst_state[name].shape:
            dst_state[name] = param.clone()
            transferred += 1

    dst_model.load_state_dict(dst_state)
    return transferred


# ===================== 蒸馏训练 (RobustEEGCNN → BIOT) =====================

def distill_to_biot(teacher_model, X_train, X_test, y_train, y_test,
                    save_dir=None, epochs=80, batch_size=32, lr=5e-4,
                    T=3.0, alpha=0.7, n_channels=4, n_timepoints=500,
                    patch_size=50, d_model=128, nhead=8, num_layers=6,
                    verbose=False):
    """
    用 RobustEEGCNN teacher 蒸馏指导 BIOT student。

    两者处理相同的 (N, 4, 500) 数据。
    蒸馏损失: alpha * CE(student, label) + (1-alpha) * KL(soft_teacher || soft_student) * T^2

    Args:
        teacher_model: 已训练的 RobustEEGCNN(4ch, 500pt, 2cls)
        X_train, X_test: (N, 4, 500) numpy arrays
        y_train, y_test: (N,) numpy arrays
        T: 蒸馏温度 (越大 soft label 越平滑)
        alpha: 硬标签 CE 权重 (1-alpha 为 soft label KL 权重, 乘以 T^2)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    log(f"\n{'='*50}")
    log(f"知识蒸馏: RobustEEGCNN Teacher → BIOT Student")
    log(f"{'='*50}")
    log(f"Teacher: RobustEEGCNN(4ch, {n_timepoints}pt, 2cls)")
    log(f"Student: BIOT(4ch, {n_timepoints}pt, 2cls)")
    log(f"蒸馏温度 T={T}, alpha={alpha} (硬标签权重)")
    log(f"设备: {device}")

    # 类别权重
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    class_weights = torch.tensor(
        [n_pos / max(n_neg, 1), 1.0], dtype=torch.float32).to(device)

    # Teacher 预测整个训练集的 soft labels (避免每 batch 重复推理)
    log("预计算 Teacher soft labels...")
    teacher_train_logits = _predict_teacher(
        teacher_model, X_train, device, batch_size)
    teacher_test_logits = _predict_teacher(
        teacher_model, X_test, device, batch_size)
    log(f"Teacher soft labels: train {teacher_train_logits.shape}, test {teacher_test_logits.shape}")

    # DataLoader (含 teacher logits)
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(teacher_train_logits, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(teacher_test_logits, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # BIOT Student
    student = BIOT(
        n_channels=n_channels, patch_size=patch_size, d_model=d_model,
        nhead=nhead, num_layers=num_layers, n_classes=2, dropout=0.1,
        max_seq_len=n_timepoints,
    ).to(device)

    total_params = sum(p.numel() for p in student.parameters())
    log(f"Student 参数量: {total_params:,}")

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    best_f1 = 0
    best_state = None
    patience_counter = 0
    patience_limit = 10

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        total_ce = 0
        total_kl = 0

        for X_batch, teacher_logits_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            teacher_logits_batch = teacher_logits_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            student_logits = student(X_batch)

            # 硬标签损失
            ce_loss = F.cross_entropy(student_logits, y_batch, weight=class_weights)

            # 蒸馏 KL 散度损失
            with torch.no_grad():
                teacher_soft = F.softmax(teacher_logits_batch / T, dim=1)
            student_log_soft = F.log_softmax(student_logits / T, dim=1)
            kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')

            loss = alpha * ce_loss + (1 - alpha) * kl_loss * (T * T)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            total_ce += ce_loss.item() * len(y_batch)
            total_kl += kl_loss.item() * len(y_batch)

        scheduler.step()
        avg_loss = total_loss / len(y_train)

        # 每 5 轮评估
        if (epoch + 1) % 5 == 0 or epoch == 0:
            student.eval()
            all_preds = []
            with torch.no_grad():
                for X_batch, _, _ in test_loader:
                    X_batch = X_batch.to(device)
                    output = student(X_batch)
                    all_preds.extend(output.argmax(1).cpu().numpy())

            val_f1 = f1_score(y_test, all_preds)
            log(f"  Epoch {epoch+1:3d}/{epochs}  "
                f"Loss={avg_loss:.4f}  CE={total_ce/len(y_train):.4f}  "
                f"KL={total_kl/len(y_train):.4f}  Val_F1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone()
                              for k, v in student.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                log(f"  早停于 Epoch {epoch+1}")
                break

    # 最优权重评估
    if best_state:
        student.load_state_dict(best_state)
    student.eval()

    all_preds = []
    all_probs = []
    with torch.no_grad():
        for X_batch, _, _ in test_loader:
            X_batch = X_batch.to(device)
            logits = student(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(logits.argmax(1).cpu().numpy())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    log(f"\n{'='*50}")
    log(f"蒸馏 BIOT Student 测试集评估结果")
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
    log(classification_report(y_test, y_pred,
                              target_names=['认知障碍(0)', '正常(1)']))

    if save_dir:
        _plot_cm(cm, os.path.join(save_dir, 'distilled_biot_confusion_matrix.png'))
        _plot_roc(y_test, y_prob, auc,
                  os.path.join(save_dir, 'distilled_biot_roc_curve.png'))
        model_path = os.path.join(save_dir, 'distilled_biot_model.pt')
        torch.save(best_state or student.state_dict(), model_path)
        log(f"\n模型已保存至: {model_path}")

    return student, "\n".join(report_lines)


def _predict_teacher(model, X, device, batch_size=64):
    """用 teacher 对整个数据集生成 logits"""
    model.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits = []
    use_amp = device.type == 'cuda'
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                logits = model(X_batch)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


# ===================== 可视化 =====================

def _plot_cm(cm, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Impaired(0)', 'Normal(1)'],
                yticklabels=['Impaired(0)', 'Normal(1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Distilled BIOT)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.close()


def _plot_roc(y_test, y_prob, auc, save_path=None):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--',
             label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve (Distilled BIOT)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.close()
