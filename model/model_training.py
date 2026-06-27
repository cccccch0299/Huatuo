import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
)
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境必须在 pyplot 之前指定后端
import matplotlib.pyplot as plt
import seaborn as sns


# ===================== 标签相关 =====================

def align_labels(features_df: pd.DataFrame, original_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    将原始数据中的 event_label 对齐到滑动窗口提取的特征中（用于无诊断标签时的探索）
    """
    if verbose:
        print("正在对齐 event_label 标签...")
    labels = []
    invalid_labels = {'', 'null', 'Null', 'Unknown', 'none', 'None'}

    for feature_time in features_df.index:
        time_window = original_df.loc[
            (original_df.index >= feature_time - pd.Timedelta(seconds=1)) &
            (original_df.index <= feature_time + pd.Timedelta(seconds=1))
        ]
        valid_labels = time_window['event_label'].dropna()
        valid_labels = valid_labels[~valid_labels.isin(invalid_labels)]

        if not valid_labels.empty:
            window_label = valid_labels.mode()[0]
        else:
            window_label = 'Unknown'
        labels.append(window_label)

    features_df = features_df.copy()
    features_df['label'] = labels
    labeled_df = features_df[features_df['label'] != 'Unknown'].copy()
    if verbose:
        print(f"标签对齐完成，获得 {len(labeled_df)} 个带标签的样本。")
    return labeled_df


def load_diagnosis_labels(csv_path: str, verbose: bool = False) -> dict[int, int]:
    """
    从 CSV/XLSX 文件加载认知障碍诊断标签
    :param csv_path: labels 文件路径，列: user_id, diagnosis (0=认知障碍, 1=正常)
    :return: {user_id: 0 或 1} 字典
    """
    # 兼容 xlsx 和 csv 两种格式
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_excel(csv_path)
    df = df.dropna(subset=['diagnosis'])
    df['diagnosis'] = pd.to_numeric(df['diagnosis'], errors='coerce')
    df = df.dropna(subset=['diagnosis'])
    df['diagnosis'] = df['diagnosis'].astype(int)
    df = df[df['diagnosis'].isin([0, 1])]
    label_map = dict(zip(df['user_id'].astype(int), df['diagnosis']))
    n_0 = sum(1 for v in label_map.values() if v == 0)
    n_1 = sum(1 for v in label_map.values() if v == 1)
    if verbose:
        print(f"从 {csv_path} 加载了 {len(label_map)} 个用户的诊断标签 (0={n_0}, 1={n_1})。")
    return label_map


def align_labels_from_csv(features_df: pd.DataFrame, label_map: dict[int, int], verbose: bool = False) -> pd.DataFrame:
    """
    根据 CSV 诊断标签对齐：每个 user_id 的所有特征窗口标记为对应的 diagnosis。
    要求 features_df 中包含 'user_id' 列。
    注意：保留 user_id 列，供后续按用户划分训练集/测试集使用。
    """
    if verbose:
        print("正在根据 CSV 诊断标签对齐...")
    features_df = features_df.copy()
    features_df['label'] = features_df['user_id'].map(label_map)
    labeled_df = features_df.dropna(subset=['label']).copy()
    labeled_df['label'] = labeled_df['label'].astype(int)
    if verbose:
        print(f"标签对齐完成，获得 {len(labeled_df)} 个带标签的样本。")
    return labeled_df


# ===================== 数据准备 =====================

def prepare_data(labeled_df: pd.DataFrame, test_size: float = 0.2, verbose: bool = False):
    """
    按用户划分训练集/测试集并进行标准化归一化。
    保证训练集和测试集中的用户完全不重叠，避免数据泄漏。
    :param labeled_df: 包含 'label' 列和 'user_id' 列的 DataFrame
    :param test_size: 测试集用户比例
    :return: X_train, X_test, y_train, y_test
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"数据划分 (按用户)")
        print(f"{'='*50}")

    has_user_id = 'user_id' in labeled_df.columns

    if has_user_id:
        # ---- 按用户划分 ----
        user_labels = labeled_df.groupby('user_id')['label'].first()
        users = user_labels.index.values
        user_y = user_labels.values

        if verbose:
            print(f"总用户数: {len(users)}")
            print(f"用户标签分布: 0(认知障碍)={sum(user_y==0)}, 1(正常)={sum(user_y==1)}")

        if sum(user_y == 0) < 1 or sum(user_y == 1) < 1:
            print("错误: 每个类别至少需要 1 个用户才能进行训练。")
            return None, None, None, None

        # 按用户分层抽样
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

        # 丢弃 user_id 列，不参与模型训练
        X_train = train_df.drop(columns=['label', 'user_id'])
        y_train = train_df['label']
        X_test = test_df.drop(columns=['label', 'user_id'])
        y_test = test_df['label']

        if verbose:
            print(f"训练集用户 ({len(train_users)}): {sorted(train_users)}")
            print(f"测试集用户 ({len(test_users)}): {sorted(test_users)}")

    else:
        # ---- 无 user_id 时按窗口划分 (兼容旧逻辑) ----
        X = labeled_df.drop(columns=['label'])
        y = labeled_df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)

    if verbose:
        print(f"\n训练集: {len(y_train)} 窗口 (0={sum(y_train==0)}, 1={sum(y_train==1)})")
        print(f"测试集: {len(y_test)} 窗口 (0={sum(y_test==0)}, 1={sum(y_test==1)})")

    # 特征标准化 (Z-score 归一化)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ===================== 模型训练与评估 =====================

MODEL_REGISTRY = {
    'rf': 'RandomForest',
    'xgb': 'XGBoost',
    'lgb': 'LightGBM',
    'eegnet': 'EEGNet',
    'multimodal': 'MultimodalTemporal',
    'biot': 'BIOT',
}


def create_model(model_name: str, y_train):
    """
    根据名称创建模型实例，自动处理类别不平衡
    :param model_name: 'rf' / 'xgb' / 'lgb'
    :param y_train: 训练集标签，用于计算类别权重
    :return: 模型实例
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_pos / max(n_neg, 1)

    if model_name == 'rf':
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == 'xgb':
        if XGBClassifier is None:
            raise ImportError("xgboost 未安装，请运行: pip install xgboost")
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == 'lgb':
        if LGBMClassifier is None:
            raise ImportError("lightgbm 未安装，请运行: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.1,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        raise ValueError(f"不支持的模型: {model_name}，可选: {list(MODEL_REGISTRY.keys())}")


def train_and_evaluate(X_train, X_test, y_train, y_test, save_dir: str = None, model_name: str = 'rf', use_smote: bool = False, verbose: bool = False, scaler=None):
    """
    训练二分类模型，输出完整评估指标，保存模型和图表
    :param save_dir: 结果保存目录
    :param model_name: 模型名称 'rf' / 'xgb' / 'lgb'
    :return: (model, report_text) 元组，report_text 为指标报告文本
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    model_display = MODEL_REGISTRY.get(model_name, model_name)
    log("\n" + "=" * 50)
    log(f"开始训练 {model_display} 二分类模型")
    log("=" * 50)

    # ---------- SMOTE 过采样 ----------
    if use_smote:
        if SMOTE is None:
            print("警告: imbalanced-learn 未安装，跳过 SMOTE。运行: pip install imbalanced-learn")
        else:
            n_before = len(y_train)
            n_neg_before = int((y_train == 0).sum())
            smote = SMOTE(random_state=42)
            X_train_arr, y_train_arr = smote.fit_resample(X_train.values, y_train.values)
            X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
            y_train = pd.Series(y_train_arr)
            n_after = len(y_train)
            n_neg_after = int((y_train == 0).sum())
            log(f"SMOTE 过采样: 训练样本 {n_before} -> {n_after} (障碍样本 {n_neg_before} -> {n_neg_after})")

    # ---------- 训练 ----------
    model = create_model(model_name, y_train)
    model.fit(X_train, y_train)

    # ---------- 交叉验证 ----------
    log("\n--- 5 折交叉验证 ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    log(f"训练集 5-Fold F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ---------- 阈值优化（在训练集上寻找最优阈值） ----------
    y_train_prob = model.predict_proba(X_train)[:, 1]
    best_threshold = 0.5
    best_f1_neg = 0.0
    for thr in np.arange(0.1, 0.9, 0.01):
        y_train_pred_thr = (y_train_prob >= thr).astype(int)
        # 计算障碍类(0)的 F1
        y_train_pred_neg = 1 - y_train_pred_thr
        y_train_neg = 1 - y_train.values
        f1_neg = f1_score(y_train_neg, y_train_pred_neg, zero_division=0)
        if f1_neg > best_f1_neg:
            best_f1_neg = f1_neg
            best_threshold = thr
    log(f"最优决策阈值: {best_threshold:.2f} (训练集障碍类 F1={best_f1_neg:.4f})")

    # ---------- 测试集预测 ----------
    y_prob = model.predict_proba(X_test)[:, 1]  # 类别 1 的概率
    y_pred = (y_prob >= best_threshold).astype(int)

    # ---------- 评估指标 ----------
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    log(f"\n{'=' * 50}")
    log(f"测试集评估结果 (阈值={best_threshold:.2f})")
    log(f"{'=' * 50}")
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
        plot_confusion_matrix(cm, save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        plot_roc_curve(y_test, y_prob, auc, save_path=os.path.join(save_dir, 'roc_curve.png'))
        plot_feature_importance(model, X_train.columns,
                               save_path=os.path.join(save_dir, 'feature_importance.png'))

    # ---------- 保存模型和 scaler ----------
    if save_dir:
        model_path = os.path.join(save_dir, f'{model_name}_model.joblib')
        joblib.dump(model, model_path)
        log(f"\n模型已保存至: {model_path}")

        if scaler is not None:
            scaler_path = os.path.join(save_dir, 'scaler.joblib')
            joblib.dump(scaler, scaler_path)
            log(f"Scaler 已保存至: {scaler_path}")

            inference_meta = {
                'threshold': best_threshold,
                'feature_names': list(X_train.columns),
                'model_name': model_name,
            }
            meta_path = os.path.join(save_dir, 'inference_meta.joblib')
            joblib.dump(inference_meta, meta_path)
            log(f"推理元数据已保存至: {meta_path}")

    report_text = "\n".join(report_lines)
    return model, report_text


# ===================== 可视化 =====================

def plot_confusion_matrix(cm, save_path: str = None):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Impaired(0)', 'Normal(1)'],
                yticklabels=['Impaired(0)', 'Normal(1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    _save_or_show(save_path)


def plot_roc_curve(y_test, y_prob, auc, save_path: str = None):
    """绘制 ROC 曲线"""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_feature_importance(model, feature_names, save_path: str = None, top_n: int = 15):
    """绘制特征重要性柱状图"""
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, hue='Feature', legend=False, palette='viridis')
    plt.title(f'Top {top_n} Most Important EEG/EMG Features')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    _save_or_show(save_path)


def _save_or_show(save_path: str = None):
    """保存图片或尝试显示"""
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    else:
        try:
            plt.show()
        except Exception:
            print("无 GUI 环境，跳过显示。")
    plt.close()


# ===================== 测试代码 =====================
if __name__ == "__main__":
    pass
