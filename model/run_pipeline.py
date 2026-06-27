"""
端到端 Pipeline：从 TimescaleDB 加载数据 -> 预处理 -> 特征提取 -> 模型训练
用法：
    conda run -n huatuo python model/run_pipeline.py
"""
import sys
import os
import argparse

# 将 model 目录加入 path，确保内部 import 可用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_eeg_data, load_game_features
from preprocess import process_pipeline
from feature_extraction import EEGFeatureExtractor, EyeTrackingFeatureExtractor
from model_training import (
    align_labels, prepare_data, train_and_evaluate,
    load_diagnosis_labels, align_labels_from_csv, MODEL_REGISTRY,
)
from eegnet_model import prepare_eegnet_data, train_and_evaluate_eegnet
from robust_eeg_cnn import RobustEEGCNN, prepare_robust_cnn_data, train_and_evaluate_robust_cnn
from multimodal_model import (
    prepare_multimodal_data, train_and_evaluate_multimodal,
    prepare_multimodal_kfold, train_and_evaluate_kfold,
    ReportPolisher, PhysiologicalReportGenerator,
)
from biot_model import prepare_biot_data, train_and_evaluate_biot, pretrain_biot
from biot_official import (
    prepare_biot_official_data, train_and_evaluate_biot_official,
)
from distillation import train_teacher_on_ds004504, transfer_backbone_weights, distill_to_biot
from multimodal_biot_model import (
    prepare_multimodal_biot_data, train_and_evaluate_multimodal_biot,
)

import pandas as pd
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm


def run_pipeline(user_ids: list[int] | None = None, min_rows: int = 5000, labels_csv: str = None, model_name: str = 'rf', epochs: int = 100, use_smote: bool = False, exclude_users: list[int] | None = None, verbose: bool = False, pretrain: bool = False, pretrain_ckpt: str = None, freeze_encoder: bool = False, use_eye_tracking: bool = False, teacher_epochs: int = 15):
    """
    完整的端到端流水线

    :param user_ids: 指定要加载的 user_id 列表。为 None 时自动选取有足够数据的用户。
    :param min_rows: 自动选取用户时的最小行数阈值
    :param labels_csv: 诊断标签 CSV 文件路径。提供后使用诊断标签训练，否则使用 event_label。
    :param model_name: 模型名称 'rf' / 'xgb' / 'lgb' / 'eegnet'
    :param verbose: 是否显示详细的处理日志
    """

    # 加载诊断标签（如果提供了 CSV）
    label_map = None
    if labels_csv:
        label_map = load_diagnosis_labels(labels_csv, verbose=verbose)
        if not label_map:
            print(f"CSV 中没有有效的诊断标签，请检查 {labels_csv}。")
            return

    # ========== 1. 加载数据 ==========
    if user_ids is None:
        if label_map:
            user_ids = list(label_map.keys())
            if verbose:
                print(f"从 CSV 中选取了 {len(user_ids)} 个有诊断标签的用户。")
        else:
            user_ids = auto_select_user_ids(min_rows)
            if not user_ids:
                print("未找到符合条件的用户数据，请检查数据库。")
                return

    # 排除指定用户
    if exclude_users:
        before = len(user_ids)
        user_ids = [u for u in user_ids if u not in exclude_users]
        print(f"已排除 {before - len(user_ids)} 个用户，剩余 {len(user_ids)} 个用户。")

    # ========== EEGNet 分支：使用原始信号 ==========
    if model_name == 'eegnet':
        _run_eegnet(user_ids, label_map, labels_csv, epochs=epochs, verbose=verbose)
        return

    # ========== RobustEEGCNN 分支：6层轻量CNN ==========
    if model_name == 'robust_cnn':
        _run_robust_cnn(user_ids, label_map, labels_csv, epochs=epochs, verbose=verbose)
        return

    # ========== RobustEEGCNN 蒸馏分支：ds004504 Teacher → Huatuo Student ==========
    if model_name == 'robust_cnn_distilled':
        _run_robust_cnn_distilled(user_ids, label_map, labels_csv,
                                  teacher_epochs=teacher_epochs,
                                  student_epochs=epochs, verbose=verbose)
        return

    # ========== RobustEEGCNN → BIOT 蒸馏分支 ==========
    if model_name == 'biot_distilled':
        _run_biot_distilled(user_ids, label_map, labels_csv,
                            teacher_epochs=teacher_epochs,
                            student_epochs=epochs, verbose=verbose)
        return

    # ========== Multimodal 分支：时序多模态模型 ==========
    if model_name == 'multimodal':
        _run_multimodal(user_ids, label_map, labels_csv, epochs=epochs, verbose=verbose)
        return

    # ========== BIOT 分支：Biosignal Transformer ==========
    if model_name == 'biot':
        if pretrain_ckpt:
            _run_biot_official(user_ids, label_map, labels_csv, epochs=epochs,
                               pretrain_ckpt=pretrain_ckpt,
                               freeze_encoder=freeze_encoder,
                               verbose=verbose)
        else:
            _run_biot(user_ids, label_map, labels_csv, epochs=epochs,
                      do_pretrain=pretrain, use_eye_tracking=use_eye_tracking,
                      verbose=verbose)
        return

    # ========== Multimodal-BIOT 分支：BIOT编码器 + 文本报告 ==========
    if model_name == 'multimodal_biot':
        _run_multimodal_biot(user_ids, label_map, labels_csv, epochs=epochs,
                             do_pretrain=pretrain, verbose=verbose)
        return

    # ========== 传统模型分支：特征提取 ==========
    all_features = []
    all_originals = []
    eye_extractor = EyeTrackingFeatureExtractor(fs=62.5, window_size_sec=4.0, overlap_sec=2.0) if use_eye_tracking else None

    for uid in tqdm(user_ids, desc="处理用户", unit="user"):
        if verbose:
            print(f"\n{'='*50}")
            print(f"处理用户: {uid}")
            print(f"{'='*50}")

        df = load_eeg_data(uid, verbose=verbose)
        if df.empty:
            if verbose:
                print(f"用户 {uid} 无数据，跳过。")
            continue

        df_clean = process_pipeline(df, verbose=verbose)
        df_original = df.copy()
        df_clean_reset = df_clean.reset_index(drop=True)

        # --- EEG 特征提取 ---
        extractor = EEGFeatureExtractor(fs=250, window_size_sec=2.0, overlap_sec=1.0)
        features_df = extractor.extract_features(df_clean_reset, verbose=verbose)

        if features_df.empty:
            if verbose:
                print(f"用户 {uid} 特征提取结果为空，跳过。")
            continue

        features_df['user_id'] = uid

        original_times = df_original.index
        feature_center_indices = features_df.index
        feature_center_times = [original_times[min(idx, len(original_times)-1)] for idx in feature_center_indices]
        features_df.index = pd.DatetimeIndex(feature_center_times)
        features_df.index.name = 'time'

        # --- 眼动特征提取（如果有且未禁用） ---
        eye_cols = ['blink_l', 'blink_r', 'gaze_x', 'gaze_y', 'gaze_z']
        has_eye = eye_extractor is not None and any(c in df_original.columns for c in eye_cols)
        if has_eye:
            eye_data = df_original[eye_cols].dropna(how='all')
            if len(eye_data) > eye_extractor.window_size:
                eye_feats = eye_extractor.extract_features(
                    df_original, verbose=verbose)
                if not eye_feats.empty:
                    # 时间对齐：用最近邻匹配 EEG 和眼动特征的时间戳
                    # eye_feats.index 来自 df_original 的 DatetimeIndex，已经是时间戳
                    eye_feats.index.name = 'time'
                    # 重采样到 EEG 特征的时间索引（取最近邻）
                    eye_feats_aligned = eye_feats.reindex(
                        features_df.index, method='nearest', tolerance=pd.Timedelta('2s'))
                    # 合并
                    features_df = pd.concat([features_df, eye_feats_aligned], axis=1)
                    features_df['has_eye_tracking'] = 1
                    if verbose:
                        print(f"用户 {uid} 已提取 {len(eye_feats_aligned.columns)} 个眼动特征。")
                else:
                    features_df['has_eye_tracking'] = 0
            else:
                features_df['has_eye_tracking'] = 0
        else:
            features_df['has_eye_tracking'] = 0

        all_features.append(features_df)
        all_originals.append(df_original)

    if not all_features:
        print("没有成功提取到任何特征，请检查数据。")
        return

    combined_features = pd.concat(all_features, axis=0)
    combined_original = pd.concat(all_originals, axis=0)

    # 填充眼动特征的 NaN（无眼动数据的用户，这些列会是 NaN → 填 0）
    if 'has_eye_tracking' in combined_features.columns:
        combined_features['has_eye_tracking'] = combined_features['has_eye_tracking'].fillna(0).astype(int)
        eye_feat_cols = [c for c in combined_features.columns
                        if c.startswith(('blink_', 'gaze_', 'saccade_'))]
        combined_features[eye_feat_cols] = combined_features[eye_feat_cols].fillna(0)
        n_with_eye = (combined_features['has_eye_tracking'] == 1).sum()
        n_without_eye = (combined_features['has_eye_tracking'] == 0).sum()
        if verbose:
            print(f"眼动特征: {len(eye_feat_cols)} 个特征列, "
                  f"有眼动 {n_with_eye} 样本, 无眼动 {n_without_eye} 样本")

    if verbose:
        print(f"\n合并后特征总数: {len(combined_features)}")
        print(f"合并后原始数据总数: {len(combined_original)}")

    # ========== 3.5 加载游戏特征并合并 ==========
    game_data = load_game_features(user_ids, verbose=verbose)
    if game_data:
        game_cols = {'game_hit_accuracy': np.nan, 'game_score': np.nan}
        for col, default in game_cols.items():
            if col not in combined_features.columns:
                combined_features[col] = default

        if 'user_id' in combined_features.columns:
            for uid, feats in game_data.items():
                mask = combined_features['user_id'] == uid
                for col, val in feats.items():
                    combined_features.loc[mask, col] = val
            combined_features[['game_hit_accuracy', 'game_score']] = combined_features[['game_hit_accuracy', 'game_score']].fillna(0.0)
            if verbose:
                print(f"已合并游戏特征 (game_hit_accuracy, game_score)，覆盖 {len(game_data)} 个用户。")
        else:
            print("警告: 特征中缺少 user_id 列，无法合并游戏特征。")
    else:
        combined_features['game_hit_accuracy'] = 0.0
        combined_features['game_score'] = 0.0
        if verbose:
            print("未找到游戏数据，游戏特征填充为 0。")

    # ========== 4. 标签对齐 ==========
    if label_map:
        labeled_df = align_labels_from_csv(combined_features, label_map, verbose=verbose)
    else:
        labeled_df = align_labels(combined_features, combined_original, verbose=verbose)

    if labeled_df.empty:
        if verbose:
            print("标签对齐后无有效样本。")
        return

    if verbose:
        print(f"\n标签分布:")
        print(labeled_df['label'].value_counts())

    # ========== 5. 准备数据并训练 ==========
    X_train, X_test, y_train, y_test, scaler = prepare_data(labeled_df, verbose=verbose)
    if X_train is None:
        return
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test')
    _, report_text = train_and_evaluate(X_train, X_test, y_train, y_test, save_dir=save_dir, model_name=model_name, use_smote=use_smote, verbose=verbose, scaler=scaler)

    # ========== 6. 保存指标报告到 result 文件夹 ==========
    _save_report(report_text, model_name, len(labeled_df))


def _run_eegnet(user_ids, label_map, labels_csv, epochs: int = 100, verbose: bool = False):
    """EEGNet 分支：使用原始 EEG 信号训练"""
    from model_training import load_diagnosis_labels as _unused  # 避免循环导入

    all_cleaned = []
    all_originals = []

    for uid in tqdm(user_ids, desc="处理用户", unit="user"):
        if verbose:
            print(f"\n{'='*50}")
            print(f"处理用户: {uid}")
            print(f"{'='*50}")

        df = load_eeg_data(uid, verbose=verbose)
        if df.empty:
            if verbose:
                print(f"用户 {uid} 无数据，跳过。")
            continue

        df_clean = process_pipeline(df, verbose=verbose)
        df_clean['user_id'] = uid
        all_cleaned.append(df_clean.reset_index(drop=True))
        all_originals.append(df.copy())

    if not all_cleaned:
        print("没有可用的数据。")
        return

    combined_clean = pd.concat(all_cleaned, axis=0)
    combined_original = pd.concat(all_originals, axis=0)

    # 标签对齐：对原始数据中的每个时间点标记 label
    invalid_labels = {'', 'null', 'Null', 'Unknown', 'none', 'None'}
    if label_map:
        combined_clean['label'] = combined_clean['user_id'].map(label_map)
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean['label'] = combined_clean['label'].astype(int)
    else:
        # 使用 event_label
        original_labels = combined_original['event_label'].reset_index(drop=True)
        combined_clean['label'] = original_labels
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean = combined_clean[~combined_clean['label'].isin(invalid_labels)]
        # 将 event_label 映射为数值：Start/Knock -> 1, 其他 -> 0
        positive_labels = {'Start', 'Knock'}
        combined_clean['label'] = combined_clean['label'].apply(lambda x: 1 if x in positive_labels else 0)

    if combined_clean.empty:
        if verbose:
            print("标签对齐后无有效样本。")
        return

    if verbose:
        print(f"\n标签分布:")
        print(combined_clean['label'].value_counts())

    # 准备 EEGNet 数据
    X_train, X_test, y_train, y_test = prepare_eegnet_data(combined_clean, verbose=verbose)
    if X_train is None:
        return

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test')
    _, report_text = train_and_evaluate_eegnet(X_train, X_test, y_train, y_test, save_dir=save_dir, epochs=epochs)

    _save_report(report_text, 'eegnet', len(combined_clean))


def _run_robust_cnn(user_ids, label_map, labels_csv, epochs: int = 100, verbose: bool = False):
    """RobustEEGCNN 分支：6层轻量CNN，使用原始 EEG+EMG 信号训练"""
    all_cleaned = []
    all_originals = []

    for uid in tqdm(user_ids, desc="处理用户", unit="user"):
        if verbose:
            print(f"\n{'='*50}")
            print(f"处理用户: {uid}")
            print(f"{'='*50}")

        df = load_eeg_data(uid, verbose=verbose)
        if df.empty:
            if verbose:
                print(f"用户 {uid} 无数据，跳过。")
            continue

        df_clean = process_pipeline(df, verbose=verbose)
        df_clean['user_id'] = uid
        all_cleaned.append(df_clean.reset_index(drop=True))
        all_originals.append(df.copy())

    if not all_cleaned:
        print("没有可用的数据。")
        return

    combined_clean = pd.concat(all_cleaned, axis=0)
    combined_original = pd.concat(all_originals, axis=0)

    # 标签对齐
    invalid_labels = {'', 'null', 'Null', 'Unknown', 'none', 'None'}
    if label_map:
        combined_clean['label'] = combined_clean['user_id'].map(label_map)
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean['label'] = combined_clean['label'].astype(int)
    else:
        original_labels = combined_original['event_label'].reset_index(drop=True)
        combined_clean['label'] = original_labels
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean = combined_clean[~combined_clean['label'].isin(invalid_labels)]
        positive_labels = {'Start', 'Knock'}
        combined_clean['label'] = combined_clean['label'].apply(lambda x: 1 if x in positive_labels else 0)

    if combined_clean.empty:
        if verbose:
            print("标签对齐后无有效样本。")
        return

    if verbose:
        print(f"\n标签分布:")
        print(combined_clean['label'].value_counts())

    # 准备数据
    X_train, X_test, y_train, y_test = prepare_robust_cnn_data(combined_clean, verbose=verbose)
    if X_train is None:
        return

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test')
    _, report_text = train_and_evaluate_robust_cnn(
        X_train, X_test, y_train, y_test, save_dir=save_dir, epochs=epochs)

    _save_report(report_text, 'robust_cnn', len(combined_clean))


def _run_robust_cnn_distilled(user_ids, label_map, labels_csv,
                               teacher_epochs: int = 15,
                               student_epochs: int = 100,
                               verbose: bool = False):
    """RobustEEGCNN 蒸馏分支：ds004504 Teacher → Huatuo Student"""
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test')
    teacher_path = os.path.join(save_dir, 'teacher_ds004504.pt')

    # Step 1: 训练/加载 Teacher
    if os.path.exists(teacher_path):
        print(f"加载已有 Teacher 模型: {teacher_path}")
        teacher = RobustEEGCNN(n_channels=19, n_timepoints=1280, num_classes=3)
        teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        teacher.eval()
    else:
        print("在 ds004504 上训练 Teacher 模型...")
        teacher = train_teacher(
            epochs=teacher_epochs, save_path=teacher_path, verbose=verbose)
        if teacher is None:
            print("Teacher 训练失败，无法进行蒸馏。")
            return

    # Step 2: 准备 Huatuo 数据
    all_cleaned = []
    all_originals = []

    for uid in tqdm(user_ids, desc="处理用户", unit="user"):
        if verbose:
            print(f"\n{'='*50}")
            print(f"处理用户: {uid}")
            print(f"{'='*50}")

        df = load_eeg_data(uid, verbose=verbose)
        if df.empty:
            if verbose:
                print(f"用户 {uid} 无数据，跳过。")
            continue

        df_clean = process_pipeline(df, verbose=verbose)
        df_clean['user_id'] = uid
        all_cleaned.append(df_clean.reset_index(drop=True))
        all_originals.append(df.copy())

    if not all_cleaned:
        print("没有可用的数据。")
        return

    combined_clean = pd.concat(all_cleaned, axis=0)
    combined_original = pd.concat(all_originals, axis=0)

    # 标签对齐
    invalid_labels = {'', 'null', 'Null', 'Unknown', 'none', 'None'}
    if label_map:
        combined_clean['label'] = combined_clean['user_id'].map(label_map)
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean['label'] = combined_clean['label'].astype(int)
    else:
        original_labels = combined_original['event_label'].reset_index(drop=True)
        combined_clean['label'] = original_labels
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean = combined_clean[~combined_clean['label'].isin(invalid_labels)]
        positive_labels = {'Start', 'Knock'}
        combined_clean['label'] = combined_clean['label'].apply(
            lambda x: 1 if x in positive_labels else 0)

    if combined_clean.empty:
        if verbose:
            print("标签对齐后无有效样本。")
        return

    if verbose:
        print(f"\n标签分布:")
        print(combined_clean['label'].value_counts())

    # Step 3: 准备 Student 数据
    from robust_eeg_cnn import prepare_robust_cnn_data
    X_train, X_test, y_train, y_test = prepare_robust_cnn_data(
        combined_clean, verbose=verbose)
    if X_train is None:
        return

    # Step 4: 蒸馏训练
    _, report_text = train_student_with_teacher_soft_labels(
        teacher, X_train, X_test, y_train, y_test,
        save_dir=save_dir, epochs=student_epochs)

    _save_report(report_text, 'robust_cnn_distilled', len(combined_clean))


def _run_biot_distilled(user_ids, label_map, labels_csv,
                         teacher_epochs: int = 15,
                         student_epochs: int = 80,
                         verbose: bool = False):
    """RobustEEGCNN → BIOT 蒸馏分支"""
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test')
    teacher_path = os.path.join(save_dir, 'teacher_huatuo_cnn.pt')
    ds004504_path = os.path.join(save_dir, 'teacher_ds004504.pt')

    # Step 1: 准备 Huatuo 数据
    all_cleaned = []
    all_originals = []

    for uid in tqdm(user_ids, desc="处理用户", unit="user"):
        if verbose:
            print(f"\n{'='*50}")
            print(f"处理用户: {uid}")
            print(f"{'='*50}")

        df = load_eeg_data(uid, verbose=verbose)
        if df.empty:
            if verbose:
                print(f"用户 {uid} 无数据，跳过。")
            continue

        df_clean = process_pipeline(df, verbose=verbose)
        df_clean['user_id'] = uid
        all_cleaned.append(df_clean.reset_index(drop=True))
        all_originals.append(df.copy())

    if not all_cleaned:
        print("没有可用的数据。")
        return

    combined_clean = pd.concat(all_cleaned, axis=0)
    combined_original = pd.concat(all_originals, axis=0)

    # 标签对齐
    invalid_labels = {'', 'null', 'Null', 'Unknown', 'none', 'None'}
    if label_map:
        combined_clean['label'] = combined_clean['user_id'].map(label_map)
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean['label'] = combined_clean['label'].astype(int)
    else:
        original_labels = combined_original['event_label'].reset_index(drop=True)
        combined_clean['label'] = original_labels
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean = combined_clean[~combined_clean['label'].isin(invalid_labels)]
        positive_labels = {'Start', 'Knock'}
        combined_clean['label'] = combined_clean['label'].apply(
            lambda x: 1 if x in positive_labels else 0)

    if combined_clean.empty:
        if verbose:
            print("标签对齐后无有效样本。")
        return

    if verbose:
        print(f"\n标签分布:")
        print(combined_clean['label'].value_counts())

    # Step 2: 训练 Teacher (RobustEEGCNN on Huatuo)
    teacher = None

    # 尝试加载已有的 Huatuo teacher
    if os.path.exists(teacher_path):
        print(f"加载已有 Huatuo Teacher: {teacher_path}")
        teacher = RobustEEGCNN(n_channels=4, n_timepoints=500, num_classes=2)
        teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        teacher.eval()
    else:
        # 先尝试用 ds004504 预训练权重初始化
        from robust_eeg_cnn import prepare_robust_cnn_data
        X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = \
            prepare_robust_cnn_data(combined_clean, verbose=verbose)
        if X_train_cnn is None:
            return

        teacher = RobustEEGCNN(n_channels=4, n_timepoints=500, num_classes=2)

        # 尝试加载 ds004504 预训练 backbone
        if os.path.exists(ds004504_path):
            print(f"加载 ds004504 预训练权重初始化 Teacher backbone...")
            ds004504_model = RobustEEGCNN(n_channels=19, n_timepoints=1280, num_classes=3)
            ds004504_model.load_state_dict(
                torch.load(ds004504_path, map_location='cpu'))
            n_transferred = transfer_backbone_weights(ds004504_model, teacher)
            print(f"迁移了 {n_transferred} 层 backbone 权重")
        else:
            print("ds004504 预训练权重不存在，从零训练 Teacher...")
            ds004504_model = train_teacher_on_ds004504(
                epochs=teacher_epochs, save_path=ds004504_path, verbose=verbose)
            if ds004504_model is not None:
                n_transferred = transfer_backbone_weights(ds004504_model, teacher)
                print(f"迁移了 {n_transferred} 层 backbone 权重")

        # 在 Huatuo 数据上微调 Teacher
        print("在 Huatuo 数据上微调 Teacher...")
        from robust_eeg_cnn import train_and_evaluate_robust_cnn
        teacher, _ = train_and_evaluate_robust_cnn(
            X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn,
            save_dir=None, n_channels=4, n_timepoints=500,
            epochs=teacher_epochs)

        # 保存 Teacher
        torch.save(teacher.state_dict(), teacher_path)
        print(f"Huatuo Teacher 已保存: {teacher_path}")

    # Step 3: 用 BIOT 数据准备 (500pt 窗口，与 Teacher 对齐)
    from biot_model import prepare_biot_data
    data = prepare_biot_data(
        combined_clean, patch_size=50, target_fs=250,
        window_sec=2.0, overlap_sec=1.0,
        combined_original=combined_original, verbose=verbose)
    X_train_biot, X_test_biot, y_train_biot, y_test_biot = data[:4]
    if X_train_biot is None:
        return

    # Step 4: 蒸馏训练
    _, report_text = distill_to_biot(
        teacher, X_train_biot, X_test_biot, y_train_biot, y_test_biot,
        save_dir=save_dir, epochs=student_epochs, verbose=verbose)

    _save_report(report_text, 'biot_distilled', len(combined_clean))


def _run_multimodal(user_ids, label_map, labels_csv, epochs: int = 80, verbose: bool = False):
    """Multimodal 分支：时序多模态模型，使用原始 EEG 特征序列 + 文本报告"""
    from transformers import BertTokenizer

    all_features = []
    all_originals = []

    for uid in tqdm(user_ids, desc="处理用户", unit="user"):
        if verbose:
            print(f"\n{'='*50}")
            print(f"处理用户: {uid}")
            print(f"{'='*50}")

        df = load_eeg_data(uid, verbose=verbose)
        if df.empty:
            if verbose:
                print(f"用户 {uid} 无数据，跳过。")
            continue

        df_clean = process_pipeline(df, verbose=verbose)
        df_original = df.copy()
        df_clean_reset = df_clean.reset_index(drop=True)

        extractor = EEGFeatureExtractor(fs=250, window_size_sec=2.0, overlap_sec=1.0)
        features_df = extractor.extract_features(df_clean_reset, verbose=verbose)

        if features_df.empty:
            if verbose:
                print(f"用户 {uid} 特征提取结果为空，跳过。")
            continue

        features_df['user_id'] = uid

        original_times = df_original.index
        feature_center_indices = features_df.index
        feature_center_times = [original_times[min(idx, len(original_times)-1)] for idx in feature_center_indices]
        features_df.index = pd.DatetimeIndex(feature_center_times)
        features_df.index.name = 'time'

        all_features.append(features_df)
        all_originals.append(df_original)

    if not all_features:
        print("没有成功提取到任何特征，请检查数据。")
        return

    combined_features = pd.concat(all_features, axis=0)
    combined_original = pd.concat(all_originals, axis=0)

    if verbose:
        print(f"\n合并后特征总数: {len(combined_features)}")

    # 加载游戏特征并合并
    game_data = load_game_features(user_ids, verbose=verbose)
    if game_data:
        game_cols = {'game_hit_accuracy': np.nan, 'game_score': np.nan}
        for col, default in game_cols.items():
            if col not in combined_features.columns:
                combined_features[col] = default

        if 'user_id' in combined_features.columns:
            for uid, feats in game_data.items():
                mask = combined_features['user_id'] == uid
                for col, val in feats.items():
                    combined_features.loc[mask, col] = val
            combined_features[['game_hit_accuracy', 'game_score']] = combined_features[['game_hit_accuracy', 'game_score']].fillna(0.0)
            if verbose:
                print(f"已合并游戏特征，覆盖 {len(game_data)} 个用户。")
    else:
        combined_features['game_hit_accuracy'] = 0.0
        combined_features['game_score'] = 0.0

    # 标签对齐（仅支持 CSV 标签模式）
    if not label_map:
        print("错误: 多模态模型仅支持 CSV 标签模式，请提供 --labels-csv 参数。")
        return
    labeled_df = align_labels_from_csv(combined_features, label_map, verbose=verbose)

    if labeled_df.empty:
        if verbose:
            print("标签对齐后无有效样本。")
        return

    if verbose:
        print(f"\n标签分布:")
        print(labeled_df['label'].value_counts())

    # 加载 tokenizer
    print("[Multimodal] 加载 chinese-roberta-wwm-ext tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    feature_names = [c for c in labeled_df.columns if c not in ('label', 'user_id')]
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test')

    # 分层 5 折交叉验证
    fold_pairs, n_features = prepare_multimodal_kfold(
        labeled_df,
        feature_names=feature_names,
        tokenizer=tokenizer,
        n_splits=5,
        max_seq_len=300,
        verbose=verbose,
    )
    if fold_pairs is None:
        return

    report_text = train_and_evaluate_kfold(
        fold_pairs, n_features,
        save_dir=save_dir, epochs=epochs, batch_size=8, verbose=verbose,
    )

    _save_report(report_text, 'multimodal', len(labeled_df))


def _run_biot(user_ids, label_map, labels_csv, epochs: int = 80,
              do_pretrain: bool = False, use_eye_tracking: bool = False,
              verbose: bool = False):
    """BIOT 分支：Biosignal Transformer，使用原始 EEG+EMG 信号训练"""
    eye_extractor = EyeTrackingFeatureExtractor(fs=62.5, window_size_sec=4.0, overlap_sec=2.0) if use_eye_tracking else None
    all_cleaned = []
    all_originals = []

    for uid in tqdm(user_ids, desc="处理用户", unit="user"):
        if verbose:
            print(f"\n{'='*50}")
            print(f"处理用户: {uid}")
            print(f"{'='*50}")

        df = load_eeg_data(uid, verbose=verbose)
        if df.empty:
            if verbose:
                print(f"用户 {uid} 无数据，跳过。")
            continue

        df_clean = process_pipeline(df, verbose=verbose)
        df_clean['user_id'] = uid
        all_cleaned.append(df_clean.reset_index(drop=True))
        all_originals.append(df.copy())

    if not all_cleaned:
        print("没有可用的数据。")
        return

    combined_clean = pd.concat(all_cleaned, axis=0)
    combined_original = pd.concat(all_originals, axis=0)

    # 标签对齐
    invalid_labels = {'', 'null', 'Null', 'Unknown', 'none', 'None'}
    if label_map:
        combined_clean['label'] = combined_clean['user_id'].map(label_map)
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean['label'] = combined_clean['label'].astype(int)
    else:
        original_labels = combined_original['event_label'].reset_index(drop=True)
        combined_clean['label'] = original_labels
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean = combined_clean[~combined_clean['label'].isin(invalid_labels)]
        positive_labels = {'Start', 'Knock'}
        combined_clean['label'] = combined_clean['label'].apply(
            lambda x: 1 if x in positive_labels else 0)

    if combined_clean.empty:
        if verbose:
            print("标签对齐后无有效样本。")
        return

    if verbose:
        print(f"\n标签分布:")
        print(combined_clean['label'].value_counts())

    # 准备 BIOT 数据
    X_train, X_test, y_train, y_test, eye_train, eye_test = prepare_biot_data(
        combined_clean, combined_original=combined_original,
        eye_extractor=eye_extractor, verbose=verbose)
    if X_train is None:
        return

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test')

    # 可选：自监督预训练
    pretrained_model = None
    if do_pretrain:
        pretrained_model = pretrain_biot(
            X_train, save_dir=save_dir, epochs=50, verbose=verbose)

    # 训练和评估
    _, report_text = train_and_evaluate_biot(
        X_train, X_test, y_train, y_test, save_dir=save_dir,
        epochs=epochs, pretrained_model=pretrained_model,
        eye_train=eye_train, eye_test=eye_test, verbose=verbose)

    _save_report(report_text, 'biot', len(combined_clean))


def _run_multimodal_biot(user_ids, label_map, labels_csv, epochs: int = 80,
                          do_pretrain: bool = False, verbose: bool = False):
    """Multimodal-BIOT 分支：BIOT编码器 + 中文文本报告的多模态模型"""
    from transformers import BertTokenizer

    all_cleaned = []
    all_originals = []

    for uid in tqdm(user_ids, desc="处理用户", unit="user"):
        if verbose:
            print(f"\n{'='*50}")
            print(f"处理用户: {uid}")
            print(f"{'='*50}")

        df = load_eeg_data(uid, verbose=verbose)
        if df.empty:
            if verbose:
                print(f"用户 {uid} 无数据，跳过。")
            continue

        df_clean = process_pipeline(df, verbose=verbose)
        df_clean['user_id'] = uid
        all_cleaned.append(df_clean.reset_index(drop=True))
        all_originals.append(df.copy())

    if not all_cleaned:
        print("没有可用的数据。")
        return

    combined_clean = pd.concat(all_cleaned, axis=0)
    combined_original = pd.concat(all_originals, axis=0)

    # 标签对齐
    if not label_map:
        print("错误: Multimodal-BIOT 仅支持 CSV 标签模式，请提供 --labels-csv 参数。")
        return

    combined_clean['label'] = combined_clean['user_id'].map(label_map)
    combined_clean = combined_clean.dropna(subset=['label'])
    combined_clean['label'] = combined_clean['label'].astype(int)

    if combined_clean.empty:
        if verbose:
            print("标签对齐后无有效样本。")
        return

    if verbose:
        print(f"\n标签分布:")
        print(combined_clean['label'].value_counts())

    # 加载 tokenizer
    print("[Multimodal-BIOT] 加载 chinese-roberta-wwm-ext tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    # 准备数据
    data = prepare_multimodal_biot_data(
        combined_clean, combined_original=combined_original,
        label_map=label_map, tokenizer=tokenizer,
        verbose=verbose)
    if data is None:
        return

    X_train, X_test, ids_train, ids_test, masks_train, masks_test, y_train, y_test, feature_names = data

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test')

    # 可选: 自监督预训练
    pretrained_model = None
    if do_pretrain:
        pretrained_model = pretrain_biot(
            X_train, save_dir=save_dir, epochs=50, verbose=verbose)

    # 训练和评估
    _, report_text = train_and_evaluate_multimodal_biot(
        X_train, X_test, ids_train, ids_test, masks_train, masks_test,
        y_train, y_test, feature_names=feature_names, save_dir=save_dir,
        epochs=epochs, pretrained_model=pretrained_model, verbose=verbose)

    _save_report(report_text, 'multimodal_biot', len(combined_clean))


def _run_biot_official(user_ids, label_map, labels_csv, epochs: int = 80,
                        pretrain_ckpt: str = None, freeze_encoder: bool = False,
                        verbose: bool = False):
    """Official BIOT 分支：使用官方预训练权重的 STFT-based Transformer"""
    all_cleaned = []
    all_originals = []

    for uid in tqdm(user_ids, desc="处理用户", unit="user"):
        if verbose:
            print(f"\n{'='*50}")
            print(f"处理用户: {uid}")
            print(f"{'='*50}")

        df = load_eeg_data(uid, verbose=verbose)
        if df.empty:
            if verbose:
                print(f"用户 {uid} 无数据，跳过。")
            continue

        df_clean = process_pipeline(df, verbose=verbose)
        df_clean['user_id'] = uid
        all_cleaned.append(df_clean.reset_index(drop=True))
        all_originals.append(df.copy())

    if not all_cleaned:
        print("没有可用的数据。")
        return

    combined_clean = pd.concat(all_cleaned, axis=0)
    combined_original = pd.concat(all_originals, axis=0)

    # 标签对齐
    invalid_labels = {'', 'null', 'Null', 'Unknown', 'none', 'None'}
    if label_map:
        combined_clean['label'] = combined_clean['user_id'].map(label_map)
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean['label'] = combined_clean['label'].astype(int)
    else:
        original_labels = combined_original['event_label'].reset_index(drop=True)
        combined_clean['label'] = original_labels
        combined_clean = combined_clean.dropna(subset=['label'])
        combined_clean = combined_clean[~combined_clean['label'].isin(invalid_labels)]
        positive_labels = {'Start', 'Knock'}
        combined_clean['label'] = combined_clean['label'].apply(
            lambda x: 1 if x in positive_labels else 0)

    if combined_clean.empty:
        if verbose:
            print("标签对齐后无有效样本。")
        return

    if verbose:
        print(f"\n标签分布:")
        print(combined_clean['label'].value_counts())

    # 准备数据 (200Hz, 4秒窗口)
    X_train, X_test, y_train, y_test = prepare_biot_official_data(
        combined_clean, verbose=verbose)
    if X_train is None:
        return

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test')

    # 训练和评估
    _, report_text = train_and_evaluate_biot_official(
        X_train, X_test, y_train, y_test, save_dir=save_dir,
        n_channels=X_train.shape[1],
        pretrained_ckpt=pretrain_ckpt,
        freeze_encoder=freeze_encoder,
        epochs=epochs, verbose=verbose)

    _save_report(report_text, 'biot_official', len(combined_clean))


def _save_report(report_text, model_name, total_samples):
    """保存指标报告到 result 文件夹"""
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'result')
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{date_str}_{model_name}_n{total_samples}.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n指标报告已保存至: {result_file}")


def auto_select_user_ids(min_rows: int = 5000) -> list[int]:
    """
    自动从数据库中选取有足够数据且包含有效标签的用户
    """
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
    db_url = os.getenv("DATABASE_URL")
    engine = create_engine(db_url)

    query = f"""
        SELECT user_id, COUNT(*) as cnt
        FROM eeg_data
        WHERE event_label IS NOT NULL
          AND event_label != ''
          AND event_label NOT IN ('null', 'Null')
        GROUP BY user_id
        HAVING COUNT(*) >= {min_rows}
        ORDER BY cnt DESC
    """

    with engine.connect() as conn:
        result = conn.execute(text(query))
        user_ids = [row[0] for row in result]

    print(f"自动选取了 {len(user_ids)} 个用户: {user_ids}")
    return user_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG 数据端到端处理流水线")
    parser.add_argument("--user-ids", nargs="+", type=int, default=None,
                        help="指定要处理的 user_id 列表，不指定则自动选取")
    parser.add_argument("--min-rows", type=int, default=5000,
                        help="自动选取用户的最小行数阈值 (默认 5000)")
    parser.add_argument("--labels-csv", type=str, default=None,
                        help="诊断标签 CSV 文件路径 (如 labels.csv)，提供后使用诊断标签训练")
    parser.add_argument("--model", type=str, default="rf",
                        choices=list(MODEL_REGISTRY.keys()) + ['eegnet', 'robust_cnn', 'robust_cnn_distilled', 'biot_distilled', 'multimodal', 'biot', 'biot_official', 'multimodal_biot'],
                        help="模型选择: rf/xgb/lgb=传统模型, eegnet=EEGNet, multimodal=时序多模态, biot=BIOT, multimodal_biot=多模态BIOT (默认 rf)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认 100，对 eegnet/multimodal 模型生效)")
    parser.add_argument("--teacher-epochs", type=int, default=15,
                        help="Teacher 训练轮数 (默认 15，对 robust_cnn_distilled 模型生效)")
    parser.add_argument("--smote", action="store_true", default=False,
                        help="对训练集使用 SMOTE 过采样平衡类别 (仅对传统模型生效)")
    parser.add_argument("--pretrain", action="store_true", default=False,
                        help="对 BIOT 模型进行自监督预训练 (对 biot 和 multimodal_biot 模型生效)")
    parser.add_argument("--pretrain-ckpt", type=str, default=None,
                        help="官方 BIOT 预训练权重路径 (.ckpt)，如 model/pretrained/EEG-six-datasets-18-channels.ckpt")
    parser.add_argument("--freeze-encoder", action="store_true", default=False,
                        help="冻结 BIOT 编码器，只训练分类头 (需配合 --pretrain-ckpt 使用)")
    parser.add_argument("--eye-tracking", action="store_true", default=False,
                        help="启用眼动特征融合 (默认禁用，仅使用 EEG/EMG 特征，对传统模型和 BIOT 生效)")
    parser.add_argument("--exclude-users", nargs="+", type=int, default=None,
                        help="要排除的 user_id 列表 (如弱信号用户)")
    parser.add_argument("--verbose", "-v", action="store_true", default=False,
                        help="显示详细的处理日志")

    args = parser.parse_args()
    run_pipeline(user_ids=args.user_ids, min_rows=args.min_rows, labels_csv=args.labels_csv, model_name=args.model, epochs=args.epochs, use_smote=args.smote, exclude_users=args.exclude_users, verbose=args.verbose, pretrain=args.pretrain, pretrain_ckpt=args.pretrain_ckpt, freeze_encoder=args.freeze_encoder, use_eye_tracking=args.eye_tracking, teacher_epochs=args.teacher_epochs)
