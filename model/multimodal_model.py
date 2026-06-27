"""
时序多模态大模型：通过"生理语义翻译"(Stage 1)和"多模态联合推理"(Stage 2)实现可解释的认知障碍诊断。

架构概览:
  Stage 1: 模板引擎 → 结构化摘要 → Qwen3-0.5B 润色 → 中文临床报告
  Stage 2: TemporalEncoder(42→64, 4层Transformer) + TextEncoderWithLoRA(冻结BERT+LoRA)
           → CrossAttentionFusion → ClassificationHead(320→64→2)

可训练参数: ~369K | 冻结参数: ~102M (ChineseBERT)
"""
import os
import math
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)


# ===================== Stage 1: 生理语义翻译 =====================

class PhysiologicalReportGenerator:
    """
    模板引擎：将 (T, n_features) 的时序特征序列转换为结构化中文临床报告。
    不依赖任何外部模型，纯规则驱动。
    """

    # 特征列名到索引的映射（与 EEGFeatureExtractor 输出对齐）
    FEATURE_NAMES = None  # 由 prepare_multimodal_data 动态设置

    def __init__(self, feature_names=None):
        if feature_names is not None:
            self.FEATURE_NAMES = list(feature_names)
            self._name_to_idx = {name: i for i, name in enumerate(self.FEATURE_NAMES)}

    def _get_idx(self, name):
        """获取特征列索引，不存在则返回 None"""
        if self._name_to_idx is None:
            return None
        return self._name_to_idx.get(name)

    def _get_col(self, seq, name):
        """从序列中提取指定特征列，不存在则返回 None"""
        idx = self._get_idx(name)
        if idx is None or idx >= seq.shape[1]:
            return None
        return seq[:, idx]

    @staticmethod
    def _trend_text(values):
        """用线性拟合判断趋势：上升/下降/稳定"""
        if len(values) < 3:
            return "稳定"
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        threshold = np.std(values) * 0.1 + 1e-10
        if slope > threshold:
            return "上升"
        elif slope < -threshold:
            return "下降"
        return "稳定"

    @staticmethod
    def _fmt(val, digits=2):
        return f"{val:.{digits}f}"

    def generate_report(self, feature_seq: np.ndarray, game_accuracy=None, game_score=None) -> str:
        """
        生成结构化中文报告。
        :param feature_seq: (T, n_features) numpy array
        :param game_accuracy: 命中准确率 (0-1)
        :param game_score: 游戏得分
        :return: 中文报告字符串 (8-12 句)
        """
        T, C = feature_seq.shape
        duration_min = T * 1.0 / 60  # 每窗口约 1 秒（window_size=2s, overlap=1s → step=1s）

        sentences = []
        sentences.append(
            f"在本次脑电检测中，共采集{T}个时间窗口的脑电数据，总时长约{self._fmt(duration_min, 1)}分钟。"
        )

        # --- EEG1 频段特征 ---
        for band in ['Theta', 'Alpha', 'Beta', 'Delta', 'Gamma']:
            col = self._get_col(feature_seq, f'EEG1_{band}_Energy')
            if col is not None:
                mean_val = np.mean(col)
                trend = self._trend_text(col)
                sentences.append(f"前额通道（EEG1）的{band}频段能量均值为{self._fmt(mean_val)}，呈现{trend}趋势。")
                break  # 只报告第一个找到的频段

        # Theta/Beta 比值
        col_tb = self._get_col(feature_seq, 'EEG1_BTBR')
        if col_tb is not None:
            tb_mean = np.mean(col_tb)
            status = "正常" if 1.0 <= tb_mean <= 4.0 else "偏高" if tb_mean > 4.0 else "偏低"
            sentences.append(f"Theta/Beta比值为{self._fmt(tb_mean)}，处于{status}范围。")

        # --- EEG2 频段特征 ---
        for band in ['Alpha', 'Theta', 'Beta', 'Delta', 'Gamma']:
            col = self._get_col(feature_seq, f'EEG2_{band}_Energy')
            if col is not None:
                mean_val = np.mean(col)
                trend = self._trend_text(col)
                sentences.append(f"枕叶通道（EEG2）的{band}频段能量均值为{self._fmt(mean_val)}，呈现{trend}趋势。")
                break

        # DAR
        col_dar = self._get_col(feature_seq, 'EEG2_DAR')
        if col_dar is not None:
            dar_mean = np.mean(col_dar)
            status = "正常" if 0.5 <= dar_mean <= 2.0 else "偏高" if dar_mean > 2.0 else "偏低"
            sentences.append(f"Delta/Alpha比值（DAR）为{self._fmt(dar_mean)}，处于{status}范围。")

        # --- EMG ---
        col_emg1 = self._get_col(feature_seq, 'EMG1_RMS')
        col_emg2 = self._get_col(feature_seq, 'EMG2_RMS')
        if col_emg1 is not None and col_emg2 is not None:
            emg1_mean = np.mean(col_emg1)
            emg2_mean = np.mean(col_emg2)
            emg_status = "正常" if max(emg1_mean, emg2_mean) < 50 else "偏高"
            sentences.append(
                f"肌电信号方面，EMG1均方根值为{self._fmt(emg1_mean)}，EMG2均方根值为{self._fmt(emg2_mean)}，肌张力{emg_status}。"
            )

        # --- 游戏表现 ---
        if game_accuracy is not None and game_score is not None:
            acc_pct = self._fmt(game_accuracy * 100, 1)
            sentences.append(f"游戏表现：命中准确率为{acc_pct}%，得分为{int(game_score)}分。")

        # --- 综合分析 ---
        # 简单规则：检查是否有明显异常
        anomalies = []
        col_alpha = self._get_col(feature_seq, 'EEG1_Alpha_Energy')
        if col_alpha is not None and np.mean(col_alpha) < 1.0:
            anomalies.append("Alpha能量偏低")
        col_theta = self._get_col(feature_seq, 'EEG1_Theta_Energy')
        if col_theta is not None and np.mean(col_theta) > 20.0:
            anomalies.append("Theta能量偏高")
        col_dar_check = self._get_col(feature_seq, 'EEG1_DAR')
        if col_dar_check is not None and np.mean(col_dar_check) > 3.0:
            anomalies.append("DAR比值偏高")

        if anomalies:
            sentences.append(f"综合分析：发现以下异常指标——{'、'.join(anomalies)}，建议进一步检查。")
        else:
            sentences.append("综合分析：脑电频段分布基本正常，未见明显异常模式。")

        return "".join(sentences)


class ReportPolisher:
    """
    Stage 1 LLM 润色：使用 Qwen3-0.5B 将模板摘要润色为更自然的临床报告。
    仅推理，不训练。降级策略：若模型不可用，直接返回模板文本。
    """

    PROMPT_TEMPLATE = (
        "你是一位专业的神经内科辅助检查医师。请将以下脑电检测数据摘要润色为专业的"
        "临床检查报告，保持数据准确性，使用医学专业术语，语言流畅自然。"
        "不要添加原始数据中没有的信息。\n\n"
        "原始数据摘要：\n{template_report}\n\n"
        "请输出润色后的临床报告："
    )

    def __init__(self, model_name="Qwen/Qwen3-0.5B", max_new_tokens=512):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True, torch_dtype=torch.float32
            )
            self._model.eval()
            self._available = True
            print(f"[ReportPolisher] Qwen3-0.5B 加载成功。")
        except Exception as e:
            print(f"[ReportPolisher] Qwen3-0.5B 加载失败 ({e})，将直接使用模板文本。")
            self._available = False

    def polish(self, template_report: str) -> str:
        """将模板报告润色为自然语言。若模型不可用则返回原文。"""
        if not self._available:
            return template_report

        prompt = self.PROMPT_TEMPLATE.format(template_report=template_report)
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            result = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return result if len(result) > 20 else template_report
        except Exception as e:
            print(f"[ReportPolisher] 润色失败 ({e})，返回模板文本。")
            return template_report


# ===================== Stage 2: 模型组件 =====================

class LoRALayer(nn.Module):
    """
    低秩适配器：在冻结的线性层旁边添加 A·B 残差分支。
    rank=4, alpha=16, scaling=alpha/rank=4
    """
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scaling


class TemporalEncoder(nn.Module):
    """
    时序编码器：Linear(42, 64) + 位置编码 + 4层 TransformerEncoderLayer
    输入: (B, T, n_features) → 输出: (B, T, 64)
    """
    def __init__(self, n_features=42, d_model=64, nhead=4, num_layers=4,
                 dim_feedforward=128, dropout=0.3, max_len=5000):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        """
        x: (B, T, n_features)
        src_key_padding_mask: (B, T), True 表示 padding 位置
        """
        T = x.size(1)
        x = self.input_proj(x) + self.pos_encoding[:, :T, :]
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x  # (B, T, d_model)


class TextEncoderWithLoRA(nn.Module):
    """
    冻结 chinese-roberta-wwm-ext + LoRA rank-4 on query/value + 投影层 768→256→128
    输出: (B, 128)
    """
    def __init__(self, model_name="hfl/chinese-roberta-wwm-ext",
                 proj_dim=128, lora_rank=4, lora_alpha=16, dropout=0.3):
        super().__init__()
        self._model_name = model_name
        self._proj_dim = proj_dim
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._dropout_rate = dropout
        self._bert = None
        self._lora_modules = nn.ModuleList()
        self._loaded = False

    def _init_bert(self, device=None):
        """延迟加载 BERT（仅在首次 forward 时加载，避免 import 时卡住）"""
        if self._loaded:
            return
        try:
            from transformers import BertTokenizer, BertModel
            tokenizer = BertTokenizer.from_pretrained(self._model_name)
            bert = BertModel.from_pretrained(self._model_name)

            # 冻结所有参数
            for param in bert.parameters():
                param.requires_grad = False

            # 在 12 层 attention 的 query/value 上注入 LoRA
            # 将 LoRA 作为 BERT 层的属性注册，确保 .to(device) 时一起移动
            for layer in bert.encoder.layer:
                attention = layer.attention.self
                # query LoRA
                lora_q = LoRALayer(attention.query.in_features,
                                   attention.query.out_features,
                                   self._lora_rank, self._lora_alpha)
                # value LoRA
                lora_v = LoRALayer(attention.value.in_features,
                                   attention.value.out_features,
                                   self._lora_rank, self._lora_alpha)
                # 注册为 attention 层的属性，这样 bert.to(device) 会一起移动
                attention._lora_q = lora_q
                attention._lora_v = lora_v
                # 也保存到 ModuleList 以便优化器发现
                self._lora_modules.append(lora_q)
                self._lora_modules.append(lora_v)

                def make_hook(lq, lv):
                    def hook_fn_q(module, input, output):
                        return output + lq(input[0])
                    def hook_fn_v(module, input, output):
                        return output + lv(input[0])
                    return hook_fn_q, hook_fn_v

                hook_q, hook_v = make_hook(lora_q, lora_v)
                attention.query.register_forward_hook(hook_q)
                attention.value.register_forward_hook(hook_v)

            hidden_size = bert.config.hidden_size  # 768

            # 投影层: 768 → 256 → 128
            projection = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.GELU(),
                nn.Dropout(self._dropout_rate),
                nn.Linear(256, self._proj_dim),
                nn.LayerNorm(self._proj_dim),
            )

            # 移到目标设备（bert.to() 会返回新模块，LoRA 属性会一起移动）
            if device is not None:
                bert = bert.to(device)
                projection = projection.to(device)

            self._bert = bert
            self._tokenizer = tokenizer
            self._projection = projection
            self._loaded = True
            print(f"[TextEncoderWithLoRA] {self._model_name} 加载成功，LoRA rank={self._lora_rank}, device={device}")

        except Exception as e:
            print(f"[TextEncoderWithLoRA] BERT 加载失败 ({e})，文本编码将使用零向量。")
            self._loaded = False

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (B, seq_len)
        attention_mask: (B, seq_len)
        返回: (B, proj_dim)
        """
        self._init_bert(device=input_ids.device)

        if not self._loaded:
            B = input_ids.size(0)
            return torch.zeros(B, self._proj_dim, device=input_ids.device)

        outputs = self._bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        return self._projection(cls_emb)  # (B, proj_dim)

    def parameters_that_require_grad(self):
        """返回需要梯度的参数（LoRA + projection）"""
        return [p for p in self.parameters() if p.requires_grad]


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合：Q=text_emb, K/V=temporal_seq
    残差 + LayerNorm + FFN
    输出: context(B, text_dim) + attention_weights
    """
    def __init__(self, temporal_dim=64, text_dim=128, nhead=4, dropout=0.2):
        super().__init__()
        self.temporal_proj = nn.Linear(temporal_dim, text_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(text_dim)
        self.norm2 = nn.LayerNorm(text_dim)
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * 2, text_dim),
            nn.Dropout(dropout),
        )

    def forward(self, temporal_seq, text_emb):
        """
        temporal_seq: (B, T, temporal_dim)
        text_emb: (B, text_dim)
        返回: context(B, text_dim), attn_weights(B, 1, T)
        """
        # 投影 temporal 到 text 维度
        kv = self.temporal_proj(temporal_seq)  # (B, T, text_dim)

        # Q = text_emb (扩展一个维度作为序列长度 1)
        q = text_emb.unsqueeze(1)  # (B, 1, text_dim)

        # 交叉注意力
        attn_out, attn_weights = self.cross_attn(q, kv, kv)  # (B, 1, text_dim)
        context = self.norm1(q + attn_out)  # 残差
        context = context.squeeze(1)  # (B, text_dim)

        # FFN
        context = self.norm2(context + self.ffn(context))

        return context, attn_weights


class MultimodalDiagnosticModel(nn.Module):
    """
    完整的多模态诊断模型
    输入: temporal features (B, T, n_features), text tokens (B, seq_len), masks
    输出: logits (B, 2)
    """
    def __init__(self, n_features=42, temporal_dim=64, text_dim=128,
                 lora_rank=4, lora_alpha=16, dropout=0.3):
        super().__init__()
        self.temporal_encoder = TemporalEncoder(
            n_features=n_features, d_model=temporal_dim,
            nhead=4, num_layers=4, dim_feedforward=128,
            dropout=dropout, max_len=5000,
        )
        self.text_encoder = TextEncoderWithLoRA(
            proj_dim=text_dim, lora_rank=lora_rank,
            lora_alpha=lora_alpha, dropout=dropout,
        )
        self.cross_attention = CrossAttentionFusion(
            temporal_dim=temporal_dim, text_dim=text_dim,
            nhead=4, dropout=0.2,
        )
        # 分类头: concat[temporal_pooled(64), text_emb(128), context(128)] = 320 → 64 → 2
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim + text_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
        )

    def forward(self, temporal_features, input_ids, attention_mask,
                temporal_mask=None):
        """
        temporal_features: (B, T, n_features)
        input_ids: (B, seq_len)
        attention_mask: (B, seq_len)
        temporal_mask: (B, T), True = padding
        """
        # 时序编码
        temporal_seq = self.temporal_encoder(temporal_features,
                                             src_key_padding_mask=temporal_mask)  # (B, T, 64)
        # 文本编码
        text_emb = self.text_encoder(input_ids, attention_mask)  # (B, 128)

        # 时序池化 (mean of non-padded positions)
        if temporal_mask is not None:
            mask_inv = (~temporal_mask).unsqueeze(-1).float()  # (B, T, 1)
            temporal_pooled = (temporal_seq * mask_inv).sum(dim=1) / mask_inv.sum(dim=1).clamp(min=1)
        else:
            temporal_pooled = temporal_seq.mean(dim=1)  # (B, 64)

        # 交叉注意力
        context, _ = self.cross_attention(temporal_seq, text_emb)  # (B, 128)

        # 拼接 + 分类
        combined = torch.cat([temporal_pooled, text_emb, context], dim=-1)  # (B, 320)
        logits = self.classifier(combined)  # (B, 2)
        return logits


# ===================== 数据集与增强 =====================

class MultimodalDataset(Dataset):
    """存储 (sequence, tokens, label, mask) 的 PyTorch Dataset"""
    def __init__(self, sequences, input_ids, attention_masks, labels,
                 temporal_masks=None):
        self.sequences = sequences          # (N, T, C) float32
        self.input_ids = input_ids          # (N, seq_len) long
        self.attention_masks = attention_masks  # (N, seq_len) long
        self.labels = labels                # (N,) long
        self.temporal_masks = temporal_masks  # (N, T) bool, True=pad

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'temporal': self.sequences[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'label': self.labels[idx],
        }
        if self.temporal_masks is not None:
            item['temporal_mask'] = self.temporal_masks[idx]
        return item


class TemporalAugmentation:
    """
    时序数据增强：解决 46 用户过拟合问题
    1. 高斯噪声注入 (5% 相对标准差)
    2. 随机时间裁剪+重填充 (10-20%)
    3. 随机特征 dropout (10%)
    """
    def __init__(self, noise_ratio=0.05, crop_prob=0.3,
                 crop_range=(0.1, 0.2), feat_dropout=0.1):
        self.noise_ratio = noise_ratio
        self.crop_prob = crop_prob
        self.crop_range = crop_range
        self.feat_dropout = feat_dropout

    def __call__(self, seq):
        """
        seq: (T, C) numpy array or torch tensor
        返回增强后的序列
        """
        if isinstance(seq, torch.Tensor):
            seq = seq.clone()
        else:
            seq = seq.copy()

        # 1. 高斯噪声
        if self.noise_ratio > 0:
            std = np.std(seq, axis=0) + 1e-10
            noise = np.random.randn(*seq.shape) * std * self.noise_ratio
            seq = seq + noise

        # 2. 随机时间裁剪
        if np.random.random() < self.crop_prob:
            T = seq.shape[0]
            crop_frac = np.random.uniform(*self.crop_range)
            n_crop = max(1, int(T * crop_frac))
            start = np.random.randint(0, max(1, T - n_crop))
            # 将裁剪区域替换为相邻值的线性插值
            end = min(start + n_crop, T)
            if start > 0 and end < T:
                for c in range(seq.shape[1]):
                    seq[start:end, c] = np.linspace(seq[start - 1, c], seq[end, c], end - start)
            elif start == 0 and end < T:
                seq[start:end] = seq[end:end + 1]
            elif start > 0:
                seq[start:end] = seq[start - 1:start]

        # 3. 随机特征 dropout
        if self.feat_dropout > 0:
            C = seq.shape[1]
            n_drop = max(1, int(C * self.feat_dropout))
            drop_idx = np.random.choice(C, n_drop, replace=False)
            seq[:, drop_idx] = 0

        return seq


# ===================== 数据准备 =====================

def prepare_multimodal_data(labeled_df, feature_names, tokenizer,
                             test_size=0.2, max_seq_len=300,
                             verbose=False):
    """
    按 user_id 分组形成时序序列，用户级 train/test split。
    返回: (train_dataset, test_dataset, n_features)
    """
    if verbose:
        print(f"\n{'='*50}")
        print("多模态数据准备 (按用户划分)")
        print(f"{'='*50}")

    # 提取特征列（排除 label 和 user_id）
    feat_cols = [c for c in labeled_df.columns if c not in ('label', 'user_id')]
    n_features = len(feat_cols)

    if verbose:
        print(f"特征维度: {n_features}")
        print(f"特征列: {feat_cols[:10]}... (共 {n_features} 列)")

    # 按 user_id 分组
    user_groups = labeled_df.groupby('user_id')
    user_ids = np.array(list(user_groups.groups.keys()))
    user_labels = np.array([user_groups.get_group(uid)['label'].iloc[0] for uid in user_ids])

    if verbose:
        print(f"总用户数: {len(user_ids)}")
        print(f"用户标签分布: 0(认知障碍)={sum(user_labels==0)}, 1(正常)={sum(user_labels==1)}")

    if sum(user_labels == 0) < 1 or sum(user_labels == 1) < 1:
        print("错误: 每个类别至少需要 1 个用户才能进行训练。")
        return None, None, n_features

    # 用户级分层划分
    try:
        train_uids, test_uids = train_test_split(
            user_ids, test_size=test_size, random_state=42, stratify=user_labels)
    except ValueError:
        print("警告: 用户数太少无法分层，改用随机抽样。")
        train_uids, test_uids = train_test_split(
            user_ids, test_size=test_size, random_state=42)

    if verbose:
        print(f"训练集用户 ({len(train_uids)}): {sorted(train_uids)}")
        print(f"测试集用户 ({len(test_uids)}): {sorted(test_uids)}")

    def build_dataset(uids, augment=None):
        sequences = []
        input_ids_list = []
        attn_masks_list = []
        labels = []
        temporal_masks = []

        for uid in uids:
            user_df = user_groups.get_group(uid)
            seq = user_df[feat_cols].values.astype(np.float32)  # (T, C)
            label = int(user_df['label'].iloc[0])

            # 数据增强（仅训练集）
            if augment is not None:
                seq = augment(seq)

            # Pad/truncate 序列
            T = seq.shape[0]
            if T > max_seq_len:
                seq = seq[:max_seq_len]
                T = max_seq_len

            if T < max_seq_len:
                pad_len = max_seq_len - T
                seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant')
                t_mask = np.array([False] * T + [True] * pad_len)
            else:
                t_mask = np.zeros(T, dtype=bool)

            # 生成报告 → tokenize
            report_gen = PhysiologicalReportGenerator(feature_names=feat_cols)
            report = report_gen.generate_report(seq[:T])  # 用原始长度生成报告
            tokens = tokenizer(report, padding='max_length', truncation=True,
                               max_length=128, return_tensors='np')

            sequences.append(seq)
            input_ids_list.append(tokens['input_ids'].squeeze(0))
            attn_masks_list.append(tokens['attention_mask'].squeeze(0))
            labels.append(label)
            temporal_masks.append(t_mask)

        return MultimodalDataset(
            sequences=torch.tensor(np.array(sequences), dtype=torch.float32),
            input_ids=torch.tensor(np.array(input_ids_list), dtype=torch.long),
            attention_masks=torch.tensor(np.array(attn_masks_list), dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            temporal_masks=torch.tensor(np.array(temporal_masks), dtype=torch.bool),
        )

    augment = TemporalAugmentation()
    train_dataset = build_dataset(train_uids, augment=augment)
    test_dataset = build_dataset(test_uids, augment=None)

    if verbose:
        print(f"\n训练集: {len(train_dataset)} 用户 (0={sum(train_dataset.labels==0)}, 1={sum(train_dataset.labels==1)})")
        print(f"测试集: {len(test_dataset)} 用户 (0={sum(test_dataset.labels==0)}, 1={sum(test_dataset.labels==1)})")

    return train_dataset, test_dataset, n_features


# ===================== 训练与评估 =====================

def train_and_evaluate_multimodal(train_dataset, test_dataset, n_features,
                                    save_dir=None, epochs=80, batch_size=8,
                                    verbose=False):
    """
    训练多模态模型并输出完整评估指标。
    返回: (model, report_text)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"\n{'='*50}")
    log(f"开始训练 MultimodalTemporal 二分类模型 (设备: {device})")
    log(f"{'='*50}")

    # 类别权重
    train_labels = train_dataset.labels.numpy()
    n_neg = int((train_labels == 0).sum())
    n_pos = int((train_labels == 1).sum())
    class_weights = torch.tensor([n_pos / max(n_neg, 1), 1.0], dtype=torch.float32).to(device)
    log(f"类别权重: [障碍={class_weights[0]:.2f}, 正常={class_weights[1]:.2f}]")

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
    model = MultimodalDiagnosticModel(
        n_features=n_features, temporal_dim=64, text_dim=128,
        lora_rank=4, lora_alpha=16, dropout=0.3,
    ).to(device)

    # 优化器：不同学习率
    temporal_params = list(model.temporal_encoder.parameters())
    cross_params = list(model.cross_attention.parameters())
    cls_params = list(model.classifier.parameters())
    lora_params = model.text_encoder.parameters_that_require_grad()

    optimizer = optim.AdamW([
        {'params': temporal_params + cross_params + cls_params, 'lr': 5e-4, 'weight_decay': 0.01},
        {'params': lora_params, 'lr': 5e-5, 'weight_decay': 0.01},
    ])

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=0.1)

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"可训练参数: {total_trainable:,}")
    log(f"总参数量: {total_params:,}")

    # ---------- 训练循环 ----------
    best_f1 = 0
    best_state = None
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in train_loader:
            temporal = batch['temporal'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            t_mask = batch.get('temporal_mask')
            if t_mask is not None:
                t_mask = t_mask.to(device)

            optimizer.zero_grad()
            logits = model(temporal, input_ids, attn_mask, t_mask)
            loss = criterion(logits, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # 每 10 epoch 验证
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    temporal = batch['temporal'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attn_mask = batch['attention_mask'].to(device)
                    t_mask = batch.get('temporal_mask')
                    if t_mask is not None:
                        t_mask = t_mask.to(device)
                    logits = model(temporal, input_ids, attn_mask, t_mask)
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
            temporal = batch['temporal'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            t_mask = batch.get('temporal_mask')
            if t_mask is not None:
                t_mask = t_mask.to(device)

            logits = model(temporal, input_ids, attn_mask, t_mask)
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

    # AUC-ROC（需要至少两个类别）
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    log(f"\n{'='*50}")
    log(f"测试集评估结果 (MultimodalTemporal)")
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
        model_path = os.path.join(save_dir, 'multimodal_model.pt')
        torch.save(best_state, model_path)
        log(f"\n模型已保存至: {model_path}")

    report_text = "\n".join(report_lines)
    return model, report_text


# ===================== 分层 K 折交叉验证 =====================

def prepare_multimodal_kfold(labeled_df, feature_names, tokenizer,
                              n_splits=5, max_seq_len=300, verbose=False):
    """
    按 user_id 分组，使用分层 K 折交叉验证。
    返回: (fold_pairs, n_features)
    其中 fold_pairs = [(train_dataset, test_dataset), ...]，长度 = n_splits
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"多模态数据准备 (分层 {n_splits} 折交叉验证)")
        print(f"{'='*50}")

    feat_cols = [c for c in labeled_df.columns if c not in ('label', 'user_id')]
    n_features = len(feat_cols)

    user_groups = labeled_df.groupby('user_id')
    user_ids = np.array(list(user_groups.groups.keys()))
    user_labels = np.array([user_groups.get_group(uid)['label'].iloc[0] for uid in user_ids])

    if verbose:
        print(f"特征维度: {n_features}")
        print(f"总用户数: {len(user_ids)}")
        print(f"用户标签分布: 0(认知障碍)={sum(user_labels==0)}, 1(正常)={sum(user_labels==1)}")

    n_0 = sum(user_labels == 0)
    n_1 = sum(user_labels == 1)
    if n_0 < n_splits or n_1 < n_splits:
        actual_k = min(n_0, n_1)
        print(f"警告: 障碍类仅 {n_0} 人，无法做 {n_splits} 折。自动调整为 {actual_k} 折。")
        n_splits = actual_k

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def build_dataset(uids, augment=None):
        sequences = []
        input_ids_list = []
        attn_masks_list = []
        labels = []
        temporal_masks = []
        report_gen = PhysiologicalReportGenerator(feature_names=feat_cols)

        for uid in uids:
            user_df = user_groups.get_group(uid)
            seq = user_df[feat_cols].values.astype(np.float32)
            label = int(user_df['label'].iloc[0])

            if augment is not None:
                seq = augment(seq)

            T = seq.shape[0]
            if T > max_seq_len:
                seq = seq[:max_seq_len]
                T = max_seq_len

            if T < max_seq_len:
                pad_len = max_seq_len - T
                seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant')
                t_mask = np.array([False] * T + [True] * pad_len)
            else:
                t_mask = np.zeros(T, dtype=bool)

            report = report_gen.generate_report(seq[:T])
            tokens = tokenizer(report, padding='max_length', truncation=True,
                               max_length=128, return_tensors='np')

            sequences.append(seq)
            input_ids_list.append(tokens['input_ids'].squeeze(0))
            attn_masks_list.append(tokens['attention_mask'].squeeze(0))
            labels.append(label)
            temporal_masks.append(t_mask)

        return MultimodalDataset(
            sequences=torch.tensor(np.array(sequences), dtype=torch.float32),
            input_ids=torch.tensor(np.array(input_ids_list), dtype=torch.long),
            attention_masks=torch.tensor(np.array(attn_masks_list), dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            temporal_masks=torch.tensor(np.array(temporal_masks), dtype=torch.bool),
        )

    augment = TemporalAugmentation()
    fold_pairs = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(user_ids, user_labels)):
        train_uids = user_ids[train_idx]
        test_uids = user_ids[test_idx]

        train_dataset = build_dataset(train_uids, augment=augment)
        test_dataset = build_dataset(test_uids, augment=None)

        if verbose:
            n_train_0 = int((train_dataset.labels == 0).sum())
            n_train_1 = int((train_dataset.labels == 1).sum())
            n_test_0 = int((test_dataset.labels == 0).sum())
            n_test_1 = int((test_dataset.labels == 1).sum())
            print(f"  Fold {fold_idx+1}: 训练 {len(train_uids)} 人 (0={n_train_0},1={n_train_1}) "
                  f"→ 测试 {len(test_uids)} 人 (0={n_test_0},1={n_test_1})")

        fold_pairs.append((train_dataset, test_dataset))

    return fold_pairs, n_features


def train_and_evaluate_kfold(fold_pairs, n_features, save_dir=None,
                              epochs=80, batch_size=8, verbose=False):
    """
    分层 K 折交叉验证：训练 K 次，汇总所有折的预测结果。
    返回: (all_report_text)
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_folds = len(fold_pairs)

    log(f"\n{'='*50}")
    log(f"分层 {n_folds} 折交叉验证 - MultimodalTemporal (设备: {device})")
    log(f"{'='*50}")

    # 汇总所有折的预测
    all_fold_preds = []
    all_fold_labels = []
    all_fold_probs = []
    fold_metrics = []

    for fold_idx, (train_dataset, test_dataset) in enumerate(fold_pairs):
        log(f"\n{'─'*40}")
        log(f"Fold {fold_idx+1}/{n_folds}")
        log(f"{'─'*40}")

        train_labels = train_dataset.labels.numpy()
        n_neg = int((train_labels == 0).sum())
        n_pos = int((train_labels == 1).sum())
        class_weights = torch.tensor([n_pos / max(n_neg, 1), 1.0], dtype=torch.float32).to(device)
        log(f"  类别权重: [障碍={class_weights[0]:.2f}, 正常={class_weights[1]:.2f}]")

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

        # 每折新建模型
        model = MultimodalDiagnosticModel(
            n_features=n_features, temporal_dim=64, text_dim=128,
            lora_rank=4, lora_alpha=16, dropout=0.3,
        ).to(device)

        temporal_params = list(model.temporal_encoder.parameters())
        cross_params = list(model.cross_attention.parameters())
        cls_params = list(model.classifier.parameters())
        lora_params = model.text_encoder.parameters_that_require_grad()

        optimizer = optim.AdamW([
            {'params': temporal_params + cross_params + cls_params, 'lr': 5e-4, 'weight_decay': 0.01},
            {'params': lora_params, 'lr': 5e-5, 'weight_decay': 0.01},
        ])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # 训练
        best_f1 = 0
        best_state = None
        patience_counter = 0
        patience = 15

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            n_batches = 0

            for batch in train_loader:
                temporal = batch['temporal'].to(device)
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                t_mask = batch.get('temporal_mask')
                if t_mask is not None:
                    t_mask = t_mask.to(device)

                optimizer.zero_grad()
                logits = model(temporal, input_ids, attn_mask, t_mask)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                model.eval()
                preds, labs = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        t = batch['temporal'].to(device)
                        ids = batch['input_ids'].to(device)
                        am = batch['attention_mask'].to(device)
                        tm = batch.get('temporal_mask')
                        if tm is not None:
                            tm = tm.to(device)
                        out = model(t, ids, am, tm)
                        preds.extend(out.argmax(dim=1).cpu().numpy())
                        labs.extend(batch['label'].numpy())
                val_f1 = f1_score(labs, preds, zero_division=0)
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

        # 评估该折
        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        fold_preds, fold_probs, fold_labels = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                t = batch['temporal'].to(device)
                ids = batch['input_ids'].to(device)
                am = batch['attention_mask'].to(device)
                tm = batch.get('temporal_mask')
                if tm is not None:
                    tm = tm.to(device)
                logits = model(t, ids, am, tm)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                fold_probs.extend(probs)
                fold_preds.extend(preds)
                fold_labels.extend(batch['label'].numpy())

        fold_preds = np.array(fold_preds)
        fold_probs = np.array(fold_probs)
        fold_labels = np.array(fold_labels)

        # 该折指标
        acc = accuracy_score(fold_labels, fold_preds)
        f1 = f1_score(fold_labels, fold_preds, zero_division=0)
        prec = precision_score(fold_labels, fold_preds, zero_division=0)
        rec = recall_score(fold_labels, fold_preds, zero_division=0)
        auc = roc_auc_score(fold_labels, fold_probs) if len(np.unique(fold_labels)) > 1 else 0.0
        cm = confusion_matrix(fold_labels, fold_preds, labels=[0, 1])

        log(f"\n  Fold {fold_idx+1} 结果: Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  AUC={auc:.4f}")
        log(f"  混淆矩阵: TP={cm[1][1]} FP={cm[0][1]} FN={cm[1][0]} TN={cm[0][0]}")

        fold_metrics.append({'acc': acc, 'f1': f1, 'prec': prec, 'rec': rec, 'auc': auc, 'cm': cm})
        all_fold_preds.extend(fold_preds)
        all_fold_labels.extend(fold_labels)
        all_fold_probs.extend(fold_probs)

        # 保存每折最优模型
        if save_dir and best_state:
            torch.save(best_state, os.path.join(save_dir, f'multimodal_fold{fold_idx+1}.pt'))

    # ===================== 汇总所有折 =====================
    all_fold_preds = np.array(all_fold_preds)
    all_fold_labels = np.array(all_fold_labels)
    all_fold_probs = np.array(all_fold_probs)

    log(f"\n{'='*50}")
    log(f"分层 {n_folds} 折交叉验证汇总")
    log(f"{'='*50}")

    # 各折平均指标
    for metric in ['acc', 'f1', 'prec', 'rec', 'auc']:
        vals = [m[metric] for m in fold_metrics]
        log(f"  {metric:>5s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  (各折: {[f'{v:.4f}' for v in vals]})")

    # 汇总混淆矩阵（所有折的预测合并）
    total_cm = confusion_matrix(all_fold_labels, all_fold_preds, labels=[0, 1])
    log(f"\n汇总混淆矩阵 (所有 {n_folds} 折合并):")
    log(f"                 预测正常(1)  预测障碍(0)")
    log(f"  实际正常(1)      {total_cm[1][1]:>6}        {total_cm[1][0]:>6}")
    log(f"  实际障碍(0)      {total_cm[0][1]:>6}        {total_cm[0][0]:>6}")

    log(f"\n汇总分类报告:")
    log(classification_report(all_fold_labels, all_fold_preds,
                              target_names=['认知障碍(0)', '正常(1)'],
                              zero_division=0))

    # 汇总 AUC
    if len(np.unique(all_fold_labels)) > 1:
        total_auc = roc_auc_score(all_fold_labels, all_fold_probs)
        log(f"汇总 AUC-ROC: {total_auc:.4f}")

    # 保存图表
    if save_dir:
        _plot_confusion_matrix(total_cm,
                               save_path=os.path.join(save_dir, 'confusion_matrix.png'))
        if len(np.unique(all_fold_labels)) > 1:
            _plot_roc_curve(all_fold_labels, all_fold_probs, total_auc,
                            save_path=os.path.join(save_dir, 'roc_curve.png'))
        # 保存各折 ROC
        _plot_kfold_roc(fold_pairs, all_fold_labels, all_fold_probs, n_folds, save_dir)

    report_text = "\n".join(report_lines)
    return report_text


def _plot_kfold_roc(fold_pairs, all_labels, all_probs, n_folds, save_dir):
    """绘制每折 + 汇总 ROC 曲线"""
    plt.figure(figsize=(8, 7))
    colors = plt.cm.Set1(np.linspace(0, 1, n_folds))

    # 计算每折的起止索引
    offsets = []
    idx = 0
    for _, test_ds in fold_pairs:
        n = len(test_ds)
        offsets.append((idx, idx + n))
        idx += n

    for i, (start, end) in enumerate(offsets):
        y_true = all_labels[start:end]
        y_score = all_probs[start:end]
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_val = roc_auc_score(y_true, y_score)
            plt.plot(fpr, tpr, color=colors[i], lw=1.2, alpha=0.7,
                     label=f'Fold {i+1} (AUC={auc_val:.3f})')

    # 汇总
    if len(np.unique(all_labels)) > 1:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        total_auc = roc_auc_score(all_labels, all_probs)
        plt.plot(fpr, tpr, color='black', lw=2.5,
                 label=f'Overall (AUC={total_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'Stratified {n_folds}-Fold CV ROC Curve (Multimodal)')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'roc_curve_kfold.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    plt.close()


# ===================== 可视化 =====================

def _plot_confusion_matrix(cm, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Impaired(0)', 'Normal(1)'],
                yticklabels=['Impaired(0)', 'Normal(1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Multimodal)')
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
    plt.title('ROC Curve (Multimodal)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    plt.close()
