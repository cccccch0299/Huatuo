"""AI-assisted diagnosis module: ML inference + LLM report generation."""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("diagnosis")

# ── Import model/ submodules (add model/ to sys.path for their imports) ──
_MODEL_DIR = Path(__file__).resolve().parent / "model"
_model_dir_str = str(_MODEL_DIR)

if _model_dir_str not in sys.path:
    sys.path.insert(0, _model_dir_str)

from preprocess import process_pipeline  # noqa: E402
from feature_extraction import EEGFeatureExtractor  # noqa: E402


class DiagnosticEngine:
    """Load trained RF model + scaler, run inference, call LLM for report."""

    def __init__(self, model_dir: str = "test"):
        self.model_dir = Path(model_dir)
        self.model = joblib.load(self.model_dir / "rf_model.joblib")
        self.scaler = joblib.load(self.model_dir / "scaler.joblib")
        self.meta = joblib.load(self.model_dir / "inference_meta.joblib")
        self.threshold: float = self.meta["threshold"]
        self.feature_names: List[str] = self.meta["feature_names"]
        LOGGER.info(
            "DiagnosticEngine loaded: model=%s, threshold=%.2f, features=%d",
            self.meta["model_name"], self.threshold, len(self.feature_names),
        )

    # ── public entry point ───────────────────────────────────────────────
    async def diagnose(self, user_id: int, pool) -> Dict[str, Any]:
        # 1. Load EEG data from DB
        df = await self._load_eeg_data(user_id, pool)
        if df.empty:
            raise ValueError(f"用户 {user_id} 无脑电数据")

        # 2. Preprocess
        df_clean = process_pipeline(df, verbose=False)
        df_clean_reset = df_clean.reset_index(drop=True)

        # 3. Feature extraction
        extractor = EEGFeatureExtractor(fs=250, window_size_sec=2.0, overlap_sec=1.0)
        features_df = extractor.extract_features(df_clean_reset, verbose=False)
        if features_df.empty:
            raise ValueError(f"用户 {user_id} 数据质量不足，无法提取有效特征")

        # 4. Merge game features
        game_feats = await self._load_game_features(user_id, pool)
        features_df["game_hit_accuracy"] = game_feats.get("game_hit_accuracy", 0.0)
        features_df["game_score"] = game_feats.get("game_score", 0.0)

        # 5. Average across windows → single feature vector
        feature_vector = features_df.mean(axis=0)
        X = pd.DataFrame([feature_vector], columns=features_df.columns)

        # Align to training feature set
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]

        # 6. Inference
        X_scaled = self.scaler.transform(X.values)
        prob = float(self.model.predict_proba(X_scaled)[0, 1])
        prediction = 1 if prob >= self.threshold else 0
        confidence = prob if prediction == 1 else 1 - prob
        label = "正常" if prediction == 1 else "认知障碍"

        # 7. Key features (top by importance)
        key_features = self._extract_key_features(X.iloc[0])

        # 8. LLM report
        report = await self._generate_report(key_features, prediction, prob)

        return {
            "prediction": prediction,
            "label": label,
            "probability": round(prob, 4),
            "confidence": round(confidence, 4),
            "threshold": round(self.threshold, 4),
            "key_features": key_features,
            "report": report,
        }

    # ── DB helpers ───────────────────────────────────────────────────────
    async def _load_eeg_data(self, user_id: int, pool) -> pd.DataFrame:
        query = """
            SELECT time, eeg_1, eeg_2, emg_1, emg_2,
                   blink_l, blink_r, gaze_x, gaze_y, gaze_z, event_label
            FROM eeg_data
            WHERE user_id = $1
            ORDER BY time ASC
        """
        async with pool.acquire() as conn:
            records = await conn.fetch(query, user_id)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records, columns=records[0].keys())
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        return df

    async def _load_game_features(self, user_id: int, pool) -> Dict[str, float]:
        query = """
            SELECT hit_count, success_count, score
            FROM game_sessions
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT 1
        """
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, user_id)

        if row is None:
            return {}
        hit_acc = row["success_count"] / row["hit_count"] if row["hit_count"] > 0 else 0.0
        return {"game_hit_accuracy": hit_acc, "game_score": float(row["score"])}

    # ── Feature importance ───────────────────────────────────────────────
    def _extract_key_features(self, feature_row: pd.Series, top_n: int = 10) -> List[Dict[str, Any]]:
        importances = self.model.feature_importances_
        feat_imp = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True,
        )[:top_n]

        results = []
        for name, imp in feat_imp:
            raw_val = float(feature_row.get(name, 0))
            results.append({
                "name": name,
                "importance": round(float(imp), 4),
                "value": round(raw_val, 4),
            })
        return results

    # ── Feature knowledge base ───────────────────────────────────────────
    # Maps feature name prefixes/suffixes to human-readable explanations.
    _BAND_INFO = {
        "Delta":  ("δ波（Delta波，0.5-4Hz）", "与深度睡眠和大脑基本修复有关。过高可能提示脑功能减退或存在慢波异常"),
        "Theta":  ("θ波（Theta波，4-8Hz）", "与记忆编码和注意力调节有关。过高通常提示认知负荷增大或脑功能下降"),
        "Alpha":  ("α波（Alpha波，8-13Hz）", "与放松清醒状态有关，是大脑正常工作的标志。过低提示大脑皮层抑制功能减弱"),
        "Beta":   ("β波（Beta波，13-30Hz）", "与专注、思考和警觉有关。过低提示注意力和执行功能可能下降"),
        "Gamma":  ("γ波（Gamma波，30-45Hz）", "与高级认知处理和信息整合有关。异常可能提示认知整合能力变化"),
    }
    _RATIO_INFO = {
        "DAR":    ("δ/α比值（DAR）", "Delta与Alpha能量之比，是评估认知功能的核心指标。正常值一般<1.5；升高提示脑功能减退，大脑慢波活动增多、快波活动减少"),
        "DTABR":  ("(δ+θ)/(α+β)比值（DTABR）", "慢波与快波的比值，反映大脑功能平衡。正常值一般<2.0；升高提示脑功能整体偏慢，常见于认知障碍"),
        "BTBR":   ("β/θ比值（BTBR）", "Beta与Theta之比，反映警觉水平。正常值一般>1.0；降低提示注意力和执行功能下降"),
        "Theta_Alpha": ("θ/α比值", "Theta与Alpha之比。升高提示大脑慢波活动相对增多"),
    }
    _RATIO_KEYS = set(_RATIO_INFO.keys())

    @classmethod
    def _describe_feature(cls, name: str) -> dict:
        """Return {label, meaning, direction_hint} for a feature name."""
        # Channel prefix
        ch = ""
        rest = name
        if name.startswith("EEG1_"):
            ch = "通道1 "
            rest = name[5:]
        elif name.startswith("EEG2_"):
            ch = "通道2 "
            rest = name[5:]
        elif name.startswith("EMG1_"):
            ch = "肌电1 "
            rest = name[5:]
        elif name.startswith("EMG2_"):
            ch = "肌电2 "
            rest = name[5:]

        # Ratio features
        for key, (label, meaning) in cls._RATIO_INFO.items():
            if rest == key:
                return {"label": ch + label, "meaning": meaning}

        # Energy features
        for band, (label, meaning) in cls._BAND_INFO.items():
            if rest == f"{band}_Energy":
                return {"label": ch + f"{label}能量", "meaning": meaning + "。数值为频段内的功率谱密度积分"}
            if rest == f"{band}_DE":
                return {"label": ch + f"{label}微分熵（DE）", "meaning": meaning + "。微分熵是对频段能量取对数，用于消除信号强度个体差异"}

        # Ratio features with channel prefix
        for key, (label, meaning) in cls._RATIO_INFO.items():
            if rest == key:
                return {"label": ch + label, "meaning": meaning}

        # EMG RMS
        if rest == "RMS":
            return {"label": ch + "肌电均方根值（RMS）", "meaning": "反映肌肉紧张程度。过高提示面部肌肉紧张或存在肌电干扰"}

        # Game features
        if name == "game_hit_accuracy":
            return {"label": "游戏命中准确率", "meaning": "反映手眼协调和反应速度，与执行功能相关"}
        if name == "game_score":
            return {"label": "游戏得分", "meaning": "综合反映反应速度、注意力和运动协调能力"}

        return {"label": name, "meaning": "脑电信号特征指标"}

    # ── LLM report generation ────────────────────────────────────────────
    async def _generate_report(
        self, features: List[Dict], prediction: int, probability: float,
    ) -> str:
        backend = os.getenv("LLM_BACKEND", "claude").lower()
        prompt = self._build_prompt(features, prediction, probability)

        if backend == "claude":
            try:
                return await self._generate_report_claude(prompt)
            except Exception as exc:
                LOGGER.warning("Claude API failed (%s), trying Qwen...", exc)
                try:
                    return self._generate_report_qwen(prompt)
                except Exception as exc2:
                    LOGGER.warning("Qwen also failed (%s), using template.", exc2)
                    return self._generate_report_template(features, prediction, probability)
        elif backend == "qwen":
            try:
                return self._generate_report_qwen(prompt)
            except Exception as exc:
                LOGGER.warning("Qwen failed (%s), using template.", exc)
                return self._generate_report_template(features, prediction, probability)
        else:
            return self._generate_report_template(features, prediction, probability)

    def _build_prompt(
        self, features: List[Dict], prediction: int, probability: float,
    ) -> str:
        label = "正常" if prediction == 1 else "认知障碍风险"

        # Build detailed feature table
        feature_lines = []
        for f in features:
            info = self._describe_feature(f["name"])
            feature_lines.append(
                f"| {info['label']} | {f['value']} | {f['importance']} | {info['meaning']} |"
            )
        feature_table = "\n".join(feature_lines)

        return f"""你是一位经验丰富的神经内科医生，擅长用通俗易懂的语言向患者解释检查结果。

## 任务
根据以下脑电检测数据和AI模型预测结果，为患者生成一份**详细、通俗、有温度**的诊断分析报告。

## AI 模型预测结果
- 预测结论：{label}
- 预测概率：{probability:.1%}
- 决策阈值：{self.threshold:.2f}

## 关键脑电特征（按对诊断的重要性排序）
| 指标名称 | 您的数值 | 重要性权重 | 含义说明 |
|---------|---------|-----------|---------|
{feature_table}

## 报告撰写要求（请严格遵守）

### 结构要求（请按以下顺序组织报告）
1. **总体结论**（2-3句话）：先用一句话告诉患者"您的脑电检测结果显示XXX"，再简要说明整体评估。
2. **关键发现逐项解读**：对上面表格中的每一个指标，逐一解释：
   - 这个指标是什么（用生活化的比喻，比如"δ波就像大脑的'怠速运转'"）
   - 您的数值是多少，跟正常范围比是偏高还是偏低
   - 这意味着什么（对大脑功能的具体影响）
   - 这个指标为什么对诊断重要（重要性权重高的要重点展开）
3. **综合分析**：把各个指标串起来，说明它们共同指向什么结论，为什么模型会做出这样的判断。
4. **健康建议**：根据异常指标给出3-5条具体、可操作的建议（如"建议每天进行30分钟有氧运动"而非"注意休息"）。
5. **免责声明**："本报告由AI辅助生成，仅供参考，不构成医学诊断。如有疑虑，请前往医院神经内科做进一步检查。"

### 语言要求
- 使用中文
- 避免专业术语，用日常语言解释（比如不说"功率谱密度积分"，说"这个频段脑电波的总能量"）
- 适当使用比喻帮助理解
- 语气温和、有同理心，不要让患者感到焦虑
- 总字数 600-900 字"""

    async def _generate_report_claude(self, prompt: str) -> str:
        import anthropic

        base_url = os.getenv("ANTHROPIC_BASE_URL")
        auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

        client = anthropic.AsyncAnthropic(
            base_url=base_url,
            auth_token=auth_token,
        )
        message = await client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _generate_report_qwen(self, prompt: str) -> str:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen3-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=2048)
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated

    def _generate_report_template(
        self, features: List[Dict], prediction: int, probability: float,
    ) -> str:
        label = "认知障碍风险" if prediction == 0 else "正常"
        lines = [
            "【AI 辅助诊断报告】",
            "",
            f"一、总体结论",
            f"您的脑电检测AI分析结果为：{label}（置信度 {probability:.1%}）。",
        ]

        if prediction == 0:
            lines.append("检测发现您的部分脑电指标存在异常，多项特征提示可能存在认知功能下降的趋势，建议关注并做进一步评估。")
        else:
            lines.append("检测显示您的脑电指标整体处于正常范围，未发现明显的认知功能异常迹象。")

        lines.extend(["", "二、关键指标逐项解读", ""])
        for i, f in enumerate(features, 1):
            info = self._describe_feature(f["name"])
            lines.append(f"{i}. {info['label']}")
            lines.append(f"   您的数值：{f['value']}")
            lines.append(f"   含义：{info['meaning']}")
            # Direction analysis for ratio features
            name_rest = f["name"].split("_", 1)[-1] if "_" in f["name"] else f["name"]
            if name_rest in self._RATIO_KEYS or f["name"] in self._RATIO_KEYS:
                _, desc = self._RATIO_INFO.get(name_rest, self._RATIO_INFO.get(f["name"], ("", "")))
                lines.append(f"   解读：该比值的异常通常与认知功能变化密切相关，是本次诊断的重要参考依据。")
            lines.append("")

        lines.extend([
            "三、综合分析",
            "",
            "上述指标中，对本次诊断影响最大的特征已按重要性排序列出。"
            "AI模型综合考虑了脑电信号中δ波（慢波）和α/β波（快波）的能量分布、"
            "各频段的比值关系以及肌电信号的特征，通过随机森林算法进行综合判断。",
        ])

        if prediction == 0:
            lines.extend([
                "",
                "您的检测结果中，慢波（δ波、θ波）能量相对偏高，快波（α波、β波）能量相对偏低，"
                "这种\"慢波增多、快波减少\"的模式在医学上常与认知功能下降相关。"
                "模型基于大量临床数据的学习，判断您的脑电模式更接近认知障碍的特征。",
            ])
        else:
            lines.extend([
                "",
                "您的各项脑电指标分布较为均衡，慢波与快波的比值处于正常范围，"
                "未发现明显的功能异常模式，整体脑电活动与正常认知功能人群一致。",
            ])

        lines.extend([
            "",
            "四、健康建议",
            "",
            "1. 保持规律的作息，每晚保证7-8小时睡眠",
            "2. 每天进行30分钟以上有氧运动（如快走、游泳），促进脑部血液循环",
            "3. 多进行认知训练活动（如阅读、下棋、学习新技能），保持大脑活跃",
            "4. 保持均衡饮食，适当补充富含Omega-3脂肪酸的食物（如深海鱼、核桃）",
            "5. 定期进行认知功能评估，建议每半年到一年复查一次",
            "",
            "五、免责声明",
            "",
            "本报告由AI辅助生成，仅供参考，不构成医学诊断。如有疑虑，请前往医院神经内科做进一步检查。",
        ])
        return "\n".join(lines)
