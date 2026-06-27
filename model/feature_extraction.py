import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid
import math

from data_loader import load_eeg_data  
from preprocess import process_pipeline

class EEGFeatureExtractor:
    def __init__(self, fs=250, window_size_sec=2.0, overlap_sec=1.0):
        """
        初始化特征提取器
        :param fs: 采样率 (Hz)
        :param window_size_sec: 滑动窗口长度 (秒)。建议 2-4 秒，能较好地捕捉低频(Delta/Theta)特征
        :param overlap_sec: 窗口重叠长度 (秒)
        """
        self.fs = fs
        self.window_size = int(window_size_sec * fs)
        self.step_size = int((window_size_sec - overlap_sec) * fs)
        
        # 定义脑电经典频段
        self.bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 45)
        }

    def calculate_psd_and_de(self, segment: np.ndarray):
        """
        计算单个数据窗口内各频段的 能量(Energy) 和 微分熵(DE)
        :param segment: 形状为 (window_size,) 的一维脑电信号片段
        :return: 包含各频段特征的字典
        """
        # 使用 Welch 方法计算功率谱密度 (PSD)
        # nperseg 设置为窗口长度，保证频率分辨率足够高
        freqs, psd = signal.welch(segment, fs=self.fs, nperseg=len(segment))
        
        features = {}
        for band_name, (low, high) in self.bands.items():
            # 找到对应频段的频率索引
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            
            # 1. 计算频段能量 (Energy / Band Power)
            # 通过对该频段内的 PSD 积分(近似求和)得到频段能量
            #band_energy = np.trapz(psd[idx_band], freqs[idx_band])
            band_energy = trapezoid(psd[idx_band], freqs[idx_band])
            features[f'{band_name}_Energy'] = band_energy
            
            # 2. 计算微分熵 (Differential Entropy, DE)
            # 在特定频段内，假设信号服从高斯分布，DE 可以用频段能量的对数来近似计算
            # 为防止 log(0) 的错误，加入一个极小值 1e-10
            band_de = math.log2(band_energy + 1e-10) 
            features[f'{band_name}_DE'] = band_de
            
        return features

    def calculate_emg_rms(self, segment: np.ndarray) -> float:
        """
        计算肌电信号的均方根 (RMS)，反映面部肌肉的紧张程度
        """
        return np.sqrt(np.mean(segment**2))

    @staticmethod
    def calculate_power_ratios(feats: dict) -> dict:
        """
        根据已有的频段能量计算功率比值指标 + 各频段能量占比
        :param feats: 包含 *_Energy 键的特征字典
        :return: 包含比值特征的字典
        """
        eps = 1e-10
        delta = feats.get('Delta_Energy', 0)
        theta = feats.get('Theta_Energy', 0)
        alpha = feats.get('Alpha_Energy', 0)
        beta  = feats.get('Beta_Energy', 0)
        gamma = feats.get('Gamma_Energy', 0)
        total = delta + theta + alpha + beta + gamma + eps

        return {
            # 经典功率比值
            'DAR':   delta / (alpha + eps),
            'DTABR': (delta + theta) / (alpha + beta + eps),
            'BTBR':  beta / (theta + eps),
            'Theta_Alpha': theta / (alpha + eps),
            # 各频段能量占总能量的比例（相对特征，消除信号强度差异）
            'Delta_Ratio': delta / total,
            'Theta_Ratio': theta / total,
            'Alpha_Ratio': alpha / total,
            'Beta_Ratio':  beta / total,
            'Gamma_Ratio': gamma / total,
        }

    def extract_features(self, df_clean: pd.DataFrame, verbose=False) -> pd.DataFrame:
        """
        对整段清理后的 DataFrame 执行滑动窗口特征提取
        """
        num_samples = len(df_clean)
        feature_list = []
        skipped_windows = 0

        if verbose:
            print(f"开始特征提取，窗口大小: {self.window_size} 采样点，步长: {self.step_size} 采样点...")

        # 使用滑动窗口遍历时间序列
        for start_idx in range(0, num_samples - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            window_df = df_clean.iloc[start_idx:end_idx]

            # ==== 数据质量检查 (Artifact Rejection) ====
            # 判断该窗口内，脑电信号的极差 (Peak-to-Peak) 是否过大(接触不良) 或 过小(直线/死机)
            is_bad_window = False
            for col in ['eeg_1_clean', 'eeg_2_clean']:
                signal_data = window_df[col].values
                ptp_amplitude = np.max(signal_data) - np.min(signal_data)

                # 规则1：如果振幅极差小于 0.1 微伏，说明电极断开了/短路了，是一条直线
                # 规则2：如果滤波后波动依然异常大(>250)，说明噪音极强
                if ptp_amplitude < 0.1 or ptp_amplitude > 250.0:
                    is_bad_window = True
                    break

            if is_bad_window:
                skipped_windows += 1
                continue
            # ==================================================

            # 记录当前窗口的中心时间戳，方便后续对其标签
            center_time = window_df.index[self.window_size // 2]

            window_features = {'time': center_time}

            # 提取 eeg_1_clean 的特征
            eeg1_feats = self.calculate_psd_and_de(window_df['eeg_1_clean'].values)
            for k, v in eeg1_feats.items():
                window_features[f'EEG1_{k}'] = v
            # 功率比值指标 (DAR, DTABR, β/θ)
            for k, v in self.calculate_power_ratios(eeg1_feats).items():
                window_features[f'EEG1_{k}'] = v

            # 提取 eeg_2_clean 的特征
            eeg2_feats = self.calculate_psd_and_de(window_df['eeg_2_clean'].values)
            for k, v in eeg2_feats.items():
                window_features[f'EEG2_{k}'] = v
            # 功率比值指标 (DAR, DTABR, β/θ)
            for k, v in self.calculate_power_ratios(eeg2_feats).items():
                window_features[f'EEG2_{k}'] = v

            # (可选) 提取 EMG 特征作为额外的生理参考
            window_features['EMG1_RMS'] = self.calculate_emg_rms(window_df['emg_1'].values)
            window_features['EMG2_RMS'] = self.calculate_emg_rms(window_df['emg_2'].values)

            feature_list.append(window_features)

        # 将提取出的特征列表转换为新的 DataFrame
        if not feature_list:
            if verbose:
                print(f"特征提取完成！所有窗口均不合格，共丢弃了 {skipped_windows} 个窗口。")
            return pd.DataFrame()

        features_df = pd.DataFrame(feature_list)
        features_df.set_index('time', inplace=True)

        if verbose:
            print(f"特征提取完成！共提取了 {len(features_df)} 个时间片段的特征。")
        if skipped_windows > 0 and verbose:
            print(f"因数据质量不合格，共丢弃了 {skipped_windows} 个窗口。")
        return features_df

# ================= 眼动特征提取 =================

class EyeTrackingFeatureExtractor:
    """
    从眼动数据（blink_l/r, gaze_x/y/z）中提取认知障碍相关特征。
    眼动采样率约 62.5Hz，远低于 EEG 的 250Hz。
    """

    def __init__(self, fs=62.5, window_size_sec=4.0, overlap_sec=2.0):
        self.fs = fs
        self.window_size = int(window_size_sec * fs)
        self.step_size = int((window_size_sec - overlap_sec) * fs)

    def extract_features(self, df_with_eye: pd.DataFrame, verbose=False) -> pd.DataFrame:
        """
        从包含眼动列的 DataFrame 中提取眼动特征。
        输入 df 应包含: blink_l, blink_r, gaze_x, gaze_y, gaze_z
        返回的 DataFrame 以时间索引，列为眼动特征。
        """
        eye_cols = ['blink_l', 'blink_r', 'gaze_x', 'gaze_y', 'gaze_z']

        # 检查眼动列是否存在
        available_cols = [c for c in eye_cols if c in df_with_eye.columns]
        if not available_cols:
            if verbose:
                print("无眼动数据列，跳过眼动特征提取。")
            return pd.DataFrame()

        num_samples = len(df_with_eye)
        feature_list = []
        skipped = 0

        for start_idx in range(0, num_samples - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            window = df_with_eye.iloc[start_idx:end_idx]

            center_time = window.index[self.window_size // 2]
            feats = {'time': center_time}

            # === 眨眼特征 ===
            if 'blink_l' in available_cols and 'blink_r' in available_cols:
                bl = window['blink_l'].values
                br = window['blink_r'].values

                # 有效数据比例
                bl_valid = np.isfinite(bl) & (bl != 0) | (bl == 0)
                br_valid = np.isfinite(br) & (br != 0) | (br == 0)
                feats['blink_valid_ratio'] = float(np.mean(bl_valid & br_valid))

                # 眨眼次数（从 0→1 的跳变）
                bl_blinks = np.sum(np.diff(bl.astype(int)) == 1)
                br_blinks = np.sum(np.diff(br.astype(int)) == 1)
                duration_sec = self.window_size / self.fs
                feats['blink_rate_l'] = float(bl_blinks / duration_sec)
                feats['blink_rate_r'] = float(br_blinks / duration_sec)
                feats['blink_rate'] = float((bl_blinks + br_blinks) / 2 / duration_sec)

                # 眨眼间隔统计
                blink_combined = np.maximum(bl, br)
                blink_starts = np.where(np.diff(blink_combined.astype(int)) == 1)[0]
                if len(blink_starts) > 1:
                    intervals = np.diff(blink_starts) / self.fs  # 秒
                    feats['blink_interval_mean'] = float(np.mean(intervals))
                    feats['blink_interval_std'] = float(np.std(intervals))
                else:
                    feats['blink_interval_mean'] = 0.0
                    feats['blink_interval_std'] = 0.0

                # 左右不对称性
                feats['blink_asymmetry'] = float(abs(
                    float(bl_blinks) - float(br_blinks)))

            # === 注视特征 ===
            if 'gaze_x' in available_cols and 'gaze_y' in available_cols:
                gx = window['gaze_x'].values
                gy = window['gaze_y'].values

                # 过滤掉全零段（追踪丢失）
                gx_valid = gx[gx != 0]
                gy_valid = gy[gy != 0]

                if len(gx_valid) > 10:
                    feats['gaze_valid_ratio'] = float(len(gx_valid) / len(gx))

                    # 注视中心
                    feats['gaze_x_mean'] = float(np.mean(gx_valid))
                    feats['gaze_y_mean'] = float(np.mean(gy_valid))

                    # 注视稳定性（标准差越小越稳定）
                    feats['gaze_x_std'] = float(np.std(gx_valid))
                    feats['gaze_y_std'] = float(np.std(gy_valid))

                    # 变异系数
                    gx_mean_abs = abs(np.mean(gx_valid)) + 1e-8
                    gy_mean_abs = abs(np.mean(gy_valid)) + 1e-8
                    feats['gaze_x_cv'] = float(np.std(gx_valid) / gx_mean_abs)
                    feats['gaze_y_cv'] = float(np.std(gy_valid) / gy_mean_abs)

                    # 注视空间覆盖面积代理
                    feats['gaze_dispersion'] = float(np.std(gx_valid) * np.std(gy_valid))

                    # 扫描路径长度
                    dx = np.diff(gx_valid)
                    dy = np.diff(gy_valid)
                    path_length = np.sum(np.sqrt(dx**2 + dy**2))
                    feats['gaze_path_length'] = float(path_length)

                    # 扫视特征（基于一阶差分的速度代理）
                    velocity = np.sqrt(dx**2 + dy**2) * self.fs  # 速度估计
                    feats['gaze_velocity_mean'] = float(np.mean(velocity))
                    feats['gaze_velocity_std'] = float(np.std(velocity))
                    feats['gaze_velocity_max'] = float(np.max(velocity))

                    # 扫视计数（速度超过中位数 3 倍的点）
                    vel_threshold = np.median(velocity) * 3
                    feats['saccade_count'] = int(np.sum(velocity > vel_threshold))
                else:
                    # 有效数据不足
                    feats['gaze_valid_ratio'] = float(len(gx_valid) / len(gx))
                    for k in ['gaze_x_mean', 'gaze_y_mean', 'gaze_x_std', 'gaze_y_std',
                              'gaze_x_cv', 'gaze_y_cv', 'gaze_dispersion',
                              'gaze_path_length', 'gaze_velocity_mean',
                              'gaze_velocity_std', 'gaze_velocity_max', 'saccade_count']:
                        feats[k] = 0.0

            # === gaze_z 特征 ===
            if 'gaze_z' in available_cols:
                gz = window['gaze_z'].values
                gz_valid = gz[gz != 0]
                if len(gz_valid) > 10:
                    feats['gaze_z_mean'] = float(np.mean(gz_valid))
                    feats['gaze_z_std'] = float(np.std(gz_valid))
                else:
                    feats['gaze_z_mean'] = 0.0
                    feats['gaze_z_std'] = 0.0

            feature_list.append(feats)

        if not feature_list:
            if verbose:
                print("眼动特征提取完成！所有窗口均不合格。")
            return pd.DataFrame()

        features_df = pd.DataFrame(feature_list)
        features_df.set_index('time', inplace=True)

        if verbose:
            print(f"眼动特征提取完成！共 {len(features_df)} 个窗口，"
                  f"跳过 {skipped} 个不合格窗口。")
        return features_df


# ================= 测试代码 =================
if __name__ == "__main__":
    target_user_id = 4141653
    eeg_df = load_eeg_data(target_user_id)
    df_clean = process_pipeline(eeg_df)
    # 假设 df_clean 是上一步 process_pipeline 返回的包含 eeg_1_clean 的 DataFrame
    extractor = EEGFeatureExtractor(fs=250, window_size_sec=2.0, overlap_sec=1.0)
    features_df = extractor.extract_features(df_clean)
    print(features_df.head())
    #pass