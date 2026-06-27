import numpy as np
import pandas as pd
from scipy import signal
from sklearn.linear_model import LinearRegression

from data_loader import load_eeg_data  

class EEGPreprocessor:
    def __init__(self, fs=250):
        """
        初始化预处理器
        :param fs: 采样率 (Hz)，根据你的 .env 文件，默认应该是 250
        """
        self.fs = fs

    def handle_extreme_values(self, data: np.ndarray, threshold, verbose=False) -> np.ndarray:
        """
        处理超出正常生理范围的极端异常值（电极接触不良）
        :param data: 一维或二维的脑电数据
        :param threshold: 异常值判定阈值 (微伏)。
                          传入单个 float 时，所有通道共用该阈值；
                          传入 list/np.ndarray 时，每个通道使用对应位置的阈值。
        """
        clean_data = data.copy()
        n_channels = clean_data.shape[1] if clean_data.ndim > 1 else 1

        # 将 threshold 统一转为数组形式，方便逐通道索引
        if np.isscalar(threshold):
            thresholds = np.full(n_channels, float(threshold))
        else:
            thresholds = np.asarray(threshold, dtype=float)

        # 遍历每一个通道
        for i in range(n_channels):
            channel_data = clean_data[:, i] if clean_data.ndim > 1 else clean_data
            ch_threshold = thresholds[i]

            # 找到超出阈值的点的索引
            outlier_mask = np.abs(channel_data) > ch_threshold

            if np.any(outlier_mask):
                outlier_count = np.sum(outlier_mask)
                if verbose:
                    print(f"通道 {i}: 发现 {outlier_count} 个极端异常采样点...")

                # 创建一个 pandas Series 利用其强大的插值功能
                s = pd.Series(channel_data)

                # 将异常值替换为 NaN
                s[outlier_mask] = np.nan

                # 1. 首先尝试线性插值 (适用于短暂脱落)
                # limit=25 表示如果连续超过25个点(0.1秒)都是异常的，就不插值了
                s = s.interpolate(method='linear', limit=25, limit_direction='both')

                # 2. 如果还有大片连续的 NaN (说明电极彻底掉了很久)，
                # 只能进行前向/后向填充或者将其限制在阈值边界，防止后续滤波器报错
                s = s.ffill().bfill()

                # 最后做一次硬裁剪，确保绝对没有超出阈值的数据进入滤波器
                s = s.clip(lower=-ch_threshold, upper=ch_threshold)

                if clean_data.ndim > 1:
                    clean_data[:, i] = s.values
                else:
                    clean_data = s.values

        return clean_data

    def apply_basic_filters(self, data: np.ndarray) -> np.ndarray:
        """
        第一步：基础滤波 (陷波 + 带通)
        """
        # 1. 50Hz 陷波滤波器 (去除工频干扰)
        # 考虑到中国国内的交流电频率为 50Hz
        notch_freq = 50.0 
        quality_factor = 30.0 
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, self.fs)
        data_notched = signal.filtfilt(b_notch, a_notch, data, axis=0)

        # 2. 0.5 - 45Hz 带通滤波器 (保留 Delta 到 Gamma 的有效频段，去除基线漂移)
        lowcut = 0.5
        highcut = 45.0
        order = 4
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b_band, a_band = signal.butter(order, [low, high], btype='band')
        data_filtered = signal.filtfilt(b_band, a_band, data_notched, axis=0)

        return data_filtered

    def remove_artifacts_regression(self, eeg_data: np.ndarray, ref_data: np.ndarray, verbose=False) -> np.ndarray:
        """
        第二步：伪迹消除 (使用多元线性回归)
        原理：假设 EEG_raw = EEG_clean + w * Artifact_ref
        通过拟合求出权重 w，然后从 raw 中减去 w * Artifact_ref
        """
        clean_eeg = np.zeros_like(eeg_data)
        
        # 逐通道处理 EEG
        for i in range(eeg_data.shape[1]):
            target_eeg = eeg_data[:, i]
            
            # 使用线性回归模型拟合参考信号到 EEG 信号的映射
            reg = LinearRegression().fit(ref_data, target_eeg)
            
            # 预测出伪迹在 EEG 中的成分
            artifact_component = reg.predict(ref_data)
            
            # 从原始 EEG 中减去伪迹成分
            clean_eeg[:, i] = target_eeg - artifact_component
            
        return clean_eeg

def _check_emg_quality_per_window(emg_filtered: np.ndarray, fs: int = 250,
                                   window_sec: float = 2.0, overlap_sec: float = 1.0) -> np.ndarray:
    """
    检查每个滑动窗口内 EMG 信号的质量。
    返回一个布尔数组，长度等于信号采样点数。
    True = 该窗口 EMG 质量合格，可以用于伪迹回归；
    False = 该窗口 EMG 质量不合格（PTP < 0.1 或 > 250），跳过回归。
    """
    n_samples = emg_filtered.shape[0]
    window_size = int(window_sec * fs)
    step_size = int((window_sec - overlap_sec) * fs)

    # 默认所有采样点标记为合格
    emg_good = np.ones(n_samples, dtype=bool)

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        window = emg_filtered[start:end]

        # 检查两个 EMG 通道的 PTP
        for ch in range(emg_filtered.shape[1]):
            ptp = np.max(window[:, ch]) - np.min(window[:, ch])
            if ptp < 0.1 or ptp > 250.0:
                emg_good[start:end] = False
                break

    return emg_good


def process_pipeline(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    """
    执行完整的预处理流水线
    """
    # 假设你的采样率是 250Hz
    fs = 250
    preprocessor = EEGPreprocessor(fs=fs)
    df = df.reset_index()
    # 只处理 4 个通道：eeg_1, eeg_2, emg_1, emg_2
    target_cols = ['eeg_1', 'eeg_2', 'emg_1', 'emg_2']
    df_numeric = df[target_cols].copy()

    # 强制转换为 float，如果有无法转换的（比如空字符串），会变成 NaN
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

    # 插值和填充
    df.update(df_numeric.interpolate(method='linear').fillna(0))

    # 提取 EEG 数据 (2通道)
    eeg_cols = ['eeg_1', 'eeg_2']
    eeg_raw = df[eeg_cols].values

    # 提取参考伪迹数据 (仅 EMG 2通道)
    ref_cols = ['emg_1', 'emg_2']
    ref_raw = df[ref_cols].values

    # ==== 在滤波前清理电极接触不良造成的极端异常值 ====
    if verbose:
        print("正在清理电极接触不良造成的极端异常值...")
    eeg_raw_fixed = preprocessor.handle_extreme_values(eeg_raw, threshold=150.0, verbose=verbose)
    ref_raw_fixed = preprocessor.handle_extreme_values(ref_raw, threshold=300.0, verbose=verbose)

    # 确保处理后没有残留的 NaN（滤波器和回归器不接受 NaN）
    eeg_raw_fixed = np.nan_to_num(eeg_raw_fixed, nan=0.0)
    ref_raw_fixed = np.nan_to_num(ref_raw_fixed, nan=0.0)

    if verbose:
        print("正在进行基础滤波 (50Hz陷波 + 0.5-45Hz带通)...")
    # 注意这里传入的是 fixed 之后的数据！
    eeg_filtered = preprocessor.apply_basic_filters(eeg_raw_fixed)
    ref_filtered = preprocessor.apply_basic_filters(ref_raw_fixed)

    # ==== 检查原始 EEG 信号强度，决定是否进行伪迹回归 ====
    eeg_ptp = np.max(eeg_raw_fixed, axis=0) - np.min(eeg_raw_fixed, axis=0)
    min_ptp = np.min(eeg_ptp)
    if verbose:
        print(f"原始 EEG 信号 PTP: eeg_1={eeg_ptp[0]:.2f}μV, eeg_2={eeg_ptp[1]:.2f}μV")

    if min_ptp >= 10.0:
        if verbose:
            print("正在使用自适应回归法去除 EMG 和 EOG 伪迹...")
        eeg_clean = preprocessor.remove_artifacts_regression(eeg_filtered, ref_filtered, verbose=verbose)
    else:
        if verbose:
            print("原始信号过弱，跳过伪迹回归...")
        eeg_clean = eeg_filtered.copy()

    # ==== 检查 EMG 质量，对 EMG 不合格的窗口回退到未经回归的 EEG ====
    emg_good_mask = _check_emg_quality_per_window(ref_filtered, fs=fs)
    bad_emg_count = np.sum(~emg_good_mask)
    if bad_emg_count > 0:
        if verbose:
            print(f"检测到 EMG 质量不合格的采样点: {bad_emg_count}/{len(emg_good_mask)}，"
                  f"这些窗口将跳过伪迹回归...")
        # 对 EMG 不合格的采样点，用未经回归的 eeg_filtered 替换 eeg_clean
        eeg_final = np.where(emg_good_mask[:, np.newaxis], eeg_clean, eeg_filtered)
    else:
        eeg_final = eeg_clean

    # 将清理后的数据写回 DataFrame 便于后续特征提取
    df_clean = df.copy()
    df_clean['eeg_1_clean'] = eeg_final[:, 0]
    df_clean['eeg_2_clean'] = eeg_final[:, 1]

    if verbose:
        print("预处理完成！")
    return df_clean

# ================= 测试代码 =================
if __name__ == "__main__":
    target_user_id = 4141653
    eeg_df = load_eeg_data(target_user_id)
    
    # 模拟一个测试用的 DataFrame
    eeg_df_clean = process_pipeline(eeg_df)
    # pass