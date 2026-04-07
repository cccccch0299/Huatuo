import numpy as np
from scipy import signal
import os

# ==========================================
# 1. 定义从 C# 移植过来的滤波器系数
# ==========================================

# 50Hz 陷波器 (Notch)
bn = [0.8768538896732492, -2.170443392130918, 5.522071837488863, -7.342463971738635, 9.419015530970857, -7.342463971738641, 5.522071837488865, -2.17044339213092, 0.8768538896732502]
an = [1.0, -2.393997411915579, 5.887335233759609, -7.573856502280153, 9.39809263934246, -7.092316711041379, 5.162566368326356, -1.9656441025019993, 0.7688727438666524]

# 0.5-40 Hz 宽带通滤波器 (EEG主频段 & EMG使用)
b_main = [0.021961263433741933, 0.0, -0.08784505373496773, 0.0, 0.13176758060245158, 0.0, -0.08784505373496773, 0.0, 0.021961263433741933]
a_main = [1.0, -5.406154514350461, 12.768907456916438, -17.41964841062876, 15.19071652785795, -8.716442114872699, 3.1973474109967395, -0.680353753924367, 0.06562740718021454]

# 各子频段滤波器 (可用于深度学习的特征提取)
# 0.35-3.5 Hz (Delta)
b_delta = [1.833801797350947e-06, 0.0, -7.335207189403788e-06, 0.0, 1.1002810784105682e-05, 0.0, -7.335207189403788e-06, 0.0, 1.833801797350947e-06]
a_delta = [1.0, -7.798683194933513, 26.61490937573006, -51.91534646621706, 63.30708297569896, -49.41914107294798, 24.11709651175616, -6.727043265977724, 0.8211251368924549]

# 4-7 Hz (Theta)
b_theta = [1.8338017973509441e-06, 0.0, -7.335207189403777e-06, 0.0, 1.1002810784105665e-05, 0.0, -7.335207189403777e-06, 0.0, 1.8338017973509441e-06]
a_theta = [1.0, -7.73406333341227, 26.23843674755751, -50.999704891798046, 62.11708217751199, -48.547531576288975, 23.77595883698728, -6.6713030091383025, 0.8211251368924553]

# 8-13 Hz (Alpha)
b_alpha = [1.3293728898752895e-05, 0.0, -5.317491559501158e-05, 0.0, 7.976237339251737e-05, 0.0, -5.317491559501158e-05, 0.0, 1.3293728898752895e-05]
a_alpha = [1.0, -7.420789634004607, 24.333876123144265, -46.04240449933033, 54.97289539619564, -42.409697852946195, 20.645719759936725, -5.799494079895906, 0.7199103272918715]

# 14-30 Hz (Beta)
b_beta  = [0.0010161163201135588, 0.0, -0.004064465280454235, 0.0, 0.006096697920681352, 0.0, -0.004064465280454235, 0.0, 0.0010161163201135588]
a_beta  = [1.0, -6.038117050206029, 16.702378553038724, -27.536641133275687, 29.547004438127917, -21.116969164532236, 9.822890374195534, -2.724452615871032, 0.3467246215165015]

# ==========================================
# 2. 核心滤波函数
# ==========================================

def apply_filters(raw_data, extract_sub_bands=False):
    """
    对单通道数据进行滤波清理
    :param raw_data: 原始数据 (1D numpy array)
    :param extract_sub_bands: 是否同时输出 Delta, Theta, Alpha, Beta 的分离频段
    :return: 滤波后的数据字典
    """
    # 第一步：过 50Hz 陷波器去除工频干扰
    # 使用 filtfilt 进行零相移滤波（比实时系统的 lfilter 更好，不会产生时间延迟）
    notched_data = signal.filtfilt(bn, an, raw_data)
    
    # 第二步：过 0.5-40Hz 主带通滤波器
    clean_main = signal.filtfilt(b_main, a_main, notched_data)
    
    # 保留小数点后3位，与 C# 中的 OnFilterDecimal 保持一致
    clean_main = np.round(clean_main, 3)
    
    results = {
        "main_0.5_40Hz": clean_main
    }
    
    # 如果深度学习需要提取特定脑波特征，可以开启这个选项
    if extract_sub_bands:
        results["delta"] = np.round(signal.filtfilt(b_delta, a_delta, notched_data), 3)
        results["theta"] = np.round(signal.filtfilt(b_theta, a_theta, notched_data), 3)
        results["alpha"] = np.round(signal.filtfilt(b_alpha, a_alpha, notched_data), 3)
        results["beta"] = np.round(signal.filtfilt(b_beta, a_beta, notched_data), 3)
        
    return results

# ==========================================
# 3. 数据读取与处理主流程
# ==========================================

def process_all_channels(txt_ch0, txt_ch1, txt_ch2, txt_ch3, output_dir="cleaned_data"):
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("开始加载原始数据...")
    # 假设 txt 文件里面每一行是一个数字，直接用 np.loadtxt 加载
    data_ch0 = np.loadtxt(txt_ch0) # EEG 1
    data_ch1 = np.loadtxt(txt_ch1) # EEG 2
    data_ch2 = np.loadtxt(txt_ch2) # EMG 1
    data_ch3 = np.loadtxt(txt_ch3) # EMG 2
    
    print(f"数据加载完成。数据点数: {len(data_ch0)} (约 {len(data_ch0)/250:.2f} 秒)")
    
    print("正在进行滤波处理...")
    
    # 处理 EEG 通道 (Ch0, Ch1)
    # 对于脑电，通常深度学习多通道输入不仅需要主波段，各子波段作为特征也是绝佳选择，所以设为 True
    res_ch0 = apply_filters(data_ch0, extract_sub_bands=True)
    res_ch1 = apply_filters(data_ch1, extract_sub_bands=True)
    
    # 处理 EMG 肌电通道 (Ch2, Ch3)
    # 根据 C# 代码逻辑，EMG 只通过了 filter2 和 filter3，即只有 Notch 和 0.5-40Hz，不需要子频段
    res_ch2 = apply_filters(data_ch2, extract_sub_bands=False)
    res_ch3 = apply_filters(data_ch3, extract_sub_bands=False)
    
    # ==========================================
    # 4. 组合并保存为深度学习友好的格式 (.npy)
    # ==========================================
    
    # 方式 A：保存一份只包含 0.5-40Hz 主频段的基础干净数据 (形状: 4 x N)
    # 这最适合想要直接把时序信号丢给 CNN 或 LSTM 的场景
    basic_clean_data = np.vstack((
        res_ch0["main_0.5_40Hz"],
        res_ch1["main_0.5_40Hz"],
        res_ch2["main_0.5_40Hz"],
        res_ch3["main_0.5_40Hz"]
    ))
    np.save(os.path.join(output_dir, "dl_basic_clean_data.npy"), basic_clean_data)
    
    # 方式 B：保存包含脑电子频段特征的扩展数据矩阵 (针对更复杂的深度学习架构)
    # 形状: (10, N) -> [EEG0_主, EEG0_D, EEG0_T, EEG0_A, EEG0_B, EEG1_主, EEG1_D..., EMG0, EMG1]
    feature_rich_data = np.vstack((
        res_ch0["main_0.5_40Hz"], res_ch0["delta"], res_ch0["theta"], res_ch0["alpha"], res_ch0["beta"],
        res_ch1["main_0.5_40Hz"], res_ch1["delta"], res_ch1["theta"], res_ch1["alpha"], res_ch1["beta"],
        res_ch2["main_0.5_40Hz"],
        res_ch3["main_0.5_40Hz"]
    ))
    np.save(os.path.join(output_dir, "dl_feature_rich_data.npy"), feature_rich_data)
    
    print(f"处理完毕！数据已保存到 '{output_dir}' 目录下。")
    print(" - dl_basic_clean_data.npy: 包含4个通道的 0.5-40Hz 纯净信号。")
    print(" - dl_feature_rich_data.npy: 包含主频段以及 Delta/Theta/Alpha/Beta 子特征，共12个特征通道。")

# 运行示例（请将此处替换为你实际的 txt 文件名路径）
if __name__ == "__main__":
    # 确保有这四个文件在同级目录，或者写入绝对路径
    try:
        process_all_channels("data0.txt", "data1.txt", "data2.txt", "data3.txt")
    except FileNotFoundError:
        print("请将代码里的 'data0.txt' 等替换为你实际的文件路径再运行！")