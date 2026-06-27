import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

def load_eeg_data(user_id: int, verbose: bool = False) -> pd.DataFrame:
    """
    从 TimescaleDB 中提取指定用户的脑电和眼动数据
    """
    # 1. 加载 .env 文件中的环境变量
    # 获取当前文件 (main.py) 的父目录 (src/)
    BASE_DIR = Path(__file__).resolve().parent
    # 再获取项目根目录
    PROJECT_ROOT = BASE_DIR.parent
    # 构建 .env 文件的绝对路径
    ENV_PATH = PROJECT_ROOT / ".env"

    # 显式指定路径加载
    load_dotenv(dotenv_path=ENV_PATH)
    
    # 2. 获取数据库连接字符串
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("未在环境变量中找到 DATABASE_URL，请检查 .env 文件。")
    
    # 3. 创建 SQLAlchemy 引擎
    # 注意：TimescaleDB 是基于 PostgreSQL 的，所以 sqlalchemy 可以直接使用 postgresql 的连接方式
    engine = create_engine(db_url)
    
    # 4. 编写 SQL 查询语句
    # 根据你的 init.sql，我们提取所有列，并按照时间升序排序
    query = f"""
        SELECT 
            time, user_id, eeg_1, eeg_2, emg_1, emg_2, 
            blink_l, blink_r, gaze_x, gaze_y, gaze_z, event_label
        FROM 
            eeg_data
        WHERE 
            user_id = {user_id}
        ORDER BY 
            time ASC;
    """
    
    if verbose:
        print(f"正在从数据库中提取 User {user_id} 的数据...")
    
    # 5. 使用 pandas 直接读取 SQL 查询结果
    try:
        df = pd.read_sql(query, engine)
        if verbose:
            print(f"[EEG] 提取成功: {len(df)} 行。")

        # 6. 数据格式化：将 time 列设置为 DatetimeIndex
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        # 7. 合并同一时间戳的重复行
        # 设备可能把同一时刻的 EEG 和 EMG 分两行发送，需要合并
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            if verbose:
                print(f"发现 {dup_count} 行重复时间戳，正在合并...")
            # 按时间戳分组，数值列取第一个非空值，event_label 同理
            df = df.groupby(df.index).first()
            if verbose:
                print(f"合并后剩余 {len(df)} 行。")

        return df
        
    except Exception as e:
        print(f"提取数据时发生错误: {e}")
        return pd.DataFrame()

def load_game_features(user_ids: list[int], verbose: bool = False) -> dict[int, dict]:
    """
    从 game_sessions 表中加载每个用户最近一局的游戏特征
    :param user_ids: 用户 ID 列表
    :return: {user_id: {'game_hit_accuracy': float, 'game_score': float}} 字典
    """
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    ENV_PATH = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=ENV_PATH)

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("未在环境变量中找到 DATABASE_URL，请检查 .env 文件。")

    engine = create_engine(db_url)

    ids_str = ",".join(str(uid) for uid in user_ids)
    query = f"""
        SELECT DISTINCT ON (user_id)
            user_id, hit_count, success_count, score
        FROM game_sessions
        WHERE user_id IN ({ids_str})
        ORDER BY user_id, created_at DESC
    """

    try:
        df = pd.read_sql(query, engine)
        result = {}
        for _, row in df.iterrows():
            hit_acc = row['success_count'] / row['hit_count'] if row['hit_count'] > 0 else 0.0
            result[int(row['user_id'])] = {
                'game_hit_accuracy': hit_acc,
                'game_score': float(row['score']),
            }
        if verbose:
            print(f"加载了 {len(result)} 个用户的游戏特征。")
        return result
    except Exception as e:
        if verbose:
            print(f"加载游戏特征时发生错误: {e}")
        return {}


# ================= 测试代码 =================
if __name__ == "__main__":
    # 假设我们要提取 user_id = 1 的数据
    target_user_id = 4141653
    
    # 获取 DataFrame
    eeg_df = load_eeg_data(target_user_id)
    
    if not eeg_df.empty:
        print("\n数据预览:")
        print(eeg_df.head())
        print("\n数据基本信息:")
        print(eeg_df.info())