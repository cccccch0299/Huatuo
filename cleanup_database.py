"""
清理数据库中不在 labels.csv 中的 user_id 对应的数据。
删除 eeg_data 和 game_sessions 两张表中的无关数据。

用法:
    python cleanup_database.py              # 先预览，确认后才执行（事务保护，可回滚）
    python cleanup_database.py --dry-run    # 仅预览，不执行删除
"""
import os
import sys
import psycopg2
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LABELS_CSV = os.path.join(os.path.dirname(__file__), "labels.csv")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:123@localhost:5432/postgres")


def load_valid_user_ids(csv_path: str) -> set[int]:
    """从 labels.csv (或 xlsx) 读取所有有效的 user_id"""
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_excel(csv_path)
    df = df.dropna(subset=["user_id"])
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df = df.dropna(subset=["user_id"])
    return set(df["user_id"].astype(int).tolist())


def main():
    dry_run = "--dry-run" in sys.argv

    valid_ids = load_valid_user_ids(LABELS_CSV)
    print(f"labels.csv 中有效 user_id 数量: {len(valid_ids)}")
    print(f"有效 user_id: {sorted(valid_ids)}")
    print()

    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        tables_info = []
        for table in ("eeg_data", "game_sessions", "processed_eeg_data"):
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                (table,),
            )
            exists = cur.fetchone()[0]
            if not exists:
                print(f"[{table}] 表不存在，跳过")
                continue

            cur.execute(f"SELECT COUNT(*) FROM {table}")
            total_before = cur.fetchone()[0]

            cur.execute(f"SELECT DISTINCT user_id FROM {table}")
            all_user_ids_in_db = {r[0] for r in cur.fetchall()}
            invalid_ids = all_user_ids_in_db - valid_ids

            if not invalid_ids:
                print(f"[{table}] 删除前总行数: {total_before}")
                print(f"[{table}] 所有 user_id 都在 labels.csv 中，无需删除")
                print()
                continue

            cur.execute(
                f"SELECT COUNT(*) FROM {table} WHERE user_id = ANY(%s)",
                (list(invalid_ids),),
            )
            deleted_count = cur.fetchone()[0]

            tables_info.append({
                "table": table,
                "total_before": total_before,
                "invalid_ids": invalid_ids,
                "deleted_count": deleted_count,
            })

            print(f"[{table}] 删除前总行数: {total_before}")
            print(f"[{table}] 数据库中的 user_id: {sorted(all_user_ids_in_db)}")
            print(f"[{table}] 将被删除的 user_id: {sorted(invalid_ids)}")
            print(f"[{table}] 将删除的行数: {deleted_count}")
            print()

        if not tables_info:
            print("所有表的数据都在 labels.csv 中，无需清理。")
            return

        if dry_run:
            print("=== 仅预览模式 (--dry-run)，不执行删除 ===")
            return

        total_delete = sum(info["deleted_count"] for info in tables_info)
        print(f"即将在事务中删除共 {total_delete} 行数据。")
        confirm = input("确认删除？输入 yes 继续: ").strip()
        if confirm != "yes":
            print("已取消，未删除任何数据。")
            conn.rollback()
            return

        for info in tables_info:
            table = info["table"]
            cur.execute(
                f"DELETE FROM {table} WHERE user_id = ANY(%s)",
                (list(info["invalid_ids"]),),
            )
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            total_after = cur.fetchone()[0]
            print(f"[{table}] 删除后总行数: {total_after}")
            print(f"[{table}] 实际删除: {info['total_before'] - total_after} 行")

        conn.commit()
        print()
        print("清理完成！labels.csv 中的有效数据已全部保留。")

    except Exception as e:
        conn.rollback()
        print(f"\n发生错误，已回滚，未删除任何数据: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
