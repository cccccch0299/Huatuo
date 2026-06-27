"""
批量处理 labels.csv 中所有 user_id 的原始数据，
通过 ML 管线预处理后存入 processed_eeg_data 表。

用法:
    python backfill_processed_data.py              # 处理所有 user_id
    python backfill_processed_data.py --dry-run    # 仅预览，不执行
    python backfill_processed_data.py --user-id 5  # 只处理指定 user_id
"""
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

LABELS_CSV = os.path.join(os.path.dirname(__file__), "labels.csv")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:123@localhost:5432/postgres")
SAMPLE_RATE_HZ = float(os.getenv("EEG_ALIGN_SAMPLE_RATE_HZ", "250"))

CHANNEL_COLUMNS = (
    "eeg_1", "eeg_2", "emg_1", "emg_2",
    "blink_l", "blink_r", "gaze_x", "gaze_y", "gaze_z",
)
INSERT_COLUMNS = ("time", "user_id", *CHANNEL_COLUMNS, "event_label")
SELECT_COLUMNS_SQL = ", ".join(INSERT_COLUMNS)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def load_valid_user_ids(csv_path: str) -> list[int]:
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_excel(csv_path)
    df = df.dropna(subset=["user_id"])
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df = df.dropna(subset=["user_id"])
    return sorted(set(df["user_id"].astype(int).tolist()))


# Import processing functions from receiver_sender
sys.path.insert(0, str(Path(__file__).parent))
from receiver_sender import _ml_process_signals


async def process_one_user(conn: asyncpg.Connection, user_id: int) -> bool:
    """Process a single user_id and store in processed_eeg_data."""
    records = await conn.fetch(
        f"SELECT {SELECT_COLUMNS_SQL} FROM eeg_data WHERE user_id = $1 ORDER BY time ASC",
        user_id,
    )
    if not records:
        print(f"  user_id={user_id}: 无数据，跳过")
        return False

    time_axis_ms = [
        round((r["time"].astimezone(timezone.utc) - EPOCH).total_seconds() * 1000)
        for r in records
    ]

    raw_signals = {}
    for ch in CHANNEL_COLUMNS:
        raw_signals[ch] = [float(r[ch]) if r[ch] is not None else None for r in records]

    processed = _ml_process_signals(time_axis_ms, raw_signals, SAMPLE_RATE_HZ)
    if not processed:
        print(f"  user_id={user_id}: ML 处理返回空结果，跳过")
        return False

    # Store blink/gaze as-is
    for ch in ("blink_l", "blink_r", "gaze_x", "gaze_y", "gaze_z"):
        processed[ch] = [float(v) if v is not None else None for v in raw_signals[ch]]
    processed["raw_eeg_1"] = raw_signals["eeg_1"]
    processed["raw_eeg_2"] = raw_signals["eeg_2"]

    upsert = """
        INSERT INTO processed_eeg_data
            (user_id, sample_rate_hz, row_count, time_axis_ms,
             eeg_1_clean, eeg_2_clean, emg_1_clean, emg_2_clean,
             blink_l, blink_r, gaze_x, gaze_y, gaze_z,
             sub_bands, raw_eeg_1, raw_eeg_2)
        VALUES ($1,$2,$3,$4::jsonb,$5::jsonb,$6::jsonb,$7::jsonb,$8::jsonb,
                $9::jsonb,$10::jsonb,$11::jsonb,$12::jsonb,$13::jsonb,
                $14::jsonb,$15::jsonb,$16::jsonb)
        ON CONFLICT (user_id) DO UPDATE SET
            sample_rate_hz = EXCLUDED.sample_rate_hz,
            row_count      = EXCLUDED.row_count,
            time_axis_ms   = EXCLUDED.time_axis_ms,
            eeg_1_clean    = EXCLUDED.eeg_1_clean,
            eeg_2_clean    = EXCLUDED.eeg_2_clean,
            emg_1_clean    = EXCLUDED.emg_1_clean,
            emg_2_clean    = EXCLUDED.emg_2_clean,
            blink_l        = EXCLUDED.blink_l,
            blink_r        = EXCLUDED.blink_r,
            gaze_x         = EXCLUDED.gaze_x,
            gaze_y         = EXCLUDED.gaze_y,
            gaze_z         = EXCLUDED.gaze_z,
            sub_bands      = EXCLUDED.sub_bands,
            raw_eeg_1      = EXCLUDED.raw_eeg_1,
            raw_eeg_2      = EXCLUDED.raw_eeg_2,
            created_at     = NOW()
    """
    await conn.execute(
        upsert, user_id, SAMPLE_RATE_HZ, len(records),
        json.dumps(time_axis_ms),
        json.dumps(processed["eeg_1_clean"]),
        json.dumps(processed["eeg_2_clean"]),
        json.dumps(processed["emg_1_clean"]),
        json.dumps(processed["emg_2_clean"]),
        json.dumps(processed["blink_l"]),
        json.dumps(processed["blink_r"]),
        json.dumps(processed["gaze_x"]),
        json.dumps(processed["gaze_y"]),
        json.dumps(processed["gaze_z"]),
        json.dumps(processed["sub_bands"]),
        json.dumps(processed["raw_eeg_1"]),
        json.dumps(processed["raw_eeg_2"]),
    )
    return True


async def main():
    dry_run = "--dry-run" in sys.argv

    # Parse optional --user-id
    target_user_id = None
    for i, arg in enumerate(sys.argv):
        if arg == "--user-id" and i + 1 < len(sys.argv):
            target_user_id = int(sys.argv[i + 1])

    if target_user_id:
        user_ids = [target_user_id]
    else:
        user_ids = load_valid_user_ids(LABELS_CSV)

    print(f"labels.csv 中有效 user_id: {len(user_ids)} 个")
    print(f"user_id 列表: {user_ids}")
    print()

    if dry_run:
        print("=== 仅预览模式 (--dry-run)，不执行处理 ===")
        return

    conn = await asyncpg.connect(DATABASE_URL)
    try:
        # Ensure table exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_eeg_data (
                user_id          INT         PRIMARY KEY,
                sample_rate_hz   DOUBLE PRECISION NOT NULL,
                row_count        INT         NOT NULL,
                time_axis_ms     JSONB       NOT NULL,
                eeg_1_clean      JSONB       NOT NULL,
                eeg_2_clean      JSONB       NOT NULL,
                emg_1_clean      JSONB       NOT NULL,
                emg_2_clean      JSONB       NOT NULL,
                blink_l          JSONB       NOT NULL,
                blink_r          JSONB       NOT NULL,
                gaze_x           JSONB       NOT NULL,
                gaze_y           JSONB       NOT NULL,
                gaze_z           JSONB       NOT NULL,
                sub_bands        JSONB       NOT NULL DEFAULT '{}',
                raw_eeg_1        JSONB       NOT NULL DEFAULT '[]',
                raw_eeg_2        JSONB       NOT NULL DEFAULT '[]',
                created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        success = 0
        failed = 0
        for uid in user_ids:
            print(f"处理 user_id={uid} ...", end=" ", flush=True)
            try:
                ok = await process_one_user(conn, uid)
                if ok:
                    print("完成")
                    success += 1
                else:
                    failed += 1
            except Exception as exc:
                print(f"失败: {exc}")
                failed += 1

        print()
        print(f"处理完成: 成功 {success}, 失败 {failed}, 共 {len(user_ids)}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
