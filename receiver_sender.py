from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    import asyncpg
except ImportError:  # pragma: no cover - handled at runtime
    asyncpg = None


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("eeg-backend")

# CHANNEL_MAP = {
#     0: "eeg_1",
#     1: "eeg_2",
#     2: "emg_1",
#     3: "emg_2",
# }
CHANNEL_MAP = {
    1: "eeg_1",  
    2: "eeg_2",  
    3: "emg_1",  
    0: "emg_2",  
}
CHANNEL_COLUMNS = ("eeg_1", "eeg_2", "emg_1", "emg_2")
INSERT_COLUMNS = ("time", "user_id", *CHANNEL_COLUMNS, "event_label")
DEFAULT_CHANNEL_VALUES = {column: None for column in CHANNEL_COLUMNS}
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LOCAL_TIMEZONE = datetime.now().astimezone().tzinfo or timezone.utc

def load_dotenv() -> None:
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:5432/postgres")
POOL_MIN_SIZE = int(os.getenv("EEG_POOL_MIN_SIZE", "1"))
POOL_MAX_SIZE = int(os.getenv("EEG_POOL_MAX_SIZE", "8"))
INGEST_BATCH_SIZE = int(os.getenv("EEG_INGEST_BATCH_SIZE", "512"))
INGEST_FLUSH_INTERVAL_MS = int(os.getenv("EEG_FLUSH_INTERVAL_MS", "40"))
INGEST_QUEUE_TIMEOUT_SECONDS = float(os.getenv("EEG_QUEUE_TIMEOUT_SECONDS", "1.0"))
INGEST_QUEUE_MAX_BATCHES = int(os.getenv("EEG_QUEUE_MAX_BATCHES", "2048"))
LATEST_BUFFER_SIZE = int(os.getenv("EEG_BUFFER_SIZE", "2500"))
ALIGN_SAMPLE_RATE_HZ = int(os.getenv("EEG_ALIGN_SAMPLE_RATE_HZ", "250"))
AUTO_INIT_DB = os.getenv("EEG_AUTO_INIT_DB", "true").lower() not in {"0", "false", "no"}
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("EEG_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
SAMPLE_PERIOD_US = max(1, round(1_000_000 / ALIGN_SAMPLE_RATE_HZ))


class EEGPoint(BaseModel):
    ch: int
    val: float
    sTime: Union[str, int, float, datetime]


class DataWrapper(BaseModel):
    items: List[EEGPoint] = Field(default_factory=list)
    user_id: int = 1
    event_label: Optional[str] = None


@dataclass(slots=True)
class EEGRow:
    time: datetime
    user_id: int
    eeg_1: Optional[float]
    eeg_2: Optional[float]
    emg_1: Optional[float]
    emg_2: Optional[float]
    event_label: Optional[str] = None

    def as_record(self) -> tuple[Any, ...]:
        return (
            self.time,
            self.user_id,
            self.eeg_1,
            self.eeg_2,
            self.emg_1,
            self.emg_2,
            self.event_label,
        )

    def as_payload(self) -> Dict[str, Any]:
        return {
            "time": self.time.astimezone(timezone.utc).isoformat(),
            "user_id": self.user_id,
            "eeg_1": self.eeg_1,
            "eeg_2": self.eeg_2,
            "emg_1": self.emg_1,
            "emg_2": self.emg_2,
            "event_label": self.event_label,
        }


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: Dict[int, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, user_id: int) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[user_id].add(websocket)

    async def disconnect(self, websocket: WebSocket, user_id: int) -> None:
        async with self._lock:
            self._connections[user_id].discard(websocket)
            if not self._connections[user_id]:
                self._connections.pop(user_id, None)

    async def broadcast(self, user_id: int, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return

        async with self._lock:
            clients = list(self._connections.get(user_id, set()))

        if not clients:
            return

        payload = {"type": "eeg_rows", "user_id": user_id, "rows": rows}
        stale_clients: List[WebSocket] = []

        for client in clients:
            try:
                await client.send_json(payload)
            except Exception:
                stale_clients.append(client)

        for client in stale_clients:
            await self.disconnect(client, user_id)

    async def client_count(self) -> int:
        async with self._lock:
            return sum(len(clients) for clients in self._connections.values())


def parse_sample_time(value: Union[str, int, float, datetime]) -> datetime:
    if isinstance(value, datetime):
        sample_time = value
    elif isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 1e15:
            sample_time = datetime.fromtimestamp(numeric / 1_000_000, tz=timezone.utc)
        elif numeric > 1e12:
            sample_time = datetime.fromtimestamp(numeric / 1_000, tz=timezone.utc)
        else:
            sample_time = datetime.fromtimestamp(numeric, tz=timezone.utc)
    else:
        raw_value = str(value).strip()
        if not raw_value:
            raise ValueError("sTime is empty")

        candidate = raw_value.replace("Z", "+00:00")
        try:
            sample_time = datetime.fromisoformat(candidate)
        except ValueError:
            sample_time = None
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                try:
                    sample_time = datetime.strptime(raw_value, fmt)
                    break
                except ValueError:
                    continue

            if sample_time is None:
                raise ValueError(f"Unsupported sTime value: {raw_value}") from None

    if sample_time.tzinfo is None:
        sample_time = sample_time.replace(tzinfo=LOCAL_TIMEZONE)

    return sample_time.astimezone(timezone.utc)


def normalize_sample_time(sample_time: datetime) -> datetime:
    delta = sample_time.astimezone(timezone.utc) - EPOCH
    epoch_us = (
        delta.days * 24 * 60 * 60 * 1_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )
    bucket_us = round(epoch_us / SAMPLE_PERIOD_US) * SAMPLE_PERIOD_US
    return EPOCH + timedelta(microseconds=bucket_us)


def align_payload(payload: DataWrapper) -> List[EEGRow]:
    aligned_rows: Dict[datetime, Dict[str, Optional[float]]] = {}

    for point in payload.items:
        column = CHANNEL_MAP.get(point.ch)
        if column is None:
            continue

        normalized_time = normalize_sample_time(parse_sample_time(point.sTime))
        channel_values = aligned_rows.setdefault(normalized_time, DEFAULT_CHANNEL_VALUES.copy())
        channel_values[column] = point.val

    rows = [
        EEGRow(
            time=sample_time,
            user_id=payload.user_id,
            eeg_1=channels["eeg_1"],
            eeg_2=channels["eeg_2"],
            emg_1=channels["emg_1"],
            emg_2=channels["emg_2"],
            event_label=payload.event_label,
        )
        for sample_time, channels in aligned_rows.items()
        if any(channels[column] is not None for column in CHANNEL_COLUMNS)
    ]
    rows.sort(key=lambda row: row.time)
    return rows


async def create_pool() -> Any:
    if asyncpg is None:
        raise RuntimeError("asyncpg is not installed. Run `pip install -r requirements.txt` first.")

    return await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=POOL_MIN_SIZE,
        max_size=POOL_MAX_SIZE,
        command_timeout=30,
        max_inactive_connection_lifetime=60,
    )


async def ensure_schema(pool: Any) -> None:
    async with pool.acquire() as conn:
        timescaledb_installed = await conn.fetchval(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb');"
        )

        if AUTO_INIT_DB and not timescaledb_installed:
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                timescaledb_installed = True
            except Exception as exc:
                LOGGER.warning("Unable to create TimescaleDB extension automatically: %s", exc)

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eeg_data (
                time        TIMESTAMPTZ       NOT NULL,
                user_id     INT               NOT NULL,
                eeg_1       DOUBLE PRECISION  NULL,
                eeg_2       DOUBLE PRECISION  NULL,
                emg_1       DOUBLE PRECISION  NULL,
                emg_2       DOUBLE PRECISION  NULL,
                event_label TEXT              NULL
            );
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS eeg_data_user_id_time_idx
            ON eeg_data (user_id, time DESC);
            """
        )

        if not AUTO_INIT_DB or not timescaledb_installed:
            return

        try:
            await conn.execute(
                "SELECT create_hypertable('eeg_data', 'time', if_not_exists => TRUE);"
            )
        except Exception as exc:
            LOGGER.warning("Unable to convert eeg_data to hypertable automatically: %s", exc)


async def flush_rows(app: FastAPI, rows: List[EEGRow]) -> None:
    if not rows:
        return

    rows.sort(key=lambda row: (row.time, row.user_id))
    records = [row.as_record() for row in rows]

    for attempt in range(3):
        try:
            async with app.state.pool.acquire() as conn:
                await conn.copy_records_to_table(
                    "eeg_data",
                    records=records,
                    columns=INSERT_COLUMNS,
                )
            break
        except Exception as exc:
            if attempt == 2:
                app.state.writer_error = exc
                raise
            await asyncio.sleep(0.2 * (attempt + 1))

    grouped_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[row.user_id].append(row.as_payload())

    async with app.state.latest_lock:
        for user_id, payload_rows in grouped_rows.items():
            app.state.latest_rows[user_id].extend(payload_rows)

    for user_id, payload_rows in grouped_rows.items():
        await app.state.websocket_manager.broadcast(user_id, payload_rows)


async def ingest_worker(app: FastAPI) -> None:
    queue: asyncio.Queue[Optional[List[EEGRow]]] = app.state.ingest_queue
    pending_rows: List[EEGRow] = []
    flush_interval = INGEST_FLUSH_INTERVAL_MS / 1000
    shutdown_requested = False

    while not shutdown_requested:
        timeout = flush_interval if pending_rows else None

        try:
            batch = await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            await flush_rows(app, pending_rows)
            pending_rows = []
            continue

        if batch is None:
            queue.task_done()
            shutdown_requested = True
            continue

        pending_rows.extend(batch)
        queue.task_done()

        while len(pending_rows) < INGEST_BATCH_SIZE:
            try:
                next_batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if next_batch is None:
                queue.task_done()
                shutdown_requested = True
                break

            pending_rows.extend(next_batch)
            queue.task_done()

        if len(pending_rows) >= INGEST_BATCH_SIZE:
            await flush_rows(app, pending_rows)
            pending_rows = []

    if pending_rows:
        await flush_rows(app, pending_rows)


async def fetch_rows_from_db(app: FastAPI, user_id: int, limit: int) -> List[Dict[str, Any]]:
    query = """
        SELECT time, user_id, eeg_1, eeg_2, emg_1, emg_2, event_label
        FROM eeg_data
        WHERE user_id = $1
        ORDER BY time DESC
        LIMIT $2
    """
    async with app.state.pool.acquire() as conn:
        records = await conn.fetch(query, user_id, limit)

    rows = []
    for record in reversed(records):
        rows.append(
            {
                "time": record["time"].astimezone(timezone.utc).isoformat(),
                "user_id": record["user_id"],
                "eeg_1": record["eeg_1"],
                "eeg_2": record["eeg_2"],
                "emg_1": record["emg_1"],
                "emg_2": record["emg_2"],
                "event_label": record["event_label"],
            }
        )
    return rows


async def fetch_history_rows(
    app: FastAPI,
    user_id: int,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    limit: int,
) -> List[Dict[str, Any]]:
    query = """
        SELECT time, user_id, eeg_1, eeg_2, emg_1, emg_2, event_label
        FROM eeg_data
        WHERE user_id = $1
          AND ($2::timestamptz IS NULL OR time >= $2)
          AND ($3::timestamptz IS NULL OR time <= $3)
        ORDER BY time ASC
        LIMIT $4
    """
    async with app.state.pool.acquire() as conn:
        records = await conn.fetch(query, user_id, start_time, end_time, limit)

    rows = []
    for record in records:
        rows.append(
            {
                "time": record["time"].astimezone(timezone.utc).isoformat(),
                "user_id": record["user_id"],
                "eeg_1": record["eeg_1"],
                "eeg_2": record["eeg_2"],
                "emg_1": record["emg_1"],
                "emg_2": record["emg_2"],
                "event_label": record["event_label"],
            }
        )
    return rows


async def fetch_time_bounds(app: FastAPI, user_id: int) -> Dict[str, Any]:
    query = """
        SELECT
            MIN(time) AS earliest_time,
            MAX(time) AS latest_time,
            COUNT(*) AS total_rows
        FROM eeg_data
        WHERE user_id = $1
    """
    async with app.state.pool.acquire() as conn:
        record = await conn.fetchrow(query, user_id)

    earliest_time = record["earliest_time"]
    latest_time = record["latest_time"]
    total_rows = record["total_rows"] or 0

    return {
        "user_id": user_id,
        "earliest_time": earliest_time.astimezone(timezone.utc).isoformat() if earliest_time else None,
        "latest_time": latest_time.astimezone(timezone.utc).isoformat() if latest_time else None,
        "total_rows": total_rows,
    }


async def fetch_rows_from_memory(app: FastAPI, user_id: int, limit: int) -> List[Dict[str, Any]]:
    async with app.state.latest_lock:
        cached_rows: Deque[Dict[str, Any]] = app.state.latest_rows.get(user_id, deque())
        return list(cached_rows)[-limit:]


def build_app() -> FastAPI:
    app = FastAPI(title="EEG Receiver", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup() -> None:
        app.state.writer_error = None
        app.state.latest_rows = defaultdict(lambda: deque(maxlen=LATEST_BUFFER_SIZE))
        app.state.latest_lock = asyncio.Lock()
        app.state.websocket_manager = ConnectionManager()
        app.state.ingest_queue = asyncio.Queue(maxsize=INGEST_QUEUE_MAX_BATCHES)
        app.state.pool = await create_pool()
        await ensure_schema(app.state.pool)
        app.state.writer_task = asyncio.create_task(ingest_worker(app), name="eeg-ingest-worker")
        LOGGER.info("EEG backend started. Sample alignment = %sus", SAMPLE_PERIOD_US)

    @app.on_event("shutdown")
    async def shutdown() -> None:
        writer_task = getattr(app.state, "writer_task", None)
        ingest_queue = getattr(app.state, "ingest_queue", None)
        pool = getattr(app.state, "pool", None)

        if ingest_queue is not None:
            await ingest_queue.join()
            await ingest_queue.put(None)

        if writer_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await writer_task

        if pool is not None:
            await pool.close()

    if FRONTEND_DIR.exists():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        writer_task = getattr(app.state, "writer_task", None)
        writer_error = getattr(app.state, "writer_error", None)
        if writer_error is None and writer_task is not None and writer_task.done():
            with contextlib.suppress(asyncio.CancelledError):
                writer_error = writer_task.exception()
        return {
            "status": "ok" if writer_error is None else "degraded",
            "db_connected": getattr(app.state, "pool", None) is not None,
            "queue_size": app.state.ingest_queue.qsize(),
            "websocket_clients": await app.state.websocket_manager.client_count(),
            "writer_running": bool(writer_task and not writer_task.done()),
            "writer_error": str(writer_error) if writer_error else None,
        }

    @app.post("/upload_eeg", deprecated=True)
    @app.post("/api/eeg/upload")
    async def upload_eeg(data: DataWrapper) -> Dict[str, Any]:
        writer_error = getattr(app.state, "writer_error", None)
        writer_task = getattr(app.state, "writer_task", None)
        if writer_error is None and writer_task is not None and writer_task.done():
            with contextlib.suppress(asyncio.CancelledError):
                writer_error = writer_task.exception()
        if writer_error is not None:
            raise HTTPException(status_code=503, detail=f"Ingest worker unavailable: {writer_error}")

        try:
            rows = align_payload(data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not rows:
            return {
                "status": "ignored",
                "received_points": len(data.items),
                "accepted_rows": 0,
            }

        try:
            await asyncio.wait_for(
                app.state.ingest_queue.put(rows),
                timeout=INGEST_QUEUE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail="Ingest queue is full; backend cannot keep up right now.",
            ) from exc

        return {
            "status": "accepted",
            "received_points": len(data.items),
            "accepted_rows": len(rows),
        }

    @app.get("/api/eeg/latest")
    async def get_latest_eeg(
        user_id: int = Query(1, ge=1),
        limit: int = Query(750, ge=1, le=5000),
    ) -> Dict[str, Any]:
        try:
            rows = await fetch_rows_from_db(app, user_id, limit)
            source = "database"
        except Exception as exc:
            LOGGER.warning("Falling back to memory buffer for latest EEG rows: %s", exc)
            rows = await fetch_rows_from_memory(app, user_id, limit)
            source = "memory"

        return {"user_id": user_id, "rows": rows, "source": source}

    @app.get("/api/eeg/history")
    async def get_history_eeg(
        user_id: int = Query(1, ge=1),
        start_time: Optional[str] = Query(None),
        end_time: Optional[str] = Query(None),
        limit: int = Query(5000, ge=1, le=200000),
    ) -> Dict[str, Any]:
        try:
            parsed_start_time = parse_sample_time(start_time) if start_time else None
            parsed_end_time = parse_sample_time(end_time) if end_time else None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if parsed_start_time and parsed_end_time and parsed_start_time > parsed_end_time:
            raise HTTPException(status_code=422, detail="start_time must be earlier than end_time")

        rows = await fetch_history_rows(app, user_id, parsed_start_time, parsed_end_time, limit)
        return {
            "user_id": user_id,
            "rows": rows,
            "source": "database-history",
            "start_time": parsed_start_time.isoformat() if parsed_start_time else None,
            "end_time": parsed_end_time.isoformat() if parsed_end_time else None,
        }

    @app.get("/api/eeg/bounds")
    async def get_eeg_bounds(
        user_id: int = Query(1, ge=1),
    ) -> Dict[str, Any]:
        return await fetch_time_bounds(app, user_id)

    @app.websocket("/ws/eeg")
    async def eeg_stream(websocket: WebSocket) -> None:
        raw_user_id = websocket.query_params.get("user_id", "1")
        try:
            user_id = int(raw_user_id)
        except ValueError:
            await websocket.close(code=1008, reason="user_id must be an integer")
            return

        await app.state.websocket_manager.connect(websocket, user_id)
        await websocket.send_json({"type": "welcome", "user_id": user_id})

        try:
            while True:
                message = await websocket.receive_text()
                if message.lower() == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            pass
        finally:
            await app.state.websocket_manager.disconnect(websocket, user_id)

    return app


app = build_app()
