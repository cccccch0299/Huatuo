# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Huatuo is a VR-based brain-computer interface (BCI) system for real-time EEG/EMG data acquisition, storage, signal processing, visualization, and machine learning classification. The system captures EEG and EMG signals from a NeuroXess headset paired with a PICO XR VR headset.

## Tech Stack

- **VR Client**: Unity + PICO XR SDK + NeuroXess SDK (C#)
- **Backend**: FastAPI + Uvicorn + asyncpg (Python)
- **Database**: TimescaleDB (PostgreSQL extension)
- **Frontend**: Vanilla JS + ECharts 5.5.1
- **ML**: scikit-learn, XGBoost, LightGBM, PyTorch (EEGNet)

## Common Commands

### Environment Setup
```bash
conda activate huatuo
```

### Start Database
```bash
docker-compose up -d
```

### Start Backend
```bash
./start_backend.sh
# Or directly:
python run_backend.py
# Runs uvicorn on http://0.0.0.0:8000
```

### Start Simulator (for testing)
```bash
python eeg_simulator.py --backend-url http://127.0.0.1:8000/api/eeg/upload --user-id 99 --sample-rate 250 --batch-ms 100 --speed 1
```

### Run ML Pipeline
```bash
# Traditional ML (Random Forest)
python model/run_pipeline.py --labels-csv labels.csv --model rf --verbose

# EEGNet deep learning
python model/run_pipeline.py --labels-csv labels.csv --model eegnet --epochs 200 --verbose

# With SMOTE oversampling
python model/run_pipeline.py --labels-csv labels.csv --model xgb --smote --verbose
```

## Architecture

### Data Flow
```
VR Headset (PICO XR) + NeuroXess EEG Headset
    ↓ Bluetooth
Unity App (C#) - EyeNeuroManagerService.cs / DataControl.cs
    ↓ HTTP POST /api/eeg/upload (~250Hz batches)
FastAPI Backend (receiver_sender.py)
    ↓ asyncpg COPY (batch ingest)
TimescaleDB (eeg_data hypertable)
    ↓ WebSocket /ws/eeg
Web Frontend (ECharts visualization)
    ↓ SQL queries
ML Pipeline (model/) → Saved models in test/
```

### Key Files
- `receiver_sender.py`: FastAPI backend - HTTP endpoints, WebSocket, async ingest pipeline
- `run_backend.py`: Backend entry point (uvicorn runner)
- `EyeNeuroManagerService.cs`: Unity client - NeuroXess SDK + PICO eye tracking integration
- `DataControl.cs`: Unity client - game session data + eye tracking
- `data_processing.py`: Signal filter coefficients and offline processing
- `eeg_simulator.py`: Synthetic EEG/EMG data simulator for testing
- `model/run_pipeline.py`: ML pipeline orchestrator
- `model/preprocess.py`: 50Hz notch + bandpass + artifact regression
- `model/feature_extraction.py`: PSD, DE, power ratios per sliding window
- `model/model_training.py`: RF/XGB/LGB training with cross-validation
- `model/eegnet_model.py`: EEGNet CNN (PyTorch, ~1,618 parameters)

### Channel Map
- 0=eeg_1, 1=eeg_2, 2=emg_1, 3=emg_2, 4=blink_l, 5=blink_r, 6=gaze_x, 7=gaze_y, 8=gaze_z

### Database Schema
Table `eeg_eye_data` (TimescaleDB hypertable, partitioned by time):
- time (TIMESTAMPTZ), user_id (INT), eeg_1, eeg_2, emg_1, emg_2 (DOUBLE PRECISION)
- blink_l, blink_r, gaze_x, gaze_y, gaze_z (DOUBLE PRECISION)
- event_label (TEXT)

### API Endpoints
- `POST /api/eeg/upload`: Ingest EEG data from Unity client
- `GET /api/eeg/latest`: Get latest data for active users
- `GET /api/eeg/history`: Query historical data with filters
- `GET /api/eeg/bounds`: Get time range of available data
- `GET /api/eeg/active_users`: List users with recent data
- `POST /api/game/result`: Submit game session result
- `GET /api/game/results`: Query game results
- `WS /ws/eeg`: WebSocket for real-time data streaming

## Environment Variables (.env)
- `DATABASE_URL`: PostgreSQL connection string
- `EEG_AUTO_INIT_DB`: Auto-create tables on startup (true)
- `EEG_ALIGN_SAMPLE_RATE_HZ`: Expected sample rate (250)
- `EEG_INGEST_BATCH_SIZE`: Max rows before DB flush (512)
- `EEG_FLUSH_INTERVAL_MS`: Max time before DB flush (40ms)
- `EEG_BUFFER_SIZE`: In-memory ring buffer per user (2500)
- `EEG_ALLOWED_ORIGINS`: CORS origins (*)

## User Preferences
- Always respond in Chinese (使用中文回答)
- 如果 Claude 自己启动了项目相关的后台进程（如 run_backend.py、eeg_simulator.py 等），测试完成后必须主动关闭。如果进程是用户自己启动的，则不要关闭。

## Notes
- No formal test suite exists; `eeg_simulator.py` serves as the integration test tool
- The `test/` directory stores trained model artifacts (*.joblib, *.pt) and evaluation plots
- The `result/` directory stores training run reports
- ML pipeline uses user-level train/test split to prevent data leakage
- Binary classification: cognitive impairment (0) vs. normal (1)
- Frontend is served as static files from `/` by the backend
