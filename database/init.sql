CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS eeg_eye_data (
    time        TIMESTAMPTZ       NOT NULL,
    user_id     INT               NOT NULL,
    eeg_1       DOUBLE PRECISION  NULL,
    eeg_2       DOUBLE PRECISION  NULL,
    emg_1       DOUBLE PRECISION  NULL,
    emg_2       DOUBLE PRECISION  NULL,
    -- 确保眼动全都是 DOUBLE PRECISION
    blink_l     DOUBLE PRECISION  NULL,
    blink_r     DOUBLE PRECISION  NULL,
    gaze_x      DOUBLE PRECISION  NULL,
    gaze_y      DOUBLE PRECISION  NULL,
    gaze_z      DOUBLE PRECISION  NULL,
    event_label TEXT              NULL
);

SELECT create_hypertable('eeg_eye_data', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS eeg_eye_data_user_id_time_idx ON eeg_eye_data (user_id, time DESC);