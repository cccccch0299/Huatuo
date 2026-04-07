CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS eeg_data (
    time        TIMESTAMPTZ       NOT NULL,
    user_id     INT               NOT NULL,
    eeg_1       DOUBLE PRECISION  NULL,
    eeg_2       DOUBLE PRECISION  NULL,
    emg_1       DOUBLE PRECISION  NULL,
    emg_2       DOUBLE PRECISION  NULL,
    event_label TEXT              NULL
);

SELECT create_hypertable('eeg_data', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS eeg_data_user_id_time_idx
ON eeg_data (user_id, time DESC);
