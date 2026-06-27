const MAIN_CHANNELS = [
    { key: "eeg_1", label: "EEG 1", color: "#005ea8", toggleBands: true },
    { key: "eeg_2", label: "EEG 2", color: "#2048b3", toggleBands: true },
    { key: "emg_1", label: "EMG 1", color: "#087447", toggleBands: false },
    { key: "emg_2", label: "EMG 2", color: "#b75a00", toggleBands: false },
];

const BLINK_CHANNELS = [
    { key: "blink_l", label: "Blink L" },
    { key: "blink_r", label: "Blink R" },
];

const GAZE_CHANNELS = ["gaze_x", "gaze_y", "gaze_z"];

const SUB_BANDS = [
    { key: "delta", label: "Delta", color: "#4b71f2" },
    { key: "theta", label: "Theta", color: "#1f9bb4" },
    { key: "alpha", label: "Alpha", color: "#9a5de0" },
    { key: "beta", label: "Beta", color: "#d26a1b" },
];

const HISTORY_SIGNAL_KEYS = [
    ...MAIN_CHANNELS.map((channel) => channel.key),
    ...BLINK_CHANNELS.map((channel) => channel.key),
    ...GAZE_CHANNELS,
];

const BLINK_FILL_GAP_MS = 140;
const GAZE_FILL_GAP_MS = 140;
const GAZE_PATH_BREAK_MS = 180;
const GAZE_DEDUPE_MS = 32;
const BLINK_TRACK_MAX = 2.2;

const BLINK_LANES = [
    {
        key: "blink_l",
        label: "Blink L",
        shortLabel: "L",
        low: 0.16,
        high: 0.92,
        stroke: "#d84b4b",
        fill: "rgba(216, 75, 75, 0.24)",
    },
    {
        key: "blink_r",
        label: "Blink R",
        shortLabel: "R",
        low: 1.18,
        high: 1.94,
        stroke: "#8751c4",
        fill: "rgba(135, 81, 196, 0.22)",
    },
];

function defaultExpandedBands() {
    return {
        eeg_1: false,
        eeg_2: false,
    };
}

function defaultLegendSelection() {
    return {
        Raw: false,
        Clean: true,
        "Clean Clip": false,
    };
}

const state = {
    mode: "realtime",
    userId: 99,
    pixelsPerSecond: 220,
    sampleRateHz: 250,
    realtimeBufferSeconds: 12,
    rows: [],
    historyData: null,
    socket: null,
    source: "-",
    socketStatus: "offline",
    rangeStartMs: null,
    rangeEndMs: null,
    viewStartMs: null,
    viewEndMs: null,
    autoFollow: true,
    bounds: null,
    expandedBands: defaultExpandedBands(),
    legendSelection: defaultLegendSelection(),
    yAxisLock: { eeg_1: null, eeg_2: null },
    timelineRenderQueued: false,
    gazeRenderQueued: false,
};

let _gazeSourceCache = { key: null, data: null };
let _blinkTrackCache = { key: null, data: null };
let _gazeDebounceTimer = null;
let _resizeDebounceTimer = null;
let _autoDetectTimer = null;

const chartHost = document.getElementById("chartScroll");
const chartEl = document.getElementById("chart");
const gazeChartEl = document.getElementById("gazeChart");
const chart = echarts.init(chartEl, null, { renderer: "canvas" });
const gazeChart = echarts.init(gazeChartEl, null, { renderer: "canvas" });

const socketDot = document.getElementById("socketDot");
const socketText = document.getElementById("socketText");
const sampleCount = document.getElementById("sampleCount");
const windowLabel = document.getElementById("windowLabel");
const lastPointTime = document.getElementById("lastPointTime");
const sourceLabel = document.getElementById("sourceLabel");
const rangeMeta = document.getElementById("rangeMeta");
const gazeMeta = document.getElementById("gazeMeta");
const terminalMeta = document.getElementById("terminalMeta");
const terminalContainer = document.getElementById("terminalContainer");
const gazeSection = document.getElementById("gazeSection");
const diagnosisSection = document.getElementById("diagnosisSection");
const terminalSection = document.getElementById("terminalSection");

const modeInput = document.getElementById("viewMode");
const userIdInput = document.getElementById("userId");
const historyStartInput = document.getElementById("historyStart");
const historyEndInput = document.getElementById("historyEnd");
const historyStartPickerBtn = document.getElementById("historyStartPicker");
const historyEndPickerBtn = document.getElementById("historyEndPicker");
const pixelsPerSecondInput = document.getElementById("pixelsPerSecond");
const loadBtn = document.getElementById("loadBtn");
const clearBtn = document.getElementById("clearBtn");
const autoDetectInput = document.getElementById("autoDetect");

function pad(value, width = 2) {
    return String(value).padStart(width, "0");
}

function toDateTimeLocalValue(date) {
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function fromDateTimeLocalValue(value) {
    if (!value) {
        return null;
    }

    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatClock(timeMs) {
    if (!timeMs && timeMs !== 0) {
        return "-";
    }

    const date = new Date(timeMs);
    return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}.${pad(date.getMilliseconds(), 3)}`;
}

function formatAxisLabel(value) {
    const date = new Date(value);
    return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function formatRange(durationMs) {
    if (!durationMs || durationMs <= 0) {
        return "-";
    }

    const totalSeconds = Math.round(durationMs / 1000);
    if (totalSeconds < 60) {
        return `${totalSeconds}s`;
    }

    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    if (minutes < 60) {
        return `${minutes}m ${seconds}s`;
    }

    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
}

function finiteNumber(value) {
    if (value == null || value === "") {
        return null;
    }

    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue : null;
}

function hasNumericValue(values) {
    return Array.isArray(values) && values.some((value) => Number.isFinite(value));
}

function lowerBound(values, target) {
    let low = 0;
    let high = values.length;

    while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (values[middle] < target) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }

    return low;
}

function upperBound(values, target) {
    let low = 0;
    let high = values.length;

    while (low < high) {
        const middle = Math.floor((low + high) / 2);
        if (values[middle] <= target) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }

    return low;
}

function setSocketStatus(status, label) {
    state.socketStatus = status;
    socketDot.classList.remove("online", "offline");
    if (status === "online") {
        socketDot.classList.add("online");
    } else {
        socketDot.classList.add("offline");
    }
    socketText.textContent = label;
}

function updateModeUI() {
    const isRealtime = state.mode === "realtime";
    loadBtn.textContent = isRealtime ? "Start Realtime" : "Load History";

    if (isRealtime) {
        setSocketStatus(state.socket ? "online" : "offline", state.socket ? "Live" : "Offline");
    } else {
        setSocketStatus("offline", "History");
    }

    if (gazeSection) gazeSection.style.display = isRealtime ? "none" : "";
    if (diagnosisSection) diagnosisSection.style.display = isRealtime ? "none" : "";
    if (terminalSection) terminalSection.style.display = isRealtime ? "" : "none";

    if (!isRealtime) {
        stopAutoDetect();
    } else if (autoDetectInput.checked) {
        startAutoDetect();
    }
}

function fmtVal(value, decimals = 4, width = 10) {
    if (value == null || value === "") return { text: "null".padStart(width), cls: "t-null" };
    const n = Number(value);
    if (!Number.isFinite(n)) return { text: "null".padStart(width), cls: "t-null" };
    if (Math.abs(n) >= 374.5) return { text: n.toFixed(decimals).padStart(width), cls: "t-clipped" };
    return { text: n.toFixed(decimals).padStart(width), cls: "" };
}

function appendTerminalRows(rows) {
    if (!terminalContainer || state.mode !== "realtime") return;

    const frag = document.createDocumentFragment();

    const batchLine = document.createElement("div");
    batchLine.className = "terminal-line terminal-batch";
    const now = new Date();
    const h = String(now.getHours()).padStart(2, "0");
    const m = String(now.getMinutes()).padStart(2, "0");
    const s = String(now.getSeconds()).padStart(2, "0");
    const ms = String(now.getMilliseconds()).padStart(3, "0");
    batchLine.textContent = `[${h}:${m}:${s}.${ms}] +${rows.length} rows`;
    frag.appendChild(batchLine);

    for (const row of rows) {
        const line = document.createElement("div");
        line.className = "terminal-line terminal-row";

        const t = row.time ? new Date(row.time) : null;
        const timeStr = t
            ? `  ${String(t.getUTCHours()).padStart(2,"0")}:${String(t.getUTCMinutes()).padStart(2,"0")}:${String(t.getUTCSeconds()).padStart(2,"0")}.${String(t.getUTCMilliseconds()).padStart(3,"0")}`
            : "  --------.---";

        const eeg1 = fmtVal(row.eeg_1, 4, 10);
        const eeg2 = fmtVal(row.eeg_2, 4, 10);
        const emg1 = fmtVal(row.emg_1, 4, 10);
        const emg2 = fmtVal(row.emg_2, 4, 10);
        const bl = fmtVal(row.blink_l, 2, 8);
        const br = fmtVal(row.blink_r, 2, 8);
        const gx = fmtVal(row.gaze_x, 4, 10);
        const gy = fmtVal(row.gaze_y, 4, 10);
        const gz = fmtVal(row.gaze_z, 4, 10);
        const label = row.event_label || "";

        const fields = [
            { text: timeStr, cls: "t-time" },
            eeg1, eeg2, emg1, emg2,
            bl, br, gx, gy, gz,
        ];
        if (label) fields.push({ text: label, cls: "t-label" });

        let html = "";
        for (let i = 0; i < fields.length; i++) {
            const f = fields[i];
            const pad = i === 0 ? 28 : (i <= 4 ? 10 : (i <= 6 ? 8 : 10));
            const txt = i === 0 ? f.text.padEnd(pad) : f.text.padStart(pad);
            if (f.cls) {
                html += `<span class="${f.cls}">${txt}</span>`;
            } else {
                html += txt;
            }
        }
        line.innerHTML = html;
        frag.appendChild(line);
    }

    terminalContainer.appendChild(frag);

    const MAX_LINES = 800;
    while (terminalContainer.children.length > MAX_LINES + 1) {
        terminalContainer.removeChild(terminalContainer.children[1]);
    }

    terminalContainer.scrollTop = terminalContainer.scrollHeight;

    if (terminalMeta) {
        terminalMeta.textContent = `${state.rows.length} samples | user ${state.userId}`;
    }
}

function clearTerminal() {
    if (!terminalContainer) return;
    while (terminalContainer.children.length > 1) {
        terminalContainer.removeChild(terminalContainer.lastChild);
    }
    if (terminalMeta) terminalMeta.textContent = "Waiting for data";
}

function currentPointCount() {
    if (state.mode === "history") {
        return state.historyData?.timeAxisMs.length ?? 0;
    }
    return state.rows.length;
}

function currentLastPointMs() {
    if (state.mode === "history") {
        const timeAxisMs = state.historyData?.timeAxisMs ?? [];
        return timeAxisMs[timeAxisMs.length - 1] ?? null;
    }
    return state.rows[state.rows.length - 1]?.timeMs ?? null;
}

function updateMetrics() {
    const durationMs = state.rangeStartMs != null && state.rangeEndMs != null
        ? state.rangeEndMs - state.rangeStartMs
        : 0;
    const visibleDurationMs = state.viewStartMs != null && state.viewEndMs != null
        ? state.viewEndMs - state.viewStartMs
        : 0;

    sampleCount.textContent = String(currentPointCount());
    windowLabel.textContent = formatRange(durationMs);
    sourceLabel.textContent = state.source;
    lastPointTime.textContent = formatClock(currentLastPointMs());

    const modeLabel = state.mode === "realtime" ? "Realtime" : "History";
    const visibleStart = state.viewStartMs != null ? formatClock(state.viewStartMs) : "-";
    const visibleEnd = state.viewEndMs != null ? formatClock(state.viewEndMs) : "-";
    rangeMeta.textContent = `${modeLabel} | View ${visibleStart} -> ${visibleEnd} | ${formatRange(visibleDurationMs)} | ${state.pixelsPerSecond}px/s`;
}

function getRealtimeMaxRows() {
    return Math.max(1, Math.ceil(state.realtimeBufferSeconds * state.sampleRateHz));
}

function plotWidthPixels() {
    const hostWidth = chartHost.clientWidth || chart.getWidth() || 1200;
    return Math.max(460, hostWidth - 132);
}

function visibleDurationMs() {
    return Math.max(1000, Math.round((plotWidthPixels() / state.pixelsPerSecond) * 1000));
}

function normalizeRows(rows) {
    return (rows || [])
        .map((row) => ({ ...row, timeMs: new Date(row.time).getTime() }))
        .filter((row) => Number.isFinite(row.timeMs))
        .sort((left, right) => left.timeMs - right.timeMs);
}

function normalizeNumericArray(values, expectedLength) {
    const input = Array.isArray(values) ? values : [];
    const normalized = new Array(expectedLength);

    for (let index = 0; index < expectedLength; index += 1) {
        normalized[index] = finiteNumber(input[index]);
    }

    return normalized;
}

function zipSeries(timeAxisMs, values) {
    const series = new Array(timeAxisMs.length);
    for (let index = 0; index < timeAxisMs.length; index += 1) {
        series[index] = [timeAxisMs[index], values[index]];
    }
    return series;
}

function normalizeHistoryPayload(payload) {
    const timeAxisMs = Array.isArray(payload?.time_axis_ms)
        ? payload.time_axis_ms.map((value) => Number(value)).filter((value) => Number.isFinite(value))
        : [];

    const signals = {};
    for (const channelKey of HISTORY_SIGNAL_KEYS) {
        const channelPayload = payload?.signals?.[channelKey] || {};
        const mainPayload = Array.isArray(channelPayload.main)
            ? { clean: channelPayload.main, raw: [] }
            : (channelPayload.main || {});
        const rawValues = normalizeNumericArray(mainPayload.raw, timeAxisMs.length);
        const cleanValues = normalizeNumericArray(mainPayload.clean, timeAxisMs.length);
        const signalState = {
            rawValues,
            rawSeries: zipSeries(timeAxisMs, rawValues),
            cleanValues,
            cleanSeries: zipSeries(timeAxisMs, cleanValues),
            bands: {},
        };

        if (channelKey === "eeg_1" || channelKey === "eeg_2") {
            for (const band of SUB_BANDS) {
                const bandValues = normalizeNumericArray(channelPayload?.bands?.[band.key], timeAxisMs.length);
                signalState.bands[band.key] = {
                    values: bandValues,
                    series: zipSeries(timeAxisMs, bandValues),
                };
            }
        }

        signals[channelKey] = signalState;
    }

    return {
        sampleRateHz: Number(payload?.sample_rate_hz) || state.sampleRateHz,
        timeAxisMs,
        signals,
    };
}

function applyViewWindow(preferredStartMs = null, anchorToEnd = false) {
    if (state.rangeStartMs == null || state.rangeEndMs == null) {
        state.viewStartMs = null;
        state.viewEndMs = null;
        return;
    }

    const totalDuration = Math.max(1, state.rangeEndMs - state.rangeStartMs);
    const windowDuration = Math.min(totalDuration, visibleDurationMs());

    if (totalDuration <= windowDuration) {
        state.viewStartMs = state.rangeStartMs;
        state.viewEndMs = state.rangeEndMs;
        return;
    }

    let start = anchorToEnd
        ? state.rangeEndMs - windowDuration
        : preferredStartMs ?? state.rangeStartMs;

    start = Math.max(state.rangeStartMs, Math.min(start, state.rangeEndMs - windowDuration));
    state.viewStartMs = start;
    state.viewEndMs = start + windowDuration;
}

function resetHistoryState() {
    state.historyData = null;
    state.expandedBands = defaultExpandedBands();
    state.legendSelection = defaultLegendSelection();
    state.yAxisLock = { eeg_1: null, eeg_2: null };
}

function scheduleTimelineRender() {
    if (state.timelineRenderQueued) {
        return;
    }

    state.timelineRenderQueued = true;
    window.requestAnimationFrame(() => {
        state.timelineRenderQueued = false;
        renderChart();
    });
}

function scheduleGazeRender() {
    if (state.gazeRenderQueued) {
        return;
    }

    state.gazeRenderQueued = true;
    window.requestAnimationFrame(() => {
        state.gazeRenderQueued = false;
        renderGazeChart();
    });
}

function scheduleFullRender() {
    scheduleTimelineRender();
    scheduleGazeRender();
}

function commitRealtimeRows(rows) {
    resetHistoryState();
    state.rows = normalizeRows(rows);
    updateMetrics();
    scheduleFullRender();
}

function commitHistoryPayload(payload) {
    state.rows = [];
    state.historyData = normalizeHistoryPayload(payload);
    state.sampleRateHz = state.historyData.sampleRateHz;
    updateMetrics();
    scheduleFullRender();
}

function appendRealtimeRows(rows) {
    const normalized = normalizeRows(rows);
    for (const row of normalized) {
        const last = state.rows[state.rows.length - 1];
        if (last && row.timeMs < last.timeMs) {
            continue;
        }
        if (last && row.timeMs === last.timeMs) {
            state.rows[state.rows.length - 1] = row;
        } else {
            state.rows.push(row);
        }
    }

    const overflow = state.rows.length - getRealtimeMaxRows();
    if (overflow > 0) {
        state.rows.splice(0, overflow);
    }

    if (state.rows.length > 0) {
        state.rangeEndMs = state.rows[state.rows.length - 1].timeMs;
        state.rangeStartMs = state.rangeEndMs - state.realtimeBufferSeconds * 1000;
        if (state.autoFollow || state.viewStartMs == null) {
            applyViewWindow(null, true);
        }
    }

    updateMetrics();
    scheduleFullRender();
}

function closeSocket() {
    if (!state.socket) {
        return;
    }

    const socket = state.socket;
    state.socket = null;
    socket.onopen = null;
    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;
    socket.close();
}

function socketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/ws/eeg?user_id=${state.userId}`;
}

function connectSocket() {
    closeSocket();
    setSocketStatus("offline", "Connecting");

    const ws = new WebSocket(socketUrl());
    state.socket = ws;

    ws.onopen = () => {
        if (state.mode !== "realtime" || state.socket !== ws) {
            ws.close();
            return;
        }
        setSocketStatus("online", "Live");
    };

    ws.onmessage = (event) => {
        if (state.mode !== "realtime") {
            return;
        }

        const payload = JSON.parse(event.data);
        if (payload.type === "eeg_rows") {
            state.source = "websocket";
            const rows = payload.rows || [];
            appendRealtimeRows(rows);
            appendTerminalRows(rows);
        }
    };

    ws.onerror = () => {
        if (state.mode === "realtime") {
            setSocketStatus("offline", "Error");
        }
    };

    ws.onclose = () => {
        if (state.socket === ws) {
            state.socket = null;
        }
        if (state.mode !== "realtime") {
            return;
        }
        setSocketStatus("offline", "Disconnected");
        window.setTimeout(() => {
            if (state.mode === "realtime" && !state.socket) {
                connectSocket();
            }
        }, 1500);
    };
}

// ── Auto-detect active user ────────────────────────────────────────────────

function stopAutoDetect() {
    if (_autoDetectTimer) {
        clearInterval(_autoDetectTimer);
        _autoDetectTimer = null;
    }
}

function startAutoDetect() {
    stopAutoDetect();
    _autoDetectTimer = setInterval(pollActiveUsers, 3000);
    pollActiveUsers();
}

async function pollActiveUsers() {
    if (state.mode !== "realtime" || !autoDetectInput.checked) {
        stopAutoDetect();
        return;
    }

    try {
        const data = await fetchJson("/api/eeg/active_users");
        const activeIds = data.user_ids || [];
        if (activeIds.length === 0) return;

        const currentId = Number(userIdInput.value) || 0;
        if (!activeIds.includes(currentId)) {
            const newId = activeIds[0];
            userIdInput.value = String(newId);
            state.userId = newId;
            state.bounds = null;
            clearTerminal();
            state.rows = [];
            loadRealtime();
        } else if (!state.socket) {
            loadRealtime();
        }
    } catch (e) {
        console.warn("[AutoDetect] poll failed:", e.message);
    }
}

async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || `Request failed: ${response.status}`);
    }
    return response.json();
}

async function fetchBounds(userId) {
    const payload = await fetchJson(`/api/eeg/bounds?user_id=${userId}`);
    state.bounds = payload;
    return payload;
}

function computeHistoryLimit(startMs, endMs) {
    const durationSeconds = Math.max(1, Math.ceil((endMs - startMs) / 1000));
    const estimatedRows = Math.ceil(durationSeconds * state.sampleRateHz * 1.15);
    return Math.max(2000, Math.min(200000, estimatedRows));
}

function setHistoryInputFromIso(input, isoString) {
    if (!isoString) {
        input.value = "";
        return;
    }
    input.value = toDateTimeLocalValue(new Date(isoString));
}

function clearHistoryInputs() {
    historyStartInput.value = "";
    historyEndInput.value = "";
}

function openDateTimePicker(input) {
    if (!input) {
        return;
    }

    input.focus();
    if (typeof input.showPicker === "function") {
        input.showPicker();
        return;
    }
    input.click();
}

async function ensureHistoryRange(bounds) {
    const explicitStart = fromDateTimeLocalValue(historyStartInput.value);
    if (!bounds.earliest_time) {
        throw new Error("No data available for this user.");
    }
    if (!bounds.latest_time) {
        throw new Error("No latest sample found for this user.");
    }

    const explicitEnd = fromDateTimeLocalValue(historyEndInput.value);
    const startDate = explicitStart ?? new Date(bounds.earliest_time);
    const endDate = explicitEnd ?? new Date(bounds.latest_time);

    if (!explicitStart) {
        setHistoryInputFromIso(historyStartInput, bounds.earliest_time);
    }
    if (!explicitEnd) {
        setHistoryInputFromIso(historyEndInput, bounds.latest_time);
    }

    return { startDate, endDate };
}

async function loadHistory() {
    closeSocket();
    setSocketStatus("offline", "History");

    const bounds = await fetchBounds(state.userId);
    if (!bounds.earliest_time || !bounds.latest_time || !bounds.total_rows) {
        state.source = "database-history-filtered";
        state.rangeStartMs = null;
        state.rangeEndMs = null;
        state.viewStartMs = null;
        state.viewEndMs = null;
        resetHistoryState();
        commitHistoryPayload({
            sample_rate_hz: state.sampleRateHz,
            time_axis_ms: [],
            signals: {},
        });
        return;
    }

    const { startDate, endDate } = await ensureHistoryRange(bounds);
    if (startDate.getTime() > endDate.getTime()) {
        throw new Error("Start time must be earlier than End time.");
    }

    const earliestDate = new Date(bounds.earliest_time);
    const latestDate = new Date(bounds.latest_time);
    if (startDate.getTime() > latestDate.getTime()) {
        throw new Error("Start time is later than the latest database row.");
    }
    if (endDate.getTime() < earliestDate.getTime()) {
        throw new Error("End time is earlier than the earliest database row.");
    }

    const limit = computeHistoryLimit(startDate.getTime(), endDate.getTime());
    const params = new URLSearchParams({
        user_id: String(state.userId),
        start_time: startDate.toISOString(),
        end_time: endDate.toISOString(),
        limit: String(limit),
    });

    const payload = await fetchJson(`/api/eeg/history?${params.toString()}`);
    state.source = payload.source || "database-history-filtered";
    state.autoFollow = false;
    state.rangeStartMs = startDate.getTime();
    state.rangeEndMs = endDate.getTime();
    state.expandedBands = defaultExpandedBands();
    state.legendSelection = defaultLegendSelection();
    applyViewWindow(startDate.getTime(), false);
    commitHistoryPayload(payload);
}

async function loadRealtime() {
    const limit = getRealtimeMaxRows();
    const payload = await fetchJson(`/api/eeg/latest?user_id=${state.userId}&limit=${limit}`);
    state.source = payload.source || "database";
    resetHistoryState();
    state.rows = normalizeRows(payload.rows || []);

    const lastRow = state.rows[state.rows.length - 1];
    const endTime = lastRow ? lastRow.timeMs : Date.now();
    state.rangeEndMs = endTime;
    state.rangeStartMs = endTime - state.realtimeBufferSeconds * 1000;
    state.autoFollow = true;
    applyViewWindow(null, true);
    updateMetrics();
    scheduleFullRender();
    connectSocket();
}

function realtimeSeries(channelKey) {
    return state.rows.map((row) => [row.timeMs, finiteNumber(row[channelKey])]);
}

function historyHasBands(channelKey) {
    const signal = state.historyData?.signals?.[channelKey];
    if (!signal?.bands) {
        return false;
    }
    return SUB_BANDS.every((band) => Array.isArray(signal.bands[band.key]?.series));
}

function estimateStepMs(timeAxisMs, index) {
    const current = timeAxisMs[index];
    const next = timeAxisMs[index + 1];
    if (Number.isFinite(next) && next > current) {
        return next - current;
    }

    const previous = timeAxisMs[index - 1];
    if (Number.isFinite(previous) && current > previous) {
        return current - previous;
    }

    return Math.max(4, Math.round(1000 / Math.max(1, state.sampleRateHz)));
}

function forwardFillSeries(values, timeAxisMs, maxGapMs) {
    const filled = new Array(values.length).fill(null);
    let lastValue = null;
    let lastTimeMs = null;

    for (let index = 0; index < values.length; index += 1) {
        const numericValue = finiteNumber(values[index]);
        if (numericValue != null) {
            filled[index] = numericValue;
            lastValue = numericValue;
            lastTimeMs = timeAxisMs[index];
            continue;
        }

        if (lastValue == null || lastTimeMs == null) {
            continue;
        }

        if (timeAxisMs[index] - lastTimeMs <= maxGapMs) {
            filled[index] = lastValue;
        }
    }

    return filled;
}

function clampBlinkState(value) {
    if (value == null) {
        return null;
    }
    return value >= 0.5 ? 0 : 1;
}

function selectHistoryVisualValues(channelKey, fillGapMs) {
    if (!state.historyData) {
        return [];
    }

    const signal = state.historyData.signals[channelKey];
    if (!signal) {
        return new Array(state.historyData.timeAxisMs.length).fill(null);
    }

    if (hasNumericValue(signal.cleanValues)) {
        return signal.cleanValues;
    }

    return forwardFillSeries(signal.rawValues, state.historyData.timeAxisMs, fillGapMs);
}

function getBlinkTrackState() {
    const cacheKey = state.mode === "history"
        ? state.historyData
        : state.rows;
    if (_blinkTrackCache.key === cacheKey && cacheKey != null) {
        return _blinkTrackCache.data;
    }

    let result;
    if (state.mode === "history") {
        const timeAxisMs = state.historyData?.timeAxisMs ?? [];
        result = {
            timeAxisMs,
            blink_l: selectHistoryVisualValues("blink_l", BLINK_FILL_GAP_MS).map(clampBlinkState),
            blink_r: selectHistoryVisualValues("blink_r", BLINK_FILL_GAP_MS).map(clampBlinkState),
        };
    } else {
        const timeAxisMs = state.rows.map((row) => row.timeMs);
        result = {
            timeAxisMs,
            blink_l: forwardFillSeries(
                state.rows.map((row) => finiteNumber(row.blink_l)),
                timeAxisMs,
                BLINK_FILL_GAP_MS
            ).map(clampBlinkState),
            blink_r: forwardFillSeries(
                state.rows.map((row) => finiteNumber(row.blink_r)),
                timeAxisMs,
                BLINK_FILL_GAP_MS
            ).map(clampBlinkState),
        };
    }

    _blinkTrackCache = { key: cacheKey, data: result };
    return result;
}

function getGazeSource() {
    const cacheKey = state.mode === "history"
        ? state.historyData
        : state.rows;
    if (_gazeSourceCache.key === cacheKey && cacheKey != null) {
        return _gazeSourceCache.data;
    }

    let result;
    if (state.mode === "history") {
        const timeAxisMs = state.historyData?.timeAxisMs ?? [];
        result = {
            timeAxisMs,
            xValues: selectHistoryVisualValues("gaze_x", GAZE_FILL_GAP_MS),
            yValues: selectHistoryVisualValues("gaze_y", GAZE_FILL_GAP_MS),
            zValues: selectHistoryVisualValues("gaze_z", GAZE_FILL_GAP_MS),
        };
    } else {
        const timeAxisMs = state.rows.map((row) => row.timeMs);
        result = {
            timeAxisMs,
            xValues: forwardFillSeries(
                state.rows.map((row) => finiteNumber(row.gaze_x)),
                timeAxisMs,
                GAZE_FILL_GAP_MS
            ),
            yValues: forwardFillSeries(
                state.rows.map((row) => finiteNumber(row.gaze_y)),
                timeAxisMs,
                GAZE_FILL_GAP_MS
            ),
            zValues: forwardFillSeries(
                state.rows.map((row) => finiteNumber(row.gaze_z)),
                timeAxisMs,
                GAZE_FILL_GAP_MS
            ),
        };
    }

    _gazeSourceCache = { key: cacheKey, data: result };
    return result;
}

function currentViewRange() {
    return {
        startMs: state.viewStartMs ?? state.rangeStartMs,
        endMs: state.viewEndMs ?? state.rangeEndMs,
    };
}

function buildBlinkStepSeries(timeAxisMs, values, lane) {
    const series = new Array(timeAxisMs.length);
    const amplitude = lane.high - lane.low;

    for (let index = 0; index < timeAxisMs.length; index += 1) {
        const blinkState = values[index];
        const mappedValue = blinkState == null ? null : lane.low + amplitude * blinkState;
        series[index] = {
            value: [timeAxisMs[index], mappedValue],
            blinkState,
        };
    }

    return series;
}

function buildBlinkIntervals(timeAxisMs, values, lane) {
    const intervals = [];
    let activeStart = null;

    for (let index = 0; index < values.length; index += 1) {
        const isActive = values[index] === 1;
        if (isActive && activeStart == null) {
            activeStart = index;
        }

        const nextIsActive = index + 1 < values.length && values[index + 1] === 1;
        if (activeStart != null && !nextIsActive) {
            const startMs = timeAxisMs[activeStart];
            const endMs = index + 1 < timeAxisMs.length
                ? timeAxisMs[index + 1]
                : timeAxisMs[index] + estimateStepMs(timeAxisMs, index);
            intervals.push([startMs, lane.low - 0.08, endMs, lane.high + 0.08]);
            activeStart = null;
        }
    }

    return intervals;
}

function getPaneDefinitions() {
    const panes = [];

    if (state.mode === "history" && state.historyData) {
        for (const channel of MAIN_CHANNELS) {
            const signal = state.historyData.signals[channel.key];
            const canToggle = channel.toggleBands && historyHasBands(channel.key);
            panes.push({
                id: `${channel.key}:main`,
                channelKey: channel.key,
                label: channel.label,
                color: channel.color,
                rawData: signal?.rawSeries || [],
                cleanData: signal?.cleanSeries || [],
                kind: "main",
                canToggle,
                expanded: canToggle && state.expandedBands[channel.key],
            });

            if (canToggle && state.expandedBands[channel.key]) {
                for (const band of SUB_BANDS) {
                    panes.push({
                        id: `${channel.key}:${band.key}`,
                        channelKey: channel.key,
                        label: `${channel.label} ${band.label}`,
                        color: band.color,
                        data: signal?.bands?.[band.key]?.series || [],
                        kind: "sub",
                        canToggle: false,
                        expanded: false,
                    });
                }
            }
        }
    } else if (state.mode === "realtime") {
        for (const channel of MAIN_CHANNELS) {
            panes.push({
                id: `${channel.key}:main`,
                channelKey: channel.key,
                label: channel.label,
                color: channel.color,
                data: realtimeSeries(channel.key),
                kind: "main",
                canToggle: false,
                expanded: false,
            });
        }
    } else {
        for (const channel of MAIN_CHANNELS) {
            panes.push({
                id: `${channel.key}:main`,
                channelKey: channel.key,
                label: channel.label,
                color: channel.color,
                data: [],
                kind: "main",
                canToggle: false,
                expanded: false,
            });
        }
    }

    panes.push({
        id: "blink-track",
        channelKey: "blink_track",
        label: "Blink Event Track",
        color: BLINK_LANES[0].stroke,
        kind: "blink",
        blinkTrack: getBlinkTrackState(),
        canToggle: false,
        expanded: false,
    });

    return panes;
}

function paneOuterHeight(pane) {
    if (pane.kind === "sub") {
        return 104;
    }
    if (pane.kind === "blink") {
        return 82;
    }
    return 140;
}

function paneHeaderHeight(pane) {
    if (pane.kind === "sub") {
        return 28;
    }
    if (pane.kind === "blink") {
        return 24;
    }
    return 38;
}

function layoutPanes(panes, layout) {
    const positioned = [];
    let cursorTop = layout.topPadding;

    for (const pane of panes) {
        const outerHeight = paneOuterHeight(pane);
        const headerHeight = paneHeaderHeight(pane);
        const gridHeight = Math.max(pane.kind === "blink" ? 42 : 58, outerHeight - headerHeight - 12);

        positioned.push({
            ...pane,
            top: cursorTop,
            outerHeight,
            headerHeight,
            gridTop: cursorTop + headerHeight,
            gridHeight,
        });

        cursorTop += outerHeight + layout.gap;
    }

    return positioned;
}

function chartHeightForPanes(panes, layout) {
    if (!panes.length) {
        return 820;
    }

    const paneHeights = panes.reduce((total, pane) => total + paneOuterHeight(pane), 0);
    const gaps = layout.gap * Math.max(0, panes.length - 1);
    return Math.max(780, layout.topPadding + layout.bottomPadding + paneHeights + gaps);
}

function toggleSubBands(channelKey) {
    if (state.mode !== "history" || !historyHasBands(channelKey)) {
        return;
    }

    state.expandedBands[channelKey] = !state.expandedBands[channelKey];
    updateMetrics();
    scheduleTimelineRender();
}

function computeVisibleRange(channelKey) {
    const startMs = state.viewStartMs;
    const endMs = state.viewEndMs;
    if (!startMs || !endMs) {
        return null;
    }

    let min = Infinity;
    let max = -Infinity;

    if (state.mode === "history" && state.historyData) {
        const signal = state.historyData.signals[channelKey];
        const series = signal?.cleanSeries || signal?.rawSeries || [];
        for (const point of series) {
            const t = point[0];
            const v = point[1];
            if (t >= startMs && t <= endMs && Number.isFinite(v)) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
    } else if (state.mode === "realtime") {
        for (const row of state.rows) {
            const t = row.timeMs;
            const v = finiteNumber(row[channelKey]);
            if (t >= startMs && t <= endMs && Number.isFinite(v)) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
    }

    if (!Number.isFinite(min) || !Number.isFinite(max)) {
        return null;
    }

    if (min === max) {
        min -= 1;
        max += 1;
    }

    const padding = (max - min) * 0.08;
    return { min: min - padding, max: max + padding };
}

function toggleYAxisLock(channelKey) {
    if (state.yAxisLock[channelKey]) {
        state.yAxisLock[channelKey] = null;
    } else {
        state.yAxisLock[channelKey] = computeVisibleRange(channelKey);
    }
    scheduleTimelineRender();
}

function blinkLaneMarkerTop(pane, lane) {
    const laneCenter = (lane.low + lane.high) / 2;
    const ratio = 1 - (laneCenter / BLINK_TRACK_MAX);
    return pane.gridTop + pane.gridHeight * ratio - 8;
}

function buildPaneGraphics(panes, layout, paneWidth) {
    const graphics = [];

    for (const pane of panes) {
        graphics.push({
            type: "rect",
            left: layout.paneLeft,
            top: pane.top,
            shape: {
                width: paneWidth,
                height: pane.outerHeight,
                r: 14,
            },
            style: {
                fill: pane.kind === "blink"
                    ? "#fffafb"
                    : (pane.kind === "sub" ? "#f8fbff" : "#fbfdff"),
                stroke: pane.kind === "blink"
                    ? "#f0d9de"
                    : (pane.kind === "sub" ? "#dbe6f1" : "#d4dee8"),
                lineWidth: 1,
                shadowBlur: 0,
            },
            silent: true,
            z: 0,
        });

        graphics.push({
            type: "rect",
            left: layout.gridLeft + 12,
            top: pane.top + (pane.kind === "blink" ? 8 : 13),
            shape: {
                width: 10,
                height: 10,
                r: 5,
            },
            style: {
                fill: pane.color,
            },
            silent: true,
            z: 7,
        });

        graphics.push({
            type: "text",
            left: layout.gridLeft + 28,
            top: pane.top + (pane.kind === "blink" ? 5 : 8),
            style: {
                text: pane.label,
                fill: "#19232e",
                font: pane.kind === "sub"
                    ? '600 11px "IBM Plex Sans", "Segoe UI", sans-serif'
                    : '700 12px "IBM Plex Sans", "Segoe UI", sans-serif',
            },
            silent: true,
            z: 7,
        });

        if (pane.kind === "main" && (pane.channelKey === "eeg_1" || pane.channelKey === "eeg_2")) {
            const isLocked = Boolean(state.yAxisLock[pane.channelKey]);
            graphics.push({
                type: "group",
                left: layout.gridLeft + 112,
                top: pane.top + 6,
                cursor: "pointer",
                onclick: () => toggleYAxisLock(pane.channelKey),
                z: 8,
                children: [
                    {
                        type: "rect",
                        shape: {
                            x: 0,
                            y: 0,
                            width: 78,
                            height: 26,
                            r: 13,
                        },
                        style: {
                            fill: isLocked ? "#ffe8d6" : "#edf5ff",
                            stroke: isLocked ? "#e0a66b" : "#bfd5ef",
                            lineWidth: 1,
                        },
                    },
                    {
                        type: "text",
                        style: {
                            x: 39,
                            y: 13,
                            text: isLocked ? "Unlock" : "Lock Y",
                            fill: isLocked ? "#8a5100" : "#244d7d",
                            font: '600 11px "IBM Plex Sans", "Segoe UI", sans-serif',
                            textAlign: "center",
                            textVerticalAlign: "middle",
                        },
                    },
                ],
            });
        }

        if (pane.canToggle) {
            const bandsLeft = (pane.channelKey === "eeg_1" || pane.channelKey === "eeg_2")
                ? layout.gridLeft + 198
                : layout.gridLeft + 112;
            const toggleButtonText = pane.expanded ? "Hide Bands" : "Show Bands";
            graphics.push({
                type: "group",
                left: bandsLeft,
                top: pane.top + 6,
                cursor: "pointer",
                onclick: () => toggleSubBands(pane.channelKey),
                z: 8,
                children: [
                    {
                        type: "rect",
                        shape: {
                            x: 0,
                            y: 0,
                            width: 104,
                            height: 26,
                            r: 13,
                        },
                        style: {
                            fill: pane.expanded ? "#dfeeff" : "#edf5ff",
                            stroke: pane.expanded ? "#8fb6e6" : "#bfd5ef",
                            lineWidth: 1,
                        },
                    },
                    {
                        type: "text",
                        style: {
                            x: 52,
                            y: 13,
                            text: toggleButtonText,
                            fill: "#244d7d",
                            font: '600 11px "IBM Plex Sans", "Segoe UI", sans-serif',
                            textAlign: "center",
                            textVerticalAlign: "middle",
                        },
                    },
                ],
            });
        }

        if (pane.kind === "blink") {
            for (const lane of BLINK_LANES) {
                const top = blinkLaneMarkerTop(pane, lane);
                graphics.push({
                    type: "rect",
                    left: layout.gridLeft - 36,
                    top,
                    shape: {
                        width: 18,
                        height: 18,
                        r: 9,
                    },
                    style: {
                        fill: lane.fill,
                        stroke: lane.stroke,
                        lineWidth: 1,
                    },
                    silent: true,
                    z: 8,
                });
                graphics.push({
                    type: "text",
                    left: layout.gridLeft - 27,
                    top: top + 3,
                    style: {
                        text: lane.shortLabel,
                        fill: lane.stroke,
                        font: '700 11px "IBM Plex Sans", "Segoe UI", sans-serif',
                        textAlign: "center",
                    },
                    silent: true,
                    z: 9,
                });
            }
        }
    }

    return graphics;
}

function renderBlinkInterval(params, api) {
    const start = api.coord([api.value(0), api.value(1)]);
    const end = api.coord([api.value(2), api.value(3)]);
    const rect = echarts.graphic.clipRectByRect(
        {
            x: start[0],
            y: end[1],
            width: Math.max(end[0] - start[0], 2),
            height: Math.max(start[1] - end[1], 6),
        },
        {
            x: params.coordSys.x,
            y: params.coordSys.y,
            width: params.coordSys.width,
            height: params.coordSys.height,
        }
    );

    if (!rect) {
        return null;
    }

    return {
        type: "rect",
        shape: rect,
        style: api.style(),
    };
}

function formatTimelineTooltip(params) {
    const items = Array.isArray(params) ? params : [params];
    if (!items.length) {
        return "";
    }

    const lines = [`<div style="margin-bottom:6px;font-weight:700;">${formatClock(items[0].axisValue)}</div>`];
    for (const item of items) {
        if (item.seriesType === "custom") {
            continue;
        }

        let displayValue = "-";
        if (item.data && typeof item.data === "object" && "blinkState" in item.data) {
            displayValue = item.data.blinkState === 1 ? "Closed" : "Open";
        } else {
            const numericValue = Array.isArray(item.value) ? item.value[1] : item.value;
            if (Number.isFinite(numericValue)) {
                displayValue = Number(numericValue).toFixed(3);
            }
        }

        lines.push(`${item.marker}${item.seriesName}: ${displayValue}`);
    }

    return lines.join("<br>");
}

function renderChart() {
    const panes = getPaneDefinitions();
    const showHistoryLegend = state.mode === "history" && Boolean(state.historyData);
    const layout = {
        topPadding: showHistoryLegend ? 52 : 28,
        bottomPadding: 88,
        gap: 20,
        paneLeft: 72,
        paneRight: 28,
        gridLeft: 118,
        gridRight: 34,
    };

    // realtime模式下，如果没有数据则跳过渲染
    if (state.mode === "realtime" && state.rows.length === 0) {
        return;
    }

    chartEl.style.height = `${chartHeightForPanes(panes, layout)}px`;
    chart.resize();

    const positionedPanes = layoutPanes(panes, layout);
    const chartWidth = chart.getWidth() || chartHost.clientWidth || 1200;
    const paneWidth = Math.max(220, chartWidth - layout.paneLeft - layout.paneRight);
    const now = Date.now();
    const axisMin = state.rangeStartMs ?? (state.rows[0]?.timeMs ?? now - 1000);
    const axisMax = state.rangeEndMs ?? (state.rows[state.rows.length - 1]?.timeMs ?? now);
    // 确保axisMin < axisMax，避免零宽度窗口
    const safeAxisMin = axisMin >= axisMax ? axisMax - 1000 : axisMin;
    const safeAxisMax = axisMax <= axisMin ? axisMin + 1000 : axisMax;
    const zoomStartValue = state.viewStartMs ?? safeAxisMin;
    const zoomEndValue = state.viewEndMs ?? safeAxisMax;

    const historySampling = state.mode === "history" && currentPointCount() > 6000 ? "lttb" : "none";
    const series = positionedPanes.flatMap((pane, index) => {
        if (pane.kind === "main" && state.mode === "history" && state.historyData) {
            return [
                {
                    name: "Raw",
                    type: "line",
                    xAxisIndex: index,
                    yAxisIndex: index,
                    showSymbol: false,
                    smooth: false,
                    connectNulls: false,
                    sampling: historySampling,
                    progressive: 8000,
                    progressiveThreshold: 12000,
                    clip: true,
                    data: pane.rawData,
                    lineStyle: {
                        color: "rgba(110, 120, 132, 0.9)",
                        width: 1,
                        opacity: 1,
                    },
                    z: 2,
                },
                {
                    name: "Clean",
                    type: "line",
                    xAxisIndex: index,
                    yAxisIndex: index,
                    showSymbol: false,
                    smooth: false,
                    connectNulls: false,
                    sampling: historySampling,
                    progressive: 8000,
                    progressiveThreshold: 12000,
                    clip: true,
                    data: pane.cleanData,
                    lineStyle: {
                        color: pane.color,
                        width: 2.1,
                        opacity: 1,
                    },
                    z: 4,
                },
                {
                    name: "Clean Clip",
                    type: "line",
                    xAxisIndex: index,
                    yAxisIndex: index,
                    showSymbol: false,
                    smooth: false,
                    connectNulls: false,
                    sampling: historySampling,
                    progressive: 8000,
                    progressiveThreshold: 12000,
                    clip: true,
                    data: pane.cleanData.map(([t, v]) => [t, (v > 2 || v < -2) ? 0 : v]),
                    lineStyle: {
                        color: "#4c1d95",
                        width: 1.8,
                        opacity: 1,
                    },
                    z: 3,
                },
            ];
        }

        if (pane.kind === "blink") {
            return BLINK_LANES.flatMap((lane) => {
                const blinkValues = pane.blinkTrack?.[lane.key] || [];
                return [
                    {
                        name: lane.label,
                        type: "custom",
                        xAxisIndex: index,
                        yAxisIndex: index,
                        renderItem: renderBlinkInterval,
                        silent: true,
                        data: buildBlinkIntervals(pane.blinkTrack.timeAxisMs, blinkValues, lane),
                        itemStyle: {
                            color: lane.fill,
                            borderColor: lane.stroke,
                            borderWidth: 1,
                        },
                        z: 2,
                    },
                    {
                        name: lane.label,
                        type: "line",
                        xAxisIndex: index,
                        yAxisIndex: index,
                        showSymbol: false,
                        smooth: false,
                        step: "end",
                        connectNulls: true,
                        progressive: 4000,
                        progressiveThreshold: 8000,
                        clip: true,
                        data: buildBlinkStepSeries(pane.blinkTrack.timeAxisMs, blinkValues, lane),
                        lineStyle: {
                            color: lane.stroke,
                            width: 1.5,
                        },
                        z: 5,
                    },
                ];
            });
        }

        return [
            {
                name: pane.label,
                type: "line",
                xAxisIndex: index,
                yAxisIndex: index,
                showSymbol: false,
                smooth: false,
                connectNulls: false,
                sampling: historySampling,
                progressive: 8000,
                progressiveThreshold: 12000,
                clip: true,
                data: pane.data,
                lineStyle: {
                    color: pane.color,
                    width: pane.kind === "sub" ? 1.7 : 2.1,
                    opacity: 1,
                },
                z: 4,
            },
        ];
    });

    chart.clear();
    chart.setOption(
        {
            animation: false,
            backgroundColor: "#f6f9fd",
            graphic: buildPaneGraphics(positionedPanes, layout, paneWidth),
            legend: {
                show: showHistoryLegend,
                data: ["Raw", "Clean", "Clean Clip"],
                selected: state.legendSelection,
                top: 12,
                right: 20,
                itemWidth: 16,
                itemHeight: 10,
                itemGap: 16,
                icon: "roundRect",
                backgroundColor: "rgba(255,255,255,0.9)",
                borderColor: "#d8e2ec",
                borderWidth: 1,
                padding: [8, 12, 8, 12],
                textStyle: {
                    color: "#334456",
                    fontSize: 12,
                    fontWeight: 600,
                },
            },
            axisPointer: {
                link: [{ xAxisIndex: "all" }],
            },
            grid: positionedPanes.map((pane) => ({
                left: layout.gridLeft,
                right: layout.gridRight,
                top: pane.gridTop,
                height: pane.gridHeight,
            })),
            tooltip: {
                trigger: "axis",
                axisPointer: {
                    animation: false,
                    lineStyle: { color: "#5f7896", width: 1.2 },
                },
                backgroundColor: "rgba(255,255,255,0.96)",
                borderColor: "#c9d6e4",
                textStyle: { color: "#16202b" },
                formatter: formatTimelineTooltip,
            },
            xAxis: positionedPanes.map((pane, index) => ({
                type: "time",
                min: safeAxisMin,
                max: safeAxisMax,
                gridIndex: index,
                axisLine: { lineStyle: { color: "#9fb2c6", width: 1 } },
                axisTick: { show: false },
                axisLabel: {
                    color: "#425262",
                    show: index === positionedPanes.length - 1,
                    formatter: formatAxisLabel,
                },
                splitLine: { lineStyle: { color: "#d8e4ef", width: 1 } },
            })),
            yAxis: positionedPanes.map((pane, index) => {
                if (pane.kind === "blink") {
                    return {
                        type: "value",
                        gridIndex: index,
                        min: 0,
                        max: BLINK_TRACK_MAX,
                        axisLine: { show: false },
                        axisTick: { show: false },
                        axisLabel: { show: false },
                        splitLine: { show: false },
                    };
                }

                const locked = state.yAxisLock[pane.channelKey];
                const yAxisConfig = {
                    type: "value",
                    gridIndex: index,
                    name: "",
                    axisLine: { show: false },
                    axisTick: { show: false },
                    axisLabel: {
                        color: "#425262",
                        margin: 14,
                    },
                    splitLine: { lineStyle: { color: "#dde6ef", width: 1 } },
                    scale: true,
                };
                if (locked) {
                    yAxisConfig.min = locked.min;
                    yAxisConfig.max = locked.max;
                }
                return yAxisConfig;
            }),
            dataZoom: [
                {
                    type: "slider",
                    xAxisIndex: positionedPanes.map((_, index) => index),
                    filterMode: "filter",
                    showDataShadow: false,
                    brushSelect: false,
                    bottom: 16,
                    height: 28,
                    borderColor: "#c7d4e2",
                    backgroundColor: "#eef3f8",
                    fillerColor: "rgba(32, 72, 179, 0.16)",
                    handleStyle: {
                        color: "#6d8fb5",
                        borderColor: "#5d7ea4",
                    },
                    moveHandleStyle: {
                        color: "#7f9ec1",
                    },
                    textStyle: {
                        color: "#4f6071",
                    },
                    labelFormatter: (value) => formatAxisLabel(value),
                    startValue: zoomStartValue,
                    endValue: zoomEndValue,
                },
                {
                    type: "inside",
                    xAxisIndex: positionedPanes.map((_, index) => index),
                    filterMode: "filter",
                    moveOnMouseMove: false,
                    moveOnMouseWheel: true,
                    zoomOnMouseWheel: false,
                },
            ],
            series,
        },
        {
            notMerge: true,
        }
    );

    // setOption之后再resize，确保画布尺寸正确
    chart.resize();
}

function interpolateRgb(start, end, t) {
    return start.map((value, index) => Math.round(value + (end[index] - value) * t));
}

function rgba(color, alpha) {
    return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha.toFixed(3)})`;
}

function computeExtent(values) {
    if (!values.length) {
        return [-1, 1];
    }

    let min = values[0];
    let max = values[0];
    for (const value of values) {
        if (value < min) {
            min = value;
        }
        if (value > max) {
            max = value;
        }
    }

    if (min === max) {
        const padding = Math.max(1, Math.abs(min) * 0.08 || 1);
        return [min - padding, max + padding];
    }

    const padding = (max - min) * 0.08;
    return [min - padding, max + padding];
}

function extractVisibleGazePoints() {
    const { timeAxisMs, xValues, yValues, zValues } = getGazeSource();
    const { startMs, endMs } = currentViewRange();
    if (!timeAxisMs.length || startMs == null || endMs == null) {
        return [];
    }

    const startIndex = lowerBound(timeAxisMs, startMs);
    const endIndex = upperBound(timeAxisMs, endMs);
    const points = [];
    let lastAccepted = null;

    for (let index = startIndex; index < endIndex; index += 1) {
        const timeMs = timeAxisMs[index];
        const x = finiteNumber(xValues[index]);
        const y = finiteNumber(yValues[index]);
        if (x == null || y == null) {
            continue;
        }

        const z = finiteNumber(zValues[index]);
        const lastZ = lastAccepted?.z ?? null;
        const sameCoordinates = Boolean(
            lastAccepted
            && Math.abs(x - lastAccepted.x) < 1e-6
            && Math.abs(y - lastAccepted.y) < 1e-6
            && ((z == null && lastZ == null) || (z != null && lastZ != null && Math.abs(z - lastZ) < 1e-6))
        );
        if (sameCoordinates && timeMs - lastAccepted.timeMs < GAZE_DEDUPE_MS) {
            continue;
        }

        const point = { timeMs, x, y, z };
        points.push(point);
        lastAccepted = point;
    }

    return points;
}

function buildGazeSegments(points) {
    const startColor = [138, 163, 189];
    const endColor = [15, 108, 173];

    return points.flatMap((point, index) => {
        if (index === 0) {
            return [];
        }

        const previous = points[index - 1];
        if (point.timeMs - previous.timeMs > GAZE_PATH_BREAK_MS) {
            return [];
        }

        const ratio = points.length <= 1 ? 1 : index / (points.length - 1);
        return [{
            coords: [
                [previous.x, previous.y],
                [point.x, point.y],
            ],
            lineStyle: {
                color: rgba(interpolateRgb(startColor, endColor, ratio), 0.24 + ratio * 0.58),
                width: 2.2,
            },
        }];
    });
}

function buildGazeScatter(points) {
    const startColor = [142, 166, 190];
    const endColor = [11, 98, 159];
    let maxAbsZ = 1;
    for (const point of points) {
        const absZ = Math.abs(point.z ?? 0);
        if (absZ > maxAbsZ) maxAbsZ = absZ;
    }

    return points.map((point, index) => {
        const ratio = points.length <= 1 ? 1 : index / (points.length - 1);
        const absZ = Math.abs(point.z ?? 0);
        return {
            value: [point.x, point.y, point.z ?? 0, point.timeMs],
            symbolSize: 8 + (absZ / maxAbsZ) * 18,
            itemStyle: {
                color: rgba(interpolateRgb(startColor, endColor, ratio), 0.28 + ratio * 0.62),
                borderColor: "#ffffff",
                borderWidth: 1,
            },
        };
    });
}

function buildEndpointMarkers(points) {
    if (!points.length) {
        return [];
    }

    const firstPoint = points[0];
    const lastPoint = points[points.length - 1];
    const markers = [
        {
            name: "Start",
            value: [firstPoint.x, firstPoint.y, firstPoint.z ?? 0, firstPoint.timeMs],
            symbol: "circle",
            symbolSize: 12,
            itemStyle: {
                color: "#9fb4c8",
                borderColor: "#ffffff",
                borderWidth: 1.5,
            },
            label: {
                show: true,
                formatter: "Start",
                position: "left",
                color: "#63788f",
                fontWeight: 700,
            },
        },
    ];

    if (points.length > 1) {
        markers.push({
            name: "Now",
            value: [lastPoint.x, lastPoint.y, lastPoint.z ?? 0, lastPoint.timeMs],
            symbol: "diamond",
            symbolSize: 14,
            itemStyle: {
                color: "#0f6cad",
                borderColor: "#ffffff",
                borderWidth: 1.5,
            },
            label: {
                show: true,
                formatter: "Now",
                position: "right",
                color: "#0f6cad",
                fontWeight: 700,
            },
        });
    }

    return markers;
}

function renderGazeEmptyState(message) {
    gazeMeta.textContent = message;
    gazeChart.resize();
    gazeChart.setOption(
        {
            animation: false,
            backgroundColor: "transparent",
            xAxis: {
                type: "value",
                show: false,
                min: -1,
                max: 1,
            },
            yAxis: {
                type: "value",
                show: false,
                min: -1,
                max: 1,
            },
            series: [],
            graphic: [
                {
                    type: "text",
                    left: "center",
                    top: "middle",
                    style: {
                        text: message,
                        fill: "#6b7d8e",
                        font: '600 16px "IBM Plex Sans", "Segoe UI", sans-serif',
                    },
                },
            ],
        },
        {
            notMerge: true,
            lazyUpdate: true,
        }
    );
}

function renderGazeChart() {
    const points = extractVisibleGazePoints();
    if (!points.length) {
        renderGazeEmptyState("No gaze samples inside the current timeline window.");
        return;
    }

    const segments = buildGazeSegments(points);
    const scatterData = buildGazeScatter(points);
    const endpoints = buildEndpointMarkers(points);
    const xExtent = computeExtent(points.map((point) => point.x));
    const yExtent = computeExtent(points.map((point) => point.y));
    const { startMs, endMs } = currentViewRange();

    gazeMeta.textContent = `${points.length} points | ${segments.length} vectors | ${formatClock(startMs)} -> ${formatClock(endMs)}`;

    gazeChart.resize();
    gazeChart.setOption(
        {
            animation: false,
            backgroundColor: "transparent",
            grid: {
                left: 68,
                right: 26,
                top: 30,
                bottom: 46,
            },
            tooltip: {
                trigger: "item",
                backgroundColor: "rgba(255,255,255,0.96)",
                borderColor: "#c9d6e4",
                textStyle: { color: "#16202b" },
                formatter: (param) => {
                    if (param.seriesName === "Trajectory") {
                        return "";
                    }

                    const value = param.value || [];
                    const x = finiteNumber(value[0]);
                    const y = finiteNumber(value[1]);
                    const z = finiteNumber(value[2]);
                    const timeMs = finiteNumber(value[3]);
                    const lines = [];
                    if (param.seriesName === "Slice Markers") {
                        lines.push(`<strong>${param.name}</strong>`);
                    }
                    if (x != null) {
                        lines.push(`gaze_x: ${x.toFixed(3)}`);
                    }
                    if (y != null) {
                        lines.push(`gaze_y: ${y.toFixed(3)}`);
                    }
                    if (z != null) {
                        lines.push(`gaze_z: ${z.toFixed(3)}`);
                    }
                    if (timeMs != null) {
                        lines.push(`time: ${formatClock(timeMs)}`);
                    }
                    return lines.join("<br>");
                },
            },
            xAxis: {
                type: "value",
                min: xExtent[0],
                max: xExtent[1],
                name: "gaze_x",
                nameLocation: "middle",
                nameGap: 28,
                axisLine: { lineStyle: { color: "#95aac0" } },
                axisLabel: { color: "#46596c" },
                splitLine: { lineStyle: { color: "#dbe5ef" } },
                scale: true,
            },
            yAxis: {
                type: "value",
                min: yExtent[0],
                max: yExtent[1],
                name: "gaze_y",
                nameLocation: "middle",
                nameGap: 48,
                axisLine: { lineStyle: { color: "#95aac0" } },
                axisLabel: { color: "#46596c" },
                splitLine: { lineStyle: { color: "#dbe5ef" } },
                scale: true,
            },
            series: [
                {
                    name: "Trajectory",
                    type: "lines",
                    coordinateSystem: "cartesian2d",
                    symbol: ["none", "arrow"],
                    symbolSize: 7,
                    data: segments,
                    lineStyle: {
                        width: 2.2,
                        opacity: 1,
                    },
                    silent: true,
                },
                {
                    name: "Gaze Points",
                    type: "scatter",
                    data: scatterData,
                },
                {
                    name: "Slice Markers",
                    type: "scatter",
                    data: endpoints,
                },
            ],
        },
        {
            notMerge: true,
            lazyUpdate: true,
        }
    );
}

chart.off("datazoom");
chart.on("datazoom", () => {
    const zoom = chart.getOption().dataZoom?.[0];
    if (!zoom) {
        return;
    }

    const startValue = Number(zoom.startValue ?? state.rangeStartMs);
    const endValue = Number(zoom.endValue ?? state.rangeEndMs);
    if (!Number.isFinite(startValue) || !Number.isFinite(endValue)) {
        return;
    }

    state.viewStartMs = startValue;
    state.viewEndMs = endValue;
    if (state.mode === "realtime" && state.rangeEndMs != null) {
        state.autoFollow = Math.abs(state.rangeEndMs - endValue) < 250;
    }
    updateMetrics();
    if (_gazeDebounceTimer) clearTimeout(_gazeDebounceTimer);
    _gazeDebounceTimer = setTimeout(() => {
        scheduleGazeRender();
    }, 120);
});

chart.off("legendselectchanged");
chart.on("legendselectchanged", (event) => {
    state.legendSelection = {
        ...state.legendSelection,
        ...event.selected,
    };
});

async function loadCurrentMode() {
    state.userId = Number(userIdInput.value) || 1;
    state.pixelsPerSecond = Math.max(40, Number(pixelsPerSecondInput.value) || 220);
    pixelsPerSecondInput.value = String(state.pixelsPerSecond);

    try {
        if (state.mode === "history") {
            await loadHistory();
        } else {
            await loadRealtime();
        }
    } catch (error) {
        console.error(error);
        rangeMeta.textContent = error.message || "Failed to load data.";
        gazeMeta.textContent = "View unavailable";
        if (state.mode === "history") {
            setSocketStatus("offline", "History error");
        } else {
            setSocketStatus("offline", "Load error");
        }
    }
}

modeInput.addEventListener("change", () => {
    state.mode = modeInput.value;
    if (state.mode === "history") {
        closeSocket();
        setSocketStatus("offline", "History");
    }
    updateModeUI();
    updateMetrics();
    scheduleFullRender();
});

userIdInput.addEventListener("change", () => {
    const nextUserId = Number(userIdInput.value) || 1;
    state.bounds = null;
    if (nextUserId !== state.userId) {
        clearHistoryInputs();
    }
});

pixelsPerSecondInput.addEventListener("change", () => {
    state.pixelsPerSecond = Math.max(40, Number(pixelsPerSecondInput.value) || 220);
    pixelsPerSecondInput.value = String(state.pixelsPerSecond);
    if (state.mode === "realtime") {
        applyViewWindow(null, state.autoFollow);
    } else {
        applyViewWindow(state.viewStartMs ?? state.rangeStartMs, false);
    }
    updateMetrics();
    scheduleFullRender();
});

historyStartPickerBtn.addEventListener("click", () => {
    openDateTimePicker(historyStartInput);
});

historyEndPickerBtn.addEventListener("click", () => {
    openDateTimePicker(historyEndInput);
});

loadBtn.addEventListener("click", loadCurrentMode);

autoDetectInput.addEventListener("change", () => {
    if (autoDetectInput.checked && state.mode === "realtime") {
        startAutoDetect();
    } else {
        stopAutoDetect();
    }
});

clearBtn.addEventListener("click", () => {
    closeSocket();
    stopAutoDetect();
    state.rows = [];
    resetHistoryState();
    clearTerminal();
    state.source = state.mode === "history" ? "database-history-filtered" : "-";
    state.bounds = null;

    if (state.mode === "realtime") {
        const endTime = Date.now();
        state.rangeEndMs = endTime;
        state.rangeStartMs = endTime - state.realtimeBufferSeconds * 1000;
        state.autoFollow = true;
        applyViewWindow(null, true);
    } else {
        clearHistoryInputs();
        state.rangeStartMs = null;
        state.rangeEndMs = null;
        state.viewStartMs = null;
        state.viewEndMs = null;
        setSocketStatus("offline", "History");
    }

    updateMetrics();
    scheduleFullRender();
});

window.addEventListener("resize", () => {
    if (_resizeDebounceTimer) clearTimeout(_resizeDebounceTimer);
    _resizeDebounceTimer = setTimeout(() => {
        if (state.mode === "realtime") {
            applyViewWindow(state.viewStartMs, state.autoFollow);
        } else {
            applyViewWindow(state.viewStartMs ?? state.rangeStartMs, false);
        }
        updateMetrics();
        scheduleFullRender();
    }, 150);
});

updateModeUI();
updateMetrics();
scheduleFullRender();
loadCurrentMode();

// ── AI Diagnosis ──────────────────────────────────────────────────────────
const diagnosisBtn = document.getElementById("diagnosisBtn");
const diagnosisContent = document.getElementById("diagnosisContent");
const diagnosisMeta = document.getElementById("diagnosisMeta");

async function loadDiagnosis() {
    const userId = Number(userIdInput.value) || 1;
    diagnosisBtn.disabled = true;
    diagnosisBtn.textContent = "正在分析...";
    diagnosisContent.innerHTML =
        '<div class="diagnosis-loading">正在加载模型并分析数据，请稍候...</div>';

    try {
        const resp = await fetch("/api/diagnosis", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: userId }),
        });
        if (!resp.ok) {
            const text = await resp.text();
            throw new Error(text);
        }
        const data = await resp.json();
        renderDiagnosisReport(data);
    } catch (e) {
        diagnosisContent.innerHTML =
            `<div class="diagnosis-error">诊断失败: ${e.message}</div>`;
    } finally {
        diagnosisBtn.disabled = false;
        diagnosisBtn.textContent = "生成 AI 诊断报告";
    }
}

function renderDiagnosisReport(data) {
    const riskClass = data.prediction === 0 ? "risk-high" : "risk-low";
    diagnosisMeta.textContent =
        `预测: ${data.label} | 概率: ${(data.probability * 100).toFixed(1)}%`;

    const featuresHtml = data.key_features
        .map(
            (f) =>
                `<div class="feature-item">
                    <span class="feature-name">${f.name}</span>
                    <span class="feature-value">${f.value}</span>
                </div>`
        )
        .join("");

    diagnosisContent.innerHTML = `
        <div class="diagnosis-result ${riskClass}">
            <div class="diagnosis-header">
                <span class="diagnosis-label">${data.label}</span>
                <span class="diagnosis-prob">置信度 ${(data.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="diagnosis-features">
                <h4>关键指标</h4>
                <div class="feature-grid">${featuresHtml}</div>
            </div>
            <div class="diagnosis-report">
                <h4>AI 分析报告</h4>
                <div class="report-text">${data.report}</div>
            </div>
        </div>`;
}

diagnosisBtn.addEventListener("click", loadDiagnosis);
