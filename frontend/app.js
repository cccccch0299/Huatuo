const CHANNELS = [
    { key: "eeg_1", label: "EEG 1", color: "#005ea8" },
    { key: "eeg_2", label: "EEG 2", color: "#2048b3" },
    { key: "emg_1", label: "EMG 1", color: "#087447" },
    { key: "emg_2", label: "EMG 2", color: "#b75a00" },
];

const state = {
    mode: "realtime",
    userId: 99,
    pixelsPerSecond: 220,
    sampleRateHz: 250,
    realtimeBufferSeconds: 12,
    rows: [],
    socket: null,
    source: "-",
    socketStatus: "offline",
    rangeStartMs: null,
    rangeEndMs: null,
    viewStartMs: null,
    viewEndMs: null,
    autoFollow: true,
    bounds: null,
};

const chartHost = document.getElementById("chartScroll");
const chart = echarts.init(document.getElementById("chart"), null, { renderer: "canvas" });

const socketDot = document.getElementById("socketDot");
const socketText = document.getElementById("socketText");
const sampleCount = document.getElementById("sampleCount");
const windowLabel = document.getElementById("windowLabel");
const lastPointTime = document.getElementById("lastPointTime");
const sourceLabel = document.getElementById("sourceLabel");
const rangeMeta = document.getElementById("rangeMeta");

const modeInput = document.getElementById("viewMode");
const userIdInput = document.getElementById("userId");
const historyStartInput = document.getElementById("historyStart");
const historyEndInput = document.getElementById("historyEnd");
const historyStartPickerBtn = document.getElementById("historyStartPicker");
const historyEndPickerBtn = document.getElementById("historyEndPicker");
const pixelsPerSecondInput = document.getElementById("pixelsPerSecond");
const loadBtn = document.getElementById("loadBtn");
const clearBtn = document.getElementById("clearBtn");

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
    if (!timeMs) {
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
}

function updateMetrics() {
    const durationMs = state.rangeStartMs != null && state.rangeEndMs != null
        ? state.rangeEndMs - state.rangeStartMs
        : 0;
    const visibleDurationMs = state.viewStartMs != null && state.viewEndMs != null
        ? state.viewEndMs - state.viewStartMs
        : 0;

    sampleCount.textContent = String(state.rows.length);
    windowLabel.textContent = formatRange(durationMs);
    sourceLabel.textContent = state.source;
    lastPointTime.textContent = formatClock(state.rows[state.rows.length - 1]?.timeMs);

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
    return Math.max(460, hostWidth - 120);
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

function commitRows(rows) {
    state.rows = normalizeRows(rows);
    updateMetrics();
    renderChart();
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
    renderChart();
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
            appendRealtimeRows(payload.rows || []);
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
        state.source = "database-history";
        state.rangeStartMs = null;
        state.rangeEndMs = null;
        state.viewStartMs = null;
        state.viewEndMs = null;
        commitRows([]);
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
    state.source = payload.source || "database-history";
    state.autoFollow = false;
    state.rangeStartMs = startDate.getTime();
    state.rangeEndMs = endDate.getTime();
    applyViewWindow(startDate.getTime(), false);
    commitRows(payload.rows || []);
}

async function loadRealtime() {
    const limit = getRealtimeMaxRows();
    const payload = await fetchJson(`/api/eeg/latest?user_id=${state.userId}&limit=${limit}`);
    state.source = payload.source || "database";
    state.rows = normalizeRows(payload.rows || []);

    const lastRow = state.rows[state.rows.length - 1];
    const endTime = lastRow ? lastRow.timeMs : Date.now();
    state.rangeEndMs = endTime;
    state.rangeStartMs = endTime - state.realtimeBufferSeconds * 1000;
    state.autoFollow = true;
    applyViewWindow(null, true);
    updateMetrics();
    renderChart();
    connectSocket();
}

function buildSeries(channelKey) {
    return state.rows.map((row) => [row.timeMs, row[channelKey]]);
}

function paneGraphics(paneHeight, topPadding, gap, paneLeft, paneRight) {
    const chartWidth = chart.getWidth() || chartHost.clientWidth || 1200;
    const paneWidth = Math.max(220, chartWidth - paneLeft - paneRight);

    return CHANNELS.map((channel, index) => {
        const top = topPadding + index * (paneHeight + gap);
        return {
            type: "rect",
            left: paneLeft,
            top,
            shape: {
                width: paneWidth,
                height: paneHeight,
                r: 12,
            },
            style: {
                fill: index % 2 === 0 ? "#fbfdff" : "#f4f8fc",
                stroke: "#d4dee8",
                lineWidth: 1,
            },
            silent: true,
            z: 0,
        };
    });
}

function paneTitles(paneHeight, topPadding, gap, gridLeft) {
    return CHANNELS.map((channel, index) => ({
        text: channel.label,
        left: gridLeft + 12,
        top: topPadding + index * (paneHeight + gap) + 10,
        padding: 0,
        z: 6,
        textStyle: {
            color: "#19232e",
            fontSize: 12,
            fontWeight: 700,
            fontFamily: '"IBM Plex Sans", "Segoe UI", sans-serif',
        },
    }));
}

function renderChart() {
    chart.resize();

    const chartHeight = chart.getHeight() || 820;
    const topPadding = 30;
    const bottomPadding = 86;
    const gap = 26;
    const paneLeft = 72;
    const paneRight = 26;
    const gridLeft = 116;
    const gridRight = 32;
    const panePaddingTop = 30;
    const panePaddingBottom = 10;
    const paneHeight = Math.max(
        126,
        Math.floor((chartHeight - topPadding - bottomPadding - gap * (CHANNELS.length - 1)) / CHANNELS.length)
    );
    const gridHeight = Math.max(86, paneHeight - panePaddingTop - panePaddingBottom);

    const axisMin = state.rangeStartMs ?? (state.rows[0]?.timeMs ?? Date.now() - 1000);
    const axisMax = state.rangeEndMs ?? (state.rows[state.rows.length - 1]?.timeMs ?? Date.now());
    const zoomStartValue = state.viewStartMs ?? axisMin;
    const zoomEndValue = state.viewEndMs ?? axisMax;

    chart.setOption(
        {
            animation: false,
            backgroundColor: "#f6f9fd",
            graphic: paneGraphics(paneHeight, topPadding, gap, paneLeft, paneRight),
            title: paneTitles(paneHeight, topPadding, gap, gridLeft),
            grid: CHANNELS.map((channel, index) => ({
                left: gridLeft,
                right: gridRight,
                top: topPadding + index * (paneHeight + gap) + panePaddingTop,
                height: gridHeight,
            })),
            tooltip: {
                trigger: "axis",
                axisPointer: { animation: false, lineStyle: { color: "#5f7896", width: 1.2 } },
                backgroundColor: "rgba(255,255,255,0.96)",
                borderColor: "#c9d6e4",
                textStyle: { color: "#16202b" },
            },
            xAxis: CHANNELS.map((channel, index) => ({
                type: "time",
                min: axisMin,
                max: axisMax,
                gridIndex: index,
                axisLine: { lineStyle: { color: "#9fb2c6", width: 1 } },
                axisTick: { show: false },
                axisLabel: {
                    color: "#425262",
                    show: index === CHANNELS.length - 1,
                    formatter: formatAxisLabel,
                },
                splitLine: { lineStyle: { color: "#d8e4ef", width: 1 } },
            })),
            yAxis: CHANNELS.map((channel, index) => ({
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
            })),
            dataZoom: [
                {
                    type: "slider",
                    xAxisIndex: CHANNELS.map((_, index) => index),
                    filterMode: "none",
                    showDataShadow: false,
                    brushSelect: false,
                    bottom: 16,
                    height: 26,
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
            ],
            series: CHANNELS.map((channel, index) => ({
                type: "line",
                xAxisIndex: index,
                yAxisIndex: index,
                showSymbol: false,
                smooth: false,
                connectNulls: false,
                sampling: "none",
                clip: true,
                data: buildSeries(channel.key),
                lineStyle: {
                    color: channel.color,
                    width: 2.1,
                    opacity: 1,
                },
                z: 4,
            })),
        },
        true
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
});

userIdInput.addEventListener("change", () => {
    state.bounds = null;
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
    renderChart();
});

historyStartPickerBtn.addEventListener("click", () => {
    openDateTimePicker(historyStartInput);
});

historyEndPickerBtn.addEventListener("click", () => {
    openDateTimePicker(historyEndInput);
});

loadBtn.addEventListener("click", loadCurrentMode);

clearBtn.addEventListener("click", () => {
    closeSocket();
    state.rows = [];
    state.source = state.mode === "history" ? "database-history" : "-";
    state.bounds = null;

    if (state.mode === "realtime") {
        const endTime = Date.now();
        state.rangeEndMs = endTime;
        state.rangeStartMs = endTime - state.realtimeBufferSeconds * 1000;
        state.autoFollow = true;
        applyViewWindow(null, true);
    } else {
        state.rangeStartMs = null;
        state.rangeEndMs = null;
        state.viewStartMs = null;
        state.viewEndMs = null;
        setSocketStatus("offline", "History");
    }

    updateMetrics();
    renderChart();
});

window.addEventListener("resize", () => {
    if (state.mode === "realtime") {
        applyViewWindow(state.viewStartMs, state.autoFollow);
    } else {
        applyViewWindow(state.viewStartMs ?? state.rangeStartMs, false);
    }
    updateMetrics();
    renderChart();
});

updateModeUI();
updateMetrics();
renderChart();
loadCurrentMode();
