const MAIN_CHANNELS = [
    { key: "eeg_1", label: "EEG 1", color: "#005ea8", toggleBands: true },
    { key: "eeg_2", label: "EEG 2", color: "#2048b3", toggleBands: true },
    { key: "emg_1", label: "EMG 1", color: "#087447", toggleBands: false },
    { key: "emg_2", label: "EMG 2", color: "#b75a00", toggleBands: false },
];

const SUB_BANDS = [
    { key: "delta", label: "Delta", color: "#4b71f2" },
    { key: "theta", label: "Theta", color: "#1f9bb4" },
    { key: "alpha", label: "Alpha", color: "#9a5de0" },
    { key: "beta", label: "Beta", color: "#d26a1b" },
];

const CHANNEL_BY_KEY = Object.fromEntries(MAIN_CHANNELS.map((channel) => [channel.key, channel]));

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
};

const chartHost = document.getElementById("chartScroll");
const chartEl = document.getElementById("chart");
const chart = echarts.init(chartEl, null, { renderer: "canvas" });

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

function currentPointCount() {
    if (state.mode === "history" && state.historyData) {
        return state.historyData.timeAxisMs.length;
    }
    return state.rows.length;
}

function currentLastPointMs() {
    if (state.mode === "history" && state.historyData) {
        return state.historyData.timeAxisMs[state.historyData.timeAxisMs.length - 1] ?? null;
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
        const numericValue = Number(input[index]);
        normalized[index] = Number.isFinite(numericValue) ? numericValue : null;
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
    for (const channel of MAIN_CHANNELS) {
        const channelPayload = payload?.signals?.[channel.key] || {};
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

        if (channel.toggleBands) {
            for (const band of SUB_BANDS) {
                const bandValues = normalizeNumericArray(channelPayload?.bands?.[band.key], timeAxisMs.length);
                signalState.bands[band.key] = {
                    values: bandValues,
                    series: zipSeries(timeAxisMs, bandValues),
                };
            }
        }

        signals[channel.key] = signalState;
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
}

function commitRealtimeRows(rows) {
    resetHistoryState();
    state.rows = normalizeRows(rows);
    updateMetrics();
    renderChart();
}

function commitHistoryPayload(payload) {
    state.rows = [];
    state.historyData = normalizeHistoryPayload(payload);
    state.sampleRateHz = state.historyData.sampleRateHz;
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
    renderChart();
    connectSocket();
}

function realtimeSeries(channelKey) {
    return state.rows.map((row) => [row.timeMs, row[channelKey] ?? null]);
}

function historyHasBands(channelKey) {
    const signal = state.historyData?.signals?.[channelKey];
    if (!signal?.bands) {
        return false;
    }
    return SUB_BANDS.every((band) => Array.isArray(signal.bands[band.key]?.series));
}

function getPaneDefinitions() {
    if (state.mode === "history" && state.historyData) {
        const panes = [];
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
        return panes;
    }

    return MAIN_CHANNELS.map((channel) => ({
        id: `${channel.key}:main`,
        channelKey: channel.key,
        label: channel.label,
        color: channel.color,
        data: realtimeSeries(channel.key),
        kind: "main",
        canToggle: false,
        expanded: false,
    }));
}

function paneOuterHeight(pane) {
    return pane.kind === "sub" ? 104 : 140;
}

function paneHeaderHeight(pane) {
    return pane.kind === "sub" ? 28 : 38;
}

function layoutPanes(panes, layout) {
    const positioned = [];
    let cursorTop = layout.topPadding;

    for (const pane of panes) {
        const outerHeight = paneOuterHeight(pane);
        const headerHeight = paneHeaderHeight(pane);
        const gridHeight = Math.max(58, outerHeight - headerHeight - 12);

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
    return Math.max(820, layout.topPadding + layout.bottomPadding + paneHeights + gaps);
}

function toggleSubBands(channelKey) {
    if (state.mode !== "history" || !historyHasBands(channelKey)) {
        return;
    }

    state.expandedBands[channelKey] = !state.expandedBands[channelKey];
    updateMetrics();
    renderChart();
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
                fill: pane.kind === "sub" ? "#f8fbff" : "#fbfdff",
                stroke: pane.kind === "sub" ? "#dbe6f1" : "#d4dee8",
                lineWidth: 1,
                shadowBlur: 0,
            },
            silent: true,
            z: 0,
        });

        graphics.push({
            type: "rect",
            left: layout.gridLeft + 12,
            top: pane.top + 13,
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
            top: pane.top + 8,
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

        if (pane.canToggle) {
            const buttonText = pane.expanded ? "收起节律" : "展示节律";
            graphics.push({
                type: "group",
                left: layout.gridLeft + 112,
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
                            text: buttonText,
                            fill: "#244d7d",
                            font: '600 11px "IBM Plex Sans", "Segoe UI", sans-serif',
                            textAlign: "center",
                            textVerticalAlign: "middle",
                        },
                    },
                ],
            });
        }
    }

    return graphics;
}

function renderChart() {
    const panes = getPaneDefinitions();
    const showHistoryLegend = state.mode === "history" && Boolean(state.historyData);
    const layout = {
        topPadding: showHistoryLegend ? 52 : 28,
        bottomPadding: 88,
        gap: 24,
        paneLeft: 72,
        paneRight: 28,
        gridLeft: 118,
        gridRight: 34,
    };

    chartEl.style.height = `${chartHeightForPanes(panes, layout)}px`;
    chart.resize();

    const positionedPanes = layoutPanes(panes, layout);
    const chartWidth = chart.getWidth() || chartHost.clientWidth || 1200;
    const paneWidth = Math.max(220, chartWidth - layout.paneLeft - layout.paneRight);
    const axisMin = state.rangeStartMs ?? (state.rows[0]?.timeMs ?? Date.now() - 1000);
    const axisMax = state.rangeEndMs ?? (state.rows[state.rows.length - 1]?.timeMs ?? Date.now());
    const zoomStartValue = state.viewStartMs ?? axisMin;
    const zoomEndValue = state.viewEndMs ?? axisMax;
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
                        color: "rgba(110, 120, 132, 0.44)",
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
            ];
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

    chart.setOption(
        {
            animation: false,
            backgroundColor: "#f6f9fd",
            graphic: buildPaneGraphics(positionedPanes, layout, paneWidth),
            legend: {
                show: showHistoryLegend,
                data: ["Raw", "Clean"],
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
            },
            xAxis: positionedPanes.map((pane, index) => ({
                type: "time",
                min: axisMin,
                max: axisMax,
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
            yAxis: positionedPanes.map((pane, index) => ({
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
                    xAxisIndex: positionedPanes.map((_, index) => index),
                    filterMode: "none",
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
                    filterMode: "none",
                    moveOnMouseMove: true,
                    moveOnMouseWheel: true,
                    zoomOnMouseWheel: false,
                },
            ],
            series,
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
    renderChart();
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
    resetHistoryState();
    state.source = state.mode === "history" ? "database-history-filtered" : "-";
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
