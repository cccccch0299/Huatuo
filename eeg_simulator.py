from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional


TAU = math.tau
UTC = timezone.utc


@dataclass(frozen=True)
class SceneProfile:
    name: str
    duration_range_s: tuple[float, float]
    delta_uv: float
    theta_uv: float
    alpha_uv: float
    beta_uv: float
    gamma_uv: float
    noise_uv: float
    emg_base_uv: float
    asymmetry: float
    blink_rate_hz: float
    burst_rate_hz: float
    motion_rate_hz: float


SCENES = (
    SceneProfile(
        name="sim_rest",
        duration_range_s=(10.0, 18.0),
        delta_uv=4.0,
        theta_uv=8.0,
        alpha_uv=24.0,
        beta_uv=7.0,
        gamma_uv=2.5,
        noise_uv=1.8,
        emg_base_uv=5.5,
        asymmetry=0.10,
        blink_rate_hz=0.18,
        burst_rate_hz=0.05,
        motion_rate_hz=0.015,
    ),
    SceneProfile(
        name="sim_focus",
        duration_range_s=(8.0, 14.0),
        delta_uv=2.5,
        theta_uv=5.0,
        alpha_uv=10.0,
        beta_uv=18.0,
        gamma_uv=4.5,
        noise_uv=2.2,
        emg_base_uv=7.0,
        asymmetry=0.06,
        blink_rate_hz=0.10,
        burst_rate_hz=0.10,
        motion_rate_hz=0.02,
    ),
    SceneProfile(
        name="sim_drowsy",
        duration_range_s=(7.0, 12.0),
        delta_uv=10.0,
        theta_uv=17.0,
        alpha_uv=8.0,
        beta_uv=4.0,
        gamma_uv=1.0,
        noise_uv=2.8,
        emg_base_uv=4.5,
        asymmetry=0.08,
        blink_rate_hz=0.12,
        burst_rate_hz=0.03,
        motion_rate_hz=0.01,
    ),
    SceneProfile(
        name="sim_motor_imagery",
        duration_range_s=(6.0, 10.0),
        delta_uv=2.0,
        theta_uv=6.5,
        alpha_uv=13.0,
        beta_uv=14.5,
        gamma_uv=3.8,
        noise_uv=2.5,
        emg_base_uv=6.0,
        asymmetry=0.18,
        blink_rate_hz=0.08,
        burst_rate_hz=0.08,
        motion_rate_hz=0.018,
    ),
)

SCENE_WEIGHTS = (0.36, 0.30, 0.16, 0.18)


class NeuroSignalSimulator:
    def __init__(self, sample_rate_hz: int, seed: Optional[int] = None) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.random = random.Random(seed)
        self.sample_index = 0
        self.scene: Optional[SceneProfile] = None
        self.scene_samples_left = 0
        self.current_event_label = "sim_boot"

        self.eeg_noise_state = [0.0, 0.0]
        self.emg_noise_state = [0.0, 0.0]
        self.active_blinks: List[Dict[str, float]] = []
        self.active_emg_bursts: List[Dict[str, float]] = []
        self.active_motion_artifacts: List[Dict[str, float]] = []

        self.delta_phase = self.random.uniform(0.0, TAU)
        self.theta_phase = self.random.uniform(0.0, TAU)
        self.alpha_phase = self.random.uniform(0.0, TAU)
        self.beta_phase = self.random.uniform(0.0, TAU)
        self.gamma_phase = self.random.uniform(0.0, TAU)
        self.line_phase = self.random.uniform(0.0, TAU)
        self.emg_phase = self.random.uniform(0.0, TAU)
        self.emg_phase_2 = self.random.uniform(0.0, TAU)
        self.drift_phase = self.random.uniform(0.0, TAU)

        self._select_scene(force=True)

    def _samples(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.sample_rate_hz)))

    def _select_scene(self, force: bool = False) -> None:
        if not force and self.scene_samples_left > 0:
            return

        self.scene = self.random.choices(SCENES, weights=SCENE_WEIGHTS, k=1)[0]
        duration_s = self.random.uniform(*self.scene.duration_range_s)
        self.scene_samples_left = self._samples(duration_s)
        self.current_event_label = self.scene.name

    def _maybe_schedule_events(self) -> None:
        assert self.scene is not None

        if self.random.random() < self.scene.blink_rate_hz / self.sample_rate_hz:
            center = self.sample_index + self.random.randint(self._samples(0.02), self._samples(0.12))
            self.active_blinks.append(
                {
                    "center": center,
                    "width": self._samples(self.random.uniform(0.03, 0.06)),
                    "amplitude": self.random.uniform(65.0, 120.0),
                }
            )

        if self.random.random() < self.scene.burst_rate_hz / self.sample_rate_hz:
            center = self.sample_index + self.random.randint(self._samples(0.04), self._samples(0.18))
            self.active_emg_bursts.append(
                {
                    "center": center,
                    "width": self._samples(self.random.uniform(0.08, 0.35)),
                    "amplitude": self.random.uniform(45.0, 180.0),
                    "balance": self.random.uniform(-0.25, 0.25),
                }
            )

        if self.random.random() < self.scene.motion_rate_hz / self.sample_rate_hz:
            center = self.sample_index + self.random.randint(self._samples(0.08), self._samples(0.3))
            self.active_motion_artifacts.append(
                {
                    "center": center,
                    "width": self._samples(self.random.uniform(0.12, 0.4)),
                    "amplitude": self.random.uniform(18.0, 42.0),
                }
            )

    def _gaussian_envelope(self, sample: int, center: float, width: float) -> float:
        x = (sample - center) / max(1.0, width)
        return math.exp(-0.5 * x * x)

    def _cleanup_events(self) -> None:
        self.active_blinks = [
            event for event in self.active_blinks
            if self.sample_index <= event["center"] + event["width"] * 4
        ]
        self.active_emg_bursts = [
            event for event in self.active_emg_bursts
            if self.sample_index <= event["center"] + event["width"] * 4
        ]
        self.active_motion_artifacts = [
            event for event in self.active_motion_artifacts
            if self.sample_index <= event["center"] + event["width"] * 4
        ]

    def next_sample(self) -> tuple[float, float, float, float, str]:
        self._select_scene()
        assert self.scene is not None

        self._maybe_schedule_events()

        t = self.sample_index / self.sample_rate_hz
        drift = 7.0 * math.sin(TAU * 0.18 * t + self.drift_phase)
        line_noise = 1.4 * math.sin(TAU * 50.0 * t + self.line_phase)

        self.eeg_noise_state[0] = 0.975 * self.eeg_noise_state[0] + self.random.gauss(0.0, self.scene.noise_uv)
        self.eeg_noise_state[1] = 0.972 * self.eeg_noise_state[1] + self.random.gauss(0.0, self.scene.noise_uv * 1.05)
        self.emg_noise_state[0] = 0.78 * self.emg_noise_state[0] + self.random.gauss(0.0, self.scene.emg_base_uv * 0.7)
        self.emg_noise_state[1] = 0.81 * self.emg_noise_state[1] + self.random.gauss(0.0, self.scene.emg_base_uv * 0.72)

        common_eeg = (
            self.scene.delta_uv * math.sin(TAU * 1.2 * t + self.delta_phase)
            + self.scene.theta_uv * math.sin(TAU * 5.8 * t + self.theta_phase)
            + self.scene.alpha_uv * math.sin(TAU * 9.8 * t + self.alpha_phase)
            + self.scene.beta_uv * math.sin(TAU * 19.5 * t + self.beta_phase)
            + self.scene.gamma_uv * math.sin(TAU * 37.0 * t + self.gamma_phase)
            + drift
            + line_noise
        )

        asymmetry = self.scene.asymmetry * self.scene.alpha_uv * math.sin(TAU * 10.4 * t + self.alpha_phase / 2.0)

        blink_left = 0.0
        blink_right = 0.0
        for event in self.active_blinks:
            env = self._gaussian_envelope(self.sample_index, event["center"], event["width"])
            blink_left += event["amplitude"] * env
            blink_right += event["amplitude"] * env * self.random.uniform(0.74, 0.92)

        motion = 0.0
        for event in self.active_motion_artifacts:
            env = self._gaussian_envelope(self.sample_index, event["center"], event["width"])
            motion += event["amplitude"] * env * math.sin(TAU * 2.4 * t)

        emg_burst_1 = 0.0
        emg_burst_2 = 0.0
        for event in self.active_emg_bursts:
            env = self._gaussian_envelope(self.sample_index, event["center"], event["width"])
            carrier_1 = math.sin(TAU * 62.0 * t + self.emg_phase) + 0.45 * math.sin(TAU * 87.0 * t + self.emg_phase_2)
            carrier_2 = math.sin(TAU * 69.0 * t + self.emg_phase_2) + 0.40 * math.sin(TAU * 96.0 * t + self.emg_phase)
            emg_burst_1 += event["amplitude"] * env * carrier_1 * (1.0 + event["balance"])
            emg_burst_2 += event["amplitude"] * env * carrier_2 * (1.0 - event["balance"])

        eeg_1 = common_eeg + asymmetry + self.eeg_noise_state[0] + blink_left + motion
        eeg_2 = common_eeg * 0.93 - asymmetry + self.eeg_noise_state[1] - blink_right + motion * 0.8

        emg_base_1 = self.scene.emg_base_uv * (
            math.sin(TAU * 42.0 * t + self.emg_phase)
            + 0.55 * math.sin(TAU * 71.0 * t + self.emg_phase_2)
        )
        emg_base_2 = self.scene.emg_base_uv * (
            math.sin(TAU * 47.0 * t + self.emg_phase_2)
            + 0.48 * math.sin(TAU * 76.0 * t + self.emg_phase)
        )

        emg_1 = emg_base_1 + self.emg_noise_state[0] + emg_burst_1 + motion * 0.35
        emg_2 = emg_base_2 + self.emg_noise_state[1] + emg_burst_2 + motion * 0.35

        self.sample_index += 1
        self.scene_samples_left -= 1
        self._cleanup_events()

        if self.scene_samples_left <= 0:
            self._select_scene(force=True)

        return (
            round(eeg_1, 4),
            round(eeg_2, 4),
            round(emg_1, 4),
            round(emg_2, 4),
            self.current_event_label,
        )


def iso_timestamp(base_time: datetime, sample_index: int, sample_rate_hz: int) -> str:
    sample_time = base_time + timedelta(seconds=sample_index / sample_rate_hz)
    return sample_time.isoformat(timespec="microseconds").replace("+00:00", "Z")


def build_batch(
    simulator: NeuroSignalSimulator,
    base_time: datetime,
    user_id: int,
    batch_samples: int,
) -> Dict[str, object]:
    items: List[Dict[str, object]] = []
    current_sample_index = simulator.sample_index
    batch_event_label = simulator.current_event_label

    for offset in range(batch_samples):
        eeg_1, eeg_2, emg_1, emg_2, batch_event_label = simulator.next_sample()
        timestamp = iso_timestamp(base_time, current_sample_index + offset, simulator.sample_rate_hz)
        items.extend(
            (
                {"ch": 0, "val": eeg_1, "sTime": timestamp},
                {"ch": 1, "val": eeg_2, "sTime": timestamp},
                {"ch": 2, "val": emg_1, "sTime": timestamp},
                {"ch": 3, "val": emg_2, "sTime": timestamp},
            )
        )

    return {
        "user_id": user_id,
        "event_label": batch_event_label,
        "items": items,
    }


def post_payload(url: str, payload: Dict[str, object], timeout_s: float) -> Dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate Unity EEG DataWrapper uploads.")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000/api/eeg/upload")
    parser.add_argument("--user-id", type=int, default=99)
    parser.add_argument("--sample-rate", type=int, default=250)
    parser.add_argument("--batch-ms", type=int, default=100)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = realtime, 10.0 = 10x faster backfill")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_rate <= 0 or args.batch_ms <= 0 or args.speed <= 0:
        print("sample-rate, batch-ms and speed must be positive.", file=sys.stderr)
        return 2

    batch_samples = max(1, round(args.sample_rate * args.batch_ms / 1000))
    batch_duration_s = batch_samples / args.sample_rate
    simulator = NeuroSignalSimulator(sample_rate_hz=args.sample_rate, seed=args.seed)
    stream_base_time = datetime.now(UTC)
    wall_clock_start = time.perf_counter()
    max_batches = None
    if args.duration_seconds > 0:
        max_batches = max(1, int(math.ceil(args.duration_seconds / batch_duration_s)))

    stop_requested = False

    def handle_stop(signum, frame) -> None:  # type: ignore[override]
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_stop)

    batch_index = 0
    sent_points = 0
    print(
        f"Simulator started: user_id={args.user_id}, sample_rate={args.sample_rate}Hz, "
        f"batch={batch_samples} samples/channel, speed={args.speed}x, backend={args.backend_url}"
    )
    print("Frontend: choose Realtime mode and set User ID to", args.user_id)

    while not stop_requested:
        if max_batches is not None and batch_index >= max_batches:
            break

        payload = build_batch(
            simulator=simulator,
            base_time=stream_base_time,
            user_id=args.user_id,
            batch_samples=batch_samples,
        )

        try:
            response = post_payload(args.backend_url, payload, timeout_s=args.timeout_seconds)
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            print(f"Upload failed with HTTP {exc.code}: {error_text}", file=sys.stderr)
            return 1
        except urllib.error.URLError as exc:
            print(f"Upload failed: {exc}", file=sys.stderr)
            return 1

        batch_index += 1
        sent_points += len(payload["items"])  # type: ignore[arg-type]

        if batch_index == 1 or batch_index % max(1, round(1.0 / batch_duration_s)) == 0:
            print(
                f"batch={batch_index} scene={payload['event_label']} "
                f"points={len(payload['items'])} accepted_rows={response.get('accepted_rows')}"
            )

        target_elapsed = batch_index * batch_duration_s / args.speed
        sleep_seconds = target_elapsed - (time.perf_counter() - wall_clock_start)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    total_seconds = time.perf_counter() - wall_clock_start
    print(
        f"Simulator stopped. batches={batch_index}, points={sent_points}, "
        f"elapsed={total_seconds:.2f}s, last_scene={simulator.current_event_label}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
