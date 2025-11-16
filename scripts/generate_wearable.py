#!/usr/bin/env python3
"""
Synthetic wearable data generator (1-min resolution for 7 days by default).

Signals:
- Heart Rate (bpm): baseline ~75, circadian swing (~±6), sleep drop (~-8 at night), noise (~2.5 SD).
  Episode bump: +8..+22 bpm.
- Pupil diameter (mm): baseline ~3.5 with small noise (~0.08 SD). Episode bump: +0.3..+0.8 mm.
- Skin Conductance Level (µS): day baseline ~3.0, night ~2.2, slow drift, noise (~0.12 SD).
  Episode bump: +0.5..+4.0 µS + extra jitter during episode minutes.
- Body Temperature (°C): mean ~36.8, circadian amplitude ~0.35°C (trough ~05:00),
  noise (~0.03 SD). Episode bump: +0.10..+0.30 °C.

Episodes:
- Duration 6..22 minutes (smooth half-cosine onset).
- Scheduled during wake hours only.
- Per-day count can be a fixed integer or Poisson(mean).

Outputs:
- CSV with columns:
  timestamp, heart_rate_bpm, pupil_mm, eda_scl_uS, body_temp_C, episode

Usage examples:
  python generate_wearable.py
  python generate_wearable.py --episodes-per-day 3 --episodes-mode fixed
  python generate_wearable.py --start "2025-11-03 08:00" --out wearable.csv
"""

import argparse
from datetime import datetime
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic 1-min wearable data for a week."
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--start",
        type=str,
        default=None,
        help='Start datetime like "YYYY-MM-DD HH:MM". Defaults to now (minute-rounded).',
    )
    p.add_argument(
        "--minutes",
        type=int,
        default=7 * 24 * 60,
        help="Number of minutes (default: one week)",
    )
    p.add_argument(
        "--sleep-start",
        type=int,
        default=23,
        help="Sleep start hour (0-23), default 23",
    )
    p.add_argument(
        "--sleep-end", type=int, default=7, help="Sleep end hour (0-23), default 7"
    )
    p.add_argument(
        "--episodes-per-day",
        type=float,
        default=2.0,
        help="If episodes-mode=poisson, this is the Poisson mean; if fixed, use an integer.",
    )
    p.add_argument(
        "--episodes-mode",
        choices=["poisson", "fixed"],
        default="poisson",
        help="Episode count scheduling mode per day",
    )
    p.add_argument(
        "--episode-min-dur", type=int, default=6, help="Min episode duration in minutes"
    )
    p.add_argument(
        "--episode-max-dur",
        type=int,
        default=22,
        help="Max episode duration in minutes",
    )
    p.add_argument(
        "--out", type=str, default="wearable_week_1min.csv", help="Output CSV path"
    )
    return p.parse_args()


def circadian(
    minute_idx: np.ndarray, amplitude: float, phase_shift_minutes: int = 0
) -> np.ndarray:
    """Simple 24h sine wave with phase shift in minutes."""
    period = 24 * 60
    x = 2 * np.pi * ((minute_idx - phase_shift_minutes) % period) / period
    return amplitude * np.sin(x)


def smooth(x: np.ndarray, win: int = 3) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Start time
    if args.start:
        start_time = datetime.strptime(args.start, "%Y-%m-%d %H:%M")
    else:
        now = datetime.now()
        start_time = now.replace(second=0, microsecond=0)

    minutes = args.minutes
    time_index = pd.date_range(start=start_time, periods=minutes, freq="1min")

    def is_sleep_hour(t):
        # True if within [sleep_start .. 23] ∪ [0 .. sleep_end)
        return (t.hour >= args.sleep_start) or (t.hour < args.sleep_end)

    minute_idx = np.arange(minutes)

    # ---------------- Baselines & circadian ----------------
    # Heart rate (bpm)
    hr_mean = 75.0
    hr_circ = circadian(
        minute_idx, amplitude=6.0, phase_shift_minutes=16 * 60
    )  # peak ~16:00
    hr_noise = rng.normal(0, 2.5, size=minutes)
    hr_sleep_drop = np.array([-8.0 if is_sleep_hour(t) else 0.0 for t in time_index])
    hr = hr_mean + hr_circ + hr_sleep_drop + hr_noise
    hr = np.clip(hr, 40, 180)

    # Pupil diameter (mm) — assume steady lighting
    pupil_baseline = 3.5
    pupil_noise = rng.normal(0, 0.08, size=minutes)
    pupil = pupil_baseline + pupil_noise
    pupil = np.clip(pupil, 2.0, 7.0)

    # Skin Conductance Level (µS)
    eda_day_base = 3.0
    eda_night_base = 2.2
    eda_base = np.array(
        [eda_night_base if is_sleep_hour(t) else eda_day_base for t in time_index]
    )
    eda_drift = np.cumsum(rng.normal(0, 0.002, size=minutes))  # slow random drift
    eda_noise = rng.normal(0, 0.12, size=minutes)  # tonic noise
    eda_scl = np.maximum(0.01, eda_base + eda_drift + eda_noise)

    # Body temperature (°C)
    temp_mean = 36.8
    temp_circ = circadian(
        minute_idx, amplitude=0.35, phase_shift_minutes=5 * 60
    )  # trough ~05:00
    temp_noise = rng.normal(0, 0.03, size=minutes)
    temp = temp_mean + temp_circ + temp_noise
    temp = np.clip(temp, 35.5, 39.5)

    # ---------------- Episode scheduling ----------------
    # Build daily episode windows (start index, duration)
    episode_mask = np.zeros(minutes, dtype=bool)
    episode_windows = []

    def schedule_daily(num, day_start_idx):
        slots = []
        attempts = 0
        while len(slots) < num and attempts < 500:
            attempts += 1
            m = rng.integers(0, 24 * 60)
            t = time_index[day_start_idx + m]
            if is_sleep_hour(t):
                continue
            dur = int(rng.integers(args.episode_min_dur, args.episode_max_dur + 1))
            # space starts ~30 minutes apart to reduce overlap
            if all(abs(m - s) > 30 for s, _ in slots):
                slots.append((m, dur))
        return [(day_start_idx + s, d) for s, d in slots]

    days = minutes // (24 * 60)
    for d in range(days):
        day_start = d * 24 * 60
        if args.episodes_mode == "fixed":
            n_today = int(round(args.episodes_per_day))
        else:
            n_today = int(rng.poisson(args.episodes_per_day))
        episode_windows.extend(schedule_daily(n_today, day_start))

    # ---------------- Apply responses ----------------
    def apply_bump(series: np.ndarray, start_idx: int, duration: int, peak: float):
        idx = np.arange(start_idx, min(start_idx + duration, minutes))
        if len(idx) == 0:
            return
        # half-cosine rise to peak over duration (smooth onset)
        t = np.linspace(0, np.pi, len(idx))
        bump = 0.5 * (1 - np.cos(t))  # 0 -> 1
        series[idx] += peak * bump
        episode_mask[idx] = True

    for start_idx, dur in episode_windows:
        # Heart Rate bump
        hr_peak = float(rng.uniform(8, 22))
        apply_bump(hr, start_idx, dur, hr_peak)
        # Pupil bump
        pupil_peak = float(rng.uniform(0.3, 0.8))
        apply_bump(pupil, start_idx, dur, pupil_peak)
        # EDA bump + extra jitter during episode minutes (mimic frequent SCRs)
        eda_peak = float(rng.uniform(0.5, 4.0))
        apply_bump(eda_scl, start_idx, dur, eda_peak)
        end_idx = min(start_idx + dur, minutes)
        eda_scl[start_idx:end_idx] += rng.normal(0, 0.2, size=end_idx - start_idx)
        # Temperature bump
        temp_peak = float(rng.uniform(0.10, 0.30))
        apply_bump(temp, start_idx, dur, temp_peak)

    # Smooth + clip for plausibility
    hr = np.clip(smooth(hr, 3), 35, 200)
    pupil = np.clip(smooth(pupil, 3), 1.5, 8.0)
    eda_scl = np.clip(smooth(eda_scl, 3), 0.01, 40.0)
    temp = np.clip(smooth(temp, 3), 34.0, 41.5)

    # ---------------- Assemble and write ----------------
    df = pd.DataFrame(
        {
            "timestamp": time_index,
            "heart_rate_bpm": np.round(hr, 1),
            "pupil_mm": np.round(pupil, 3),
            "eda_scl_uS": np.round(eda_scl, 3),
            "body_temp_C": np.round(temp, 3),
            "episode": episode_mask,
        }
    )
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(
        f"Episode minutes: {int(df['episode'].sum())} | Episodes scheduled: {len(episode_windows)}"
    )


if __name__ == "__main__":
    main()
