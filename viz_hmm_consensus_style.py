# viz_hmm_consensus_style.py
import argparse, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    p = argparse.ArgumentParser(
        description="Plot HMM sleep probability & hmm_state over first 4 hours with duration metrics (consensus-style)."
    )
    p.add_argument("file", help="CSV with dataTimestamp, hmm_state, [optional] hmm_sleep_prob")
    p.add_argument("--baseline", default=None,
                   help="If dataTimestamp is seconds, provide baseline datetime (e.g., '2025-02-03 21:00:00').")
    p.add_argument("--hours", type=float, default=4.0,
                   help="Hours to show from start (default: 4)")
    args = p.parse_args()

    # Load
    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        sys.exit(f"Error loading '{args.file}': {e}")

    # Validate
    if "dataTimestamp" not in df.columns:
        sys.exit("CSV must contain 'dataTimestamp'")
    if "hmm_state" not in df.columns:
        sys.exit("CSV must contain 'hmm_state' (values 'S'/'W')")
    has_prob = "hmm_sleep_prob" in df.columns

    # Parse/convert timestamps
    ts = df["dataTimestamp"]
    if np.issubdtype(ts.dtype, np.number):
        if not args.baseline:
            sys.exit("Numeric timestamps detected. Provide --baseline (e.g., '2025-02-03 21:00:00').")
        base = pd.to_datetime(args.baseline)
        df["time"] = base + pd.to_timedelta(ts, unit="s")
    else:
        # already strings -> try to parse as datetimes
        df["time"] = pd.to_datetime(ts, errors="coerce")
        if df["time"].isna().any():
            sys.exit("Failed to parse 'dataTimestamp' to datetime. If it's in seconds, pass --baseline.")

    # Binary map
    df["state_num"] = df["hmm_state"].map({"S": 0, "W": 1})
    if df["state_num"].isnull().any():
        bad = df.loc[df["state_num"].isnull(), "hmm_state"].unique()
        sys.exit(f"'hmm_state' must be only 'S'/'W'. Found: {bad}")

    # Epoch length (minutes)
    if len(df) > 1:
        epoch_min = (df["time"].iloc[1] - df["time"].iloc[0]).total_seconds() / 60.0
    else:
        epoch_min = 1.0

    # Totals
    total_awake = int(df["state_num"].sum() * epoch_min)
    total_sleep = int((len(df) - df["state_num"].sum()) * epoch_min)

    # Window
    start = df["time"].min()
    end = start + pd.Timedelta(hours=args.hours)
    view = df[(df["time"] >= start) & (df["time"] <= end)].copy()
    if view.empty:
        sys.exit("No data in requested window.")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(14, 4))

    # Continuous curve (use hmm_sleep_prob if present; else fall back to 1 - state_num for a pseudo-prob)
    if has_prob:
        curve = view["hmm_sleep_prob"].astype(float).clip(0, 1)
        ylabel = "HMM Sleep Probability"
        top = 1.02
    else:
        # not ideal, but gives a visual similar to consensus index
        curve = 1.0 - view["state_num"].astype(float)
        ylabel = "Sleep Signal (proxy)"
        top = 1.02

    ax.plot(
        view["time"], curve,
        label="HMM Sleep Probability" if has_prob else "Sleep (proxy)",
        linewidth=1.6, alpha=0.85
    )

    # Threshold at 0.5
    thresh = 1.0
    ax.axhline(thresh, color="gray", linestyle="--", linewidth=1, label="Threshold = 1.0")

    # Sleep/Wake fill (red wake, blue sleep), consensus-style
    # Scale bands to same vertical range
    ax.fill_between(
        view["time"], 0, view["state_num"] * top,
        step="post", color="red", alpha=0.28, label="Wake"
    )
    ax.fill_between(
        view["time"], 0, (1 - view["state_num"]) * top,
        step="post", color="blue", alpha=0.10, label="Sleep"
    )

    # Annotate contiguous wake segments with duration (minutes)
    awake_bool = view["state_num"].to_numpy().astype(bool)
    times = view["time"].to_numpy()
    segments = []
    in_seg = False
    for i, on in enumerate(awake_bool):
        if on and not in_seg:
            s = i
            in_seg = True
        elif not on and in_seg:
            segments.append((s, i - 1))
            in_seg = False
    if in_seg:
        segments.append((s, len(awake_bool) - 1))

    span = top - thresh
    for idx, (s, e) in enumerate(segments):
        t0, t1 = times[s], times[e]
        dur_min = (e - s + 1) * epoch_min
        mid = t0 + (t1 - t0) / 2
        offset = span * (0.05 + 0.12 * (idx % 2))
        y = thresh + offset
        ax.text(mid, y, f"{int(dur_min)} m.", color="red", ha="center", va="bottom")

    # Legend & metrics
    leg = ax.legend(loc="upper right")
    leg.get_frame().set_linewidth(2)

    ax.text(
        0.98, 0.60,
        f"Total Awake: {total_awake} min\nTotal Sleep: {total_sleep} min",
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.4")
    )

    # Axes formatting
    ax.set_xlim(start, end)
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel(ylabel)
    ax.set_title("HMM-Smoothed Sleep/Wake Classification (Consensus-Style)")

    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":", alpha=0.3)
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
