# compare_results.py
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def _to_datetime(series, baseline_str=None):
    """Convert dataTimestamp into pandas datetime."""
    # numeric seconds since baseline
    if np.issubdtype(series.dtype, np.number):
        if baseline_str is None:
            sys.exit("dataTimestamp is numeric; please provide --baseline (e.g., '2025-02-03 21:00:00').")
        baseline = pd.to_datetime(baseline_str)
        return baseline + pd.to_timedelta(series, unit="s")
    # string timestamps
    try:
        return pd.to_datetime(series)
    except Exception:
        sys.exit("Failed to parse 'dataTimestamp' as datetime. If it's seconds, pass --baseline.")

def _sleep_to_num(series, name):
    m = series.map({"S": 0, "W": 1})
    if m.isnull().any():
        bad = series[series.map(lambda x: x not in ("S", "W"))].unique()
        sys.exit(f"Error: {name} must be only 'S'/'W'. Found: {bad}")
    return m.astype(int)

def main():
    ap = argparse.ArgumentParser(description="Visualize CK baseline vs HMM-smoothed results, with stats.")
    ap.add_argument("--a", required=True, help="Baseline CSV (e.g., cole_mult_results.csv)")
    ap.add_argument("--b", required=True, help="HMM CSV (e.g., cole_mult_hmm_results.csv)")
    ap.add_argument("--label_a", default="baseline", help="Label for baseline (default: baseline)")
    ap.add_argument("--label_b", default="hmm", help="Label for HMM (default: hmm)")
    ap.add_argument("--out", default="ck_vs_hmm_comparison.csv", help="Merged CSV output")
    ap.add_argument("--png", default="ck_vs_hmm_plot.png", help="PNG file to save plot (use '-' to only show)")
    ap.add_argument("--baseline", default=None, help="Baseline datetime for numeric timestamps (e.g., '2025-02-03 21:00:00')")
    ap.add_argument("--hours", type=float, default=None, help="Limit x-axis to first N hours (optional)")
    args = ap.parse_args()

    # Load
    try:
        A = pd.read_csv(args.a)
        B = pd.read_csv(args.b)
    except Exception as e:
        sys.exit(f"Error loading inputs: {e}")

    key = "dataTimestamp"
    if key not in A.columns or key not in B.columns:
        sys.exit(f"Both CSVs must contain '{key}'")

    # Find baseline label col in A
    if "sleep" in A.columns:
        a_label_col = "sleep"
    elif "consensus_sleep" in A.columns:
        a_label_col = "consensus_sleep"
    else:
        sys.exit("Baseline CSV must have 'sleep' or 'consensus_sleep' column.")

    # HMM column in B
    if "hmm_state" not in B.columns:
        sys.exit("HMM CSV must have 'hmm_state' column.")
    has_hmm_prob = "hmm_sleep_prob" in B.columns

    # Select/rename
    A2 = A[[key, a_label_col]].rename(columns={a_label_col: f"{args.label_a}_label"})
    B2 = B[[key, "hmm_state"] + (["hmm_sleep_prob"] if has_hmm_prob else [])] \
            .rename(columns={"hmm_state": f"{args.label_b}_label"})

    # Merge
    M = A2.merge(B2, on=key, how="inner")
    if M.empty:
        sys.exit("No overlapping timestamps after merge.")

    # Time index
    M["time"] = _to_datetime(M[key], args.baseline)

    # Map to numeric 0=S,1=W
    M[f"{args.label_a}_num"] = _sleep_to_num(M[f"{args.label_a}_label"], f"{args.label_a}_label")
    M[f"{args.label_b}_num"] = _sleep_to_num(M[f"{args.label_b}_label"], f"{args.label_b}_label")

    # Agreement and simple counts
    M["agree"] = (M[f"{args.label_a}_label"] == M[f"{args.label_b}_label"]).astype(int)
    agree_rate = float(M["agree"].mean())

    s_to_w = ((M[f"{args.label_a}_label"] == "S") & (M[f"{args.label_b}_label"] == "W")).sum()
    w_to_s = ((M[f"{args.label_a}_label"] == "W") & (M[f"{args.label_b}_label"] == "S")).sum()

    # Epoch minutes (estimate from first delta)
    if len(M) > 1:
        epoch_min = (M["time"].iloc[1] - M["time"].iloc[0]).total_seconds() / 60.0
    else:
        epoch_min = 1.0

    total_awake_a = M[f"{args.label_a}_num"].sum() * epoch_min
    total_sleep_a = (len(M) - M[f"{args.label_a}_num"].sum()) * epoch_min

    total_awake_b = M[f"{args.label_b}_num"].sum() * epoch_min
    total_sleep_b = (len(M) - M[f"{args.label_b}_num"].sum()) * epoch_min

    # Save merged CSV
    M_out = M[[key, "time", f"{args.label_a}_label", f"{args.label_b}_label", "agree"] +
              ([ "hmm_sleep_prob"] if has_hmm_prob else [])]
    M_out.to_csv(args.out, index=False)
    print(f"Rows compared: {len(M)}")
    print(f"Agreement: {agree_rate:.3f}")
    print(f"{args.label_a} S -> {args.label_b} W: {s_to_w}")
    print(f"{args.label_a} W -> {args.label_b} S: {w_to_s}")
    print(f"Merged CSV saved to {args.out}")

    # ---- Visualization ----
    # Limit window if requested
    t0 = M["time"].min()
    if args.hours is not None:
        t1 = t0 + pd.Timedelta(hours=args.hours)
        plot_df = M[(M["time"] >= t0) & (M["time"] <= t1)].copy()
    else:
        plot_df = M.copy()

    if plot_df.empty:
        sys.exit("No data to plot after applying hours window.")

    fig_height = 5.5 if has_hmm_prob else 4.2
    fig, axes = plt.subplots(
        nrows=(2 if has_hmm_prob else 1),
        ncols=1,
        figsize=(14, fig_height),
        sharex=True
    )
    ax = axes if not has_hmm_prob else axes[0]

    # Step plots for states
    # 0 = Sleep, 1 = Wake
    ax.step(plot_df["time"], plot_df[f"{args.label_a}_num"], where="post", lw=1.5, label=f"{args.label_a} (0=S,1=W)")
    ax.step(plot_df["time"], plot_df[f"{args.label_b}_num"], where="post", lw=1.2, linestyle="--", label=f"{args.label_b} (0=S,1=W)")

    # Shade disagreements
    disagree = plot_df[f"{args.label_a}_num"] != plot_df[f"{args.label_b}_num"]
    if disagree.any():
        # build intervals of disagreement
        times = plot_df["time"].to_numpy()
        mask = disagree.to_numpy().astype(int)
        # find contiguous regions
        start_idx = None
        for i in range(len(mask)):
            if mask[i] and start_idx is None:
                start_idx = i
            if (not mask[i] or i == len(mask)-1) and start_idx is not None:
                end_idx = i if mask[i] else i-1
                ax.axvspan(times[start_idx], times[end_idx], alpha=0.15, label="disagree" if start_idx == np.where(mask==1)[0][0] else None)
                start_idx = None

    # Y formatting
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sleep (0)", "Wake (1)"])
    ax.set_ylabel("State")
    ax.grid(which="major", linestyle="--", alpha=0.5)
    ax.grid(which="minor", linestyle=":", alpha=0.3)
    ax.set_title(f"Baseline vs HMM Sleep/Wake")

    # Legend
    leg = ax.legend(loc="upper right")
    leg.get_frame().set_linewidth(1.5)

    # Metrics box
    ax.text(
        0.995, 0.02,
        "Agreement: {:.1f}%\n{} Awake: {} min\n{} Awake: {} min\nEpoch: {:.2f} min".format(
            100.0*agree_rate, args.label_a, int(total_awake_a), args.label_b, int(total_awake_b), epoch_min
        ),
        transform=ax.transAxes, ha="right", va="bottom",
        bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.4")
    )

    # Optional HMM probability subplot
    if has_hmm_prob:
        ax2 = axes[1]
        ax2.plot(plot_df["time"], plot_df["hmm_sleep_prob"], lw=1.5, label="HMM P(Sleep)")
        ax2.axhline(0.5, linestyle="--", lw=1)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("P(Sleep)")
        ax2.grid(which="major", linestyle="--", alpha=0.5)
        ax2.grid(which="minor", linestyle=":", alpha=0.3)
        ax2.legend(loc="upper right")

    # X formatting
    ax.set_xlabel("Time")
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if args.png != "-":
        plt.savefig(args.png, dpi=180)
        print(f"Plot saved to {args.png}")
    plt.show()

if __name__ == "__main__":
    main()
