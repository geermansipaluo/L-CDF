#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
csv_root = "/home/guo/L-CDF/src/sensor_cdf/scripts/saved_evaldata"
out_dir = "/home/guo/L-CDF/image/eval/compare"
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# GP-CBF point cloud resolution sweep result
# CSV should contain at least:
#   num_rays, success_rate, arrival_rate, collision_rate
# Each row can be one 100-episode averaged result, or repeated runs.
# If repeated rows exist for the same num_rays, this script averages them.
# ============================================================
csv_path = os.path.join(csv_root, "eval_metrics_gpcbf_pc.csv")
out_png = os.path.join(out_dir, "gpcbf_rates_vs_num_rays.png")

# ============================================================
# Visual settings
# ============================================================
show_std_band = False  # For one 100-episode average per num_rays, keep this False.

metrics = {
    "success_rate": {
        "label": "Success rate",
        "marker": "o",
        "linestyle": "-",
    },
    "arrival_rate": {
        "label": "Arrival rate",
        "marker": "s",
        "linestyle": "--",
    },
    "collision_rate": {
        "label": "Collision rate",
        "marker": "^",
        "linestyle": "-.",
    },
}

# ============================================================
# Read CSV
# ============================================================
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df = pd.read_csv(csv_path)

required_cols = ["num_rays", "success_rate", "arrival_rate", "collision_rate"]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise RuntimeError(f"CSV missing required columns: {missing_cols}")

# Keep only valid rows.
df = df.copy()
df["num_rays"] = pd.to_numeric(df["num_rays"], errors="coerce")
for metric_key in metrics.keys():
    df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce")

df = df.dropna(subset=required_cols)
if df.empty:
    raise RuntimeError("No valid rows after filtering. Please check the CSV content.")

# ============================================================
# Aggregate mean / std by num_rays
# ============================================================
agg_dict = {}
for metric_key in metrics.keys():
    agg_dict[metric_key] = ["mean", "std", "count"]

summary = (
    df
    .groupby("num_rays")
    .agg(agg_dict)
    .reset_index()
    .sort_values("num_rays")
)

x = summary["num_rays"].values.astype(float)
x_ticks = sorted(df["num_rays"].unique())

# ============================================================
# Plot three metrics in one figure
# ============================================================
plt.figure(figsize=(8.5, 5.2))

for metric_key, style in metrics.items():
    mean = summary[(metric_key, "mean")].fillna(0.0).values
    std = summary[(metric_key, "std")].fillna(0.0).values
    count = summary[(metric_key, "count")].fillna(0).values

    plt.plot(
        x,
        mean,
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=2.2,
        markersize=6.0,
        label=style["label"],
    )

    # Only meaningful if you have repeated independent runs for the same num_rays.
    if show_std_band and (count > 1).any():
        lower = (mean - std).clip(min=0.0)
        upper = (mean + std).clip(max=1.0)
        plt.fill_between(x, lower, upper, alpha=0.08)

plt.xlabel("Number of lidar rays", fontsize=20)
plt.ylabel("Rate", fontsize=20)
plt.xticks(x_ticks, fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0.0, 1.05)

# No grid and no title, consistent with your figure style.
plt.grid(False)
plt.legend(frameon=True, fontsize=15, loc="best")
plt.tight_layout()

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()

print(f"saved: {out_png}")