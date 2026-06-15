#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/home/guo/L-CDF/src/sensor_cdf/scripts/saved_evaldata/eval_metrics_repair.csv"
out_dir = "/home/guo/L-CDF/image/eval"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(csv_path)

summary = df.groupby("num_demos").agg({
    "success_rate": ["mean", "std"],
    "arrival_rate": ["mean", "std"],
    "collision_rate": ["mean", "std"],
    "avg_collision_events": ["mean", "std"],
}).reset_index()

x = summary["num_demos"].values

plot_items = [
    ("success_rate", "Success Rate", "success_rate_curve.png", 0.0, 1.05),
    ("arrival_rate", "Arrival Rate", "arrival_rate_curve.png", 0.0, 1.05),
    ("collision_rate", "Collision Rate", "collision_rate_curve.png", 0.0, 1.05),
    ("avg_collision_events", "Average Collision Events", "avg_collision_events_curve.png", None, None),
]

for key, label, filename, ymin, ymax in plot_items:
    mean = summary[(key, "mean")].fillna(0.0).values
    std = summary[(key, "std")].fillna(0.0).values

    plt.figure(figsize=(7, 5))
    plt.plot(x, mean, marker="o", linewidth=2.5, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.25)
    plt.xlabel("Demonstrations")
    plt.ylabel(label)
    plt.title(f"{label} vs Demonstrations")
    plt.grid(True, linestyle=":", alpha=0.6)
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"saved: {save_path}")

# 三个 rate 放在一张图
plt.figure(figsize=(8, 5))
for key, label in [
    ("success_rate", "Success Rate"),
    ("arrival_rate", "Arrival Rate"),
    ("collision_rate", "Collision Rate"),
]:
    mean = summary[(key, "mean")].fillna(0.0).values
    std = summary[(key, "std")].fillna(0.0).values
    plt.plot(x, mean, marker="o", linewidth=2.5, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.18)

plt.xlabel("Demonstrations")
plt.ylabel("Rate")
plt.title("Policy Evaluation vs Demonstrations")
plt.grid(True, linestyle=":", alpha=0.6)
plt.ylim(0.0, 1.05)
plt.legend()
plt.tight_layout()
save_path = os.path.join(out_dir, "all_rates_curve.png")
plt.savefig(save_path, dpi=300)
plt.close()
print(f"saved: {save_path}")