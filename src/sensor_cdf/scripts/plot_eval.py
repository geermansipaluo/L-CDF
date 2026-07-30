#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
csv_root = "/home/guo/L-CDF/src/sensor_cdf/scripts/saved_evaldata"
# out_dir = "/home/guo/L-CDF/image/eval/ablation"
out_dir = "/home/guo/L-CDF/image/eval/compare"
os.makedirs(out_dir, exist_ok=True)

# 消融实验
# method_files = {
#     "Ours": os.path.join(csv_root, "eval_metrics_baseline.csv"),
#     "w/o dual branch": os.path.join(csv_root, "eval_metrics_nodual.csv"),
#     "w/o learnable lambda": os.path.join(csv_root, "eval_metrics_nolambda.csv"),
#     "w/o G/h loss": os.path.join(csv_root, "eval_metrics_noloss.csv"),
#     "w/o learnable CDF params": os.path.join(csv_root, "eval_metrics_nolearnable.csv"),
#     "w/o safety layer": os.path.join(csv_root, "eval_metrics_nosafe.csv"),
# }
# 对比实验
method_files = {
    "Ours": os.path.join(csv_root, "eval_metrics_baseline.csv"),
    "BC": os.path.join(csv_root, "eval_metrics_nosafe.csv"),
    "PointNet++": os.path.join(csv_root, "eval_metrics_pointnet2.csv")
}

# ============================================================
# 可视化开关
# ============================================================
show_std_band = True

# 多条曲线重叠时，给每种方法一个很小的 x 方向偏移。
# 只影响显示，不影响统计结果。
use_x_offset = True
x_offset_step = 0.12

markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

# ============================================================
# 读取 CSV
# ============================================================
all_df = []

for method_name, csv_path in method_files.items():
    if not os.path.exists(csv_path):
        print(f"[WARN] File not found, skip: {method_name} -> {csv_path}")
        continue

    df = pd.read_csv(csv_path)
    df["method"] = method_name
    all_df.append(df)

if len(all_df) == 0:
    raise RuntimeError("No CSV files loaded. Please check csv_root and file names.")

df_all = pd.concat(all_df, ignore_index=True)

required_cols = [
    "method",
    "num_demos",
    "success_rate",
    "arrival_rate",
    "collision_rate",
]
missing_cols = [c for c in required_cols if c not in df_all.columns]
if missing_cols:
    raise RuntimeError(f"CSV missing required columns: {missing_cols}")

# ============================================================
# 统计 mean / std
# ============================================================
summary = (
    df_all
    .groupby(["method", "num_demos"])
    .agg({
        "success_rate": ["mean", "std", "count"],
        "arrival_rate": ["mean", "std", "count"],
        "collision_rate": ["mean", "std", "count"],
    })
    .reset_index()
)

# summary_path = os.path.join(out_dir, "ablation_summary.csv")
summary_path = os.path.join(out_dir, "compare_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"saved: {summary_path}")

method_names = list(method_files.keys())
demo_ticks = sorted(df_all["num_demos"].unique())


def plot_metric(metric_key, y_label, filename, ymin=0.0, ymax=1.05):
    plt.figure(figsize=(8.5, 5.2))

    num_methods = len(method_names)
    center = (num_methods - 1) / 2.0

    for idx, method_name in enumerate(method_names):
        sub = summary[summary["method"] == method_name].sort_values("num_demos")
        if len(sub) == 0:
            continue

        x = sub["num_demos"].values.astype(float)
        mean = sub[(metric_key, "mean")].fillna(0.0).values
        std = sub[(metric_key, "std")].fillna(0.0).values

        if use_x_offset:
            x_plot = x + (idx - center) * x_offset_step
        else:
            x_plot = x

        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]

        plt.plot(
            x_plot,
            mean,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.2,
            markersize=6.0,
            label=method_name,
        )

        if show_std_band:
            lower = mean - std
            upper = mean + std

            if ymin is not None:
                lower = lower.clip(min=ymin)
            if ymax is not None:
                upper = upper.clip(max=ymax)

            plt.fill_between(
                x_plot,
                lower,
                upper,
                alpha=0.08,
            )

    plt.xlabel("Demonstrations", fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.xticks(demo_ticks, fontsize=20)
    plt.yticks(fontsize=20)

    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)

    # 不加网格
    plt.grid(False)

    # 不加标题
    # plt.title(...)

    plt.legend(
        frameon=True,
        fontsize=15,
        loc="right",
    )

    plt.tight_layout()

    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved: {save_path}")


plot_metric(
    metric_key="success_rate",
    y_label="Success rate",
    # filename="ablation_success_rate_curve.png",
    filename="compare_success_rate_curve.png",
)

plot_metric(
    metric_key="collision_rate",
    y_label="Collision rate",
    # filename="ablation_collision_rate_curve.png",
    filename="compare_collision_rate_curve.png",
)

plot_metric(
    metric_key="arrival_rate",
    y_label="Arrival rate",
    # filename="ablation_arrival_rate_curve.png",
    filename="compare_arrival_rate_curve.png",
)