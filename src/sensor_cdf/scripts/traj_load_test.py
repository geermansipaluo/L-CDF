#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
# ============================================================
# 1. Path configuration
# ============================================================
trajectory_dir = "/home/guo/L-CDF/src/sensor_cdf/scripts/eval_trajectories"
out_dir = "/home/guo/L-CDF/image/eval/trajectory_compare"
os.makedirs(out_dir, exist_ok=True)

# Fixed target used in your current evaluation.
# If target_mode is fixed_y0, change this to [15.0, 0.0].
target_pos = np.array([15.0, -1.0], dtype=np.float32)

# Which episode to draw from each saved file.
# If the selected episode is empty or missing, the first non-empty one is used.
run_index = 0

# Keep this False for paper figures. True is only useful for debugging all episodes.
plot_all_episodes = False

# For clarity: do not draw point markers along every trajectory.
show_path_markers = False

# Put legend outside the axes to reduce visual clutter.
legend_outside = False

# ============================================================
# 2. Obstacle and robot size configuration
# ============================================================
# Fixed evaluation scene. Each item is (x, y, obstacle_radius).
obstacles = [
    (5.0, 0.05, 0.5),
    (6.5, -0.5, 0.5),
    (10.0, -0.5, 0.5),
]

# Jackal / ego radius used for collision audit.
r_ego = 0.31

# Draw inflated obstacle boundary. If the robot center trajectory enters this circle,
# then the robot body overlaps the obstacle.
draw_collision_boundary = True

# Mark sampled trajectory points that are inside the inflated obstacle boundary.
draw_collision_samples = True

# Draw sparse ego footprints to show robot radius.
# To avoid making the ablation figure too messy, only selected labels are drawn.
draw_ego_footprints = False
num_footprints_per_curve = 6
footprint_labels_ablation = {"Full", "No safety"}
footprint_labels_comparison = {"L-CDF", "Pure BC", "PointNet++", "GP-CBF"}

# ============================================================
# 3. Trajectory files
# ============================================================
ablation_files = {
    "Full": "trajectory_densitynet_mode-full_ablation-full_runtime-qpth_target-fixed.pt",
    "No dual": "trajectory_densitynet_mode-wo_dual_ablation-no_dual_runtime-qpth_target-fixed.pt",
    "No lambda": "trajectory_densitynet_mode-wo_lambda_ablation-full_runtime-qpth_target-fixed.pt",
    "No learnable": "trajectory_densitynet_mode-wo_learnable_ablation-full_runtime-qpth_target-fixed.pt",
    "No safety": "trajectory_densitynet_mode-wo_safety_ablation-no_safety_runtime-nominal_target-fixed.pt",
    "No loss": "trajectory_densitynet_mode-ablation_no_loss_ablation-no_loss_runtime-qpth_target-fixed.pt",
}

comparison_files = {
    "L-CDF": "trajectory_densitynet_mode-full_ablation-full_runtime-qpth_target-fixed.pt",
    # Pure BC and no_safety are the same execution mode, so reuse the no_safety trajectory.
    "Pure BC": "trajectory_densitynet_mode-wo_safety_ablation-no_safety_runtime-nominal_target-fixed.pt",
    "PointNet++": "trajectory_pointnet2_mode-wo_safety_ablation-no_safety_runtime-nominal_target-fixed.pt",
    "GP-CBF": "trajectory_mode-gpcbf_pc256_target-fixed.pt",
}

# Line style configuration. The ablation figure is intentionally lighter to reduce clutter.
line_styles = {
    "Full": {"linewidth": 2.5, "linestyle": "-", "alpha": 1.0, "zorder": 4},
    "No dual": {"linewidth": 1.8, "linestyle": "--", "alpha": 0.85, "zorder": 3},
    "No lambda": {"linewidth": 1.8, "linestyle": "-.", "alpha": 0.85, "zorder": 3},
    "No learnable": {"linewidth": 1.8, "linestyle": ":", "alpha": 0.85, "zorder": 3},
    "No safety": {"linewidth": 2.2, "linestyle": "--", "alpha": 0.95, "zorder": 4},
    "No loss": {"linewidth": 1.8, "linestyle": "-.", "alpha": 0.85, "zorder": 3},
    "L-CDF": {"linewidth": 2.5, "linestyle": "-", "alpha": 1.0, "zorder": 5},
    "Pure BC": {"linewidth": 2.2, "linestyle": "--", "alpha": 0.95, "zorder": 4},
    "PointNet++": {"linewidth": 2.2, "linestyle": "-.", "alpha": 0.95, "zorder": 4},
    "GP-CBF": {"linewidth": 2.2, "linestyle": ":", "alpha": 0.95, "zorder": 4},
}

# ============================================================
# 4. Utilities
# ============================================================
def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_trajectory_file(path):
    data = safe_torch_load(path)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy().tolist()
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Unsupported trajectory data type: {type(data)}")
    return list(data)


def select_trajectory(all_runs, preferred_index=0):
    if len(all_runs) == 0:
        return None
    if 0 <= preferred_index < len(all_runs) and len(all_runs[preferred_index]) > 0:
        return all_runs[preferred_index]
    for traj in all_runs:
        if len(traj) > 0:
            return traj
    return None


def trajectory_to_xy(trajectory):
    arr = np.asarray(trajectory, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return None, None
    return arr[:, 0], arr[:, 1]


def collision_mask_for_xy(x, y):
    mask = np.zeros_like(x, dtype=bool)
    for ox, oy, radius in obstacles:
        inflated_radius = radius + r_ego
        dist = np.hypot(x - ox, y - oy)
        mask |= dist <= inflated_radius
    return mask


def draw_obstacles(ax):
    obstacle_labeled = False
    boundary_labeled = False

    for ox, oy, radius in obstacles:
        if draw_collision_boundary:
            boundary = Circle(
                (ox, oy),
                radius + r_ego,
                fill=False,
                linestyle="--",
                linewidth=1.3,
                alpha=0.8,
                label="Collision boundary" if not boundary_labeled else None,
                zorder=1,
            )
            ax.add_patch(boundary)
            boundary_labeled = True

        obs = Circle(
            (ox, oy),
            radius,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=1.2,
            alpha=0.9,
            label="Obstacle" if not obstacle_labeled else None,
            zorder=2,
        )
        ax.add_patch(obs)
        obstacle_labeled = True


def draw_start_and_goal(ax, start_xy=None):
    if start_xy is None:
        start_xy = np.array([0.0, 0.0], dtype=np.float32)

    ax.scatter(
        [start_xy[0]],
        [start_xy[1]],
        marker="o",
        s=90,
        label="Start",
        zorder=8,
        color = 'red'
    )
    ax.scatter(
        [target_pos[0]],
        [target_pos[1]],
        marker="*",
        s=90,
        label="Goal",
        zorder=8,
        color = 'red'
    )


def draw_ego_footprints_for_curve(ax, x, y, label, allowed_labels, add_label_flag):
    if not draw_ego_footprints or label not in allowed_labels or len(x) == 0:
        return add_label_flag

    if len(x) <= num_footprints_per_curve:
        idxs = np.arange(len(x), dtype=int)
    else:
        idxs = np.linspace(0, len(x) - 1, num_footprints_per_curve, dtype=int)

    for j, idx in enumerate(idxs):
        footprint = Circle(
            (float(x[idx]), float(y[idx])),
            r_ego,
            fill=False,
            linestyle=":",
            linewidth=0.9,
            alpha=0.32,
            label="Ego footprint" if add_label_flag else None,
            zorder=3,
        )
        ax.add_patch(footprint)
        add_label_flag = False

    return add_label_flag


def mark_collision_samples(ax, x, y, add_label_flag):
    if not draw_collision_samples or len(x) == 0:
        return add_label_flag

    mask = collision_mask_for_xy(x, y)
    if not np.any(mask):
        return add_label_flag

    # Downsample collision samples if there are too many, otherwise the figure becomes noisy.
    idxs = np.where(mask)[0]
    if len(idxs) > 60:
        idxs = idxs[:: max(len(idxs) // 60, 1)]

    # 每隔 collision_marker_stride 个点画一个 x
    collision_marker_stride = 10
    idxs_plot = idxs[::collision_marker_stride]

    # ax.scatter(
    #     x[idxs_plot],
    #     y[idxs_plot],
    #     marker="x",
    #     s=30,
    #     linewidths=1.1,
    #     label="Collision samples" if add_label_flag else None,
    #     zorder=9,
    # )
    return False


def load_group(group_files):
    loaded = {}
    for label, filename in group_files.items():
        path = os.path.join(trajectory_dir, filename)
        if not os.path.exists(path):
            print(f"[WARN] Missing file, skip {label}: {path}")
            continue
        try:
            all_runs = load_trajectory_file(path)
            main_traj = select_trajectory(all_runs, run_index)
            if main_traj is None:
                print(f"[WARN] Empty trajectory, skip {label}: {path}")
                continue
            loaded[label] = {
                "path": path,
                "all_runs": all_runs,
                "main_traj": main_traj,
            }
            print(f"[OK] Loaded {label}: {len(all_runs)} episodes -> {path}")
        except Exception as e:
            print(f"[WARN] Failed to load {label}: {path}, error={e}")

    if len(loaded) == 0:
        raise RuntimeError("No trajectory files loaded. Please check trajectory_dir and file names.")
    return loaded


def plot_group(group_data, save_name, footprint_labels, legend_ncol=2):
    fig, ax = plt.subplots(figsize=(9.4, 5.6))

    draw_obstacles(ax)

    first_start = None
    collision_label_needed = True
    footprint_label_needed = True

    for label, item in group_data.items():
        style = line_styles.get(label, {"linewidth": 2.0, "linestyle": "-", "alpha": 0.9, "zorder": 4})

        if plot_all_episodes:
            for traj in item["all_runs"]:
                x_all, y_all = trajectory_to_xy(traj)
                if x_all is None:
                    continue
                ax.plot(
                    x_all,
                    y_all,
                    linewidth=0.7,
                    linestyle=style.get("linestyle", "-"),
                    alpha=0.10,
                    zorder=2,
                )

        x, y = trajectory_to_xy(item["main_traj"])
        if x is None:
            continue

        if first_start is None and len(x) > 0:
            first_start = np.array([x[0], y[0]], dtype=np.float32)

        if show_path_markers:
            marker = "o"
            markersize = 2.0
            markevery = max(len(x) // 28, 1)
        else:
            marker = None
            markersize = 0
            markevery = None

        ax.plot(
            x,
            y,
            linewidth=style.get("linewidth", 2.0),
            linestyle=style.get("linestyle", "-"),
            alpha=style.get("alpha", 0.9),
            marker=marker,
            markersize=markersize,
            markevery=markevery,
            label=label,
            zorder=style.get("zorder", 4),
        )

        footprint_label_needed = draw_ego_footprints_for_curve(
            ax=ax,
            x=x,
            y=y,
            label=label,
            allowed_labels=footprint_labels,
            add_label_flag=footprint_label_needed,
        )

        collision_label_needed = mark_collision_samples(
            ax=ax,
            x=x,
            y=y,
            add_label_flag=collision_label_needed,
        )

    draw_start_and_goal(ax, first_start)

    ax.set_xlabel("X (m)", fontsize=20)
    ax.set_ylabel("Y (m)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.grid(False)
    ax.axis("equal")

    # Fixed view for your current scene.
    ax.set_xlim(-0.5, 16.0)
    ax.set_ylim(-3.2, 1.8)

    if legend_outside:
        ax.legend(
            frameon=True,
            fontsize=25,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            ncol=1,
        )
    else:
        ax.legend(frameon=True, fontsize=12, loc="best", ncol=legend_ncol)

    fig.tight_layout()
    save_path = os.path.join(out_dir, save_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save_path}")


def main():
    ablation_data = load_group(ablation_files)
    comparison_data = load_group(comparison_files)

    plot_group(
        ablation_data,
        save_name="trajectory_ablation_compare_clear_radius.png",
        footprint_labels=footprint_labels_ablation,
        legend_ncol=2,
    )

    plot_group(
        comparison_data,
        save_name="trajectory_method_compare_clear_radius.png",
        footprint_labels=footprint_labels_comparison,
        legend_ncol=2,
    )


if __name__ == "__main__":
    main()