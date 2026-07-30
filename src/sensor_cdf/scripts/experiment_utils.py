#!/usr/bin/env python3
"""Pure-Python helpers for reproducible LCDF experiments.

This module deliberately has no ROS dependency so scenario generation and
result aggregation can be checked without launching Gazebo.
"""

import csv
import json
import math
import os
from datetime import datetime

import numpy as np


def to_builtin(value):
    """Convert numpy/torch-like values into JSON-serializable Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return to_builtin(value.detach().cpu().numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, ensure_ascii=False, indent=2, sort_keys=True)


def write_csv(path, rows):
    rows = [to_builtin(row) for row in rows]
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_random_circle_scenarios(
    num_scenarios,
    seed,
    targets,
    min_obstacles=3,
    max_obstacles=6,
    obstacle_radius=0.25,
    robot_radius=0.31,
    x_min=2.0,
    x_max=13.0,
    y_min=-3.0,
    y_max=3.0,
    min_center_separation=1.35,
    start_clearance=1.5,
    goal_clearance=1.2,
    force_path_obstacle=True,
):
    """Generate deterministic, individually reproducible circular scenes.

    Each scene uses ``seed + scene_id``. The first obstacle is sampled near the
    straight start-to-goal line so the benchmark is not dominated by empty,
    trivial paths. Remaining obstacles are sampled uniformly with clearance
    checks. The returned scenes can be regenerated independently.
    """
    num_scenarios = int(num_scenarios)
    targets = np.asarray(targets, dtype=np.float32)
    if targets.shape != (num_scenarios, 2):
        raise ValueError(
            f"targets must have shape ({num_scenarios}, 2), got {targets.shape}"
        )
    if int(min_obstacles) < 1 or int(max_obstacles) < int(min_obstacles):
        raise ValueError("invalid obstacle count range")
    if int(max_obstacles) > 8:
        raise ValueError("Gazebo world only provides cylinder_0 ... cylinder_7")

    start = np.array([0.0, 0.0], dtype=np.float64)
    scenarios = []
    for scene_id in range(num_scenarios):
        scene_seed = int(seed) + scene_id
        rng = np.random.default_rng(scene_seed)
        target = targets[scene_id].astype(np.float64)
        count = int(rng.integers(int(min_obstacles), int(max_obstacles) + 1))
        centers = []

        for obstacle_id in range(count):
            accepted = None
            for _ in range(5000):
                if obstacle_id == 0 and force_path_obstacle:
                    x = float(rng.uniform(max(float(x_min), 3.0), min(float(x_max), 12.0)))
                    line_y = float(target[1] * (x / max(float(target[0]), 1e-6)))
                    y = float(np.clip(line_y + rng.uniform(-0.65, 0.65), y_min, y_max))
                else:
                    x = float(rng.uniform(x_min, x_max))
                    y = float(rng.uniform(y_min, y_max))
                candidate = np.array([x, y], dtype=np.float64)

                if np.linalg.norm(candidate - start) < float(start_clearance):
                    continue
                if np.linalg.norm(candidate - target) < float(goal_clearance):
                    continue
                if any(
                    np.linalg.norm(candidate - prev) < float(min_center_separation)
                    for prev in centers
                ):
                    continue
                accepted = candidate
                break

            if accepted is None:
                raise RuntimeError(
                    f"failed to sample scene={scene_id}, obstacle={obstacle_id}; "
                    "relax random-scene clearance parameters"
                )
            centers.append(accepted)

        scenarios.append(
            {
                "environment_id": scene_id,
                "seed": scene_seed,
                "start": [0.0, 0.0, 0.0],
                "target": target.astype(np.float32).tolist(),
                "obstacle_radius": float(obstacle_radius),
                "obstacles_static": True,
                "robot_radius": float(robot_radius),
                "obstacles": [
                    {
                        "model_name": f"cylinder_{i}",
                        "center": center.astype(np.float32).tolist(),
                        "radius": float(obstacle_radius),
                    }
                    for i, center in enumerate(centers)
                ],
            }
        )
    return scenarios


def _numeric_summary(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def summarize_episode_results(episodes):
    episodes = [to_builtin(ep) for ep in episodes]
    n = len(episodes)
    if n == 0:
        return {
            "num_completed_episodes": 0,
            "generated_at": datetime.now().astimezone().isoformat(),
        }

    count_true = lambda key: sum(bool(ep.get(key, False)) for ep in episodes)
    arrival_count = count_true("arrived")
    success_count = count_true("success")
    collision_count = count_true("collision")
    physical_collision_count = count_true("physical_collision")
    cdf_violation_count = count_true("cdf_envelope_violated")
    timeout_count = sum(ep.get("finish_reason") == "timeout" for ep in episodes)
    fallback_steps = sum(int(ep.get("qpth_fallback_steps", 0)) for ep in episodes)

    metric_keys = [
        "duration_sec",
        "path_length_m",
        "straight_line_distance_m",
        "direct_progress_m",
        "path_efficiency",
        "endpoint_error_m",
        "min_center_distance_m",
        "min_physical_clearance_m",
        "min_audit_clearance_m",
        "min_cdf_clearance_m",
        "mean_linear_speed_mps",
        "max_linear_speed_mps",
        "mean_abs_angular_speed_rps",
        "max_abs_angular_speed_rps",
        "mean_lidar_points",
        "min_lidar_range_m",
        "mean_control_correction",
        "max_control_correction",
        "mean_inference_ms",
        "p95_inference_ms",
        "max_inference_ms",
        "control_steps",
        "collision_events",
        "qpth_failures",
    ]

    return {
        "num_completed_episodes": n,
        "arrival_count": arrival_count,
        "success_count": success_count,
        "collision_count": collision_count,
        "physical_collision_count": physical_collision_count,
        "cdf_envelope_violation_count": cdf_violation_count,
        "timeout_count": timeout_count,
        "arrival_rate": arrival_count / n,
        "success_rate": success_count / n,
        "collision_rate": collision_count / n,
        "physical_collision_rate": physical_collision_count / n,
        "cdf_envelope_violation_rate": cdf_violation_count / n,
        "timeout_rate": timeout_count / n,
        "qpth_fallback_steps": fallback_steps,
        "metrics": {
            key: _numeric_summary(ep.get(key) for ep in episodes)
            for key in metric_keys
        },
        "generated_at": datetime.now().astimezone().isoformat(),
    }


def render_summary_markdown(summary, run_config):
    cfg = to_builtin(run_config)
    lines = [
        f"# LCDF随机环境评测结果：{cfg.get('run_id', 'unknown')}",
        "",
        "## 核心配置",
        "",
        f"- 实验版本：`{cfg.get('experiment_version')}`",
        f"- 完成时间：`{summary.get('generated_at')}`",
        f"- 模型：`{cfg.get('model_path')}`",
        f"- 场景模式：`{cfg.get('environment_mode')}`",
        f"- 场景数：{cfg.get('num_eval_episodes')}",
        f"- 起点：`[0, 0, 0]`",
        f"- 目标模式：`{cfg.get('target_mode')}`",
        f"- 物理机器人半径：{cfg.get('robot_physical_radius')} m",
        f"- CDF有效半径：{cfg.get('cdf_r_ego')} m",
        f"- CDF额外裕量：{cfg.get('cdf_safety_margin')} m",
        f"- 随机场景种子：{cfg.get('random_env_seed')}",
        "",
        "## 汇总指标",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 到达率 | {summary.get('arrival_rate', 0.0):.4f} |",
        f"| 成功率（按评测阈值零碰撞到达） | {summary.get('success_rate', 0.0):.4f} |",
        f"| 评测碰撞率 | {summary.get('collision_rate', 0.0):.4f} |",
        f"| 真实几何碰撞率 | {summary.get('physical_collision_rate', 0.0):.4f} |",
        f"| CDF安全包络违反率 | {summary.get('cdf_envelope_violation_rate', 0.0):.4f} |",
        f"| 超时率 | {summary.get('timeout_rate', 0.0):.4f} |",
        f"| qpth回退步数 | {summary.get('qpth_fallback_steps', 0)} |",
        "",
        "## 连续指标",
        "",
        "| 指标 | 均值 | 标准差 | 最小 | 最大 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, stat in summary.get("metrics", {}).items():
        def fmt(value):
            return "N/A" if value is None else f"{value:.6f}"
        lines.append(
            f"| {key} | {fmt(stat.get('mean'))} | {fmt(stat.get('std'))} | "
            f"{fmt(stat.get('min'))} | {fmt(stat.get('max'))} |"
        )
    lines.extend(
        [
            "",
            "## 文件说明",
            "",
            "- `config.json`：完整运行参数、模型路径、Git状态。",
            "- `scenarios.json`：20个随机场景的目标、障碍物和随机种子。",
            "- `episodes.csv` / `episodes.json`：逐场景指标。",
            "- `steps.jsonl`：逐控制周期状态、控制量、距离与推理耗时。",
            "- `summary.json` / `summary.md`：聚合结果。",
            "- `trajectories.pt`：兼容原绘图代码的轨迹数据。",
            "- `run.log`：ROS与终端完整日志，由启动脚本保存。",
            "",
        ]
    )
    return "\n".join(lines)
