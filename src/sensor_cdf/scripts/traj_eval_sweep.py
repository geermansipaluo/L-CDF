#!/usr/bin/env python3
import os
import sys
import csv
import signal
import time
import inspect

import rospy
import torch
import numpy as np

from std_msgs.msg import Float32MultiArray, Header
from geometry_msgs.msg import Twist
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from torch_geometric.data import Data, Batch

from model import UNet
from data_generate import LocalSdfCdfPlanner


def ros_bool(value, default=False):
    """Robust ROS bool parser: supports bool/int/float/'true'/'false'."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y", "t"):
        return True
    if value in ("false", "0", "no", "n", "f"):
        return False
    return bool(default)


def safe_torch_load(path, map_location):
    """兼容不同 torch 版本的权重加载。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_state_dict(loaded):
    """兼容直接保存 state_dict、或 checkpoint dict 的情况。"""
    if isinstance(loaded, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            val = loaded.get(key, None)
            if isinstance(val, dict):
                loaded = val
                break

    if not isinstance(loaded, dict):
        raise TypeError(f"checkpoint 不是 state_dict/dict，实际类型: {type(loaded)}")

    # 兼容 DataParallel 保存的 module.xxx 前缀。
    if any(str(k).startswith("module.") for k in loaded.keys()):
        loaded = {str(k).replace("module.", "", 1): v for k, v in loaded.items()}

    return loaded


class ParametricEllipseTracker:
    """DensityNet 单模型闭环评测节点。"""

    def __init__(self):
        self.start_time = time.time()
        rospy.init_node("parametric_ellipse_tracker_node", anonymous=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"当前 DensityNet 测试设备: [{self.device}]")

        # ============================================================
        # 1. ROS 私有参数：批量评测从 launch / shell 传入
        # ============================================================
        self.model_path = rospy.get_param(
            "~model_path",
            "/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models/new_loss/DensityNet-demo48-dseed0-seed0/model_best_parametric_bc.pt",
        )
        self.output_csv = rospy.get_param(
            "~output_csv",
            "/home/guo/L-CDF/src/sensor_cdf/scripts/eval_metrics.csv",
        )
        self.trajectory_save_dir = rospy.get_param(
            "~trajectory_save_dir",
            "/home/guo/L-CDF/src/sensor_cdf/scripts/eval_trajectories",
        )

        self.num_demos = int(rospy.get_param("~num_demos", -1))
        self.demo_seed = int(rospy.get_param("~demo_seed", -1))
        self.train_seed = int(rospy.get_param("~train_seed", -1))

        self.total_eval_episodes = int(rospy.get_param("~num_eval_episodes", 10))
        self.test_target_seed = int(rospy.get_param("~test_target_seed", 2026))
        self.target_mode = str(rospy.get_param("~target_mode", "fixed")).lower()
        self.fixed_target_x = float(rospy.get_param("~fixed_target_x", 15.0))
        self.fixed_target_y = float(rospy.get_param("~fixed_target_y", -1.1))
        self.target_x_min = float(rospy.get_param("~target_x_min", 14.0))
        self.target_x_max = float(rospy.get_param("~target_x_max", 16.0))
        self.target_y_min = float(rospy.get_param("~target_y_min", -2.0))
        self.target_y_max = float(rospy.get_param("~target_y_max", 2.0))
        self.goal_radius = float(rospy.get_param("~goal_radius", 0.4))
        self.max_episode_time = float(rospy.get_param("~max_episode_time", 80.0))
        self.hold_before_episode = float(rospy.get_param("~hold_before_episode", 2.0))
        self.terminate_on_collision = ros_bool(rospy.get_param("~terminate_on_collision", False), False)

        # ============================================================
        # 1.1 连续多回合 reset 防旧数据保护
        # ============================================================
        # reset_simulation 后，ROS subscriber 队列中可能仍有 reset 前的 pose/pointcloud。
        # 因此每次 reset 后先进入 hold，且只接受 sensor_accept_wall_time 之后到达的新消息。
        self.sensor_accept_delay = float(rospy.get_param("~sensor_accept_delay", 0.6))
        self.require_fresh_pose_after_reset = ros_bool(
            rospy.get_param("~require_fresh_pose_after_reset", True), True
        )
        self.require_fresh_cloud_after_reset = ros_bool(
            rospy.get_param("~require_fresh_cloud_after_reset", True), True
        )
        self.fresh_sensor_wait_timeout = float(rospy.get_param("~fresh_sensor_wait_timeout", 5.0))
        self.reset_stop_burst_count = int(rospy.get_param("~reset_stop_burst_count", 8))
        self.reset_stop_burst_dt = float(rospy.get_param("~reset_stop_burst_dt", 0.05))

        # 可选：reset 后显式把 robot model 放回原点。模型名按你的 Gazebo 世界设置；
        # 如果为空字符串则不做显式重置。
        self.robot_model_name = str(rospy.get_param("~robot_model_name", "jackal"))
        self.robot_reset_z = float(rospy.get_param("~robot_reset_z", 0.15))

        # ============================================================
        # 2. 与当前训练 / model.py 对齐的网络结构参数
        # ============================================================
        self.model_state_dim = int(rospy.get_param("~state_dim", 4))
        self.model_hidden_dim = int(rospy.get_param("~hidden_dim", 256))
        self.model_graph_k = int(rospy.get_param("~graph_k", 5))
        self.model_lambda_smooth = float(rospy.get_param("~lambda_smooth", 25.0))
        self.model_qp_limit = float(rospy.get_param("~qp_limit", 1.2))
        self.model_ablation = str(rospy.get_param("~ablation", "full"))
        self.nominal_speed = float(rospy.get_param("~nominal_speed", 1.2))

        # runtime_qp_mode:
        #   qpth    : 完整评估当前模型内部 learned PyTorch CDF-G/h + qpth safety layer
        #   jax     : 网络 head 输出 u_nom，再调用 JAX/ProxQP 输出最终 u_safe；用于稳定部署对照
        #   nominal : 只执行网络 head 输出，不做安全投影；用于 no_safety/BC 对照
        self.runtime_qp_mode = str(rospy.get_param("~runtime_qp_mode", "jax")).lower()
        if self.runtime_qp_mode not in ("qpth", "jax", "nominal"):
            rospy.logwarn(f"未知 runtime_qp_mode={self.runtime_qp_mode}，自动改成 qpth")
            self.runtime_qp_mode = "qpth"

        # qpth safety layer 参数。注意这些不会进入 state_dict，但必须和训练时的结构/数值设定一致。
        self.use_qp_box_constraints = ros_bool(rospy.get_param("~use_qp_box_constraints", True), True)
        self.qp_jitter = float(rospy.get_param("~qp_jitter", 1e-4))
        self.qp_normalize_constraints = ros_bool(rospy.get_param("~qp_normalize_constraints", False), False)
        self.qp_constraint_scale_floor = float(rospy.get_param("~qp_constraint_scale_floor", 1.0))
        self.qp_box_eps = float(rospy.get_param("~qp_box_eps", 1e-4))
        self.qp_max_iter = int(rospy.get_param("~qp_max_iter", 100))
        self.qp_eps = float(rospy.get_param("~qp_eps", 1e-4))
        self.qp_not_improved_lim = int(rospy.get_param("~qp_not_improved_lim", 20))
        self.qp_fail_mode = str(rospy.get_param("~qp_fail_mode", "fallback")).lower()
        self.qp_debug_max_print = int(rospy.get_param("~qp_debug_max_print", 20))
        self.qp_check_invalid_constraints = ros_bool(rospy.get_param("~qp_check_invalid_constraints", True), True)
        self.qp_invalid_g_norm_eps = float(rospy.get_param("~qp_invalid_g_norm_eps", 1e-8))
        self.qp_invalid_h_eps = float(rospy.get_param("~qp_invalid_h_eps", 1e-6))
        self.qp_invalid_constraint_mode = str(rospy.get_param("~qp_invalid_constraint_mode", "warn")).lower()
        self.qp_invalid_debug_max_print = int(rospy.get_param("~qp_invalid_debug_max_print", 20))
        self.qp_sanitize_redundant_constraints = ros_bool(rospy.get_param("~qp_sanitize_redundant_constraints", True), True)
        self.qp_redundant_constraint_h = float(rospy.get_param("~qp_redundant_constraint_h", 1.0))
        self.qp_verify_solution = ros_bool(rospy.get_param("~qp_verify_solution", True), True)
        self.qp_solution_violation_tol = float(rospy.get_param("~qp_solution_violation_tol", 1e-3))
        self.qp_solution_debug_max_print = int(rospy.get_param("~qp_solution_debug_max_print", 20))
        self.qp_suppress_qpth_warnings = ros_bool(rospy.get_param("~qp_suppress_qpth_warnings", True), True)

        # checkpoint 如果包含 lambda_raw/lambda_prior，会在 load_model() 中自动强制打开。
        self.learnable_lambda_smooth = ros_bool(rospy.get_param("~learnable_lambda_smooth", False), False)
        self.lambda_smooth_min = float(rospy.get_param("~lambda_smooth_min", 0.1))
        self.lambda_smooth_max = float(rospy.get_param("~lambda_smooth_max", 50.0))
        self.lambda_reg_weight = float(rospy.get_param("~lambda_reg_weight", 1e-4))
        self.qpth_fail_fallback_to_jax = ros_bool(rospy.get_param("~qpth_fail_fallback_to_jax", True), True)

        # ============================================================
        # 2.1 端到端 PyTorch CDF-G/h 构造参数：与 learnable 训练版本对齐
        # ============================================================
        self.use_learned_cdf_constraints = ros_bool(rospy.get_param("~use_learned_cdf_constraints", True), True)
        self.cdf_l_k = float(rospy.get_param("~cdf_l_k", 0.33))
        self.cdf_r_ego = float(rospy.get_param("~cdf_r_ego", 0.31))
        self.cdf_sense_range = float(rospy.get_param("~cdf_sense_range", 3.0))
        self.cdf_alpha_init = float(rospy.get_param("~cdf_alpha_init", 0.25))
        self.cdf_alpha_min = float(rospy.get_param("~cdf_alpha_min", 0.10))
        self.cdf_alpha_max = float(rospy.get_param("~cdf_alpha_max", 0.55))
        self.learnable_cdf_alpha = ros_bool(rospy.get_param("~learnable_cdf_alpha", True), True)
        self.cdf_epsilon_init = float(rospy.get_param("~cdf_epsilon_init", 0.1))
        self.cdf_epsilon_min = float(rospy.get_param("~cdf_epsilon_min", 0.05))
        self.cdf_epsilon_max = float(rospy.get_param("~cdf_epsilon_max", 0.20))
        self.learnable_cdf_epsilon = ros_bool(rospy.get_param("~learnable_cdf_epsilon", True), True)
        self.cdf_rho_floor_init = float(rospy.get_param("~cdf_rho_floor_init", 0.0))
        self.learnable_cdf_rho_floor = ros_bool(rospy.get_param("~learnable_cdf_rho_floor", False), False)
        self.cdf_margin_init = float(rospy.get_param("~cdf_margin_init", 0.0))
        self.learnable_cdf_margin = ros_bool(rospy.get_param("~learnable_cdf_margin", False), False)
        self.cdf_valid_point_abs_max = float(rospy.get_param("~cdf_valid_point_abs_max", 50.0))
        self.cdf_padding_value = float(rospy.get_param("~cdf_padding_value", 99.0))
        self.lambda_gh = float(rospy.get_param("~lambda_gh", 0.001))

        os.makedirs(self.trajectory_save_dir, exist_ok=True)

        # ============================================================
        # 3. 控制和评估配置：与当前 traj_generate.py 的 u=[v,L*w] 表示保持一致
        # ============================================================
        self.cbf_config = {
            "l_k": 0.33,
            "r_ego": 0.31,
            "v_min": -1.2,
            "v_max": 1.2,
            "w_min": -2.0,
            "w_max": 2.0,
        }

        self.current_pose = [0.0, 0.0, 0.0]
        self.pointcloud_local = np.zeros((0, 2), dtype=np.float32)
        self.last_executed_v = 0.0
        self.last_executed_w = 0.0      # 这里存的是 u_y=L*omega，不是 omega

        # reset / fresh sensor 状态量。首次启动也要求在 hold 后收到新 pose/cloud。
        now_sensor = time.time()
        self.is_resetting = False
        self.reset_wall_time = now_sensor
        self.sensor_accept_wall_time = now_sensor + self.sensor_accept_delay
        self.last_pose_wall_time = 0.0
        self.last_cloud_wall_time = 0.0
        self.last_cloud_num_points = 0

        self.test_targets = self.build_test_targets()
        self.episode_index = 0
        self.target_pos = self.test_targets[self.episode_index].tolist()
        now = time.time()
        self.episode_hold_until = now + self.hold_before_episode
        self.current_episode_start_time = self.episode_hold_until
        self.episode_finish_lock = False

        # 固定评测场景中的圆柱障碍物中心；用于碰撞统计。
        self.obstacles = np.array(
            [
                [5.0, 0.05],
                [6.5, -0.5],
                # [8.0, -2.5],
                [10.0, -0.5],
            ],
            dtype=np.float32,
        )
        self.safety_threshold = 0.5 + self.cbf_config["r_ego"]

        self.all_runs_trajectories = []
        self.current_run_trajectory = []
        self.reached_goals_count = 0
        self.perfect_runs_count = 0
        self.collision_runs_count = 0
        self.timeout_runs_count = 0
        self.total_collision_events = 0
        self.collision_happened_in_current_run = False
        self.last_collision_time = 0.0

        # JAX 专家只用于：根据网络 u_nom 和当前局部点云生成 SDF-CDF-QP 的 G/h；
        # runtime_qp_mode='jax' 时也可用它作为最终安全投影。
        self.local_expert = LocalSdfCdfPlanner()
        self.warmup_expert()

        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.debug_pub = rospy.Publisher("/processed_cloud", PointCloud2, queue_size=1)
        self.marker_pub = rospy.Publisher("/target_goal_marker", Marker, queue_size=1)

        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=5.0)
            self.set_model_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            rospy.loginfo("Gazebo /set_model_state 服务已连接")
        except Exception as e:
            self.set_model_state_srv = None
            rospy.logwarn(f"Gazebo /set_model_state 服务不可用: {e}")

        self.apply_fixed_obstacles_to_gazebo()
        self.model = self.load_model()

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.__sub_curr_state = rospy.Subscriber(
            "/curr_state", Float32MultiArray, self.pose_callback, queue_size=1
        )
        self.__sub_global_cloud = rospy.Subscriber(
            "/densitynet_input_points", PointCloud2, self.cloud_callback, queue_size=1
        )

        self.ctrl_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        self.goal_timer = rospy.Timer(rospy.Duration(0.05), self.check_reach_goal)

        rospy.loginfo("=" * 70)
        rospy.loginfo("DensityNet 自动批量评测节点启动")
        rospy.loginfo(f"model_path              = {self.model_path}")
        rospy.loginfo(f"num_demos/dseed/seed    = {self.num_demos}/{self.demo_seed}/{self.train_seed}")
        rospy.loginfo(f"runtime_qp_mode         = {self.runtime_qp_mode}")
        rospy.loginfo(f"graph_k/hidden/lambda   = {self.model_graph_k}/{self.model_hidden_dim}/{self.model_lambda_smooth}")
        rospy.loginfo(f"box/normalize/fail_mode = {self.use_qp_box_constraints}/{self.qp_normalize_constraints}/{self.qp_fail_mode}")
        rospy.loginfo(f"target_mode             = {self.target_mode}")
        rospy.loginfo(f"first_target            = {self.target_pos}")
        rospy.loginfo(f"num_eval_episodes       = {self.total_eval_episodes}")
        rospy.loginfo(f"output_csv              = {self.output_csv}")
        rospy.loginfo("=" * 70)

    # ============================================================
    # 目标采样
    # ============================================================
    def build_test_targets(self):
        rng = np.random.default_rng(self.test_target_seed)
        mode = self.target_mode

        if mode == "random":
            targets = np.stack(
                [
                    rng.uniform(self.target_x_min, self.target_x_max, size=self.total_eval_episodes),
                    rng.uniform(self.target_y_min, self.target_y_max, size=self.total_eval_episodes),
                ],
                axis=1,
            ).astype(np.float32)
        elif mode == "random_y0":
            xs = rng.uniform(self.target_x_min, self.target_x_max, size=self.total_eval_episodes)
            ys = np.zeros(self.total_eval_episodes, dtype=np.float32)
            targets = np.stack([xs, ys], axis=1).astype(np.float32)
        elif mode == "fixed":
            fixed_target = np.array([self.fixed_target_x, self.fixed_target_y], dtype=np.float32)
            targets = np.tile(fixed_target[None, :], (self.total_eval_episodes, 1)).astype(np.float32)
        elif mode == "fixed_y0":
            fixed_target = np.array([self.fixed_target_x, 0.0], dtype=np.float32)
            targets = np.tile(fixed_target[None, :], (self.total_eval_episodes, 1)).astype(np.float32)
        else:
            rospy.logwarn(f"未知 target_mode={mode}，自动使用 fixed_y0")
            fixed_target = np.array([self.fixed_target_x, 0.0], dtype=np.float32)
            targets = np.tile(fixed_target[None, :], (self.total_eval_episodes, 1)).astype(np.float32)

        return targets

    # ============================================================
    # 工具函数
    # ============================================================
    def warmup_expert(self):
        rospy.loginfo("JAX 专家 QP 热身中...")
        t_warm = time.time()
        fake_ego_p = np.array([self.cbf_config["l_k"], 0.0], dtype=np.float32)
        fake_u_nom = np.array([0.5, 0.0], dtype=np.float32)
        fake_pts = np.random.uniform(-1.0, 1.0, (200, 2)).astype(np.float32)
        fake_target = np.array([5.0, 0.0], dtype=np.float32)
        _, _, _ = self.local_expert.solve_agent_qp_local(fake_ego_p, fake_u_nom, fake_pts, fake_target)
        rospy.loginfo(f"JAX 专家 QP 热身完成，耗时 {time.time() - t_warm:.2f}s")

    def resolve_model_path(self, path):
        if os.path.isdir(path):
            candidate = os.path.join(path, "model_best_parametric_bc.pt")
            if os.path.exists(candidate):
                return candidate
        return path

    def build_model_kwargs(self):
        candidate_kwargs = {
            "state_dim": self.model_state_dim,
            "hidden_dim": self.model_hidden_dim,
            "graph_k": self.model_graph_k,
            "lambda_smooth": self.model_lambda_smooth,
            "qp_limit": self.model_qp_limit,
            "use_qp_box_constraints": self.use_qp_box_constraints,
            "qp_jitter": self.qp_jitter,
            "qp_normalize_constraints": self.qp_normalize_constraints,
            "qp_constraint_scale_floor": self.qp_constraint_scale_floor,
            "qp_box_eps": self.qp_box_eps,
            "qp_max_iter": self.qp_max_iter,
            "qp_eps": self.qp_eps,
            "qp_not_improved_lim": self.qp_not_improved_lim,
            "qp_fail_mode": self.qp_fail_mode,
            "qp_debug_max_print": self.qp_debug_max_print,
            "qp_check_invalid_constraints": self.qp_check_invalid_constraints,
            "qp_invalid_g_norm_eps": self.qp_invalid_g_norm_eps,
            "qp_invalid_h_eps": self.qp_invalid_h_eps,
            "qp_invalid_constraint_mode": self.qp_invalid_constraint_mode,
            "qp_invalid_debug_max_print": self.qp_invalid_debug_max_print,
            "qp_sanitize_redundant_constraints": self.qp_sanitize_redundant_constraints,
            "qp_redundant_constraint_h": self.qp_redundant_constraint_h,
            "qp_verify_solution": self.qp_verify_solution,
            "qp_solution_violation_tol": self.qp_solution_violation_tol,
            "qp_solution_debug_max_print": self.qp_solution_debug_max_print,
            "qp_suppress_qpth_warnings": self.qp_suppress_qpth_warnings,
            "learnable_lambda_smooth": self.learnable_lambda_smooth,
            "lambda_smooth_min": self.lambda_smooth_min,
            "lambda_smooth_max": self.lambda_smooth_max,
            "lambda_reg_weight": self.lambda_reg_weight,
            "use_learned_cdf_constraints": self.use_learned_cdf_constraints,
            "cdf_l_k": self.cdf_l_k,
            "cdf_r_ego": self.cdf_r_ego,
            "cdf_sense_range": self.cdf_sense_range,
            "cdf_alpha_init": self.cdf_alpha_init,
            "cdf_alpha_min": self.cdf_alpha_min,
            "cdf_alpha_max": self.cdf_alpha_max,
            "cdf_epsilon_init": self.cdf_epsilon_init,
            "cdf_epsilon_min": self.cdf_epsilon_min,
            "cdf_epsilon_max": self.cdf_epsilon_max,
            "cdf_rho_floor_init": self.cdf_rho_floor_init,
            "cdf_margin_init": self.cdf_margin_init,
            "learnable_cdf_alpha": self.learnable_cdf_alpha,
            "learnable_cdf_epsilon": self.learnable_cdf_epsilon,
            "learnable_cdf_rho_floor": self.learnable_cdf_rho_floor,
            "learnable_cdf_margin": self.learnable_cdf_margin,
            "cdf_valid_point_abs_max": self.cdf_valid_point_abs_max,
            "cdf_padding_value": self.cdf_padding_value,
            "gh_loss_weight": self.lambda_gh,
            "enable_timing_debug": False,
            "timing_sync_cuda": False,
            "ablation": self.model_ablation,
        }
        signature = inspect.signature(UNet.__init__)
        supported_keys = set(signature.parameters.keys())
        return {k: v for k, v in candidate_kwargs.items() if k in supported_keys}

    def load_model(self):
        model_path = self.resolve_model_path(self.model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        try:
            state_dict = extract_state_dict(safe_torch_load(model_path, map_location=self.device))
            ckpt_has_learnable_lambda = (
                "safety_layer.lambda_raw" in state_dict
                or "safety_layer.lambda_prior" in state_dict
            )
            ckpt_has_learned_cdf = any(str(k).startswith("cdf_constraint_layer.") for k in state_dict.keys())

            if ckpt_has_learned_cdf and not self.use_learned_cdf_constraints:
                rospy.logwarn(
                    "checkpoint 包含 cdf_constraint_layer.*，"
                    "自动强制 use_learned_cdf_constraints=True 以匹配模型结构。"
                )
                self.use_learned_cdf_constraints = True

            if ckpt_has_learnable_lambda and not self.learnable_lambda_smooth:
                rospy.logwarn(
                    "checkpoint 包含 safety_layer.lambda_raw/lambda_prior，"
                    "自动强制 learnable_lambda_smooth=True 以匹配模型结构。"
                )
                self.learnable_lambda_smooth = True
            elif (not ckpt_has_learnable_lambda) and self.learnable_lambda_smooth:
                rospy.logwarn(
                    "checkpoint 不包含 learnable lambda 参数，"
                    "自动强制 learnable_lambda_smooth=False 以避免 strict=True 缺 key。"
                )
                self.learnable_lambda_smooth = False

            model_kwargs = self.build_model_kwargs()
            model = UNet(**model_kwargs)

            try:
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as strict_e:
                rospy.logerr("strict=True 加载失败：通常是测试端 model.py 与训练端 model.py 不一致。")
                rospy.logerr(f"ckpt_has_learnable_lambda={ckpt_has_learnable_lambda}, ckpt_has_learned_cdf={ckpt_has_learned_cdf}")
                rospy.logerr(f"runtime_learnable_lambda_smooth={self.learnable_lambda_smooth}, use_learned_cdf={self.use_learned_cdf_constraints}")
                rospy.logerr(f"model_kwargs={model_kwargs}")
                raise strict_e

            model = model.to(self.device).eval()

            lambda_info = self.model_lambda_smooth
            try:
                if hasattr(model, "safety_layer") and hasattr(model.safety_layer, "lambda_smooth_value"):
                    lambda_info = model.safety_layer.lambda_smooth_value().detach().cpu().item()
            except Exception:
                pass

            cdf_info = {}
            try:
                if hasattr(model, "cdf_parameter_dict"):
                    cdf_info = model.cdf_parameter_dict()
            except Exception:
                cdf_info = {}

            rospy.loginfo(
                f"模型装载成功: {model_path}; mode={self.runtime_qp_mode}; "
                f"graph_k={self.model_graph_k}; hidden={self.model_hidden_dim}; "
                f"lambda={float(lambda_info):.4f}; learnable_lambda={self.learnable_lambda_smooth}; "
                f"use_learned_cdf={self.use_learned_cdf_constraints}; cdf={cdf_info}; "
                f"box={self.use_qp_box_constraints}; normalize={self.qp_normalize_constraints}"
            )
            return model
        except Exception as e:
            rospy.logerr(f"模型加载失败: {str(e)}")
            raise RuntimeError("模型加载错误") from e

    def make_points_batch(self, local_pts):
        pc_model = np.asarray(local_pts, dtype=np.float32)
        if pc_model.ndim != 2 or pc_model.shape[0] == 0:
            pc_model = np.zeros((2, 2), dtype=np.float32)

        # DynamicEdgeConv/knn_graph 要求每个 batch 至少 k+1 个点。
        min_points = max(2, int(self.model_graph_k) + 1)
        if pc_model.shape[0] < min_points:
            repeat_count = min_points - pc_model.shape[0]
            pad = np.repeat(pc_model[-1:, :], repeats=repeat_count, axis=0)
            pc_model = np.vstack([pc_model, pad]).astype(np.float32)

        pos_tensor = torch.tensor(pc_model, dtype=torch.float32)
        return Batch.from_data_list([Data(pos=pos_tensor)]).to(self.device)

    def predict_model_unom(self, state_tensor, points_batch):
        """只取网络 head 的 nominal 输出 u_nom=[v,L*omega]，不经过 safety_layer。"""
        with torch.no_grad():
            if hasattr(self.model, "forward_nominal"):
                return self.model.forward_nominal(state_tensor, points_batch)
            if hasattr(self.model, "predict_unom"):
                return self.model.predict_unom(state_tensor, points_batch)

            if getattr(self.model, "use_dual_branch", True):
                geo_feat = self.model.geo_encoder(points_batch)
                state_feat = self.model.state_encoder(state_tensor)
                fused = self.model.fusion(torch.cat([geo_feat, state_feat], dim=1))
            else:
                state_per_point = state_tensor[points_batch.batch]
                node_features = torch.cat([points_batch.pos, state_per_point], dim=1)
                geo_feat = self.model.geo_encoder(points_batch, node_features=node_features)
                fused = self.model.fusion(geo_feat)

            return self.model.head(fused) * float(self.nominal_speed)

    def apply_model_safety_layer(self, u_nom_tensor, G_cdf_tensor, h_cdf_tensor):
        """用当前模型内部 qpth safety_layer 做安全投影，避免再次走 model.forward 造成歧义。"""
        with torch.no_grad():
            if (
                hasattr(self.model, "safety_layer")
                and self.model.safety_layer is not None
                and getattr(self.model, "use_safety_layer", True)
            ):
                out = self.model.safety_layer(u_nom_tensor, G_cdf_tensor, h_cdf_tensor)
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out
            return u_nom_tensor

    def run_model_learned_cdf_qpth(self, state_tensor, points_batch):
        # model.forward 内部会根据点云和 u_nom 重新构造 PyTorch CDF-G/h；
        # dummy G/h 只是兼容 forward 签名，use_learned_cdf_constraints=True 时不会被用作 QP 主约束。
        dummy_G = torch.zeros((1, 1, 6), dtype=torch.float32, device=self.device)
        dummy_h = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(state_tensor, points_batch, dummy_G, dummy_h)
            if isinstance(out, (tuple, list)):
                u_safe_tensor, u_nom_tensor = out[0], out[1]
            else:
                u_safe_tensor = out
                u_nom_tensor = None
        u_safe_np = u_safe_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)[:2]
        u_nom_np = None if u_nom_tensor is None else u_nom_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)[:2]
        return u_safe_np, u_nom_np

    def set_gazebo_model_pose(self, model_name, x, y, z=0.25):
        if self.set_model_state_srv is None:
            rospy.logwarn(f"/gazebo/set_model_state 不可用，无法移动 {model_name}")
            return False

        state = ModelState()
        state.model_name = model_name
        state.reference_frame = "world"
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        state.pose.orientation.w = 1.0
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0

        try:
            resp = self.set_model_state_srv(state)
            if not resp.success:
                rospy.logwarn(f"移动模型失败: {model_name}, Gazebo message: {resp.status_message}")
            return resp.success
        except Exception as e:
            rospy.logwarn(f"调用 /gazebo/set_model_state 失败: {model_name}, error={e}")
            return False

    def apply_fixed_obstacles_to_gazebo(self):
        fixed_obstacles = [
            [5.0, 0.05],
            [6.5, -0.5],
            # [8.0, -2.5],
            [10.0, -0.5],
        ]
        for i, p in enumerate(fixed_obstacles):
            self.set_gazebo_model_pose(f"cylinder_{i}", float(p[0]), float(p[1]), z=0.25)
        for i in range(len(fixed_obstacles), 8):
            self.set_gazebo_model_pose(f"cylinder_{i}", 80.0 + 3.0 * i, 20.0, z=0.25)
        rospy.logwarn("固定评测场景已同步到 Gazebo: cylinder_0~2 已摆放，其余 cylinder 已隐藏")

    def publish_target_marker(self):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.target_pos[0])
        marker.pose.position.y = float(self.target_pos[1])
        marker.pose.position.z = 0.2
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def publish_twist(self, v, w):
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_vel_pub.publish(twist)

    def publish_cloud(self, points):
        if len(points) == 0:
            return
        header = Header(stamp=rospy.Time.now(), frame_id="velodyne")
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        padded = np.hstack((points, np.zeros((len(points), 1), dtype=np.float32)))
        pc_msg = point_cloud2.create_cloud(header, fields, padded)
        self.debug_pub.publish(pc_msg)

    def publish_stop_burst(self, count=None, sleep_dt=None):
        """reset 前后连续发零速度，尽量清掉底盘控制残留。"""
        if count is None:
            count = self.reset_stop_burst_count
        if sleep_dt is None:
            sleep_dt = self.reset_stop_burst_dt
        for _ in range(max(1, int(count))):
            self.publish_twist(0.0, 0.0)
            time.sleep(max(0.0, float(sleep_dt)))


    # ============================================================
    # ROS callback
    # ============================================================
    def cloud_callback(self, msg):
        if rospy.is_shutdown():
            return

        now = time.time()
        # reset 后一小段时间内丢弃队列残留消息；只接受 sensor_accept_wall_time 之后的新消息。
        if self.is_resetting or now < getattr(self, "sensor_accept_wall_time", 0.0):
            return

        pts_gen = point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True)
        pts = np.array([[p[0], p[1]] for p in pts_gen], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            self.pointcloud_local = np.zeros((0, 2), dtype=np.float32)
            self.last_cloud_num_points = 0
        else:
            self.pointcloud_local = pts[:, :2]
            self.last_cloud_num_points = int(self.pointcloud_local.shape[0])
            self.publish_cloud(self.pointcloud_local)

        self.last_cloud_wall_time = now

    def pose_callback(self, msg):
        now = time.time()
        if self.is_resetting or now < getattr(self, "sensor_accept_wall_time", 0.0):
            return

        if len(msg.data) >= 3:
            self.current_pose = [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])]
            self.last_pose_wall_time = now

    # ============================================================
    # 控制主循环
    # ============================================================
    def control_loop(self, event):
        if self.episode_finish_lock or rospy.is_shutdown():
            return

        self.publish_target_marker()
        now_loop = time.time()

        if self.is_resetting:
            self.publish_twist(0.0, 0.0)
            return

        if now_loop < self.episode_hold_until:
            self.publish_twist(0.0, 0.0)
            return

        # reset 后必须等新 pose/cloud 到达，避免上一轮队列缓存污染下一轮。
        if self.require_fresh_pose_after_reset and self.last_pose_wall_time < self.sensor_accept_wall_time:
            if now_loop - self.sensor_accept_wall_time > self.fresh_sensor_wait_timeout:
                rospy.logwarn_throttle(
                    1.0,
                    f"[WAIT FRESH POSE] episode={self.episode_index + 1}, "
                    f"last_pose_wall_time={self.last_pose_wall_time:.3f}, "
                    f"accept_after={self.sensor_accept_wall_time:.3f}",
                )
            self.publish_twist(0.0, 0.0)
            return

        if self.require_fresh_cloud_after_reset and self.last_cloud_wall_time < self.sensor_accept_wall_time:
            if now_loop - self.sensor_accept_wall_time > self.fresh_sensor_wait_timeout:
                rospy.logwarn_throttle(
                    1.0,
                    f"[WAIT FRESH CLOUD] episode={self.episode_index + 1}, "
                    f"last_cloud_wall_time={self.last_cloud_wall_time:.3f}, "
                    f"accept_after={self.sensor_accept_wall_time:.3f}",
                )
            self.publish_twist(0.0, 0.0)
            return

        x, y, theta = self.current_pose
        l_k = self.cbf_config["l_k"]
        self.current_run_trajectory.append([x, y, theta, time.time() - self.start_time])

        if time.time() - self.current_episode_start_time > self.max_episode_time:
            rospy.logwarn(f"第 {self.episode_index + 1} 回合超时，判定为未到达。")
            self.finish_episode(arrived=False, reason="timeout")
            return

        # 碰撞统计
        ego_center = np.array([x, y], dtype=np.float32)
        distances_to_obs = np.linalg.norm(self.obstacles - ego_center, axis=1)
        if np.any(distances_to_obs < self.safety_threshold):
            first_collision = not self.collision_happened_in_current_run
            self.collision_happened_in_current_run = True
            current_time = time.time()
            if current_time - self.last_collision_time > 0.5:
                self.total_collision_events += 1
                self.last_collision_time = current_time
                rospy.logerr(f"碰撞警告: 当前总碰撞事件计数 {self.total_collision_events}")
            if first_collision and self.terminate_on_collision:
                self.finish_episode(arrived=False, reason="collision")
                return

        dx = float(self.target_pos[0]) - x
        dy = float(self.target_pos[1]) - y
        dist_to_goal_val = np.hypot(dx, dy)
        target_local_x = dx * np.cos(theta) + dy * np.sin(theta)
        target_local_y = -dx * np.sin(theta) + dy * np.cos(theta)
        target_local_np = np.array([target_local_x, target_local_y], dtype=np.float32)

        state_array = np.array(
            [target_local_x, target_local_y, self.last_executed_v, self.last_executed_w],
            dtype=np.float32,
        )
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device)

        dist_local = np.linalg.norm(target_local_np)
        if dist_to_goal_val > self.goal_radius and dist_local > 0.1:
            analytic_u_nom = self.nominal_speed * target_local_np / (dist_local + 1e-6)
        else:
            analytic_u_nom = np.zeros(2, dtype=np.float32)

        is_empty = (
            self.pointcloud_local.shape[0] == 0
            or (
                self.pointcloud_local.shape[0] > 0
                and self.pointcloud_local[0, 0] == 99
                and self.pointcloud_local[0, 1] == 99
            )
        )

        if is_empty:
            # 空点云下没有可靠障碍物约束，和当前测试基准一致：解析 nominal 直驱。
            u_safe_np = analytic_u_nom.astype(np.float32)
        else:
            local_pts = self.pointcloud_local.astype(np.float32)
            points_batch = self.make_points_batch(local_pts)

            u_nom_np = None
            u_jax_np = None

            if self.runtime_qp_mode == "nominal":
                u_nom_tensor = self.predict_model_unom(state_tensor, points_batch)
                u_nom_np = u_nom_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)[:2]
                u_safe_np = np.clip(u_nom_np, -self.model_qp_limit, self.model_qp_limit).astype(np.float32)

            elif self.runtime_qp_mode == "jax":
                # 对照模式：网络 nominal + JAX/ProxQP 安全投影。
                u_nom_tensor = self.predict_model_unom(state_tensor, points_batch)
                u_nom_np = u_nom_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)[:2]
                u_nom_np = np.clip(u_nom_np, -self.model_qp_limit, self.model_qp_limit).astype(np.float32)

                fixed_size = 200
                if local_pts.shape[0] > fixed_size:
                    local_pts_for_qp = local_pts[:fixed_size]
                elif local_pts.shape[0] < fixed_size:
                    pad_box = np.full((fixed_size - local_pts.shape[0], 2), 99.0, dtype=np.float32)
                    local_pts_for_qp = np.vstack([local_pts, pad_box]).astype(np.float32)
                else:
                    local_pts_for_qp = local_pts

                ego_p_local_jax = np.array([l_k, 0.0], dtype=np.float32)
                sol_6d_raw, _, _ = self.local_expert.solve_agent_qp_local(
                    ego_p_local_jax,
                    u_nom_np,
                    local_pts_for_qp,
                    target_local_np,
                )
                u_jax_np = np.array(sol_6d_raw[:2], dtype=np.float32)
                u_safe_np = u_jax_np

            else:
                # 正式评估当前 learnable CDF-QP：模型内部根据实时点云 + 学习到的参数构造 G/h。
                try:
                    u_safe_np, u_nom_np = self.run_model_learned_cdf_qpth(state_tensor, points_batch)
                    u_safe_np = np.clip(u_safe_np, -self.model_qp_limit, self.model_qp_limit).astype(np.float32)
                except Exception as e:
                    rospy.logerr(f"learned PyTorch CDF-QP 推理失败: {e}")
                    if self.qpth_fail_fallback_to_jax:
                        rospy.logwarn("已回退到 JAX/ProxQP 安全投影输出，避免当前 episode 中断。")
                        u_nom_tensor = self.predict_model_unom(state_tensor, points_batch)
                        u_nom_np = u_nom_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)[:2]
                        u_nom_np = np.clip(u_nom_np, -self.model_qp_limit, self.model_qp_limit).astype(np.float32)
                        fixed_size = 200
                        if local_pts.shape[0] > fixed_size:
                            local_pts_for_qp = local_pts[:fixed_size]
                        elif local_pts.shape[0] < fixed_size:
                            pad_box = np.full((fixed_size - local_pts.shape[0], 2), 99.0, dtype=np.float32)
                            local_pts_for_qp = np.vstack([local_pts, pad_box]).astype(np.float32)
                        else:
                            local_pts_for_qp = local_pts
                        ego_p_local_jax = np.array([l_k, 0.0], dtype=np.float32)
                        sol_6d_raw, _, _ = self.local_expert.solve_agent_qp_local(
                            ego_p_local_jax, u_nom_np, local_pts_for_qp, target_local_np
                        )
                        u_jax_np = np.array(sol_6d_raw[:2], dtype=np.float32)
                        u_safe_np = u_jax_np
                    else:
                        u_safe_np = np.zeros(2, dtype=np.float32)

        if len(u_safe_np) < 2 or (not np.all(np.isfinite(u_safe_np[:2]))):
            v_final = 0.0
            w_final = 0.0
            u_safe_np = np.zeros(2, dtype=np.float32)
        else:
            v_final = float(np.clip(u_safe_np[0], self.cbf_config["v_min"], self.cbf_config["v_max"]))
            w_final = float(np.clip(u_safe_np[1] / l_k, self.cbf_config["w_min"], self.cbf_config["w_max"]))
            u_safe_np = np.array([v_final, w_final * l_k], dtype=np.float32)

        self.last_executed_v = v_final
        self.last_executed_w = w_final * l_k
        self.publish_twist(v_final, w_final)

    def check_reach_goal(self, event):
        if self.episode_finish_lock or rospy.is_shutdown():
            return
        dist = np.linalg.norm(np.array(self.current_pose[:2]) - np.array(self.target_pos))
        if dist < self.goal_radius:
            self.finish_episode(arrived=True, reason="arrival")

    # ============================================================
    # episode 结束、保存和 CSV
    # ============================================================
    def finish_episode(self, arrived, reason):
        if self.episode_finish_lock:
            return
        self.episode_finish_lock = True
        self.publish_twist(0.0, 0.0)

        episode_no = self.episode_index + 1
        rospy.logwarn(
            f"第 {episode_no}/{self.total_eval_episodes} 回合结束: "
            f"reason={reason}, arrived={arrived}, collided={self.collision_happened_in_current_run}"
        )

        if arrived:
            self.reached_goals_count += 1
            if not self.collision_happened_in_current_run:
                self.perfect_runs_count += 1
        if self.collision_happened_in_current_run:
            self.collision_runs_count += 1
        if reason == "timeout":
            self.timeout_runs_count += 1

        self.all_runs_trajectories.append(self.current_run_trajectory)
        self.print_final_report()

        if self.episode_index + 1 >= self.total_eval_episodes:
            rospy.logerr("所有评测回合完成，正在保存轨迹和 CSV...")
            self.save_trajectory_data()
            self.append_eval_result_to_csv()
            self.print_final_report()
            rospy.signal_shutdown("Evaluation Completed Successfully")
            sys.exit(0)

        self.reset_for_next_episode()

    def reset_for_next_episode(self):
        # reset 期间先锁住控制循环，并反复发送停止命令，避免底盘残留速度。
        self.is_resetting = True
        self.episode_finish_lock = True
        self.publish_stop_burst()

        try:
            from std_srvs.srv import Empty

            rospy.wait_for_service("/gazebo/reset_simulation", timeout=1.0)
            reset_sim = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
            reset_sim()
        except Exception:
            os.system("rosservice call /gazebo/reset_simulation '{}' > /dev/null 2>&1")

        # reset 后重新摆放障碍物，并可选显式重置机器人模型。
        time.sleep(0.3)
        self.apply_fixed_obstacles_to_gazebo()

        if self.robot_model_name.strip():
            ok = self.set_gazebo_model_pose(
                self.robot_model_name.strip(), 0.0, 0.0, z=self.robot_reset_z
            )
            if not ok:
                rospy.logwarn(
                    f"robot_model_name={self.robot_model_name} 显式复位失败；"
                    f"如果你的机器人模型名不是 jackal，请用 _robot_model_name:=实际模型名"
                )

        self.publish_stop_burst()

        # 更新 episode 状态。注意：先设置 sensor_accept_wall_time，再清空 last_*，
        # callback 会丢弃这个时间之前到达的队列残留消息。
        self.episode_index += 1
        self.target_pos = self.test_targets[self.episode_index].tolist()

        now = time.time()
        self.reset_wall_time = now
        self.sensor_accept_wall_time = now + float(self.sensor_accept_delay)
        self.last_pose_wall_time = 0.0
        self.last_cloud_wall_time = 0.0
        self.last_cloud_num_points = 0

        self.current_pose = [0.0, 0.0, 0.0]
        self.pointcloud_local = np.zeros((0, 2), dtype=np.float32)
        self.last_executed_v = 0.0
        self.last_executed_w = 0.0
        self.current_run_trajectory = []
        self.collision_happened_in_current_run = False
        self.last_collision_time = 0.0

        self.episode_hold_until = max(now + self.hold_before_episode, self.sensor_accept_wall_time)
        self.current_episode_start_time = self.episode_hold_until
        self.episode_finish_lock = False
        self.is_resetting = False
        self.publish_twist(0.0, 0.0)

        rospy.logerr(
            f"新一轮开始: {self.episode_index + 1}/{self.total_eval_episodes}, "
            f"target={self.target_pos}, accept_sensor_after={self.sensor_accept_wall_time:.3f}"
        )

    def save_trajectory_data(self):
        try:
            tag = f"demo{self.num_demos}_dseed{self.demo_seed}_seed{self.train_seed}_{self.runtime_qp_mode}_{self.target_mode}"
            trajectory_save_path = os.path.join(self.trajectory_save_dir, f"trajectory_{tag}.pt")
            torch.save(self.all_runs_trajectories, trajectory_save_path)
            rospy.loginfo(
                f"轨迹落盘成功: {len(self.all_runs_trajectories)} 回合 -> {trajectory_save_path}"
            )
        except Exception as e:
            rospy.logerr(f"保存轨迹数据失败: {str(e)}")

    def print_final_report(self):
        completed_runs = min(self.episode_index + 1, self.total_eval_episodes)
        completed_runs = max(completed_runs, 1)

        arrival_rate = self.reached_goals_count / completed_runs
        success_rate = self.perfect_runs_count / completed_runs
        collision_rate = self.collision_runs_count / completed_runs
        avg_collision_events = self.total_collision_events / completed_runs

        print("\n" + " DensityNet 自动评测报告 ".center(70, "="))
        print(f"模型路径             : {self.resolve_model_path(self.model_path)}")
        print(f"num_demos/demo_seed  : {self.num_demos}/{self.demo_seed}, train_seed={self.train_seed}")
        print(f"runtime_qp_mode      : {self.runtime_qp_mode}")
        print(f"target_mode/target   : {self.target_mode}/{self.target_pos}")
        print(f"进度                 : {completed_runs} / {self.total_eval_episodes}")
        print(f"到达率 arrival_rate  : {arrival_rate:.4f} ({self.reached_goals_count}/{completed_runs})")
        print(f"成功率 success_rate  : {success_rate:.4f} ({self.perfect_runs_count}/{completed_runs})")
        print(f"碰撞率 collision_rate: {collision_rate:.4f} ({self.collision_runs_count}/{completed_runs})")
        print(f"平均碰撞事件         : {avg_collision_events:.4f} events/episode")
        print(f"超时回合             : {self.timeout_runs_count}")
        print("=" * 70 + "\n")

    def append_eval_result_to_csv(self):
        completed_runs = self.total_eval_episodes
        arrival_rate = self.reached_goals_count / completed_runs
        success_rate = self.perfect_runs_count / completed_runs
        collision_rate = self.collision_runs_count / completed_runs
        avg_collision_events = self.total_collision_events / completed_runs

        row = {
            "num_demos": self.num_demos,
            "demo_seed": self.demo_seed,
            "train_seed": self.train_seed,
            "num_eval_episodes": completed_runs,
            "test_target_seed": self.test_target_seed,
            "target_mode": self.target_mode,
            "fixed_target_x": self.fixed_target_x,
            "fixed_target_y": self.fixed_target_y,
            "runtime_qp_mode": self.runtime_qp_mode,
            "graph_k": self.model_graph_k,
            "hidden_dim": self.model_hidden_dim,
            "lambda_smooth": self.model_lambda_smooth,
            "learnable_lambda_smooth": self.learnable_lambda_smooth,
            "use_learned_cdf_constraints": self.use_learned_cdf_constraints,
            "cdf_alpha": getattr(getattr(self.model, "cdf_constraint_layer", None), "_last_alpha", None).detach().cpu().item() if getattr(getattr(self.model, "cdf_constraint_layer", None), "_last_alpha", None) is not None else None,
            "cdf_epsilon": getattr(getattr(self.model, "cdf_constraint_layer", None), "_last_epsilon", None).detach().cpu().item() if getattr(getattr(self.model, "cdf_constraint_layer", None), "_last_epsilon", None) is not None else None,
            "qp_box": self.use_qp_box_constraints,
            "qp_normalize": self.qp_normalize_constraints,
            "arrival_count": self.reached_goals_count,
            "success_count": self.perfect_runs_count,
            "collision_runs_count": self.collision_runs_count,
            "collision_events": self.total_collision_events,
            "timeout_runs_count": self.timeout_runs_count,
            "arrival_rate": arrival_rate,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "avg_collision_events": avg_collision_events,
            "model_path": self.resolve_model_path(self.model_path),
        }

        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        file_exists = os.path.exists(self.output_csv)
        with open(self.output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        rospy.loginfo(f"评估结果已追加写入 CSV: {self.output_csv}")

    def signal_handler(self, sig, frame):
        print("\n" + "!" * 70)
        rospy.logwarn("捕获中断信号，保存当前轨迹与阶段性报告。")
        self.publish_twist(0.0, 0.0)
        self.save_trajectory_data()
        self.print_final_report()
        print("!" * 70 + "\n")
        rospy.signal_shutdown("User interrupt")
        sys.exit(0)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        tracker = ParametricEllipseTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass