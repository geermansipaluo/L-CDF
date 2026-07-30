#!/usr/bin/env python3
import os
import sys
import csv
import time
import signal

import rospy
import numpy as np
import cvxpy as cp
import tf

from std_msgs.msg import Float32MultiArray, Header, ColorRGBA
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def ros_bool(value, default=False):
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


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class GaussianCbfEvalPlanner:
    """
    GP-CBF closed-loop evaluation node.

    Input point cloud convention:
        - Subscribe to /densitynet_input_points.
        - This topic should be produced by perfect_lidar_bridge.py.
        - Points are local 2D obstacle points in the robot / velodyne frame.
        - The fake empty point [99, 99, 0] is treated as no obstacle.

    Control convention:
        - The GP-CBF QP solves a desired world-frame velocity for the lookahead point.
        - The velocity is mapped to Jackal Twist by the same lookahead length l_k used in your
          current evaluation baseline.
    """

    def __init__(self):
        self.start_time = time.time()
        rospy.init_node("gaussian_cbf_eval_node", anonymous=True)

        # ============================================================
        # 1. Evaluation parameters, kept close to traj_eval_sweep.py
        # ============================================================
        self.output_csv = rospy.get_param(
            "~output_csv",
            "/home/guo/L-CDF/src/sensor_cdf/scripts/eval_metrics_gpcbf.csv",
        )
        self.trajectory_save_dir = rospy.get_param(
            "~trajectory_save_dir",
            "/home/guo/L-CDF/src/sensor_cdf/scripts/eval_trajectories",
        )
        os.makedirs(self.trajectory_save_dir, exist_ok=True)

        # These tags are not used by GP-CBF itself, but make CSV rows align with your sweep scripts.
        self.num_demos = int(rospy.get_param("~num_demos", -1))
        self.demo_seed = int(rospy.get_param("~demo_seed", -1))
        self.train_seed = int(rospy.get_param("~train_seed", -1))

        self.total_eval_episodes = int(rospy.get_param("~num_eval_episodes", 10))
        self.test_target_seed = int(rospy.get_param("~test_target_seed", 2026))
        self.target_mode = str(rospy.get_param("~target_mode", "fixed")).lower()
        self.fixed_target_x = float(rospy.get_param("~fixed_target_x", 15.0))
        self.fixed_target_y = float(rospy.get_param("~fixed_target_y", -1.0))
        self.target_x_min = float(rospy.get_param("~target_x_min", 14.0))
        self.target_x_max = float(rospy.get_param("~target_x_max", 16.0))
        self.target_y_min = float(rospy.get_param("~target_y_min", -2.0))
        self.target_y_max = float(rospy.get_param("~target_y_max", 2.0))
        self.goal_radius = float(rospy.get_param("~goal_radius", 0.4))
        self.max_episode_time = float(rospy.get_param("~max_episode_time", 80.0))
        self.hold_before_episode = float(rospy.get_param("~hold_before_episode", 2.0))
        self.terminate_on_collision = ros_bool(rospy.get_param("~terminate_on_collision", False), False)
        self.num_rays = int(rospy.get_param("~num_rays", -1))

        # Reset / stale message guard. This mirrors your DensityNet evaluator.
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
        self.robot_model_name = str(rospy.get_param("~robot_model_name", "jackal"))
        self.robot_reset_z = float(rospy.get_param("~robot_reset_z", 0.15))

        # ============================================================
        # 2. GP-CBF and robot parameters
        # ============================================================
        self.l_k = float(rospy.get_param("~l_k", 0.33))
        self.r_ego = float(rospy.get_param("~r_ego", 0.31))
        self.obstacle_radius = float(rospy.get_param("~obstacle_radius", 0.5))
        self.safety_threshold = self.obstacle_radius + self.r_ego

        self.v_min = float(rospy.get_param("~v_min", -1.2))
        self.v_max = float(rospy.get_param("~v_max", 1.2))
        self.w_min = float(rospy.get_param("~w_min", -2.0))
        self.w_max = float(rospy.get_param("~w_max", 2.0))

        self.nominal_speed = float(rospy.get_param("~nominal_speed", 1.0))
        self.world_u_limit = float(rospy.get_param("~world_u_limit", 1.0))
        self.control_dt = float(rospy.get_param("~control_dt", 0.1))
        self.goal_check_dt = float(rospy.get_param("~goal_check_dt", 0.05))

        self.gp_length_scale = float(rospy.get_param("~gp_length_scale", 0.6))
        self.gp_noise = float(rospy.get_param("~gp_noise", 0.01))
        self.gp_h_shift = float(rospy.get_param("~gp_h_shift", 0.6))
        self.cbf_gamma = float(rospy.get_param("~cbf_gamma", 0.8))
        self.clf_gamma = float(rospy.get_param("~clf_gamma", 2.0))
        self.cbf_slack_weight = float(rospy.get_param("~cbf_slack_weight", 1000.0))
        self.clf_slack_weight = float(rospy.get_param("~clf_slack_weight", 1.0))
        self.min_gp_points = int(rospy.get_param("~min_gp_points", 3))
        self.max_gp_points = int(rospy.get_param("~max_gp_points", 200))
        self.voxel_size = float(rospy.get_param("~voxel_size", 0.05))
        self.gp_visualize = ros_bool(rospy.get_param("~gp_visualize", False), False)

        kernel = RBF(length_scale=self.gp_length_scale, length_scale_bounds="fixed") + WhiteKernel(
            noise_level=self.gp_noise, noise_level_bounds="fixed"
        )
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=False)

        # Fixed evaluation obstacles. Keep this synchronized with Gazebo placement and collision audit.
        self.obstacles = np.array(
            [
                [5.0, 0.05],
                [6.5, -0.5],
                [10.0, -0.5],
            ],
            dtype=np.float32,
        )

        # ============================================================
        # 3. Runtime state
        # ============================================================
        self.current_pose = [0.0, 0.0, 0.0]
        self.pointcloud_local = np.zeros((0, 2), dtype=np.float32)
        self.pointcloud_world = np.zeros((0, 2), dtype=np.float32)
        self.min_distance_local = float("inf")
        self.last_cmd = np.zeros(2, dtype=np.float32)

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

        self.all_runs_trajectories = []
        self.current_run_trajectory = []
        self.reached_goals_count = 0
        self.perfect_runs_count = 0
        self.collision_runs_count = 0
        self.timeout_runs_count = 0
        self.total_collision_events = 0
        self.collision_happened_in_current_run = False
        self.last_collision_time = 0.0
        self.infeasible_count = 0

        # ============================================================
        # 3.1 Inference-time statistics, printed only in terminal.
        #     These fields are not written to CSV or trajectory files.
        #     Timing covers GP fitting/evaluation, CBF-QP solve, and
        #     conversion from the solved world velocity to Twist commands.
        # ============================================================
        self.inference_time_count = 0
        self.inference_time_total = 0.0
        self.inference_time_min = float("inf")
        self.inference_time_max = 0.0
        self.inference_time_warmup_skip = int(rospy.get_param("~inference_time_warmup_skip", 3))
        self.inference_time_raw_count = 0
        self.inference_time_print_every = int(rospy.get_param("~inference_time_print_every", 50))

        # ============================================================
        # 4. ROS interfaces
        # ============================================================
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.debug_pub = rospy.Publisher("/processed_cloud", PointCloud2, queue_size=1)
        self.marker_pub = rospy.Publisher("/target_goal_marker", Marker, queue_size=1)
        self.gaussian_cbf_pub = rospy.Publisher("/cbf", Marker, queue_size=1)

        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=5.0)
            self.set_model_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            rospy.loginfo("Gazebo /set_model_state service connected")
        except Exception as e:
            self.set_model_state_srv = None
            rospy.logwarn(f"Gazebo /set_model_state service unavailable: {e}")

        self.apply_fixed_obstacles_to_gazebo()

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # Use the same topics as your current DensityNet evaluation node.
        self.pose_sub = rospy.Subscriber("/curr_state", Float32MultiArray, self.pose_callback, queue_size=1)
        self.cloud_sub = rospy.Subscriber(
            "/densitynet_input_points", PointCloud2, self.cloud_callback, queue_size=1
        )

        self.ctrl_timer = rospy.Timer(rospy.Duration(self.control_dt), self.control_loop)
        self.goal_timer = rospy.Timer(rospy.Duration(self.goal_check_dt), self.check_reach_goal)

        rospy.loginfo("=" * 70)
        rospy.loginfo("GP-CBF automatic evaluation node started")
        rospy.loginfo(f"num_eval_episodes       = {self.total_eval_episodes}")
        rospy.loginfo(f"target_mode/first target = {self.target_mode}/{self.target_pos}")
        rospy.loginfo(f"topics: /curr_state + /densitynet_input_points -> /cmd_vel")
        rospy.loginfo(f"output_csv              = {self.output_csv}")
        rospy.loginfo("=" * 70)

    # ============================================================
    # Inference-time helpers
    # ============================================================
    def _inference_tic(self):
        return time.perf_counter()

    def _record_inference_time(self, t0):
        dt = time.perf_counter() - t0
        self.inference_time_raw_count += 1

        # Skip a few first measurements to avoid one-time solver/cache warm-up effects.
        if self.inference_time_raw_count <= self.inference_time_warmup_skip:
            return

        self.inference_time_count += 1
        self.inference_time_total += float(dt)
        self.inference_time_min = min(self.inference_time_min, float(dt))
        self.inference_time_max = max(self.inference_time_max, float(dt))

        if self.inference_time_print_every > 0 and self.inference_time_count % self.inference_time_print_every == 0:
            avg_ms = 1000.0 * self.inference_time_total / max(self.inference_time_count, 1)
            rospy.loginfo(f"[GP-CBF inference time] avg={avg_ms:.3f} ms over {self.inference_time_count} calls")

    def _inference_time_summary(self):
        if self.inference_time_count <= 0:
            return "Average inference time : N/A, no timed control calls"
        avg = self.inference_time_total / self.inference_time_count
        return (
            f"Average inference time : {1000.0 * avg:.3f} ms "
            f"({self.inference_time_count} calls, "
            f"min={1000.0 * self.inference_time_min:.3f} ms, "
            f"max={1000.0 * self.inference_time_max:.3f} ms, "
            f"warmup skipped={self.inference_time_warmup_skip})"
        )

    # ============================================================
    # Target sampling
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
            rospy.logwarn(f"Unknown target_mode={mode}; using fixed_y0")
            fixed_target = np.array([self.fixed_target_x, 0.0], dtype=np.float32)
            targets = np.tile(fixed_target[None, :], (self.total_eval_episodes, 1)).astype(np.float32)
        return targets

    # ============================================================
    # Point cloud processing
    # ============================================================
    def local_to_world_points(self, local_pts):
        if local_pts.ndim != 2 or local_pts.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        x, y, theta = self.current_pose
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        world = local_pts @ rot.T + np.array([x, y], dtype=np.float32)
        return world.astype(np.float32)

    def voxel_downsample(self, pts):
        if pts.ndim != 2 or pts.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if self.voxel_size <= 0.0:
            return pts[: self.max_gp_points].astype(np.float32)
        voxel_indices = np.floor(pts / self.voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxel_indices, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        pts = pts[unique_indices]
        if pts.shape[0] > self.max_gp_points:
            # Nearest points are usually the most useful for local GP-CBF.
            dists = np.linalg.norm(pts - self.lookahead_world_point()[None, :], axis=1)
            keep = np.argsort(dists)[: self.max_gp_points]
            pts = pts[keep]
        return pts.astype(np.float32)

    def cloud_callback(self, msg):
        if rospy.is_shutdown():
            return
        now = time.time()
        if self.is_resetting or now < getattr(self, "sensor_accept_wall_time", 0.0):
            return

        pts_gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pts3 = np.array([[p[0], p[1], p[2]] for p in pts_gen], dtype=np.float32)
        if pts3.ndim != 2 or pts3.shape[0] == 0:
            self.pointcloud_local = np.zeros((0, 2), dtype=np.float32)
            self.pointcloud_world = np.zeros((0, 2), dtype=np.float32)
            self.min_distance_local = float("inf")
            self.last_cloud_num_points = 0
            self.last_cloud_wall_time = now
            return

        local = pts3[:, :2].astype(np.float32)

        # perfect_lidar_bridge.py publishes [99, 99, 0] as the empty-scene heartbeat.
        fake_empty = (local.shape[0] == 1 and abs(local[0, 0] - 99.0) < 1e-3 and abs(local[0, 1] - 99.0) < 1e-3)
        if fake_empty:
            self.pointcloud_local = np.zeros((0, 2), dtype=np.float32)
            self.pointcloud_world = np.zeros((0, 2), dtype=np.float32)
            self.min_distance_local = float("inf")
            self.last_cloud_num_points = 0
            self.last_cloud_wall_time = now
            return

        dists = np.linalg.norm(local, axis=1)
        finite_mask = np.isfinite(dists) & (dists > 0.05) & (dists < 50.0)
        local = local[finite_mask]
        if local.shape[0] == 0:
            self.pointcloud_local = np.zeros((0, 2), dtype=np.float32)
            self.pointcloud_world = np.zeros((0, 2), dtype=np.float32)
            self.min_distance_local = float("inf")
            self.last_cloud_num_points = 0
            self.last_cloud_wall_time = now
            return

        self.min_distance_local = float(np.min(np.linalg.norm(local, axis=1)))
        world = self.local_to_world_points(local)
        world = self.voxel_downsample(world)

        self.pointcloud_local = local
        self.pointcloud_world = world
        self.last_cloud_num_points = int(world.shape[0])
        self.last_cloud_wall_time = now
        self.publish_cloud(world, frame_id="world")

    # ============================================================
    # GP-CBF core
    # ============================================================
    def lookahead_world_point(self):
        x, y, theta = self.current_pose
        return np.array([x + self.l_k * np.cos(theta), y + self.l_k * np.sin(theta)], dtype=np.float32)

    def reference_world_velocity(self, query_point):
        target = np.asarray(self.target_pos, dtype=np.float32)
        delta = target - query_point
        dist = float(np.linalg.norm(delta))
        if dist < self.goal_radius-0.1:
            return np.zeros(2, dtype=np.float32)
        return (self.nominal_speed * delta / (dist + 1e-6)).astype(np.float32)

    def fit_gp_and_eval(self, query_point, obstacle_points_world):
        if obstacle_points_world.shape[0] < self.min_gp_points:
            return None, None

        x_train = obstacle_points_world.astype(np.float64)
        y_train = -np.ones((x_train.shape[0],), dtype=np.float64)

        try:
            self.gpr.fit(x_train, y_train)
            state = query_point.reshape(1, -1).astype(np.float64)
            h_value = float(self.gpr.predict(state, return_cov=False)[0] + self.gp_h_shift)

            # Derivative of the RBF kernel wrt the query state.
            # WhiteKernel has zero derivative, so only gpr.kernel_.k1 is used.
            rbf = self.gpr.kernel_.k1
            length_scale = float(rbf.length_scale)
            k_vec = rbf(state, x_train).reshape(-1, 1)  # [N, 1]
            dk_dx = k_vec * (x_train - state) / (length_scale ** 2)  # [N, 2]
            alpha = self.gpr.alpha_.reshape(1, -1)  # [1, N]
            grad = (alpha @ dk_dx).reshape(2).astype(np.float64)
            return h_value, grad
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"GP fit/eval failed: {e}")
            return None, None

    def solve_gp_cbf_qp(self):
        query = self.lookahead_world_point().astype(np.float64)
        u_ref = self.reference_world_velocity(query).astype(np.float64)

        if self.pointcloud_world.shape[0] < self.min_gp_points:
            return u_ref.astype(np.float32), "nominal_no_obstacle"

        h_value, grad_h = self.fit_gp_and_eval(query, self.pointcloud_world)
        if h_value is None or grad_h is None or not np.all(np.isfinite(grad_h)):
            return u_ref.astype(np.float32), "nominal_gp_failed"

        u = cp.Variable(2)
        r_cbf = cp.Variable(nonneg=True)
        r_clf = cp.Variable(nonneg=True)

        target = np.asarray(self.target_pos, dtype=np.float64)
        goal_delta = query - target
        V = 0.5 * float(goal_delta @ goal_delta)
        grad_V = goal_delta.reshape(1, 2)
        grad_h_row = grad_h.reshape(1, 2)

        objective = cp.Minimize(
            cp.sum_squares(u - u_ref)
            + self.cbf_slack_weight * r_cbf
            + self.clf_slack_weight * r_clf
            + 1e-4 * cp.sum_squares(u)
        )

        constraints = [
            grad_h_row @ u + self.cbf_gamma * h_value + r_cbf >= 0.0,
            grad_V @ u + self.clf_gamma * V <= r_clf,
            u >= -self.world_u_limit,
            u <= self.world_u_limit,
        ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(verbose=False, solver=cp.OSQP, warm_start=True)
            if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or u.value is None:
                self.infeasible_count += 1
                return u_ref.astype(np.float32), f"fallback_{prob.status}"
            u_val = np.asarray(u.value, dtype=np.float32).reshape(2)
            if not np.all(np.isfinite(u_val)):
                self.infeasible_count += 1
                return u_ref.astype(np.float32), "fallback_nan"
            if self.gp_visualize:
                self.publish_gp_marker(query, h_value, grad_h)
            return u_val, "gp_cbf"
        except Exception as e:
            self.infeasible_count += 1
            rospy.logwarn_throttle(1.0, f"GP-CBF QP failed: {e}")
            return u_ref.astype(np.float32), "fallback_exception"

    def world_velocity_to_twist(self, u_world):
        vx, vy = float(u_world[0]), float(u_world[1])
        theta = float(self.current_pose[2])
        v = vx * np.cos(theta) + vy * np.sin(theta)
        w = (-vx * np.sin(theta) + vy * np.cos(theta)) / max(self.l_k, 1e-6)
        v = float(np.clip(v, self.v_min, self.v_max))
        w = float(np.clip(w, self.w_min, self.w_max))
        return v, w

    # ============================================================
    # ROS callbacks and control loop
    # ============================================================
    def pose_callback(self, msg):
        now = time.time()
        if self.is_resetting or now < getattr(self, "sensor_accept_wall_time", 0.0):
            return
        if len(msg.data) >= 3:
            self.current_pose = [float(msg.data[0]), float(msg.data[1]), wrap_angle(float(msg.data[2]))]
            self.last_pose_wall_time = now

    def control_loop(self, event):
        if self.episode_finish_lock or rospy.is_shutdown():
            return

        self.publish_target_marker()
        now_loop = time.time()

        if self.is_resetting or now_loop < self.episode_hold_until:
            self.publish_twist(0.0, 0.0)
            return

        if self.require_fresh_pose_after_reset and self.last_pose_wall_time < self.sensor_accept_wall_time:
            if now_loop - self.sensor_accept_wall_time > self.fresh_sensor_wait_timeout:
                rospy.logwarn_throttle(1.0, "[GP-CBF] waiting for fresh /curr_state after reset")
            self.publish_twist(0.0, 0.0)
            return

        if self.require_fresh_cloud_after_reset and self.last_cloud_wall_time < self.sensor_accept_wall_time:
            if now_loop - self.sensor_accept_wall_time > self.fresh_sensor_wait_timeout:
                rospy.logwarn_throttle(1.0, "[GP-CBF] waiting for fresh /densitynet_input_points after reset")
            self.publish_twist(0.0, 0.0)
            return

        x, y, theta = self.current_pose
        self.current_run_trajectory.append([x, y, theta, time.time() - self.start_time])

        if time.time() - self.current_episode_start_time > self.max_episode_time:
            rospy.logwarn(f"Episode {self.episode_index + 1} timeout")
            self.finish_episode(arrived=False, reason="timeout")
            return

        self.check_collision_runtime()
        if self.collision_happened_in_current_run and self.terminate_on_collision:
            self.finish_episode(arrived=False, reason="collision")
            return

        inference_t0 = self._inference_tic()
        u_world, mode = self.solve_gp_cbf_qp()
        v_cmd, w_cmd = self.world_velocity_to_twist(u_world)
        self._record_inference_time(inference_t0)
        self.last_cmd = np.array([v_cmd, w_cmd], dtype=np.float32)
        self.publish_twist(v_cmd, w_cmd)
        rospy.loginfo_throttle(
            1.0,
            f"[GP-CBF] ep={self.episode_index + 1}/{self.total_eval_episodes}, "
            f"mode={mode}, cmd=({v_cmd:.3f},{w_cmd:.3f}), points={self.last_cloud_num_points}",
        )

    def check_reach_goal(self, event):
        if self.episode_finish_lock or rospy.is_shutdown():
            return
        dist = float(np.linalg.norm(np.asarray(self.current_pose[:2]) - np.asarray(self.target_pos)))
        if dist < self.goal_radius+0.1:
            self.finish_episode(arrived=True, reason="arrival")

    def check_collision_runtime(self):
        ego_center = np.asarray(self.current_pose[:2], dtype=np.float32)
        distances_to_obs = np.linalg.norm(self.obstacles - ego_center[None, :], axis=1)
        if np.any(distances_to_obs < self.safety_threshold):
            first_collision = not self.collision_happened_in_current_run
            self.collision_happened_in_current_run = True
            current_time = time.time()
            if current_time - self.last_collision_time > 0.5:
                self.total_collision_events += 1
                self.last_collision_time = current_time
                rospy.logerr(f"Collision warning: total collision events={self.total_collision_events}")
            return first_collision
        return False

    # ============================================================
    # Gazebo, publishing, and episode reset
    # ============================================================
    def set_gazebo_model_pose(self, model_name, x, y, z=0.25):
        if self.set_model_state_srv is None:
            rospy.logwarn(f"/gazebo/set_model_state unavailable, cannot move {model_name}")
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
                rospy.logwarn(f"Failed to move model {model_name}: {resp.status_message}")
            return bool(resp.success)
        except Exception as e:
            rospy.logwarn(f"/gazebo/set_model_state call failed for {model_name}: {e}")
            return False

    def apply_fixed_obstacles_to_gazebo(self):
        for i, p in enumerate(self.obstacles):
            self.set_gazebo_model_pose(f"cylinder_{i}", float(p[0]), float(p[1]), z=0.25)
        for i in range(len(self.obstacles), 8):
            self.set_gazebo_model_pose(f"cylinder_{i}", 80.0 + 3.0 * i, 20.0, z=0.25)
        rospy.logwarn("GP-CBF fixed evaluation scene synced to Gazebo")

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

    def publish_cloud(self, points, frame_id="world"):
        if points is None or len(points) == 0:
            return
        header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
        ]
        padded = np.hstack((points.astype(np.float32), np.zeros((len(points), 1), dtype=np.float32)))
        pc_msg = point_cloud2.create_cloud(header, fields, padded)
        self.debug_pub.publish(pc_msg)

    def publish_gp_marker(self, query, h_value, grad_h):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "gaussian_cbf_grad"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.10
        marker.scale.z = 0.10
        marker.color.r = 0.0 if h_value >= 0.0 else 1.0
        marker.color.g = 1.0 if h_value >= 0.0 else 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        p0 = Point(x=float(query[0]), y=float(query[1]), z=0.25)
        grad = np.asarray(grad_h, dtype=np.float64)
        norm = np.linalg.norm(grad)
        if norm > 1e-6:
            p1_arr = query + 0.6 * grad / norm
        else:
            p1_arr = query
        p1 = Point(x=float(p1_arr[0]), y=float(p1_arr[1]), z=0.25)
        marker.points = [p0, p1]
        self.gaussian_cbf_pub.publish(marker)

    def publish_stop_burst(self, count=None, sleep_dt=None):
        if count is None:
            count = self.reset_stop_burst_count
        if sleep_dt is None:
            sleep_dt = self.reset_stop_burst_dt
        for _ in range(max(1, int(count))):
            self.publish_twist(0.0, 0.0)
            time.sleep(max(0.0, float(sleep_dt)))

    def finish_episode(self, arrived, reason):
        if self.episode_finish_lock:
            return
        self.episode_finish_lock = True
        self.publish_twist(0.0, 0.0)

        episode_no = self.episode_index + 1
        rospy.logwarn(
            f"GP-CBF episode {episode_no}/{self.total_eval_episodes} finished: "
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
            rospy.logerr("All GP-CBF evaluation episodes completed; saving results.")
            self.save_trajectory_data()
            self.append_eval_result_to_csv()
            self.print_final_report()
            rospy.signal_shutdown("GP-CBF Evaluation Completed")
            sys.exit(0)

        self.reset_for_next_episode()

    def reset_for_next_episode(self):
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

        time.sleep(0.3)
        self.apply_fixed_obstacles_to_gazebo()

        if self.robot_model_name.strip():
            ok = self.set_gazebo_model_pose(self.robot_model_name.strip(), 0.0, 0.0, z=self.robot_reset_z)
            if not ok:
                rospy.logwarn(
                    f"robot_model_name={self.robot_model_name} reset failed; "
                    f"set _robot_model_name:=your_model_name if it is not jackal"
                )

        self.publish_stop_burst()

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
        self.pointcloud_world = np.zeros((0, 2), dtype=np.float32)
        self.min_distance_local = float("inf")
        self.last_cmd = np.zeros(2, dtype=np.float32)
        self.current_run_trajectory = []
        self.collision_happened_in_current_run = False
        self.last_collision_time = 0.0

        self.episode_hold_until = max(now + self.hold_before_episode, self.sensor_accept_wall_time)
        self.current_episode_start_time = self.episode_hold_until
        self.episode_finish_lock = False
        self.is_resetting = False
        self.publish_twist(0.0, 0.0)

        rospy.logerr(
            f"GP-CBF next episode: {self.episode_index + 1}/{self.total_eval_episodes}, "
            f"target={self.target_pos}, accept_sensor_after={self.sensor_accept_wall_time:.3f}"
        )

    def save_trajectory_data(self):
        try:
            import torch

            # ============================================================
            # 1. GP-CBF 方法标签
            # ============================================================
            num_rays = int(getattr(self, "num_rays", -1))

            if num_rays > 0:
                mode = f"gpcbf_pc{num_rays}"
            else:
                mode = "gpcbf"

            # ============================================================
            # 2. 读取 GP-CBF 关键参数，写入文件名避免覆盖
            # ============================================================
            target_mode = str(getattr(self, "target_mode", "unknown"))

            # 文件名里避免小数点太乱，把小数点替换成 p
            def fmt_float(x):
                return f"{x:.3g}".replace(".", "p").replace("-", "m")

            tag = (
                f"mode-{mode}_"
                f"target-{target_mode}"
            )

            trajectory_save_path = os.path.join(
                self.trajectory_save_dir,
                f"trajectory_{tag}.pt"
            )

            torch.save(self.all_runs_trajectories, trajectory_save_path)

            rospy.loginfo(
                f"GP-CBF trajectories saved: {len(self.all_runs_trajectories)} episodes -> {trajectory_save_path}"
            )

        except Exception as e:
            rospy.logerr(f"Failed to save GP-CBF trajectory data: {e}")

    def print_final_report(self):
        completed_runs = min(self.episode_index + 1, self.total_eval_episodes)
        completed_runs = max(completed_runs, 1)
        arrival_rate = self.reached_goals_count / completed_runs
        success_rate = self.perfect_runs_count / completed_runs
        collision_rate = self.collision_runs_count / completed_runs
        avg_collision_events = self.total_collision_events / completed_runs

        print("\n" + " GP-CBF 自动评测报告 ".center(70, "="))
        print(f"进度                 : {completed_runs} / {self.total_eval_episodes}")
        print(f"target_mode/target   : {self.target_mode}/{self.target_pos}")
        print(f"到达率 arrival_rate  : {arrival_rate:.4f} ({self.reached_goals_count}/{completed_runs})")
        print(f"成功率 success_rate  : {success_rate:.4f} ({self.perfect_runs_count}/{completed_runs})")
        print(f"碰撞率 collision_rate: {collision_rate:.4f} ({self.collision_runs_count}/{completed_runs})")
        print(f"平均碰撞事件         : {avg_collision_events:.4f} events/episode")
        print(f"超时回合             : {self.timeout_runs_count}")
        print(f"QP 不可行/失败次数   : {self.infeasible_count}")
        print(self._inference_time_summary())
        print("=" * 70 + "\n")

    def append_eval_result_to_csv(self):
        completed_runs = self.total_eval_episodes
        arrival_rate = self.reached_goals_count / completed_runs
        success_rate = self.perfect_runs_count / completed_runs
        collision_rate = self.collision_runs_count / completed_runs
        avg_collision_events = self.total_collision_events / completed_runs

        row = {
            "algorithm": "GP-CBF",
            "num_demos": self.num_demos,
            "demo_seed": self.demo_seed,
            "train_seed": self.train_seed,
            "num_eval_episodes": completed_runs,
            "test_target_seed": self.test_target_seed,
            "target_mode": self.target_mode,
            "fixed_target_x": self.fixed_target_x,
            "fixed_target_y": self.fixed_target_y,
            "goal_radius": self.goal_radius,
            "l_k": self.l_k,
            "r_ego": self.r_ego,
            "nominal_speed": self.nominal_speed,
            "world_u_limit": self.world_u_limit,
            "gp_length_scale": self.gp_length_scale,
            "gp_noise": self.gp_noise,
            "gp_h_shift": self.gp_h_shift,
            "cbf_gamma": self.cbf_gamma,
            "clf_gamma": self.clf_gamma,
            "arrival_count": self.reached_goals_count,
            "success_count": self.perfect_runs_count,
            "collision_runs_count": self.collision_runs_count,
            "collision_events": self.total_collision_events,
            "timeout_runs_count": self.timeout_runs_count,
            "infeasible_count": self.infeasible_count,
            "arrival_rate": arrival_rate,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "avg_collision_events": avg_collision_events,
        }

        out_dir = os.path.dirname(self.output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        file_exists = os.path.exists(self.output_csv)
        with open(self.output_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        rospy.loginfo(f"GP-CBF evaluation result appended to CSV: {self.output_csv}")

    def signal_handler(self, sig, frame):
        print("\n" + "!" * 70)
        rospy.logwarn("GP-CBF interrupted; saving current trajectories and report.")
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
        planner = GaussianCbfEvalPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass