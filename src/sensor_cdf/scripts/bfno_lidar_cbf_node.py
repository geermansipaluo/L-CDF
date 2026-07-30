#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS/Gazebo online BFNO-Poisson-CBF node for Jackal + LiDAR.

Pipeline:
    /velodyne_points + /base_pose_ground_truth
        -> ego-centric world-aligned local occupancy grid
        -> BFNO/SOR Poisson safety field
        -> look-ahead-point CBF-QP over Jackal [v, w]
        -> /cmd_vel

The local grid is centered at the robot but its axes are aligned with the world
frame.  This avoids rotating gradients before using Jackal kinematics.
"""

import math
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

import rospy
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, Float32, String
from visualization_msgs.msg import Marker
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class BFNOJackalSafetyNode:
    def __init__(self):
        rospy.init_node("bfno_lidar_cbf_node", anonymous=False)

        # ------------------------------------------------------------------
        # Paths and imports
        # ------------------------------------------------------------------
        default_code_dir = os.path.expanduser("~/L-CDF/src/sensor_cdf/scripts/Born-FNO-CBF")
        self.bfno_code_dir = rospy.get_param("~bfno_code_dir", default_code_dir)
        if self.bfno_code_dir and os.path.isdir(self.bfno_code_dir):
            sys.path.insert(0, self.bfno_code_dir)
            sys.path.insert(0, os.path.join(self.bfno_code_dir, "bfno"))
        else:
            rospy.logwarn("bfno_code_dir does not exist yet: %s", self.bfno_code_dir)

        try:
            from bfno.lidar_local_map import LocalLidarGridBuilder
            from bfno.BFNOPoissonCBF import PoissonCBFFilter
        except Exception as exc:
            rospy.logerr("Cannot import BFNO modules. Set ~bfno_code_dir to your Born-FNO-CBF folder. Error: %s", exc)
            raise

        self.LocalLidarGridBuilder = LocalLidarGridBuilder
        self.PoissonCBFFilter = PoissonCBFFilter

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.pointcloud_topic = rospy.get_param("~pointcloud_topic", "/velodyne_points")
        self.odom_topic = rospy.get_param("~odom_topic", "/base_pose_ground_truth")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        self.grid_n = int(rospy.get_param("~grid_n", 128))
        self.local_size = float(rospy.get_param("~local_size", 6.0))
        self.max_range = float(rospy.get_param("~max_range", 3.0))
        self.robot_radius = float(rospy.get_param("~robot_radius", 0.35))
        self.hit_radius = float(rospy.get_param("~hit_radius", 0.08))
        self.min_z = float(rospy.get_param("~min_z", -0.25))
        self.max_z = float(rospy.get_param("~max_z", 1.20))
        self.max_points = int(rospy.get_param("~max_points", 20000))
        self.world_aligned_grid = bool(rospy.get_param("~world_aligned_grid", True))

        self.backend = rospy.get_param("~backend", "bfno")  # bfno, sor_cpu, sor_gpu
        self.model_path = rospy.get_param("~model_path", "")
        self.h_born_mode = rospy.get_param("~h_born_mode", "auto")
        self.projection_steps = int(rospy.get_param("~projection_steps", 0))
        self.device = rospy.get_param("~device", "cuda")
        self.method = rospy.get_param("~forcing_method", "constant")
        self.f_constant = float(rospy.get_param("~f_constant", -5.0))
        self.b_flux = float(rospy.get_param("~b_flux", -5.0))
        self.derivative_smooth_sigma = float(rospy.get_param("~derivative_smooth_sigma", 0.75))

        self.goal_x = float(rospy.get_param("~goal_x", 15.0))
        self.goal_y = float(rospy.get_param("~goal_y", 0.0))
        self.goal_tolerance = float(rospy.get_param("~goal_tolerance", 0.35))
        self.control_rate = float(rospy.get_param("~control_rate", 10.0))
        self.update_field_every = int(rospy.get_param("~update_field_every", 1))

        self.lookahead = float(rospy.get_param("~lookahead", 0.35))
        self.gamma = float(rospy.get_param("~gamma", 2.0))
        self.v_nom_gain = float(rospy.get_param("~v_nom_gain", 0.45))
        self.w_nom_gain = float(rospy.get_param("~w_nom_gain", 1.4))
        self.v_min = float(rospy.get_param("~v_min", -0.25))
        self.v_max = float(rospy.get_param("~v_max", 0.8))
        self.w_max = float(rospy.get_param("~w_max", 1.6))
        self.qp_w_weight = float(rospy.get_param("~qp_w_weight", 0.35))
        self.use_ht = bool(rospy.get_param("~use_ht", False))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        self.builder = self.LocalLidarGridBuilder(
            local_size=self.local_size,
            grid_n=self.grid_n,
            max_range=self.max_range,
            robot_radius=self.robot_radius,
            hit_radius=self.hit_radius,
            min_z=self.min_z,
            max_z=self.max_z,
            mark_free=True,
        )

        dummy_map = self.builder.empty_map()
        dummy_env = self.builder.to_environment(dummy_map, b_flux_value=self.b_flux)
        self.cbf = self.PoissonCBFFilter(
            dummy_env,
            method=self.method,
            f_constant=self.f_constant,
            b_flux=self.b_flux,
            backend=self.backend,
            h_born_mode=self.h_born_mode,
            pde_projection_steps=self.projection_steps,
            derivative_smooth_sigma=self.derivative_smooth_sigma,
            device=self.device,
            build_on_init=False,
            verbose=False,
        )
        if self.backend.lower() == "bfno":
            if not self.model_path:
                raise RuntimeError("backend=bfno requires ~model_path.")
            self.cbf.load_model_once(self.model_path)

        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=5.0)
            self.set_model_state_srv = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            rospy.loginfo("Gazebo /set_model_state 服务已连接")
        except Exception as e:
            self.set_model_state_srv = None
            rospy.logwarn(f"Gazebo /set_model_state 服务不可用: {e}")

        self.apply_fixed_obstacles_to_gazebo()

        # ------------------------------------------------------------------
        # ROS I/O
        # ------------------------------------------------------------------
        self.cloud_msg: Optional[PointCloud2] = None
        self.robot_xy = np.zeros(2, dtype=np.float64)
        self.robot_yaw = 0.0
        self.have_pose = False
        self.last_h_center = None
        self.last_field_time = None
        self.loop_count = 0
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.world_frame

        rospy.Subscriber(self.pointcloud_topic, PointCloud2, self.cloud_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.h_pub = rospy.Publisher("~h_value", Float32, queue_size=1)
        self.runtime_pub = rospy.Publisher("~runtime_profile", String, queue_size=1)
        self.occ_pub = rospy.Publisher("~local_occupancy", OccupancyGrid, queue_size=1, latch=False)
        self.free_pub = rospy.Publisher("~local_free", OccupancyGrid, queue_size=1, latch=False)
        self.unknown_pub = rospy.Publisher("~local_unknown", OccupancyGrid, queue_size=1, latch=False)
        self.field_pub = rospy.Publisher("~safety_field", PointCloud2, queue_size=1)
        self.path_pub = rospy.Publisher("~trajectory", Path, queue_size=1)
        self.goal_pub = rospy.Publisher("~goal_marker", Marker, queue_size=1)
        self.lookahead_pub = rospy.Publisher("~lookahead_marker", Marker, queue_size=1)

        rospy.on_shutdown(self.stop_robot)
        rospy.loginfo("BFNO LiDAR CBF node initialized. backend=%s, grid=%dx%d, rays/test points from %s", self.backend, self.grid_n, self.grid_n, self.pointcloud_topic)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def cloud_cb(self, msg: PointCloud2):
        self.cloud_msg = msg

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.robot_xy[:] = [p.x, p.y]
        self.robot_yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.have_pose = True
        ps = PoseStamped()
        ps.header = msg.header
        ps.header.frame_id = self.world_frame
        ps.pose = msg.pose.pose
        self.path_msg.header.stamp = rospy.Time.now()
        self.path_msg.poses.append(ps)
        if len(self.path_msg.poses) > 3000:
            self.path_msg.poses = self.path_msg.poses[-3000:]

    # ------------------------------------------------------------------
    # Mapping and visualization
    # ------------------------------------------------------------------
    def pointcloud_to_numpy(self, msg: PointCloud2) -> np.ndarray:
        pts = []
        for k, p in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if k >= self.max_points:
                break
            pts.append(p)
        if not pts:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32)

    def make_grid_msg(self, arr: np.ndarray, stamp: rospy.Time, unknown_value: int = 0) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.world_frame
        msg.info.resolution = self.local_size / float(self.grid_n)
        msg.info.width = self.grid_n
        msg.info.height = self.grid_n
        msg.info.origin.position.x = float(self.robot_xy[0] - self.local_size / 2.0)
        msg.info.origin.position.y = float(self.robot_xy[1] - self.local_size / 2.0)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        a = np.asarray(arr)
        data = np.zeros((self.grid_n, self.grid_n), dtype=np.int8)
        data[a > 0.5] = 100
        data[(a < 0) if np.issubdtype(a.dtype, np.signedinteger) else np.zeros_like(a, dtype=bool)] = unknown_value
        # ROS OccupancyGrid data starts at bottom-left row; our arr[y,x] uses y up in local coordinates.
        msg.data = data.reshape(-1).tolist()
        return msg

    def make_field_cloud(self, h_grid: np.ndarray, stamp: rospy.Time, stride: int = 2) -> PointCloud2:
        stride = max(1, int(stride))
        h = np.asarray(h_grid, dtype=np.float32)
        h_min, h_max = float(np.min(h)), float(np.max(h))
        denom = max(h_max - h_min, 1e-6)
        points = []
        origin_x = float(self.robot_xy[0] - self.local_size / 2.0)
        origin_y = float(self.robot_xy[1] - self.local_size / 2.0)
        dx = self.local_size / float(self.grid_n - 1)
        for iy in range(0, self.grid_n, stride):
            y = origin_y + iy * dx
            for ix in range(0, self.grid_n, stride):
                x = origin_x + ix * dx
                val = float(h[iy, ix])
                intensity = (val - h_min) / denom
                points.append((x, y, 0.05 + 0.20 * intensity, intensity))
        header = Header(stamp=stamp, frame_id=self.world_frame)
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("intensity", 12, PointField.FLOAT32, 1),
        ]
        return pc2.create_cloud(header, fields, points)

    def publish_marker(self, pub, x, y, ns, rgba, scale=0.25):
        m = Marker()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.world_frame
        m.ns = ns
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.25
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = float(scale)
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        pub.publish(m)

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
            [5.0, 0.15],
            [6.5, -0.5],
            # [8.0, -2.5],
            [10.0, -0.5],
        ]
        for i, p in enumerate(fixed_obstacles):
            self.set_gazebo_model_pose(f"cylinder_{i}", float(p[0]), float(p[1]), z=0.25)
        for i in range(len(fixed_obstacles), 8):
            self.set_gazebo_model_pose(f"cylinder_{i}", 80.0 + 3.0 * i, 20.0, z=0.25)
        rospy.logwarn("固定评测场景已同步到 Gazebo: cylinder_0~2 已摆放，其余 cylinder 已隐藏")

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------
    def nominal_cmd(self) -> Tuple[float, float, float, float]:
        goal = np.array([self.goal_x, self.goal_y], dtype=np.float64)
        vec = goal - self.robot_xy
        dist = float(np.linalg.norm(vec))
        desired_yaw = math.atan2(vec[1], vec[0]) if dist > 1e-6 else self.robot_yaw
        err = angle_wrap(desired_yaw - self.robot_yaw)
        v_nom = self.v_nom_gain * dist * max(math.cos(err), -0.4)
        w_nom = self.w_nom_gain * err
        v_nom = float(np.clip(v_nom, self.v_min, self.v_max))
        w_nom = float(np.clip(w_nom, -self.w_max, self.w_max))
        return v_nom, w_nom, dist, err

    def get_h_derivatives_at_lookahead(self):
        pc_x = self.local_size / 2.0 + self.lookahead * math.cos(self.robot_yaw)
        pc_y = self.local_size / 2.0 + self.lookahead * math.sin(self.robot_yaw)
        pc_x = float(np.clip(pc_x, 0.0, self.local_size))
        pc_y = float(np.clip(pc_y, 0.0, self.local_size))
        pt = np.array([[pc_y, pc_x]], dtype=np.float64)
        h_val = float(self.cbf.interp_h(pt)[0])
        dh = np.array([float(self.cbf.interp_hx(pt)[0]), float(self.cbf.interp_hy(pt)[0])], dtype=np.float64)
        return h_val, dh, pc_x, pc_y

    def solve_vw_qp(self, v_nom: float, w_nom: float, h_val: float, dh: np.ndarray, h_t: float = 0.0):
        theta = self.robot_yaw
        l = self.lookahead
        A = np.array([
            [math.cos(theta), -l * math.sin(theta)],
            [math.sin(theta),  l * math.cos(theta)],
        ], dtype=np.float64)
        a = dh @ A
        # CBF: h_t + dh^T p_dot_c >= -gamma h
        b = -self.gamma * h_val - float(h_t)
        u_nom = np.array([v_nom, w_nom], dtype=np.float64)
        lb = np.array([self.v_min, -self.w_max], dtype=np.float64)
        ub = np.array([self.v_max,  self.w_max], dtype=np.float64)
        W = np.diag([1.0, self.qp_w_weight])

        if np.linalg.norm(a) < 1e-9:
            return np.clip(u_nom, lb, ub)

        if minimize is not None:
            def obj(u):
                d = u - u_nom
                return float(d @ W @ d)
            cons = ({"type": "ineq", "fun": lambda u: float(a @ u - b)},)
            res = minimize(
                obj,
                x0=np.clip(u_nom, lb, ub),
                bounds=[(lb[0], ub[0]), (lb[1], ub[1])],
                constraints=cons,
                method="SLSQP",
                options={"ftol": 1e-8, "maxiter": 40, "disp": False},
            )
            if res.success and not np.any(np.isnan(res.x)):
                return np.asarray(res.x, dtype=np.float64)

        # Closed-form fallback: project onto the half-space, then clip.
        if a @ u_nom >= b:
            return np.clip(u_nom, lb, ub)
        Winv = np.diag([1.0, 1.0 / max(self.qp_w_weight, 1e-6)])
        denom = float(a @ Winv @ a.T)
        u = u_nom + ((b - a @ u_nom) / max(denom, 1e-12)) * (Winv @ a.T)
        return np.clip(u, lb, ub)

    def publish_cmd(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        try:
            self.publish_cmd(0.0, 0.0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def spin(self):
        rate = rospy.Rate(self.control_rate)
        while not rospy.is_shutdown():
            if self.cloud_msg is None or not self.have_pose:
                rate.sleep()
                continue

            stamp = rospy.Time.now()
            points = self.pointcloud_to_numpy(self.cloud_msg)
            local_map = self.builder.from_points_local(points, yaw=self.robot_yaw, world_aligned=self.world_aligned_grid)
            env = self.builder.to_environment(local_map, b_flux_value=self.b_flux)

            t0 = time.perf_counter()
            if self.loop_count % max(self.update_field_every, 1) == 0:
                profile = self.cbf.update_safety_field(env, backend=self.backend, projection_steps=self.projection_steps, profile=False)
            else:
                profile = self.cbf.profile
            field_update_time = time.perf_counter() - t0

            v_nom, w_nom, dist_goal, yaw_err = self.nominal_cmd()
            if dist_goal < self.goal_tolerance:
                self.publish_cmd(0.0, 0.0)
                rospy.loginfo_throttle(1.0, "Reached goal: distance %.3f", dist_goal)
                rate.sleep()
                continue

            h_val, dh, pc_x, pc_y = self.get_h_derivatives_at_lookahead()
            h_t = 0.0
            if self.use_ht:
                now = time.time()
                if self.last_h_center is not None and self.last_field_time is not None:
                    h_t = (h_val - self.last_h_center) / max(now - self.last_field_time, 1e-3)
                self.last_h_center = h_val
                self.last_field_time = now

            v_safe, w_safe = self.solve_vw_qp(v_nom, w_nom, h_val, dh, h_t=h_t)
            self.publish_cmd(v_safe, w_safe)

            if self.publish_debug:
                self.occ_pub.publish(self.make_grid_msg(local_map.occupancy, stamp))
                self.free_pub.publish(self.make_grid_msg(local_map.free, stamp))
                self.unknown_pub.publish(self.make_grid_msg(local_map.unknown, stamp))
                self.field_pub.publish(self.make_field_cloud(self.cbf.h_grid, stamp, stride=2))
                self.h_pub.publish(Float32(data=float(h_val)))
                self.path_pub.publish(self.path_msg)
                self.publish_marker(self.goal_pub, self.goal_x, self.goal_y, "goal", (0.0, 1.0, 0.0, 1.0), scale=0.35)
                look_x = self.robot_xy[0] + self.lookahead * math.cos(self.robot_yaw)
                look_y = self.robot_xy[1] + self.lookahead * math.sin(self.robot_yaw)
                self.publish_marker(self.lookahead_pub, look_x, look_y, "lookahead", (1.0, 0.2, 0.0, 1.0), scale=0.18)
                runtime = {
                    "backend": self.backend,
                    "field_update_ms": 1000.0 * field_update_time,
                    "h": h_val,
                    "dh_norm": float(np.linalg.norm(dh)),
                    "v_nom": v_nom,
                    "w_nom": w_nom,
                    "v_safe": float(v_safe),
                    "w_safe": float(w_safe),
                    "num_points": int(points.shape[0]),
                    "num_occ": int(local_map.occupancy.sum()),
                    **{k: float(v) * 1000.0 for k, v in profile.items()},
                }
                self.runtime_pub.publish(String(data=str(runtime)))

            rospy.loginfo_throttle(
                1.0,
                "BFNO-CBF h=%.3f | v %.2f->%.2f, w %.2f->%.2f | occ=%d | update=%.1f ms",
                h_val, v_nom, v_safe, w_nom, w_safe, int(local_map.occupancy.sum()), 1000.0 * field_update_time,
            )
            self.loop_count += 1
            rate.sleep()


if __name__ == "__main__":
    node = BFNOJackalSafetyNode()
    node.spin()
