#!/usr/bin/env python3
import os
import sys
import csv
import signal
import time

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


class ParametricEllipseTracker:
    """DensityNet 单模型闭环评测节点。

    用法示例：
      rosrun sensor_cdf traj_eval_sweep.py \
        _model_path:=/home/guo/L-CDF/src/sensor_cdf/scripts/save_models/DensityNet-demo2-dseed0-seed0/model_best_parametric_bc.pt \
        _num_demos:=2 _demo_seed:=0 _train_seed:=0 \
        _num_eval_episodes:=10 _test_target_seed:=2026 \
        _output_csv:=/home/guo/L-CDF/src/sensor_cdf/scripts/eval_metrics.csv
    """

    def __init__(self):
        self.start_time = time.time()
        rospy.init_node("parametric_ellipse_tracker_node", anonymous=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"当前 DensityNet 测试设备: [{self.device}]")

        # ============================================================
        # 1. ROS 私有参数：批量评测需要从脚本传入
        # ============================================================
        self.model_path = rospy.get_param(
            "~model_path",
            "/home/guo/L-CDF/src/sensor_cdf/scripts/save_models/DensityNet-demo2-dseed0-seed0/model_best_parametric_bc.pt",
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
        self.target_x_min = float(rospy.get_param("~target_x_min", 14.0))
        self.target_x_max = float(rospy.get_param("~target_x_max", 16.0))
        self.target_y_min = float(rospy.get_param("~target_y_min", -2.0))
        self.target_y_max = float(rospy.get_param("~target_y_max", 2.0))
        self.goal_radius = float(rospy.get_param("~goal_radius", 0.4))
        self.max_episode_time = float(rospy.get_param("~max_episode_time", 80.0))
        self.terminate_on_collision = bool(rospy.get_param("~terminate_on_collision", False))

        # 按你的要求：hidden_dim / graph_k / lambda_smooth 不在这里暴露。
        # 这里保持你本地 model.py 和训练设置一致即可。
        self.model_state_dim = 4
        self.model_hidden_dim = int(rospy.get_param("~hidden_dim", 256))

        os.makedirs(self.trajectory_save_dir, exist_ok=True)

        self.cbf_config = {
            "l_k": 0.2,
            "r_ego": 0.31,
            "v_min": -0.0,
            "v_max": 0.5,
            "w_min": -2.0,
            "w_max": 2.0,
        }

        self.current_pose = [0.0, 0.0, 0.0]
        self.pointcloud_local = np.zeros((0, 2), dtype=np.float32)
        self.last_executed_v = 0.0
        self.last_executed_w = 0.0

        # ============================================================
        # 2. 固定测试目标：所有模型必须用同一个 test_target_seed
        # ============================================================
        rng = np.random.default_rng(self.test_target_seed)
        self.test_targets = np.stack(
            [
                rng.uniform(self.target_x_min, self.target_x_max, size=self.total_eval_episodes),
                rng.uniform(self.target_y_min, self.target_y_max, size=self.total_eval_episodes),
            ],
            axis=1,
        ).astype(np.float32)

        self.episode_index = 0
        self.target_pos = self.test_targets[self.episode_index].tolist()
        self.current_episode_start_time = time.time()
        self.episode_finish_lock = False

        # ============================================================
        # 3. 固定环境障碍物和统计量
        # ============================================================
        self.obstacles = np.array(
            [
                [5.0, 0.05],
                [6.5, -0.5],
                [8.0, -2.5],
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

        # ============================================================
        # 4. 专家 QP 用于测试端生成 G/h
        # ============================================================
        self.local_expert = LocalSdfCdfPlanner()
        self.warmup_expert()

        # ============================================================
        # 5. ROS 通信
        # ============================================================
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
            "/curr_state", Float32MultiArray, self.pose_callback, queue_size=10
        )
        self.__sub_global_cloud = rospy.Subscriber(
            "/densitynet_input_points", PointCloud2, self.cloud_callback, queue_size=10
        )

        self.ctrl_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        self.goal_timer = rospy.Timer(rospy.Duration(0.05), self.check_reach_goal)

        rospy.loginfo("=" * 70)
        rospy.loginfo("DensityNet 自动评测节点启动")
        rospy.loginfo(f"model_path          = {self.model_path}")
        rospy.loginfo(f"num_demos           = {self.num_demos}")
        rospy.loginfo(f"demo_seed           = {self.demo_seed}")
        rospy.loginfo(f"train_seed          = {self.train_seed}")
        rospy.loginfo(f"num_eval_episodes   = {self.total_eval_episodes}")
        rospy.loginfo(f"test_target_seed    = {self.test_target_seed}")
        rospy.loginfo(f"output_csv          = {self.output_csv}")
        rospy.loginfo(f"first_target        = {self.target_pos}")
        rospy.loginfo("=" * 70)

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

    def load_model(self):
        model_path = self.resolve_model_path(self.model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        try:
            model = UNet(state_dim=self.model_state_dim, hidden_dim=self.model_hidden_dim)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device).eval()
            rospy.loginfo(f"模型装载成功: {model_path}")
            return model
        except Exception as e:
            rospy.logerr(f"模型加载失败: {str(e)}")
            raise RuntimeError("模型加载错误") from e

    def get_model_action(self, state_tensor, points_batch, G_cdf_tensor, h_cdf_tensor):
        """兼容两种模型输出：Tensor 或 (u_safe, u_nom)。

        你现在测试只需要最终控制输出；如果 model.forward 已经只返回一个 Tensor，
        这里会直接使用它。如果仍返回 tuple/list，这里自动取第一个。
        """
        with torch.no_grad():
            out = self.model(state_tensor, points_batch, G_cdf_tensor, h_cdf_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out

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
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = 0.0
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
            [8.0, -2.5],
            [10.0, -0.5],
        ]
        for i, p in enumerate(fixed_obstacles):
            self.set_gazebo_model_pose(f"cylinder_{i}", float(p[0]), float(p[1]), z=0.25)

        for i in range(len(fixed_obstacles), 8):
            self.set_gazebo_model_pose(f"cylinder_{i}", 80.0 + 3.0 * i, 20.0, z=0.25)

        rospy.logwarn("固定评测场景已同步到 Gazebo: cylinder_0~3 已摆放，其余 cylinder 已隐藏")

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
        padded = np.hstack((points, np.zeros((len(points), 1))))
        pc_msg = point_cloud2.create_cloud(header, fields, padded)
        self.debug_pub.publish(pc_msg)

    # ============================================================
    # ROS callback
    # ============================================================
    def cloud_callback(self, msg):
        if rospy.is_shutdown():
            return
        pts_gen = point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True)
        pts = np.array([[p[0], p[1]] for p in pts_gen], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] == 0:
            self.pointcloud_local = np.zeros((0, 2), dtype=np.float32)
        else:
            self.pointcloud_local = pts[:, :2]
            self.publish_cloud(self.pointcloud_local)

    def pose_callback(self, msg):
        if len(msg.data) >= 3:
            self.current_pose = [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])]

    # ============================================================
    # 控制主循环
    # ============================================================
    def control_loop(self, event):
        if self.episode_finish_lock or rospy.is_shutdown():
            return

        self.publish_target_marker()
        x, y, theta = self.current_pose
        l_k = self.cbf_config["l_k"]

        self.current_run_trajectory.append([x, y, theta, time.time() - self.start_time])

        # timeout 兜底，避免某个模型永远不到目标导致评测卡死
        if time.time() - self.current_episode_start_time > self.max_episode_time:
            rospy.logwarn(f"第 {self.episode_index + 1} 回合超时，判定为未到达。")
            self.finish_episode(arrived=False, reason="timeout")
            return

        # 碰撞检测
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

        state_array = np.array(
            [target_local_x, target_local_y, self.last_executed_v, self.last_executed_w],
            dtype=np.float32,
        )
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device)

        ego_p_x = x + l_k * np.cos(theta)
        ego_p_y = y + l_k * np.sin(theta)
        dx_p = float(self.target_pos[0]) - ego_p_x
        dy_p = float(self.target_pos[1]) - ego_p_y
        dist_p = np.hypot(dx_p, dy_p)

        if dist_to_goal_val > self.goal_radius:
            u_nom_local_np = np.array(
                [1.0 * dx_p / (dist_p + 1e-6), 1.0 * dy_p / (dist_p + 1e-6)],
                dtype=np.float32,
            )
        else:
            u_nom_local_np = np.array([0.0, 0.0], dtype=np.float32)

        is_empty = (
            self.pointcloud_local.shape[0] == 0
            or (self.pointcloud_local.shape[0] > 0 and self.pointcloud_local[0, 0] == 99 and self.pointcloud_local[0, 1] == 99)
        )

        if is_empty:
            # 空点云时保持你原来的测试逻辑：走 nominal
            v_global_x = u_nom_local_np[0]
            v_global_y = u_nom_local_np[1]
            v_pure_local_x = v_global_x * np.cos(theta) + v_global_y * np.sin(theta)
            v_pure_local_y = -v_global_x * np.sin(theta) + v_global_y * np.cos(theta)
            v_final = float(v_pure_local_x)
            w_final = float(v_pure_local_y / l_k)
        else:
            ego_p_local_jax = np.array([l_k, 0.0], dtype=np.float32)
            fixed_size = 200
            local_pts = self.pointcloud_local.astype(np.float32)
            if local_pts.shape[0] > fixed_size:
                local_pts_for_qp = local_pts[:fixed_size]
            elif local_pts.shape[0] < fixed_size:
                pad_box = np.full((fixed_size - local_pts.shape[0], 2), 99.0, dtype=np.float32)
                local_pts_for_qp = np.vstack([local_pts, pad_box])
            else:
                local_pts_for_qp = local_pts

            _, G_extracted, h_extracted = self.local_expert.solve_agent_qp_local(
                ego_p_local_jax,
                u_nom_local_np,
                local_pts_for_qp,
                np.array([target_local_x, target_local_y], dtype=np.float32),
            )

            G_cdf_tensor = torch.tensor(G_extracted, dtype=torch.float32).unsqueeze(0).to(self.device)
            h_cdf_tensor = torch.tensor(h_extracted, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 模型点云输入。点数太少时简单重复，避免 kNN 图构建极端退化。
            pc_model = local_pts
            if pc_model.shape[0] == 1:
                pc_model = np.repeat(pc_model, repeats=2, axis=0)
            pos_tensor = torch.tensor(pc_model, dtype=torch.float32)
            points_batch = Batch.from_data_list([Data(pos=pos_tensor)]).to(self.device)

            u_safe_pred = self.get_model_action(state_tensor, points_batch, G_cdf_tensor, h_cdf_tensor)
            u_safe_np = u_safe_pred.detach().cpu().numpy().flatten()

            if len(u_safe_np) < 2:
                v_final = 0.0
                w_final = 0.0
            else:
                v_final = float(u_safe_np[0])
                w_final = float(u_safe_np[1] / l_k)

        self.last_executed_v = v_final
        self.last_executed_w = w_final
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
        try:
            from std_srvs.srv import Empty

            rospy.wait_for_service("/gazebo/reset_simulation", timeout=1.0)
            reset_sim = rospy.ServiceProxy("/gazebo/reset_simulation", Empty)
            reset_sim()
        except Exception:
            os.system("rosservice call /gazebo/reset_simulation '{}' > /dev/null 2>&1")

        time.sleep(0.5)
        self.apply_fixed_obstacles_to_gazebo()

        self.episode_index += 1
        self.target_pos = self.test_targets[self.episode_index].tolist()
        self.current_pose = [0.0, 0.0, 0.0]
        self.pointcloud_local = np.zeros((0, 2), dtype=np.float32)
        self.last_executed_v = 0.0
        self.last_executed_w = 0.0
        self.current_run_trajectory = []
        self.collision_happened_in_current_run = False
        self.last_collision_time = 0.0
        self.current_episode_start_time = time.time()
        self.episode_finish_lock = False

        rospy.logerr(
            f"新一轮开始: {self.episode_index + 1}/{self.total_eval_episodes}, target={self.target_pos}"
        )

    def save_trajectory_data(self):
        try:
            tag = f"demo{self.num_demos}_dseed{self.demo_seed}_seed{self.train_seed}"
            trajectory_save_path = os.path.join(self.trajectory_save_dir, f"trajectory_{tag}.pt")
            torch.save(self.all_runs_trajectories, trajectory_save_path)
            rospy.loginfo(
                f"轨迹落盘成功: {len(self.all_runs_trajectories)} 回合 -> {trajectory_save_path}"
            )
        except Exception as e:
            rospy.logerr(f"保存轨迹数据失败: {str(e)}")

    def print_final_report(self):
        completed_runs = min(self.episode_index + 1, self.total_eval_episodes)
        if self.episode_finish_lock and completed_runs <= 0:
            completed_runs = 1
        completed_runs = max(completed_runs, 1)

        arrival_rate = self.reached_goals_count / completed_runs
        success_rate = self.perfect_runs_count / completed_runs
        collision_rate = self.collision_runs_count / completed_runs
        avg_collision_events = self.total_collision_events / completed_runs

        print("\n" + " DensityNet 自动评测报告 ".center(70, "="))
        print(f"模型路径             : {self.resolve_model_path(self.model_path)}")
        print(f"num_demos/demo_seed  : {self.num_demos}/{self.demo_seed}, train_seed={self.train_seed}")
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
