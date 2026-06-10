#!/usr/bin/env python3
import os, sys, time, signal
import rospy
import torch
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

from model import UNet
from torch_geometric.data import Data, Batch
from data_generate import LocalSdfCdfPlanner
from test_env_pool import get_env_pool

class ParametricEllipseTracker:
    def __init__(self):
        self.start_time = time.time()
        rospy.init_node('parametric_ellipse_tracker_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CBF 配置
        self.cbf_config = {'l_k':0.31,'r_ego':0.31,'v_min':0.0,'v_max':0.5,'w_min':-2,'w_max':2}

        # 初始状态
        self.current_pose = [0.0,0.0,0.0]
        self.goal_counter = 1
        seed = rospy.get_param("~random_seed",-1)
        self.rng = np.random.default_rng(None if seed<0 else int(seed))

        # 环境池
        self.env_pool = get_env_pool()
        self.current_env_id = -1
        self.current_env_name = ""
        self.target_pos = [13.0,0.0]
        self.current_meta_obstacles = []
        self.pointcloud_local = np.zeros((0, 2)) 
        self.last_executed_v = 0.0
        self.last_executed_w = 0.0

        # 数据统计
        self.all_runs_trajectories=[]
        self.current_run_trajectory=[]
        self.reached_goals_count=0
        self.perfect_runs_count=0
        self.collision_happened_in_current_run=False
        self.total_collision_events=0
        self.last_collision_time=0.0
        self.total_eval_episodes=10
        self.safety_threshold=0.5+self.cbf_config['r_ego']

        # 本地专家
        self.local_expert = LocalSdfCdfPlanner()

        # 发布器
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.debug_pub = rospy.Publisher('/processed_cloud', PointCloud2, queue_size=1)
        self.marker_pub = rospy.Publisher('/target_goal_marker', Marker, queue_size=1)

        # Gazebo 服务
        try:
            rospy.wait_for_service('/gazebo/set_model_state',timeout=5.0)
            self.set_model_state_srv = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
            rospy.loginfo("✅ Gazebo /set_model_state 服务已连接")
        except Exception as e:
            self.set_model_state_srv=None
            rospy.logwarn("❌ Gazebo /set_model_state 服务不可用: %s"%str(e))

        # 加载模型
        self.model = self.load_model()

        # 信号处理
        signal.signal(signal.SIGINT,self.signal_handler)
        signal.signal(signal.SIGTERM,self.signal_handler)

        # 订阅
        self.__sub_curr_state=rospy.Subscriber('/curr_state',Float32MultiArray,self.pose_callback,queue_size=10)
        self.__sub_global_cloud=rospy.Subscriber('/densitynet_input_points',PointCloud2,self.cloud_callback,queue_size=10)

        # 定时器
        self.ctrl_timer = rospy.Timer(rospy.Duration(0.1),self.control_loop)
        self.goal_timer = rospy.Timer(rospy.Duration(0.05),self.check_reach_goal)

        # 初始化第一轮
        self.prepare_new_episode()

    # ---------------- Gazebo 模型移动 ----------------
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
                rospy.logwarn(
                    f"移动模型失败: {model_name}, "
                    f"Gazebo message: {resp.status_message}"
                )

            return resp.success

        except Exception as e:
            rospy.logwarn(f"调用 /gazebo/set_model_state 失败: {model_name}, error={e}")
            return False

    def apply_meta_environment_to_gazebo(self, meta_obstacles):
        """
        将当前环境同步到 Gazebo。

        circle -> cylinder_0 ~ cylinder_7
        rect   -> 按尺寸选择固定 box 模型

        支持的矩形尺寸：
        a=0.6, b=0.2 -> box_060_020_0, box_060_020_1
        a=0.5, b=1.0 -> box_050_100_0, box_050_100_1
        a=0.8, b=0.8 -> box_080_080_0, box_080_080_1
        """

        max_cylinders = 8

        rect_model_pool = {
            (0.6, 0.2): ["box_060_020_0", "box_060_020_1"],
            (0.5, 1.0): ["box_050_100_0", "box_050_100_1"],
            (0.8, 0.8): ["box_080_080_0", "box_080_080_1"],
        }

        used_circle = 0
        used_rect_count = {
            (0.6, 0.2): 0,
            (0.5, 1.0): 0,
            (0.8, 0.8): 0,
        }

        # 1. 先把所有 cylinder 藏起来
        for i in range(max_cylinders):
            ok = self.set_gazebo_model_pose(
                f"cylinder_{i}",
                80.0 + 3.0 * i,
                20.0,
                z=0.25
            )
            if not ok:
                rospy.logwarn(f"未能移动 cylinder_{i}，请检查 world 中是否存在该模型")

        # 2. 先把所有 box 藏起来
        hide_idx = 0
        for model_list in rect_model_pool.values():
            for model_name in model_list:
                ok = self.set_gazebo_model_pose(
                    model_name,
                    100.0 + 3.0 * hide_idx,
                    25.0,
                    z=0.5
                )
                if not ok:
                    rospy.logwarn(f"未能移动 {model_name}，请检查 world 中是否存在该模型")
                hide_idx += 1

        # 3. 再放置当前环境真正需要的障碍物
        for obs in meta_obstacles:
            c = obs["center"]
            x, y = float(c[0]), float(c[1])

            if obs["type"] == "circle":
                if used_circle >= max_cylinders:
                    rospy.logwarn("circle 数量超过 cylinder 数量，多余 circle 被忽略")
                    continue

                model_name = f"cylinder_{used_circle}"
                ok = self.set_gazebo_model_pose(model_name, x, y, z=0.25)

                if not ok:
                    rospy.logwarn(f"设置 {model_name} 失败")

                used_circle += 1

            elif obs["type"] == "rect":
                a = round(float(obs["a"]), 2)
                b = round(float(obs["b"]), 2)
                key = (a, b)

                if key not in rect_model_pool:
                    rospy.logwarn(
                        f"未知 rect 尺寸 a={a}, b={b}，请在 rect_model_pool 和 world 中添加对应 box"
                    )
                    continue

                slot = used_rect_count[key]

                if slot >= len(rect_model_pool[key]):
                    rospy.logwarn(
                        f"rect 尺寸 a={a}, b={b} 的数量超过预设模型数量，多余 rect 被忽略"
                    )
                    continue

                model_name = rect_model_pool[key][slot]
                ok = self.set_gazebo_model_pose(model_name, x, y, z=0.5)

                if not ok:
                    rospy.logwarn(f"设置 {model_name} 失败")

                used_rect_count[key] += 1

            else:
                rospy.logwarn(f"未知障碍物类型: {obs['type']}，已忽略")

        rospy.logwarn(
            f"🌍 Gazebo 环境同步完成: "
            f"circle={used_circle}, "
            f"rect_060_020={used_rect_count[(0.6, 0.2)]}, "
            f"rect_050_100={used_rect_count[(0.5, 1.0)]}, "
            f"rect_080_080={used_rect_count[(0.8, 0.8)]}"
        )

    def prepare_new_episode(self):
        self.current_env_id = int(self.rng.integers(0, len(self.env_pool)))
        env = self.env_pool[self.current_env_id]

        self.current_env_name = env.get("name", f"env_{self.current_env_id}")
        self.current_meta_obstacles = env["meta_obstacles"]

        # 目标点仍然随机
        self.target_pos = [
            float(self.rng.uniform(12.0, 14.0)),
            float(self.rng.uniform(-2.0, 2.0))
        ]

        rospy.logwarn("=" * 60)
        rospy.logwarn(f"🌍 新回合环境 id={self.current_env_id}, name={self.current_env_name}")
        rospy.logwarn(f"🎯 target={self.target_pos}")

        for i, obs in enumerate(self.current_meta_obstacles):
            if obs["type"] == "circle":
                rospy.logwarn(
                    f"  obs[{i}] circle center={obs['center']}, r={obs['r']}"
                )
            elif obs["type"] == "rect":
                rospy.logwarn(
                    f"  obs[{i}] rect center={obs['center']}, a={obs['a']}, b={obs['b']}"
                )

        self.apply_meta_environment_to_gazebo(self.current_meta_obstacles)

    # ---------------- ROS 回调 ----------------
    def cloud_callback(self,msg):
        pts_gen=point_cloud2.read_points(msg,field_names=("x","y"),skip_nans=True)
        self.pointcloud_local=np.array([[p[0],p[1]] for p in pts_gen])
        if len(self.pointcloud_local)>0: self.publish_cloud(self.pointcloud_local)

    def pose_callback(self,msg):
        if len(msg.data)>=3: self.current_pose=[msg.data[0],msg.data[1],msg.data[2]]

    # ---------------- 控制循环 ----------------
    def control_loop(self, event):
        self.publish_target_marker()
        x, y, theta = self.current_pose[0], self.current_pose[1], self.current_pose[2]
        l_k = self.cbf_config['l_k']

        # 记录当前回合轨迹：x, y, theta, time, env_id, target_x, target_y
        self.current_run_trajectory.append([
            x, y, theta,
            time.time() - self.start_time,
            self.current_env_id,
            float(self.target_pos[0]),
            float(self.target_pos[1])
        ])

        # 在线几何碰撞检测：支持 circle 和 rect
        ego_center = np.array([x, y], dtype=np.float32)

        if self.check_collision_with_meta_obstacles(ego_center):
            self.collision_happened_in_current_run = True

            current_time = time.time()
            if current_time - self.last_collision_time > 0.5:
                self.total_collision_events += 1
                self.last_collision_time = current_time
                rospy.logerr(
                    f"💥 [碰撞警告] env={self.current_env_name}, "
                    f"target=({self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}), "
                    f"当前总碰撞计步: {self.total_collision_events}"
                )

        # --- A. 状态及意图解算 ---
        dx = float(self.target_pos[0]) - x
        dy = float(self.target_pos[1]) - y
        dist_to_goal_val = np.hypot(dx, dy)

        target_local_x = dx * np.cos(theta) + dy * np.sin(theta)
        target_local_y = -dx * np.sin(theta) + dy * np.cos(theta)

        state_array = np.array(
            [target_local_x, target_local_y, self.last_executed_v, self.last_executed_w],
            dtype=np.float32
        )
        state_tensor = torch.tensor(
            state_array,
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # 计算标称运动学前瞻控制器动作
        ego_p_x = x + l_k * np.cos(theta)
        ego_p_y = y + l_k * np.sin(theta)

        dx_p = float(self.target_pos[0]) - ego_p_x
        dy_p = float(self.target_pos[1]) - ego_p_y
        dist_p = np.hypot(dx_p, dy_p)

        if dist_to_goal_val > 0.44:
            u_nom_local_np = np.array([
                1.0 * dx_p / (dist_p + 1e-6),
                1.0 * dy_p / (dist_p + 1e-6)
            ], dtype=np.float32)
        else:
            u_nom_local_np = np.array([0.0, 0.0], dtype=np.float32)

        is_empty = (
            self.pointcloud_local.shape[0] == 0
            or (
                self.pointcloud_local[0, 0] == 99
                and self.pointcloud_local[0, 1] == 99
            )
        )

        # --- B. 双轨制控制解算 ---
        if is_empty:
            v_global_x = u_nom_local_np[0]
            v_global_y = u_nom_local_np[1]

            v_pure_local_x = v_global_x * np.cos(theta) + v_global_y * np.sin(theta)
            v_pure_local_y = -v_global_x * np.sin(theta) + v_global_y * np.cos(theta)

            v_final = float(v_pure_local_x)
            w_final = float(v_pure_local_y / l_k)

        else:
            ego_p_local_jax = np.array([l_k, 0.0])
            fixed_size = 200
            local_pts = self.pointcloud_local

            if local_pts.shape[0] > fixed_size:
                local_pts = local_pts[:fixed_size]
            elif local_pts.shape[0] < fixed_size:
                pad_box = np.full((fixed_size - local_pts.shape[0], 2), 99.0)
                local_pts = np.vstack([local_pts, pad_box])

            sol_6d_raw, G_extracted, h_extracted = self.local_expert.solve_agent_qp_local(
                ego_p_local_jax,
                u_nom_local_np,
                local_pts,
                np.array([target_local_x, target_local_y])
            )

            G_cdf_tensor = torch.tensor(
                G_extracted,
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            h_cdf_tensor = torch.tensor(
                h_extracted,
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            pos_tensor = torch.tensor(self.pointcloud_local, dtype=torch.float32)
            points_batch = Batch.from_data_list([Data(pos=pos_tensor)]).to(self.device)

            with torch.no_grad():
                u_safe_pred = self.model(
                    state_tensor,
                    points_batch,
                    G_cdf_tensor,
                    h_cdf_tensor
                )
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

    # ---------------- 碰撞检测 ----------------
    def check_collision_with_meta_obstacles(self,ego_xy):
        r_ego=self.cbf_config['r_ego']
        for obs in self.current_meta_obstacles:
            c=obs['center']
            if obs['type']=='circle':
                if np.linalg.norm(ego_xy-c)<float(obs['r'])+r_ego: return True
            elif obs['type']=='rect':
                dx=abs(ego_xy[0]-c[0]); dy=abs(ego_xy[1]-c[1])
                if dx<float(obs['a'])+r_ego and dy<float(obs['b'])+r_ego: return True
        return False

    # ---------------- 发布 ----------------
    def publish_twist(self,v,w):
        twist=Twist(); twist.linear.x=float(v); twist.angular.z=float(w)
        self.cmd_vel_pub.publish(twist)

    def publish_cloud(self,points):
        header=Header(stamp=rospy.Time.now(),frame_id="velodyne")
        fields=[PointField('x',0,PointField.FLOAT32,1),PointField('y',4,PointField.FLOAT32,1),PointField('z',8,PointField.FLOAT32,1)]
        padded=np.hstack((points,np.zeros((len(points),1))))
        pc_msg=point_cloud2.create_cloud(header,fields,padded)
        self.debug_pub.publish(pc_msg)

    def publish_target_marker(self):
        marker=Marker()
        marker.header.frame_id="world"; marker.header.stamp=rospy.Time.now()
        marker.ns="target"; marker.id=0; marker.type=Marker.SPHERE; marker.action=Marker.ADD
        marker.pose.position.x=float(self.target_pos[0]); marker.pose.position.y=float(self.target_pos[1]); marker.pose.position.z=0.2
        marker.pose.orientation.w=1.0
        marker.scale.x=marker.scale.y=marker.scale.z=0.5
        marker.color.r=0.0; marker.color.g=1.0; marker.color.b=0.0; marker.color.a=1.0
        self.marker_pub.publish(marker)

    # ---------------- 回合逻辑 ----------------
    def check_reach_goal(self,event):
        dist=np.linalg.norm(np.array(self.current_pose[:2])-np.array(self.target_pos))
        if dist<0.4:
            self.publish_twist(0.0,0.0)
            self.reached_goals_count+=1
            if not self.collision_happened_in_current_run: self.perfect_runs_count+=1
            self.save_trajectory_data()
            # Reset Gazebo 和状态
            try:
                from std_srvs.srv import Empty
                rospy.wait_for_service('/gazebo/reset_simulation', timeout=1.0)
                reset_sim=rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
                reset_sim()
            except:
                os.system("rosservice call /gazebo/reset_simulation '{}' > /dev/null 2>&1")
            self.current_pose=[0.0,0.0,0.0]
            self.current_run_trajectory=[]
            self.last_collision_time=0.0
            self.collision_happened_in_current_run=False
            # 切换到下一轮随机环境
            self.prepare_new_episode()
            self.goal_counter+=1

    # ---------------- 数据落盘 ----------------
    def save_trajectory_data(self):
        if len(self.current_run_trajectory)>0:
            self.all_runs_trajectories.append(self.current_run_trajectory)
        torch.save(self.all_runs_trajectories,"/home/guo/L-CDF/densitynet_trajectory.pt")

    # ---------------- 模型加载 ----------------
    def load_model(self):
        model_path="/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models/model_best_parametric_bc.pt"
        model=UNet(state_dim=4, hidden_dim=256)
        state_dict=torch.load(model_path,map_location=self.device,weights_only=True)
        model.load_state_dict(state_dict,strict=True)
        return model.to(self.device).eval()

    # ---------------- 信号 ----------------
    def signal_handler(self,sig,frame):
        self.save_trajectory_data()
        rospy.signal_shutdown("KeyboardInterrupt")
        sys.exit(0)

    # ---------------- 主循环 ----------------
    def run(self):
        rospy.spin()

if __name__=="__main__":
    try:
        tracker=ParametricEllipseTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass
