#!/usr/bin/env python3
import os
import sys
import signal
import rospy
import torch
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
import time
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

# 🟢【重构核心 1】引入你最新重构的同质图 UNet 以及 PyG 同质数据大闸
from model import UNet
from torch_geometric.data import Data, Batch

# 引入高性能 JAX 专家，在测试端扮演数据落盘和 DAgger 的“黄金自监督标签提取器”
from data_generate import LocalSdfCdfPlanner

# =========================================================================
# 智能体闭环追踪控制节点主类（100% 对齐 6维松弛参数化行为克隆版本）
# =========================================================================
class ParametricEllipseTracker:
    def __init__(self):
        self.start_time = time.time()
        rospy.init_node('parametric_ellipse_tracker_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo(f"显卡高速通道已拉起！当前 DensityNet 锁定的硬件计算核心为: [{self.device}]")

        self.cbf_config = {
            'l_k': 0.33,          # 前瞻点杠杆距离 L
            'r_ego': 0.31,        # 自车膨胀物理硬壳
            'v_min': -1.2, 'v_max': 1.2,
            'w_min': -2, 'w_max': 2
        }

        self.current_pose = [0.0, 0.0, 0.0]  # [x, y, theta]
        self.goal_counter = 1 
        self.target_pos = [15.0, -0.5]     
        
        self.pointcloud_local = np.zeros((0, 2)) 
        self.last_executed_v = 0.0
        self.last_executed_w = 0.0

        # -----------------------------------------------------------------
        # 📊 新增功能：测试评价与指标统计计数器
        # -----------------------------------------------------------------
        # 静态障碍物配置：圆形半径 0.5m，小车自身物理半径 r_ego=0.31m，安全临界距离阈值 = 0.5 + 0.31 = 0.81m
        self.obstacles = np.array([
            [5.0, 0.05],
            # [6.5, -0.5],
            [8.0, -2.5],
            [10.0, -0.5]
        ])
        self.safety_threshold = 0.5 + self.cbf_config['r_ego'] # 0.81 米
        
        # 统计数据结构
        self.all_runs_trajectories = []  # 存储所有回合的轨迹
        self.current_run_trajectory = [] # 当前回合的轨迹点序列 [(x, y, theta, time), ...]
        
        self.total_eval_episodes = 1    # 目标评测 10 次
        self.reached_goals_count = 0     # 到达目标的次数
        self.perfect_runs_count = 0      # 无碰撞且成功到达的次数 (成功率基数)
        
        self.collision_happened_in_current_run = False # 当前回合是否碰过
        self.total_collision_events = 0  # 10个回合总共发生的碰撞采样点数
        self.last_collision_time = 0.0   # 碰撞冷却锁，防止单次碰撞在 0.1s 循环内高频重复计步
        # -----------------------------------------------------------------

        # 实例化测试端局部专家大闸
        self.local_expert = LocalSdfCdfPlanner()

        rospy.loginfo("⏳ 检测到 JAX 专家引擎，正在执行硬件管线热身 (Warm-up)...")
        t_warm = time.time()
        fake_ego_p = np.array([self.cbf_config['l_k'], 0.0])
        fake_u_nom = np.array([0.5, 0.0], dtype=np.float32)
        fake_pts = np.random.uniform(-1.0, 1.0, (200, 2))  
        fake_target = np.array([5.0, 0.0])
        _, _, _ = self.local_expert.solve_agent_qp_local(fake_ego_p, fake_u_nom, fake_pts, fake_target)
        rospy.loginfo(f"🔥 JAX 管线热身完毕！耗时: {time.time() - t_warm:.2f}s")
        
        # 数据落盘路径
        self.source_dataset_path = "/home/guo/L-CDF/data_degenarate_test.pt"
        self.output_dataset_path = "/home/guo/L-CDF/data_degenarate_test_combine.pt"
        self.trajectory_save_path = "/home/guo/L-CDF/densitynet_trajectory.pt" # 轨迹保存路径
        self.online_collected_buffer = [] 
        self.is_saving_lock = False 
        
        if os.path.exists(self.source_dataset_path):
            try:
                rospy.loginfo(f"⏳ 正在读取 13维可微参数化历史数据集基底: {self.source_dataset_path} ...")
                self.base_dataset = torch.load(self.source_dataset_path, map_location='cpu', weights_only=False)
                self.original_data_count = len(self.base_dataset)
                rospy.loginfo(f"✅ 历史旧数据装载成功！基底容量: {self.original_data_count} 帧")
            except Exception as e:
                rospy.logerr(f"老数据集装载失败，原因: {str(e)}")
                self.base_dataset = []
                self.original_data_count = 0
        else:
            self.base_dataset = []
            self.original_data_count = 0
            
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.debug_pub = rospy.Publisher('/processed_cloud', PointCloud2, queue_size=1)

        try:
            rospy.wait_for_service('/gazebo/set_model_state', timeout=5.0)
            self.set_model_state_srv = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            rospy.loginfo("✅ Gazebo /set_model_state 服务已连接")
        except Exception as e:
            self.set_model_state_srv = None
            rospy.logwarn(f"❌ Gazebo /set_model_state 服务不可用: {e}")

        # 呼叫最新含有 qpth 屏障层的控制器
        self.model = self.load_model()

        # 中断信号锁
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.__sub_curr_state = rospy.Subscriber('/curr_state', Float32MultiArray, self.pose_callback, queue_size=10)
        self.__sub_global_cloud = rospy.Subscriber('/densitynet_input_points', PointCloud2, self.cloud_callback, queue_size=10)
        self.marker_pub = rospy.Publisher('/target_goal_marker', Marker, queue_size=1)
        self.apply_fixed_obstacles_to_gazebo()
        
        # 定时器线程拉起
        self.ctrl_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        self.goal_timer = rospy.Timer(rospy.Duration(0.05), self.check_reach_goal) # 🟢 恢复并开启目标/碰撞监测定时器

        rospy.loginfo("🚀【DAgger 可微参数化自监督测试系统】部署就位！已挂载 10 回合全自动化在线指标考核大闸")

    def load_model(self):
        model_path = "/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models/new_loss/DensityNet-demo48-dseed1-seed1/model_best_parametric_bc.pt"
        try:
            model = UNet(
                state_dim=4,
                hidden_dim=256,
                graph_k=5,
                lambda_smooth=25,
                ablation='full',
            )
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device).eval()
            rospy.loginfo("🎉【6维松弛参数化可微控制网络 UNet】装载成功！")
            return model
        except Exception as e:
            rospy.logerr(f"参数化模型加载灾难性断层: {str(e)}")
            raise RuntimeError(f"模型加载错误") from e

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
                    f"移动模型失败: {model_name}, Gazebo message: {resp.status_message}"
                )
            return resp.success

        except Exception as e:
            rospy.logwarn(f"调用 /gazebo/set_model_state 失败: {model_name}, error={e}")
            return False


    def apply_fixed_obstacles_to_gazebo(self):
        fixed_obstacles = [
            [5.0, 0.05],
            # [6.5, -0.5],
            [8.0, -2.5],
            [10.0, -0.5],
        ]

        for i, p in enumerate(fixed_obstacles):
            self.set_gazebo_model_pose(
                f"cylinder_{i}",
                float(p[0]),
                float(p[1]),
                z=0.25
            )

        # 把多余 cylinder 藏远，避免上一版环境残留
        for i in range(len(fixed_obstacles), 8):
            self.set_gazebo_model_pose(
                f"cylinder_{i}",
                80.0 + 3.0 * i,
                20.0,
                z=0.25
            )

        rospy.logwarn("🌍 固定轨迹场景已同步到 Gazebo: cylinder_0~3 已摆放，其余 cylinder 已隐藏")


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
    
    def signal_handler(self, sig, frame):
        print("\n" + "!"*60)
        rospy.logwarn("[控制台紧急拦截] 捕获到手动结束信号！正在紧急将测试轨迹及增量冲刷至磁盘...")
        self.save_trajectory_data() # 中断时同步保存轨迹
        self.execute_data_flush()
        self.print_final_report()   # 强行打印当前的统计阶段成果
        print("!"*60 + "\n")
        rospy.signal_shutdown("User KeyboardInterrupt")
        sys.exit(0)

    def save_trajectory_data(self):
        """ 🟢 功能1：将包含多回合的完整测试轨迹数据落盘保存 """
        try:
            # 如果当前回合还有残留轨迹，先并入总池
            if len(self.current_run_trajectory) > 0:
                self.all_runs_trajectories.append(self.current_run_trajectory)
            
            torch.save(self.all_runs_trajectories, self.trajectory_save_path)
            rospy.loginfo(f"💾 [轨迹落盘] 成功保存 {len(self.all_runs_trajectories)} 回合轨迹数据至: {self.trajectory_save_path}")
        except Exception as e:
            rospy.logerr(f"保存轨迹数据时发生灾难性错误: {str(e)}")

    def print_final_report(self):
        """ 📊 聚合打印当前实验考核大看板 """
        completed_runs = self.goal_counter - 1 if self.goal_counter <= self.total_eval_episodes else self.total_eval_episodes
        if completed_runs == 0: completed_runs = 1 # 防止除 0
        
        arrival_rate = (self.reached_goals_count / completed_runs) * 100.0
        success_rate = (self.perfect_runs_count / completed_runs) * 100.0
        avg_collisions = self.total_collision_events / completed_runs

        print("\n" + " 🎯 DensityNet 在线实机闭环评测报告 🎯 ".center(60, "="))
        print(f" 当前总考核进度 (Progress)   : {completed_runs} / {self.total_eval_episodes} 回合")
        print(f" 🎯 到达率 (Arrival Rate)    : {arrival_rate:.2f}%  ({self.reached_goals_count}/{completed_runs})")
        print(f" 🏆 成功率 (Success Rate)    : {success_rate:.2f}%  ({self.perfect_runs_count}/{completed_runs}, 含义：无碰撞到达)")
        print(f" 🛑 平均碰撞次数 (Avg Collide): {avg_collisions:.2f} 次/每回合 (总计触发 {self.total_collision_events} 帧采样点违规)")
        print("="*60 + "\n")

    def execute_data_flush(self):
        if self.is_saving_lock:
            return
        self.is_saving_lock = True
        self.publish_twist(0.0, 0.0) 
        
        if len(self.online_collected_buffer) > 0:
            rospy.loginfo(f"💾 正在将在线捕获到的 {len(self.online_collected_buffer)} 帧 13维黄金样本并入全局池...")
            self.base_dataset.extend(self.online_collected_buffer)
            torch.save(self.base_dataset, self.output_dataset_path)
            
            new_data_count = len(self.base_dataset)
            self.original_data_count = new_data_count
            self.online_collected_buffer = [] 
        self.is_saving_lock = False

    def cloud_callback(self, msg):
        if rospy.is_shutdown():
            return
        pts_gen = point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True)
        self.pointcloud_local = np.array([[p[0], p[1]] for p in pts_gen])
        if len(self.pointcloud_local) > 0:
            self.publish_cloud(self.pointcloud_local)

    def pose_callback(self, msg):
        if len(msg.data) >= 3:
            self.current_pose = [msg.data[0], msg.data[1], msg.data[2]]

    def control_loop(self, event):
        self.publish_target_marker()
        x, y, theta = self.current_pose[0], self.current_pose[1], self.current_pose[2]
        l_k = self.cbf_config['l_k']

        # 🟢 功能1：增量式记录当前控制周期的物理轨迹点
        self.current_run_trajectory.append([x, y, theta, time.time() - self.start_time])

        # 🟢 功能2：在线高保真几何碰撞状态检测
        # 计算当前自车中心 (x, y) 到 4 个圆形障碍物圆心的几何距离
        ego_center = np.array([x, y])
        distances_to_obs = np.linalg.norm(self.obstacles - ego_center, axis=1)
        
        # 只要任意一个距离小于临界安全半径（障碍半径 + 自车膨胀外壳），即判定发生碰撞
        if np.any(distances_to_obs < self.safety_threshold):
            self.collision_happened_in_current_run = True
            
            # 引入 0.5s 硬件碰撞计步冷却锁，防止 10Hz 循环内单次撞墙疯狂刷几十次碰撞计数
            current_time = time.time()
            if current_time - self.last_collision_time > 0.5:
                self.total_collision_events += 1
                self.last_collision_time = current_time
                rospy.logerr(f"💥 [碰撞警告] 检测到车体侵入障碍物安全红线！当前总碰撞计步: {self.total_collision_events}")

        # --- A. 状态及意图解算 ---
        dx = float(self.target_pos[0]) - x
        dy = float(self.target_pos[1]) - y
        dist_to_goal_val = np.hypot(dx, dy)

        target_local_x = dx * np.cos(theta) + dy * np.sin(theta)
        target_local_y = -dx * np.sin(theta) + dy * np.cos(theta)
        
        state_array = np.array([
            target_local_x,
            target_local_y,
            self.last_executed_v,
            self.last_executed_w,
        ], dtype=np.float32)
        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device) 

        target_local_np = np.array([target_local_x, target_local_y], dtype=np.float32)
        dist_local = np.linalg.norm(target_local_np)
        NOMINAL_SPEED = 1.0

        # 计算标称运动学前瞻控制器动作
        if dist_to_goal_val > 0.44 and dist_local > 0.1:
            u_nom_local_np = NOMINAL_SPEED * target_local_np / (dist_local + 1e-6)
        else:
            u_nom_local_np = np.zeros(2, dtype=np.float32)

        is_empty = (self.pointcloud_local.shape[0] == 0) or (self.pointcloud_local[0,0] == 99 and self.pointcloud_local[0,1] == 99)

        # --- B. 双轨制控制解算 ---
        if is_empty:
            u_safe_np = u_nom_local_np
            v_final = u_safe_np[0]
            w_final = u_safe_np[1]/l_k

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
                ego_p_local_jax, u_nom_local_np, local_pts, np.array([target_local_x, target_local_y])
            )
            
            G_cdf_tensor = torch.tensor(G_extracted, dtype=torch.float32).unsqueeze(0).to(self.device) 
            h_cdf_tensor = torch.tensor(h_extracted, dtype=torch.float32).unsqueeze(0).to(self.device) 
            
            pos_tensor = torch.tensor(self.pointcloud_local, dtype=torch.float32)
            points_batch = Batch.from_data_list([Data(pos=pos_tensor)]).to(self.device)

            with torch.no_grad():
                u_safe_pred, _ = self.model(state_tensor, points_batch, G_cdf_tensor, h_cdf_tensor)
                u_safe_np = u_safe_pred.detach().cpu().numpy().flatten()
                if len(u_safe_np) < 2:
                    v_final = 0.0
                    w_final = 0.0
                v_final = u_safe_np[0]
                w_final = u_safe_np[1]/l_k

        self.last_executed_v = u_safe_np[0]
        self.last_executed_w = u_safe_np[1]
        
        self.publish_twist(v_final, w_final)

    def publish_twist(self, v, w):
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_vel_pub.publish(twist)

    def publish_cloud(self, points):
        header = Header(stamp=rospy.Time.now(), frame_id="velodyne") 
        fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1), PointField('z', 8, PointField.FLOAT32, 1)]
        padded = np.hstack((points, np.zeros((len(points), 1))))
        pc_msg = point_cloud2.create_cloud(header, fields, padded)
        self.debug_pub.publish(pc_msg)

    def check_reach_goal(self, event):
        """ 🟢 核心重构逻辑：判定到达靶点，执行多回合统计滚动与重置 """
        dist = np.linalg.norm(np.array(self.current_pose[:2]) - np.array(self.target_pos))
        
        # 触发到达阈值
        if dist < 0.4:
            self.publish_twist(0.0, 0.0)
            rospy.loginfo(f"🎉【第 {self.goal_counter} 轮特种兵冲刺圆满结束】自车完美切入靶点状态！")
            
            # 更新到达率和成功率的基础计数
            self.reached_goals_count += 1
            if not self.collision_happened_in_current_run:
                self.perfect_runs_count += 1
                rospy.loginfo("🏆 达成一次完美通关（零碰撞到达目标点）！")
            else:
                rospy.logwarn("⚠️ 虽然成功到达目标点，但由于中途发生过碰撞，此回合不计入绝对成功率。")
            
            # 将当前回合的单条轨迹归档到总池中
            self.all_runs_trajectories.append(self.current_run_trajectory)
            
            # 实时更新并打印当前的阶段性实验看板
            self.print_final_report()

            # 🟢 检验是否跑满了 10 次的终极大考
            if self.goal_counter >= self.total_eval_episodes:
                rospy.logerr(f"🏁【10 回合全自动化在线指标考核圆满结束】正在封盘数据...")
                self.save_trajectory_data() # 跑满 10 次，落盘轨迹文件
                self.execute_data_flush()
                self.print_final_report()   # 最终看板定格打印
                rospy.signal_shutdown("Evaluation Completed Successfully")
                sys.exit(0)

            # 每轮切入目标后，触发 DAgger 内存池与磁盘老数据的融合
            self.execute_data_flush()

            # 一键空间传送复位 Gazebo
            rospy.logwarn("🔄 [空间闪现大闸启动] -> 正在重置仿真世界，瞬移小车至原点 (0,0)...")
            try:
                from std_srvs.srv import Empty
                rospy.wait_for_service('/gazebo/reset_simulation', timeout=1.0)
                reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
                reset_sim()
            except Exception:
                os.system("rosservice call /gazebo/reset_simulation '{}' > /dev/null 2>&1")
                pass

            # 环境和临时变量状态彻底复位
            self.current_pose = [0.0, 0.0, 0.0]
            self.last_executed_v = 0.0
            self.last_executed_w = 0.0
            self.current_run_trajectory = [] # 清空当前回合轨迹缓冲区
            self.collision_happened_in_current_run = False # 重置单回合碰撞状态位
            time.sleep(0.5) 

            # 固定靶点测试（如需随机，可换回下方随机注释）
            self.target_pos = [15.0, 0.0]
            # next_target_x = np.random.uniform(10.0, 12.0)
            # next_target_y = np.random.uniform(-3.0, 3.0)
            # self.target_pos = [next_target_x, next_target_y]
            
            self.goal_counter += 1
            rospy.logerr(f"🔥 [新一轮突围拉满] -> 第 {self.goal_counter} 回合考核开始，冲刺开始！")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        tracker = ParametricEllipseTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass