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
from std_msgs.msg import Header
import time

# 引入三实体异质图模型与 PyG 容器
from model import DensityNet 
from torch_geometric.data import HeteroData

# 导入高性能 CPU 凸优化求解器
from cvxopt.solvers import qp, options
from cvxopt import matrix
options['show_progress'] = False

# =========================================================================
# 1. 基于前瞻点平面单积分器解耦的 3约束无箱体 LiDAR-CBF 安全过滤器
# =========================================================================
class CvxoptLiDARCBFSafetyFilter:
    def __init__(self, k_closest=3):
        self.k_closest = k_closest

    def filter_control(self, u_nom, local_points, config, current_risk):
        """
        前瞻点平面单积分器解耦安全过滤器（3路并行硬约束）
        """
        v_nom, w_nom = float(u_nom[0]), float(u_nom[1])
        l_k = config['l_k']
        r_ego = config['r_ego']
        
        # 恢复网络动态认知风险 Gamma 自适应
        adaptive_gamma = config['gamma'] * (1.0 - float(current_risk) * 0.5)

        # 1. 【正向投影】：转换至前瞻点 P 载体系下的 2D 平面标称速度
        vx_p_nom = v_nom
        vy_p_nom = l_k * w_nom

        P = matrix(np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64))
        q = matrix(np.array([-2.0 * vx_p_nom, -2.0 * vy_p_nom], dtype=np.float64))

        G_list = []
        h_list = []

        # 提取最近的 3 个激光点，独立组装平面单积分器 CBF 约束
        if len(local_points) > 0:
            dists = np.hypot(local_points[:, 0], local_points[:, 1])
            closest_indices = np.argsort(dists)[:self.k_closest]
            
            for idx in closest_indices:
                if idx >= len(local_points): continue
                x_i, y_i = local_points[idx, 0], local_points[idx, 1]
                
                h_i = (l_k - x_i)**2 + (0.0 - y_i)**2 - r_ego**2
                dx_l = l_k - x_i
                dy_l = 0.0 - y_i
                
                # 平面单积分器标准型: -2*dx*vx - 2*dy*vy <= gamma * h
                G_row = np.array([[-2.0 * dx_l, -2.0 * dy_l]], dtype=np.float64)
                h_val_array = np.array([adaptive_gamma * h_i], dtype=np.float64)
                
                G_list.append(G_row)
                h_list.append(h_val_array)

        if len(G_list) > 0:
            G = matrix(np.vstack(G_list))
            h = matrix(np.concatenate(h_list))
        else:
            G, h = None, None

        try:
            result = qp(P, q, G, h)
            if result['status'] == 'optimal':
                u_opt = np.array(result['x']).flatten()
                # 2. 【逆向投影】：无缝映射回差速底盘实际物理执行量
                v_final = u_opt[0]
                w_final = u_opt[1] / l_k
                return v_final, w_final
            else:
                return v_nom, w_nom  
        except Exception as e:
            rospy.logerr_throttle(1.0, f"CVXOPT 平面解耦拦截器求解异常: {str(e)}")
            return v_nom, w_nom

# =========================================================================
# 2. 智能体闭环追踪控制节点主类
# =========================================================================
class EllipseTracker:
    def __init__(self):
        self.start_time = time.time()
        rospy.init_node('ellipse_tracker_node')
        self.device = torch.device('cpu') 

        self.cbf_config = {
            'l_k': 0.15,          
            'r_ego': 0.31,        
            'gamma': 1.0,           
            'safety_margin': 1,  
            'v_min': -0.2, 'v_max': 0.8,
            'w_min': -1.5, 'w_max': 1.5
        }

        self.current_pose = [0.0, 0.0, 0.0]  
        self.goal_counter = 1 
        # 初始目标靶点锚定在密集障碍物攻坚区
        self.target_pos = [np.random.uniform(10.0, 12.0), np.random.uniform(-3.0, 3.0)]        
        
        self.pointcloud_local = np.zeros((0, 2)) 
        self.last_executed_v = 0.0
        self.last_executed_w = 0.0

        self.total_echo = 0
        self.inf_echo = 0
        
        # 数据集增量追加系统参数
        self.source_dataset_path = "/home/guo/L-CDF/src/sensor_cdf/scripts/saved_data/dataset.pt"
        self.output_dataset_path = "/home/guo/L-CDF/src/sensor_cdf/scripts/saved_data/dataset_combine.pt"
        self.online_collected_buffer = [] 
        self.is_saving_lock = False 
        
        if os.path.exists(self.source_dataset_path):
            try:
                rospy.loginfo(f"⏳ 正在读取历史离线旧数据集: {self.source_dataset_path} ...")
                self.base_dataset = torch.load(self.source_dataset_path, map_location='cpu',weights_only=False)
                self.original_data_count = len(self.base_dataset)
                rospy.loginfo(f"✅ 历史旧数据装载成功！基底样本量: {self.original_data_count} 帧")
            except Exception as e:
                rospy.logerr(f"老数据集读取失败，已创建全新空集。原因: {str(e)}")
                self.base_dataset = []
                self.original_data_count = 0
        else:
            rospy.logwarn(f"未找到指定的旧数据集，系统将从0开始收集新数据。")
            self.base_dataset = []
            self.original_data_count = 0
            
        self.safety_filter = CvxoptLiDARCBFSafetyFilter(k_closest=3)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.debug_pub = rospy.Publisher('/processed_cloud', PointCloud2, queue_size=1)

        self.model_u = self.load_model()

        # 中断信号锁
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.__sub_curr_state = rospy.Subscriber('/curr_state', Float32MultiArray, self.pose_callback, queue_size=10)
        self.__sub_global_cloud = rospy.Subscriber('/densitynet_input_points', PointCloud2, self.cloud_callback, queue_size=10)

        self.ctrl_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        self.goal_timer = rospy.Timer(rospy.Duration(0.05), self.check_reach_goal)

        rospy.loginfo("🚀【DAgger 标称增强型自监督系统】挂载成功！磁盘基底容量: %d 帧", self.original_data_count)
        rospy.logwarn(f"🎯 [初始攻坚步指派] -> 目标靶点已锁定: [{self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}]")

    def signal_handler(self, sig, frame):
        print("\n" + "!"*60)
        rospy.logwarn("[控制台紧急拦截] 捕获到用户执行了 Ctrl+C 退出指令！正在打包保存内存样本...")
        self.execute_data_flush()
        print("!"*60 + "\n")
        rospy.signal_shutdown("User KeyboardInterrupt")
        sys.exit(0)

    def execute_data_flush(self):
        if self.is_saving_lock:
            return
        self.is_saving_lock = True
        self.publish_twist(0.0, 0.0) 
        
        if len(self.online_collected_buffer) > 0:
            rospy.loginfo(f"💾 正在将本次运行新捕获到的 {len(self.online_collected_buffer)} 帧危险边界样本并入老数据集...")
            
            # 🟢 在老数据集列表的屁股后面追加这一个回合捕获到的新数据
            self.base_dataset.extend(self.online_collected_buffer)
            
            # 🟢 保存合并后的完整大图集
            torch.save(self.base_dataset, self.output_dataset_path)
            
            new_data_count = len(self.base_dataset)
            print("\n" + " DAgger Online Dataset Fusion Report ".center(60, "="))
            print(f"📊 离线老数据集基底容量 (Old Baseline): {self.original_data_count} 帧")
            print(f"➕ 本次在线交互拦截新增 (Online Added): {len(self.online_collected_buffer)} 帧")
            print(f"🔥 完美融合后最终数据集容量 (Total Fused): {new_data_count} 帧")
            print(f"💾 最终落盘二进制大对象存储于: {self.output_dataset_path}")
            print("="*60 + "\n")
            
            # 同步更新计数，防止重复累加，清空缓冲区
            self.original_data_count = new_data_count
            self.online_collected_buffer = [] 
        else:
            rospy.loginfo("💡 检查缓冲区：中途未产生任何边界拦截样本，无需覆写磁盘。")
        self.is_saving_lock = False

    def load_model(self):
        model_path = "/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models/model_best.pt"
        try:
            model = DensityNet(hidden_dim=512)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model = model.eval()
            rospy.loginfo("🎉【三实体对齐异质图 DensityNet】最优指导模型装载成功！")
            return model
        except Exception as e:
            rospy.logerr(f"模型权重装载灾难性断层: {str(e)}")
            raise RuntimeError(f"模型加载错误") from e

    def cloud_callback(self, msg):
        pts_gen = point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True)
        self.pointcloud_local = np.array([[p[0], p[1]] for p in pts_gen])
        if len(self.pointcloud_local) > 0:
            print("接收到点云数据了")
            self.publish_cloud(self.pointcloud_local)

    def pose_callback(self, msg):
        if len(msg.data) >= 3:
            self.current_pose = [msg.data[0], msg.data[1], msg.data[2]]

    def calculate_nominal_input(self):
        """
        📐 标准前瞻点名义引力路径追踪算法
        """
        x, y, theta = self.current_pose[0], self.current_pose[1], self.current_pose[2]
        l_k = self.cbf_config['l_k']
        ego_p_x = x + l_k * np.cos(theta)
        ego_p_y = y + l_k * np.sin(theta)
        dx = float(self.target_pos[0]) - ego_p_x
        dy = float(self.target_pos[1]) - ego_p_y
        dist_to_goal = np.hypot(dx, dy)
        if dist_to_goal > 0.1:
            u_nom_x = 1.2 * dx / (dist_to_goal + 1e-6)
            u_nom_y = 1.2 * dy / (dist_to_goal + 1e-6)
        else:
            u_nom_x = 0.0; u_nom_y = 0.0
        return np.array([u_nom_x, u_nom_y], dtype=np.float32)

    def convert_to_diff_drive(self, u_control):
        theta = self.current_pose[2]
        l_k = self.cbf_config['l_k']
        u_x, u_y = u_control[0], u_control[1]
        v_nom_kin = u_x * np.cos(theta) + u_y * np.sin(theta)
        omega_nom_kin = (-u_x * np.sin(theta) + u_y * np.cos(theta)) / l_k
        return float(v_nom_kin), float(omega_nom_kin)

    def control_loop(self, event):
        self.total_echo += 1
        x, y, theta = self.current_pose[0], self.current_pose[1], self.current_pose[2]

        dx = float(self.target_pos[0]) - x
        dy = float(self.target_pos[1]) - y
        dist_to_goal_val = np.hypot(dx, dy)

        target_local_x = dx * np.cos(theta) + dy * np.sin(theta)
        target_local_y = -dx * np.sin(theta) + dy * np.cos(theta)

        X_step = np.array([x, y, theta, dist_to_goal_val], dtype=np.float32)

        # 编制单样本图数据骨架
        graph_sample = HeteroData()
        graph_sample['ego'].x = torch.tensor(X_step, dtype=torch.float32).unsqueeze(0).to(self.device)
        graph_sample['goal'].x = torch.tensor([target_local_x, target_local_y], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        num_pts = len(self.pointcloud_local)
        
        # 1. 优先提取当前纯粹运动学的标称控制器动作解算值
        nominal_vector = self.calculate_nominal_input()
        v_nominal_kin, omega_nominal_kin = self.convert_to_diff_drive(nominal_vector)
        v_nominal_kin = np.clip(v_nominal_kin, self.cbf_config['v_min'], self.cbf_config['v_max'])
        omega_nominal_kin = np.clip(omega_nominal_kin, self.cbf_config['w_min'], self.cbf_config['w_max'])

        # 2. 🟢【核心重构双轨制】：空旷平缓地带直接由标称控制器直驱接管，避免网络盲目打转！
        if len(self.pointcloud_local)>0:
            if self.pointcloud_local[0,0] == 99 and self.pointcloud_local[0,1]==99:
                print("当前点云数据为空，激活标称控制器")
                graph_sample['point'].x = torch.zeros((0, 2), dtype=torch.float32).to(self.device)
                graph_sample['point', 'to', 'ego'].edge_index = torch.zeros((2, 0), dtype=torch.long).to(self.device)
                graph_sample['goal', 'to', 'ego'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)
                graph_sample['ego', 'to', 'goal'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)
                
                # 无障碍物时，绝对名义控制量直驱
                v_final = float(v_nominal_kin)
                w_final = float(omega_nominal_kin)
                
                # 平衡下采样：只保留 1% 的空旷巡航直行数据，避免数据集中充斥太多直行样本导致过拟合
                if np.random.rand() < 0.01:
                    y_step = np.array([v_final, w_final, 0.0, float(v_nominal_kin), float(omega_nominal_kin)], dtype=np.float32)
                    graph_sample.y = torch.tensor(y_step, dtype=torch.float32).unsqueeze(0)
                    self.online_collected_buffer.append(graph_sample)
                    
                self.last_executed_v = v_final
                self.last_executed_w = w_final
                self.publish_twist(v_final, w_final)
                return

        # 3. 有障碍物时：走神经网络 + 后置平面解耦 QP 联合防线
        graph_sample['point'].x = torch.tensor(self.pointcloud_local, dtype=torch.float32).to(self.device)
        senders_p = torch.arange(num_pts, dtype=torch.long)
        receivers_p = torch.zeros(num_pts, dtype=torch.long)
        graph_sample['point', 'to', 'ego'].edge_index = torch.stack([senders_p, receivers_p], dim=0).to(self.device)
        graph_sample['goal', 'to', 'ego'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)
        graph_sample['ego', 'to', 'goal'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            pred_action, pred_risk = self.model_u(graph_sample)
            v_nom, w_nom = pred_action.squeeze().cpu().numpy()
            current_risk = float(pred_risk.squeeze().cpu().item())
            print(f"当前风险度：{current_risk}")

        dists = np.hypot(self.pointcloud_local[:, 0], self.pointcloud_local[:, 1])
        min_physical_dist = np.min(dists)
        trigger_bound = self.cbf_config['r_ego'] + self.cbf_config['safety_margin']
        
        v_final, w_final = float(v_nom), float(w_nom)
        
        # if min_physical_dist <= trigger_bound:
        #     try:
        #         # print("⚠️ 触发物理探针边界，开始在线纠正动作错误！！！！")
        #         u_nom_vec = np.array([v_nom, w_nom], dtype=np.float64)
        #         v_final, w_final = self.safety_filter.filter_control(
        #             u_nom=u_nom_vec, local_points=self.pointcloud_local, config=self.cbf_config, current_risk=current_risk
        #         )
                
        #         # 挂载真实的纠偏修正动作标签作为自监督补课样本
        #         y_step = np.array([float(v_final), float(w_final), 0.0, float(v_nominal_kin), float(omega_nominal_kin)], dtype=np.float32)
        #         graph_sample.y = torch.tensor(y_step, dtype=torch.float32).unsqueeze(0)
        #         self.online_collected_buffer.append(graph_sample)
                
        #         if np.hypot(v_final - v_nom, w_final - w_nom) > 1e-2:
        #             self.inf_echo += 1
        #     except Exception as e:
        #         rospy.logerr_throttle(1.0, f"CVXOPT 拦截器故障: {str(e)}")
        #         v_final, w_final = v_nom, w_nom

        # 执行动作限幅，防止执行器超限震荡
        v_final = np.clip(v_final, self.cbf_config['v_min'], self.cbf_config['v_max'])
        w_final = np.clip(w_final, self.cbf_config['w_min'], self.cbf_config['w_max'])

        self.last_executed_v = v_final
        self.last_executed_w = w_final
        self.publish_twist(v_final, w_final/2)

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
        dist = np.linalg.norm(np.array(self.current_pose[:2]) - np.array(self.target_pos))
        if dist < 0.4:
            self.publish_twist(0.0, 0.0)
            rospy.loginfo(f"🎉【第 {self.goal_counter} 轮攻坚步圆满结束】自车切入靶点目标状态！")
            
            # 数据增量大回流
            self.execute_data_flush()

            # 🟢 一键空间传送复位：呼叫 ROS 后门，秒回绝对零点重新冲刺
            rospy.logwarn("🔄 [空间闪现大闸启动] -> 正在重置仿真世界，瞬移小车至原点 (0,0)...")
            try:
                from std_srvs.srv import Empty
                rospy.wait_for_service('/gazebo/reset_simulation', timeout=1.0)
                reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
                reset_sim()
            except Exception:
                os.system("rosservice call /gazebo/reset_simulation '{}' > /dev/null 2>&1")
                pass

            # 消除突冲延迟
            self.current_pose = [0.0, 0.0, 0.0]
            self.last_executed_v = 0.0
            self.last_executed_w = 0.0
            time.sleep(0.3) 

            # 刷新下一个位于密集障碍区的全新攻坚目标点：横轴 10~12，纵轴 -3~3
            next_target_x = np.random.uniform(10.0, 12.0)
            next_target_y = np.random.uniform(-3.0, 3.0)
            self.target_pos = [next_target_x, next_target_y]
            
            self.goal_counter += 1
            rospy.logerr(f"🔥 [新一轮突围拉满] -> 下一个靶点已下发: [{next_target_x:.2f}, {next_target_y:.2f}]，小车开始全力冲刺！")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        tracker = EllipseTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass