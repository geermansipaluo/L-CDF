#!/usr/bin/env python3
import rospy
import torch
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import time

# 🔴 关键替换：引入你完全重构后的、对齐 GCBF+ 星形多边异质图架构的网络
from model import DensityNet 
# 🔴 引入纯正的 PyG 异质图数据容器，彻底杜绝半图非图的割裂传参
from torch_geometric.data import HeteroData

# 导入高性能 CPU 凸优化求解器
from cvxopt.solvers import qp, options
from cvxopt import matrix
options['show_progress'] = False

# =========================================================================
# 1. CVXOPT 隐式激光点云安全过滤器 (纯 CPU 零拷贝运行)
# =========================================================================
class CvxoptLiDARCBFSafetyFilter:
    def __init__(self, k_closest=3):
        self.k_closest = k_closest

    def filter_control(self, u_nom, local_points, config, current_risk):
        """
        利用网络安全风险头直出的认知风险场强 current_risk 在线动态收缩安全边界
        """
        v_nom, w_nom = float(u_nom[0]), float(u_nom[1])
        l_k = config['l_k']
        r_ego = config['r_ego']
        
        # 风险场自适应融合：自车周围危险度越高，SDF-CBF收敛常数越趋向于它的收缩上限
        adaptive_gamma = config['gamma'] * (1.0 - float(current_risk) * 0.5)

        # 1. 构建 QP 目标代价函数: min ||u - u_nom||^2
        P = matrix(np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64))
        q = matrix(np.array([-2.0 * v_nom, -2.0 * w_nom], dtype=np.float64))

        G_list = []
        h_list = []

        # 2. 物理边界硬约束注入
        G_box = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float64)
        h_box = np.array([config['v_max'], -config['v_min'], config['w_max'], -config['w_min']], dtype=np.float64)
        G_list.append(G_box)
        h_list.append(h_box)

        # 3. 动态提取最临近的 K 个点构建隐式 CBF 不等式
        if len(local_points) > 0:
            dists = np.hypot(local_points[:, 0], local_points[:, 1])
            closest_indices = np.argsort(dists)[:self.k_closest]
            
            for idx in closest_indices:
                if idx >= len(local_points): continue
                x_i, y_i = local_points[idx, 0], local_points[idx, 1]
                
                h_i = (l_k - x_i)**2 + (0.0 - y_i)**2 - r_ego**2
                G_row = np.array([[-2.0 * (l_k - x_i), 2.0 * l_k * y_i]], dtype=np.float64)
                h_val = np.array([adaptive_gamma * h_i], dtype=np.float64)
                
                G_list.append(G_row)
                h_list.append(h_val)

        G = matrix(np.vstack(G_list))
        h = matrix(np.concatenate(h_list))

        try:
            result = qp(P, q, G, h)
            if result['status'] == 'optimal':
                u_opt = np.array(result['x']).flatten()
                return u_opt[0], u_opt[1]
            else:
                return 0.0, 0.0 
        except Exception as e:
            rospy.logerr_throttle(1.0, f"CVXOPT 拦截器求解异常: {str(e)}")
            return 0.0, 0.0

# =========================================================================
# 2. 智能体闭环追踪控制节点主类
# =========================================================================
class EllipseTracker:
    def __init__(self):
        self.start_time = time.time()
        rospy.init_node('ellipse_tracker_node')
        # 显式锁定 CPU 运行，防止 PCI-E 跨总线搬运延迟
        self.device = torch.device('cpu') 

        self.cbf_config = {
            'l_k': 0.15,          
            'r_ego': 0.31,        
            'gamma': 0.6,         
            'safety_margin': 0.14, 
            'v_min': -0.2, 'v_max': 0.8,
            'w_min': -1.5, 'w_max': 1.5
        }

        self.current_pose = [0.0, 0.0, 0.0]  
        self.target_pos = [11.0, 1.0]        
        self.pointcloud_local = np.zeros((0, 2)) 

        self.last_executed_v = 0.0
        self.last_executed_w = 0.0

        self.total_echo = 0
        self.inf_echo = 0

        # 先装配拦截器与发布器
        self.safety_filter = CvxoptLiDARCBFSafetyFilter(k_closest=3)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.debug_pub = rospy.Publisher('/processed_cloud', PointCloud2, queue_size=1)

        # 🟢【终极对接】：最后实例化模型，完全剔除已经废弃的 7维 state_dim 显式定义
        self.model_u = self.load_model()

        self.__sub_curr_state = rospy.Subscriber('/curr_state', Float32MultiArray, self.pose_callback, queue_size=10)
        self.__sub_global_cloud = rospy.Subscriber('/densitynet_input_points', PointCloud2, self.cloud_callback, queue_size=10)

        self.ctrl_timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        self.goal_timer = rospy.Timer(rospy.Duration(0.05), self.check_reach_goal)

        rospy.loginfo("🚀【三实体异质图 DensityNet 部署测试大闸】已全线合龙就位！")

    def load_model(self):
        model_path = "/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models/model_best.pt"
        try:
            # 🟢 完全兼容新版 model.py：隐藏层与训练时严格一致，不再需要向外暴露外层状态特征维
            model = DensityNet(hidden_dim=256)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)
            model = model.eval()
            rospy.loginfo("🎉【三实体对齐异质图 DensityNet】最优指导模型加载成功！")
            return model
        except Exception as e:
            rospy.logerr(f"模型权重装载灾难性断层: {str(e)}")
            raise RuntimeError(f"模型加载错误") from e

    def cloud_callback(self, msg):
        pts_gen = point_cloud2.read_points(msg, field_names=("x", "y"), skip_nans=True)
        self.pointcloud_local = np.array([[p[0], p[1]] for p in pts_gen])
        if len(self.pointcloud_local) > 0:
            self.publish_cloud(self.pointcloud_local)

    def pose_callback(self, msg):
        if len(msg.data) >= 3:
            self.current_pose = [msg.data[0], msg.data[1], msg.data[2]]

    def control_loop(self, event):
        self.total_echo += 1
        x, y, theta = self.current_pose[0], self.current_pose[1], self.current_pose[2]

        dx = float(self.target_pos[0]) - x
        dy = float(self.target_pos[1]) - y
        dist_to_goal_val = np.hypot(dx, dy)

        # 🟢【完全消除坐标系污染】：对齐数据生成端，将导航吸引力解耦转换至载体局部系
        target_local_x = dx * np.cos(theta) + dy * np.sin(theta)
        target_local_y = -dx * np.sin(theta) + dy * np.cos(theta)

        # 🟢 组装自车 4维 固有状态特征 [x, y, theta, dist_to_goal]
        X_step = np.array([x, y, theta, dist_to_goal_val], dtype=np.float32)

        # =========================================================================
        # 🟢【全新重构】：在线单帧多实体异质图组装（完全平替旧版手写 Padding）
        # =========================================================================
        graph_sample = HeteroData()
        
        # 实体 1：自车实体节点特征 [1, 4]
        graph_sample['ego'].x = torch.tensor(X_step, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 实体 2：目标吸引力虚拟节点特征 [1, 2] -> 灌入自车局部相对航向对准输入
        graph_sample['goal'].x = torch.tensor([target_local_x, target_local_y], dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 实体 3：激光雷达局部障碍物击中点实体节点特征 [K, 2] 与边关系编制
        num_pts = len(self.pointcloud_local)
        if num_pts == 0:
            # 防御性锁死：空图状态下强制赋予干净的空张量及 [2, 0] 边骨架，确保 GNN 内部 FiLM 短路安全释放
            graph_sample['point'].x = torch.zeros((0, 2), dtype=torch.float32).to(self.device)
            graph_sample['point', 'to', 'ego'].edge_index = torch.zeros((2, 0), dtype=torch.long).to(self.device)
        else:
            graph_sample['point'].x = torch.tensor(self.pointcloud_local, dtype=torch.float32).to(self.device)
            
            senders_p = torch.arange(num_pts, dtype=torch.long)
            receivers_p = torch.zeros(num_pts, dtype=torch.long)
            # 建立有向斥力避障消息边拓扑
            graph_sample['point', 'to', 'ego'].edge_index = torch.stack([senders_p, receivers_p], dim=0).to(self.device)

        # 建立有向引力导航消息边拓扑 (目标指向自车)
        graph_sample['goal', 'to', 'ego'].edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            # 🟢 一行闭环输入：前向传播只传递统一图大 batch 对象，彻底切断维度代数不匹配报错源！
            pred_action, pred_risk = self.model_u(graph_sample)
            
            # 异质图消息汇聚后直接双头解包
            v_nom, w_nom = pred_action.squeeze().cpu().numpy()
            current_risk = float(pred_risk.squeeze().cpu().item()) # 空间认知风险概率 \sigma

        # 空间物理硬门控拦截过滤大闸
        if num_pts > 0:
            dists = np.hypot(self.pointcloud_local[:, 0], self.pointcloud_local[:, 1])
            min_physical_dist = np.min(dists)
        else:
            min_physical_dist = float('inf')

        trigger_bound = self.cbf_config['r_ego'] + self.cbf_config['safety_margin']
        
        v_final, w_final = float(v_nom), float(w_nom)
        if min_physical_dist <= trigger_bound:
            try:
                u_nom_vec = np.array([v_nom, w_nom], dtype=np.float64)
                # 传入网络预测出来的实时风险度 current_risk，以在线自适应收缩凸优化可行域边界
                v_final, w_final = self.safety_filter.filter_control(
                    u_nom=u_nom_vec, local_points=self.pointcloud_local, config=self.cbf_config, current_risk=current_risk
                )
                if np.hypot(v_final - v_nom, w_final - w_nom) > 1e-2:
                    self.inf_echo += 1
            except Exception as e:
                rospy.logerr_throttle(1.0, f"CVXOPT 拦截器故障: {str(e)}")
                v_final, w_final = 0.0, 0.0

        self.last_executed_v = v_final
        self.last_executed_w = w_final
        self.publish_twist(v_final, w_final)

    def publish_twist(self, v, w):
        twist = Twist()
        twist.linear.x = float(v)
        twist.angular.z = float(w)
        self.cmd_vel_pub.publish(twist)

    def publish_cloud(self, points):
        header = Header(stamp=rospy.Time.now(), frame_id="velodyne") 
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        padded = np.hstack((points, np.zeros((len(points), 1))))
        pc_msg = point_cloud2.create_cloud(header, fields, padded)
        self.debug_pub.publish(pc_msg)

    def check_reach_goal(self, event):
        dist = np.linalg.norm(np.array(self.current_pose[:2]) - np.array(self.target_pos))
        if dist < 0.4:
            self.publish_twist(0.0, 0.0)
            inf_rate = (self.inf_echo / self.total_echo) if self.total_echo > 0 else 0.0
            rospy.loginfo(f"🎉【对齐战报】小车成功安全突围切入终点！")
            rospy.loginfo(f"📊 CVXOPT 过滤器拦截纠偏率: {inf_rate*100:.2f}% | 全闭环实车推演总耗时: {time.time()-self.start_time:.2f}s")
            self.goal_timer.shutdown()
            self.ctrl_timer.shutdown()

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        tracker = EllipseTracker()
        tracker.run()
    except rospy.ROSInterruptException:
        pass