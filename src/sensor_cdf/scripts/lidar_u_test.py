#!/usr/bin/env python3
import rospy
import torch
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from local_planner_cdf import cdf_control
from model_u import UNet
from torch_geometric.data import Data
import signal 
import sys
import time

class PointCloudProcessor:
    # 🔴 修复 1：将最大感知距离从 2.0m 统一调整为 3.0m，与完美的桥节点和离线训练集完全一致！
    def __init__(self, max_points=200, max_range=3.0):
        self.max_points = max_points
        self.max_range = max_range
        self.min_distance = 0

    def process(self, raw_points, current_position):
        """完整点云处理流水线"""
        # 1. 滤波排序
        filtered = self.filter_and_sort_points(raw_points)

        # 2. 转换到世界坐标系（前端神经网络训练时的状态输入要求）
        points_world = filtered + current_position
        
        # 3. 填充对齐并封装为 PyG 对象
        data = self.to_pyg_data(points_world)
        return data, self.min_distance

    def filter_and_sort_points(self, points):
        """基于距离筛选和排序"""
        if points.size == 0:
            return np.zeros((0, 2))
            
        distances = np.linalg.norm(points, axis=1)
        self.min_distance = np.min(distances)
        
        # 🔴 修复 2：由原来的 2.0m 修正为 3.0m
        mask = distances <= self.max_range
        filtered = points[mask]
        sorted_indices = np.argsort(distances[mask])
        return filtered[sorted_indices] if filtered.size > 0 else np.empty((0, 2))

    def to_pyg_data(self, points):
        """转换为PyG Data对象"""
        if points.size == 0:
            return Data(pos=torch.empty(0, 2), batch=torch.empty(0, dtype=torch.long))        
        pos = torch.tensor(points, dtype=torch.float32)
        return Data(pos=pos, batch=torch.zeros(len(pos), dtype=torch.long))

class EllipseTracker:
    def __init__(self):
        # 初始化节点
        rospy.init_node('ellipse_tracker_node')
        self.start_time = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        signal.signal(signal.SIGINT, self.signal_handler)

        self.model_u = self.load_model(mask=0)
        self.max_points = 200
        self.min_distance = 0

        # 点云处理器 (统一调整为 3.0m)
        self.pc_processor = PointCloudProcessor(
            max_points=self.max_points,
            max_range=3.0
        ) 

        # 存储机器人当前位置 (x, y, theta)
        self.current_pose = [0.0, 0.0, 0.0]
        self.max_rand = 0.8
        self.target_pos = [12 , 0]
        self.k_p = 1
        self.flag = 0
        self.flag_locked = False
        self.deltaT = 0.05  
        self.r = 0.4

        self.all_points = []
        self.pointcloud = np.zeros((0,2))
        self.filter_pointcloud = np.zeros((0,2))
        self.pc = np.zeros((0,2))
        self.vall = []

        # 核心控制定时器 (50Hz刷新率，匹配您的论文声明)
        self.ctrl_timer = rospy.Timer(rospy.Duration(0.02), self.control_loop)

        # 速度发布与 RViz 调试
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.debug_pub = rospy.Publisher('/processed_cloud', PointCloud2, queue_size = 1)

        # 订阅机器人状态
        self.__sub_curr_state = rospy.Subscriber(
            '/robot/dlio/odom_node/pose',
            PoseStamped,
            self.pose_callback,
            queue_size=10
        )

        # 🔴【核心修改点】：修改订阅的话题！
        # 改为订阅由 perfect_lidar_bridge 重构出的标准化 2D 拓扑点云
        self.__sub_global_cloud = rospy.Subscriber(
            '/densitynet_input_points',
            PointCloud2,
            self.cloud_callback,
            queue_size=10
        )

    def signal_handler(self, sig, frame):
        print("\n检测到 Ctrl+C，正在保存数组...")
        end_time = time.time()
        print(f"run time is:{end_time - self.start_time}")
        if len(self.vall) > 0:
            vall = np.array(self.vall)
            avg_v = np.mean(vall)
            print(f"avg velocity is:{avg_v}")
        x = np.stack(self.all_points, axis=0)
        np.savez('/home/maslab1/L-CDF/src/sensor_cdf/scripts/saved_data/all_points_densitynet.npz', X=x, allow_pickle=True)
        print("保存完毕，程序退出。")
        sys.exit(0)

    def load_model(self, mask):
        model = None
        model_dir = "/home/maslab1/L-CDF/src/sensor_cdf/scripts/saved_models/"
        try:
            if mask == 0:
                model = UNet(hidden_dim=512)
                model_path = f"{model_dir}lidar_u_model.pt"
            else:
                raise ValueError("Invalid mask value")

            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            rospy.logerr(f"模型加载失败: {str(e)}")
            raise RuntimeError(f"模型加载错误: {str(e)}") from e

    def cloud_callback(self, msg):
        current_position = np.array(self.current_pose[:2]).reshape(1,-1)
        points_3d = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

        if points_3d is None or points_3d.size == 0:
            self.filter_pointcloud = np.zeros((0,2))
            return  
        
        self.pointcloud = points_3d[:,:2] 
        if self.pointcloud is not None and current_position is not None:
            # 1. 转换并生成 PyG 数据送给端到端神经网络模型
            self.pc, self.min_distance = self.pc_processor.process(self.pointcloud, current_position)
            
            # 2. 提取用于调试发布的局部点云
            self.filter_pointcloud = self.pc_processor.filter_and_sort_points(self.pointcloud)
            rospy.loginfo(f"DensityNet接收到的标准化二维点云长度为：{len(self.filter_pointcloud)}")
        else:
            self.filter_pointcloud = np.zeros((0,2))

        # 3. 发布调试信息
        self.publish_cloud(self.filter_pointcloud) 
            
    def pose_callback(self, msg):
        """ 机器人当前位置回调 """
        quax = msg.pose.orientation.x
        quay = msg.pose.orientation.y
        quaz = msg.pose.orientation.z
        quaw = msg.pose.orientation.w
        theta = np.arctan2(2 * (quaw * quaz + quax * quay), 1 - 2 * (quay**2 + quaz**2))
        self.current_pose = [
            msg.pose.position.x,  
            msg.pose.position.y,  
            theta   
        ]
        self.all_points.append([msg.pose.position.x, msg.pose.position.y])

    def publish_cloud(self, points):
        """发布处理后的点云供 RViz 调试"""
        # 🔴 修复 3：统一坐标系名称为 Jackal 模型的 "velodyne"
        header = Header(stamp=rospy.Time.now(), frame_id="velodyne")
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        if points.shape[0] == 0:
            padded = np.zeros((0, 3))
        else:
            padded = np.hstack((points, np.zeros((len(points), 1))))
        pc_msg = point_cloud2.create_cloud(header, fields, padded)
        self.debug_pub.publish(pc_msg)

    def control_loop(self, event):
        current_pos = np.array(self.current_pose[:2])
        nominal_input = self.calculate_nominal_input()
        target_pos = np.array(self.target_pos)
        
        if self.flag == 1:
            v, w = 0.0, 0.0  
            twist_msg = Twist()
            self.cmd_vel_pub.publish(twist_msg)
        else:
            # 视野内无障碍物：执行传统的全局标称目标趋近控制器
            if self.filter_pointcloud.shape[0] == 0:
                current_cdf = self.calculate_onedensity(current_pos, target_pos)
                current_grad = self.calculate_graddensity(current_pos, target_pos)
                x_pred = current_pos + self.deltaT * np.clip(current_cdf, 0.0, 1e3) * np.ones(2)
                pred_grad = self.calculate_graddensity(x_pred, target_pos)
                try:
                    u = cdf_control(
                        current_grad=current_grad,
                        pred_grad=pred_grad,
                        dx=nominal_input,
                        deltaT = self.deltaT
                    )
                    v, w = self.convert_to_diff_drive(u)
                    twist = Twist()
                    twist.linear.x = float(v)
                    twist.angular.z = float(w)
                    self.cmd_vel_pub.publish(twist)
                except Exception as e:
                    rospy.logerr(f"控制错误: {str(e)}")

            # 🔴【真正的 DensityNet 核心闭环】：当视野内存在任何障碍物点时，直接执行端到端前向推理
            else:
                state = torch.FloatTensor(current_pos).unsqueeze(0).to(self.device)
                points = self.pc.to(self.device)

                with torch.no_grad():
                    u = self.model_u(state, points)
                v, w = u.detach().cpu().squeeze().numpy()
                
                if self.min_distance < 1:
                    self.vall.append(v)
                
                try:
                    twist = Twist()
                    twist.linear.x = float(v)
                    twist.angular.z = float(w)
                    self.cmd_vel_pub.publish(twist)
                    
                except Exception as e:
                    rospy.logerr(f"控制错误: {str(e)}")

    def calculate_onedensity(self, x, target):
        target = np.array([20, 0])
        r = np.linalg.norm(x - target)
        return 1 / r**0.6
        
    def calculate_graddensity(self, x, target):
        r = np.linalg.norm(x - target)
        df_dx1 = -0.6 * (x[0] - target[0]) / r**2.6
        df_dx2 = -0.6 * x[1] / r**2.6
        return np.array([df_dx1, df_dx2])
            
    def calculate_nominal_input(self):
        """ Go-to-goal 标称控制器 """
        current_pos = np.array(self.current_pose[:2], dtype=np.float32)
        target_pos = np.array(self.target_pos, dtype=np.float32)
        target_vector = target_pos - current_pos
        distance = np.linalg.norm(target_vector)
        if distance < 0.5:
            self.flag = 1
        return self.k_p * (target_vector / distance)
    
    def convert_to_diff_drive(self, u_control):
        """单积分器到差速驱动转换"""
        theta = self.current_pose[2]
        u = np.array(u_control).reshape(-1)
        A = np.array([
            [np.cos(theta), np.sin(theta)],
            [-1/self.r * np.sin(theta), 1/self.r * np.cos(theta)]
        ])
        return np.dot(A, u)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    tracker = EllipseTracker()
    rospy.loginfo("【真正 DensityNet 端到端测试节点】已启动就绪！")
    tracker.run()