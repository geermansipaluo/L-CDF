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
    def __init__(self, max_points=200, voxel_size=0.05):
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.min_distance = 0

    def process(self, raw_points, current_position):
        """完整点云处理流水线"""
        # 1. 滤波排序
        filtered = self.filter_and_sort_points(raw_points, current_position)

        points_robot = filtered + current_position
        
        # 2. 体素合并
        # merged = self.merge_points(points_robot)
        
        # 3. 填充对齐
        data = self.to_pyg_data(points_robot)
        
        return data, self.min_distance

    def filter_and_sort_points(self, points, current_position):
        """基于距离筛选和排序"""
        if points.size == 0:
            return np.zeros((0, 2))
            
        # deltas = points - current_position
        distances = np.linalg.norm(points, axis=1)
        self.min_distance = np.min(distances)
        mask = distances < 2
        filtered = points[mask]
        sorted_indices = np.argsort(distances[mask])
        return filtered[sorted_indices] if filtered.size > 0 else np.empty((0, 2))

    def merge_points(self, points):
        """体素合并"""
        if points.size == 0:
            return np.zeros((0, 2))
            
        voxel_indices = (points / self.voxel_size).astype(int)
        unique_voxels, inverse_indices, counts = np.unique(
            voxel_indices, axis=0, return_inverse=True, return_counts=True)
        
        sum_points = np.zeros((len(unique_voxels), 2))
        np.add.at(sum_points, inverse_indices, points)
        return sum_points / counts[:, None]

    def to_pyg_data(self, points):
        """转换为PyG Data对象"""
        if points.size == 0:
            return Data(pos=torch.empty(0, 2), batch=torch.empty(0, dtype=torch.long))        
        # points = (points - np.mean(points, axis=0)) / (np.std(points, axis=0) + 1e-8)     
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

        # 点云处理器
        self.pc_processor = PointCloudProcessor(
            max_points=self.max_points,
            voxel_size=0.020
        ) 

        # 存储机器人当前位置 (x, y, theta)
        self.current_pose = [0.0, 0.0, 0.0]

        # 噪声幅度
        self.max_rand = 0.8

        # 存储目标位置
        self.target_pos = [12 , 0]
        # self.target_pos = jnp.array([9.0, 0.0], dtype=jnp.float64)

        # 比例增益
        self.k_p = 1

        # 到达目标位置的标志位
        self.flag = 0
        self.flag_locked = False

        self.deltaT = 0.05  # 默认初始值

        self.r = 0.4

        self.all_points = []

        # 二维点云数据
        self.pointcloud = np.zeros((0,2))
        self.filter_pointcloud = np.zeros((0,2))
        self.filter_points_3d = np.zeros((0,2))
        self.pc = np.zeros((0,2))
        self.voxel_size = rospy.get_param("~voxel_size", 0.05)  # 体素大小（米）
        self.frame_id = rospy.get_param("~frame_id", "os_sensor")

        # save v and w
        self.vall = []

        self.ctrl_timer = rospy.Timer(rospy.Duration(0.02), self.control_loop)

        # 速度发布
        self.cmd_vel_pub = rospy.Publisher(
            '/cmd_vel', 
            Twist, 
            queue_size=1
        )

        self.debug_pub = rospy.Publisher(
            '/processed_cloud',
            PointCloud2,
            queue_size = 1
        )

        # 订阅机器人状态
        self.__sub_curr_state = rospy.Subscriber(
            '/robot/dlio/odom_node/pose',
            PoseStamped,
            self.pose_callback,
            queue_size=10
        )

        self.__sub_global_cloud = rospy.Subscriber(
            '/filtered_3d',
            PointCloud2,
            self.cloud_callback,
            queue_size=10
        )

    def signal_handler(self, sig, frame):
        print("\n检测到 Ctrl+C，正在保存数组...")
        end_time = time.time()
        print(f"run time is:{end_time - self.start_time}")
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

            # 加载时启用安全模式
            state_dict = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True
            )
            
            # 参数名清洗
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # 严格加载检查
            model.load_state_dict(state_dict, strict=True)
            model = model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            rospy.logerr(f"模型加载失败: {str(e)}")
            raise RuntimeError(f"模型加载错误: {str(e)}") from e


    def cloud_callback(self, msg):
        current_position = np.array(self.current_pose[:2])
        current_position = current_position.reshape(1,-1)
        # print(f"位置形状：{current_position.shape}")
        points_3d = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

        if points_3d is None or points_3d.size == 0:
            self.pointcloud = np.zeros((0,2))
            rospy.logwarn("没有接收到有效的点云数据，跳过处理")
            return  # 没有点云数据时跳过处理
        
        self.pointcloud = points_3d[:,:2] # 雷达坐标系下的点云坐标 
        # rospy.loginfo(f"三维点云数据为：{len(self.pointcloud)}")
        if self.pointcloud is not None and current_position is not None:
            self.pc, self.min_distance = self.pc_processor.process(self.pointcloud, current_position)
            self.filter_pointcloud = self.pc.pos.numpy()
            # print(f"过滤后点云：{self.filter_pointcloud}")
            rospy.loginfo(f"二维点云数据长度为：{len(self.filter_pointcloud)}")
        else:
            rospy.loginfo("没有二维点云数据！")

        # 发布调试信息
        self.publish_cloud(self.filter_pointcloud) 
            
    def pose_callback(self, msg):
        """ 机器人当前位置回调 """
        # 假设消息格式为 [x, y, theta]
        quax = msg.pose.orientation.x
        quay = msg.pose.orientation.y
        quaz = msg.pose.orientation.z
        quaw = msg.pose.orientation.w
        theta = np.arctan2(2 * (quaw * quaz + quax * quay), 1 - 2 * (quay**2 + quaz**2))
        self.current_pose = [
            msg.pose.position.x,  # x
            msg.pose.position.y,  # y
            theta   # z
        ]
        self.all_points.append([msg.pose.position.x, msg.pose.position.y])
        # rospy.loginfo(f"当前位置为：[{msg.pose.position.x, msg.pose.position.y}]")

    def publish_cloud(self, points):
        """发布处理后的点云"""
        header = Header(stamp=rospy.Time.now(), frame_id="os_sensor")
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        # 添加虚拟z轴
        padded = np.hstack((points, np.zeros((len(points), 1))))
        pc_msg = point_cloud2.create_cloud(header, fields, padded)
        self.debug_pub.publish(pc_msg)

    def control_loop(self, event):
        current_pos = np.array(self.current_pose[:2])
        nominal_input = self.calculate_nominal_input()
        target_pos = np.array(self.target_pos)
        if self.flag == 1:
            v, w = 0.0, 0.0  # 速度为 0
            twist_msg = Twist()
            twist_msg.linear.x = v
            twist_msg.angular.z = w
            self.cmd_vel_pub.publish(twist_msg)
        else:
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
                    # rospy.loginfo("标称控制为: [%f, %f]", u[0], u[1])
                    # 转换并发布控制指令
                    v, w = self.convert_to_diff_drive(u)
                    twist = Twist()
                    twist.linear.x = float(v)
                    twist.angular.z = float(w)
                    self.cmd_vel_pub.publish(twist)
                except Exception as e:
                    rospy.logerr(f"控制错误: {str(e)}")

            else:
                state = torch.FloatTensor(current_pos).unsqueeze(0).to(self.device)
                points = self.pc
                points = points.to(self.device)

                with torch.no_grad():
                    u = self.model_u(state, points)
                v,w = u.detach().cpu().squeeze().numpy()
                if self.min_distance < 1:
                    self.vall.append(v)
                
                try:
                    twist = Twist()
                    twist.linear.x = float(v)
                    twist.angular.z = float(w)
                    # twist.linear.x = 0
                    # twist.angular.z = 0
                    self.cmd_vel_pub.publish(twist)
                    
                except Exception as e:
                    rospy.logerr(f"控制错误: {str(e)}")

    def calculate_onedensity(self, x, target):
        target = np.array([20, 0])
    
        r = np.linalg.norm(x - target)

        return 1 / r**0.6
        
    
    def calculate_graddensity(self, x, target):
        r = np.linalg.norm(x - target)

        # 计算梯度 df/dx1 和 df/dx2
        df_dx1 = -0.6 * (x[0] - target[0]) / r**2.6
        df_dx2 = -0.6 * x[1] / r**2.6

        return np.array([df_dx1, df_dx2])
            
    def calculate_nominal_input(self):
        """ Go-to-goal 标称控制器 """
        current_pos = np.array(self.current_pose[:2], dtype=np.float32)
        target_pos = np.array(self.target_pos, dtype=np.float32)
        target_vector = target_pos - current_pos
        distance = np.linalg.norm(target_vector)
        # rospy.loginfo(f"距离终点距离为：{distance}")
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
    rospy.loginfo("椭圆追踪节点已启动")
    tracker.run()