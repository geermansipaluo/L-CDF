#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from functools import partial
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from local_planner_cdf import cdf_control
from model_u import UNet
from torch_geometric.data import Data
from torch.nn.parallel import DistributedDataParallel as DDP
from visualization_msgs.msg import Marker
from cal_grad import density_grad
import time

class EllipseTracker:
    def __init__(self):
        # 初始化节点
        self.start_time = time.time()
        rospy.init_node('ellipse_tracker_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.model_cdf = self.load_model(mask=1)
        # self.model_grad = self.load_model(mask=0)
        self.model_u = self.load_model(mask=0)
        self.max_points = 200

        # 存储机器人当前位置 (x, y, theta)
        self.current_pose = [-5.0, 0.0, 0.0]

        # infeasible rate
        self.inf_rate = 0
        self.total_echo = 0
        self.inf_echo = 0

        # 噪声幅度
        self.max_rand = 0.5

        # 存储目标位置
        self.target_pos = [5, 0]

        # 比例增益
        self.k_p = 1.0

        # 虚拟控制点与智能体距离
        self.r = 0.4

        self.deltaT = 0.05  # 默认初始值

        # 二维点云数据
        self.pointcloud = np.zeros((0,2))
        self.filter_pointcloud = np.zeros((0,2))
        self.pc = np.zeros((0,1))
        # self.voxel_size = rospy.get_param("~voxel_size", 0.04)  # 体素大小（米）
        self.frame_id = rospy.get_param("~frame_id", "world")

        self.ctrl_timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)

        self.goal_timer = rospy.Timer(rospy.Duration(0.05), self.check_reach_goal)

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
            '/curr_state',
            Float32MultiArray,
            self.pose_callback,
            queue_size=10
        )

        self.__sub_global_cloud = rospy.Subscriber(
            '/densitynet_input_points', PointCloud2, self.cloud_callback, queue_size=10
        )

        self.marker_pub = rospy.Publisher('/target_goal_marker', Marker, queue_size=1)

    def load_model(self, mask):
        model = None
        model_dir = "/home/guo/L-CDF/src/sensor_cdf/scripts/saved_models/"
        
        try:
            if mask == 0:
                model = UNet(hidden_dim=512)
                model_path = f"{model_dir}lidar_u_model_15000+.pt"
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

    def publish_target_marker(self):
        marker = Marker()
        marker.header.frame_id = "world"          # 绑定绝对世界系
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE               # 设置为实心球体形状
        marker.action = Marker.ADD
        
        # 目标点坐标设定
        marker.pose.position.x = float(self.target_pos[0])
        marker.pose.position.y = float(self.target_pos[1])
        marker.pose.position.z = 0.1               # 微微悬空 10cm，防止被地平面网格压住
        marker.pose.orientation.w = 1.0
        
        # 标记大小：直径 0.3 米的小圆点
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # 颜色配置：纯红色 (RGBA)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0                       # 不透明
        
        self.marker_pub.publish(marker)

    def cloud_callback(self, msg):
        # 1. 直接读取桥节点发出来的、已经 512 线对齐且按距离排好序的局部 [x_local, y_local] 坐标
        points_3d = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

        if points_3d is None or points_3d.size == 0:
            self.pointcloud = np.zeros((0,2))
            self.filter_pointcloud = np.zeros((0,2))
            rospy.logwarn("没有接收到有效的点云数据，跳过处理")
            return  
        
        self.pointcloud = points_3d[:, :2] # 此时为干净的局部坐标
        current_position = self.current_pose[:2] # 仿真世界系平移量 [x, y]
        theta = self.current_pose[2] # 仿真世界系偏航角 yaw

        if self.pointcloud is not None and current_position is not None:
            # 🔴【核心 Debug 修复】：引入严谨的 2D 旋转平移矩阵，将局部坐标彻底转为绝对世界坐标
            # 完美对齐你训练集（scene_env.py）中存储的绝对世界系点云格式！
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            rot_matrix = np.array([
                [cos_t, -sin_t],
                [sin_t,  cos_t]
            ])
            # 矩阵乘法执行旋转 + 平移 = 绝对世界坐标 [x_world, y_world]
            points_world = np.dot(self.pointcloud, rot_matrix.T) + current_position
            
            # 将绝对世界坐标点云赋予 filter_pointcloud 供网络和 RViz 使用
            self.filter_pointcloud = points_world 
            rospy.loginfo_throttle(1.0, f"【坐标系对齐成功】已将单线点云转换为绝对世界坐标送入网络，当前点数: {len(self.filter_pointcloud)}")

            # 3. 🔴 转换为 PyG Data 直接喂给网络模型
            pos_tensor = torch.tensor(points_world, dtype=torch.float32)
            self.pc = Data(
                pos=pos_tensor, 
                batch=torch.zeros(len(pos_tensor), dtype=torch.long)
            )

        # 发布调试信息（因为你的 publish_cloud 里 frame_id 写的是 "world"，
        # 传入绝对世界坐标 points_world 后，RViz 中的点云会奇迹般地完美贴合在障碍物本体上！）
        self.publish_cloud(self.filter_pointcloud)
            
    def pose_callback(self, msg):
        """ 机器人当前位置回调 """
        # 假设消息格式为 [x, y, theta]
        if len(msg.data) >= 3:
            self.current_pose = [
                msg.data[0],  # x
                msg.data[1],  # y
                msg.data[2]   # theta (朝向角)
            ]

    def publish_cloud(self, points):
        """发布处理后的点云"""
        header = Header(stamp=rospy.Time.now(), frame_id="world")
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        # 添加虚拟z轴
        padded = np.hstack((points, np.zeros((len(points), 1))))
        pc_msg = point_cloud2.create_cloud(header, fields, padded)
        self.debug_pub.publish(pc_msg)

    def check_reach_goal(self, event):
        if self.current_pose is not None:
            dist = np.linalg.norm(np.array(self.current_pose[:2])-np.array(self.target_pos))
            if dist < 0.6:
                print(f"当前距离目标位置：{dist}")
                end_time = time.time()
                
                # 🔴【核心修复】：在注销定时器前，必须显式发布一帧绝对零速，强制底盘立刻刹车抱死！
                stop_msg = Twist()
                stop_msg.linear.x = 0.0
                stop_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(stop_msg)
                
                if self.total_echo > 0:
                    self.inf_rate = self.inf_echo / self.total_echo
                else:
                    self.inf_rate = 0.0
                rospy.loginfo(f"目标到达！infeasible 率 = {self.inf_rate:.3f}")
                rospy.loginfo(f"总优化步数为 = {self.total_echo}")
                rospy.loginfo(f"无解步数为 = {self.inf_echo}")
                rospy.loginfo(f"总优化耗时为 = {end_time-self.start_time}")
                
                # 安全注销线程
                self.goal_timer.shutdown()
                self.ctrl_timer.shutdown()

    def control_loop(self, event):
        self.publish_target_marker()
        current_pos = np.array(self.current_pose[:2])
        nominal_input = self.calculate_nominal_input()
        self.total_echo = self.total_echo + 1
        if self.filter_pointcloud.shape[0] == 0 or np.all(nominal_input == 0):
            v, w = self.convert_to_diff_drive(nominal_input)
            twist_msg = Twist()
            twist_msg.linear.x = v
            twist_msg.angular.z = w
            self.cmd_vel_pub.publish(twist_msg)
        else:
            state = torch.FloatTensor(current_pos).unsqueeze(0).to(self.device)
            points = self.pc
            points = points.to(self.device)

            with torch.no_grad():
                # grad = self.model_grad(state, points) / 1000
                # cdf = self.model_cdf(state, points) / 1000
                u = self.model_u(state, points)
                v,w = u.detach().cpu().squeeze().numpy()
                

            # x_pred = current_pos + self.deltaT * np.clip(cdf.detach().cpu().squeeze().numpy(), 0.0, 1e3) * np.ones(2)
            # state_pred = torch.FloatTensor(x_pred).unsqueeze(0).to(self.device)

            # with torch.no_grad():
            #     pred_grad = self.model_grad(state_pred, points) / 1000
            
            # noi = ((2 * self.max_rand * np.random.rand(2) - self.max_rand) * self.deltaT)
            # noise = noi.reshape((2,1))
            # nominal_input_noise = nominal_input + noise.flatten()
            try:
                # u = cdf_control(
                #     current_grad=grad.detach().cpu().squeeze().numpy(),
                #     pred_grad=pred_grad.detach().cpu().squeeze().numpy(),
                #     dx=nominal_input,
                #     deltaT = self.deltaT
                # )
                # rospy.loginfo("标称控制为: [%f, %f]", u[0], u[1])
                # 转换并发布控制指令
                # u_noise = u + noise
                # v, w = self.convert_to_diff_drive(u.detach().cpu().squeeze().numpy())
                twist = Twist()
                twist.linear.x = float(v)
                twist.angular.z = float(w)
                self.cmd_vel_pub.publish(twist)
                
            except Exception as e:
                rospy.logerr(f"控制错误: {str(e)}")
            
            
    def calculate_nominal_input(self):
        """ Go-to-goal 标称比例控制器（线性相关） """
        current_pos = np.array(self.current_pose[:2], dtype=np.float32)
        target_pos = np.array(self.target_pos, dtype=np.float32)
        target_vector = target_pos - current_pos
        distance = np.linalg.norm(target_vector)

        # 1. 基础线性映射控制输入 (纯P控制：离终点越近，速度矢量越小)
        nominal_out = self.k_p * target_vector

        # 2. 🔴 安全增益限幅（Saturated P-Control）：
        # 防止远距离启动时（如距离>20米）控制量过大导致物理底盘直接飞出去
        max_nominal_speed = 1.2  # 限制最大标称单步输入大小为 1.2
        if distance > 0.0:
            current_speed = np.linalg.norm(nominal_out)
            if current_speed > max_nominal_speed:
                nominal_out = (nominal_out / current_speed) * max_nominal_speed

        return np.where(distance < 0.1, np.zeros(2), nominal_out)
    
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