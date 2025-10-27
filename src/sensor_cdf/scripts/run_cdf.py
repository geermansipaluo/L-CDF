#!/usr/bin/env python3
import rospy
import torch
import math
import numpy as np
import sympy as sp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import threading
from functools import partial
from jax import grad, jit
from std_msgs.msg import Float32MultiArray, Bool, ColorRGBA
from sensor_msgs import point_cloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Twist, Point 
from cal_grad import density_grad
import signal 
import sys
import time
from sensor_msgs.msg import  PointCloud2

class PointCloudProcessor:
    def __init__(self, max_points=200, voxel_size=0.05):
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.min_distance = 0

    def process(self, raw_points):
        """完整点云处理流水线"""
        # 1. 滤波排序
        self.filter_and_sort_points(raw_points)
        
        return self.min_distance

    def filter_and_sort_points(self, points):
        """基于距离筛选和排序"""
        if points.size == 0:
            return np.zeros((0, 2))
        
        distances = np.linalg.norm(points, axis=1)
        self.min_distance = np.min(distances)


class EllipseTracker:
    def __init__(self):
        # 初始化节点
        rospy.init_node('ellipse_tracker_node')

        signal.signal(signal.SIGINT, self.signal_handler)
        self.start_time = time.time()
        self.min_distance = 0
        self.all_points = []
        self.vall = []
        self.infeasible = 0
        self.pc_processor = PointCloudProcessor(
            max_points=200,
            voxel_size=0.1
        )

        # 数据收集容器
        self.collected_states = []      # 存储状态 (x, y)
        self.collected_obstacles = []   # 存储障碍物参数 (5,5)
        self.collected_gradients = []   # 存储梯度 (dx, dy)

        # 存储机器人当前位置 (x, y, theta)
        self.current_pose = [0.0, 0.0, 0.0]  
        self.current_pose_jax = jnp.array([0.0, 0.0], dtype=jnp.float64)

        # 存储密度函数梯度
        self.current_grad = np.array([0.0, 0.0], dtype=np.float64)
        self.pred_grad = np.array([0.0, 0.0], dtype=np.float64)
        self.lock = threading.RLock()

        # 存储参数改为JAX数组
        self.C = jnp.zeros((0, 2, 2))  # 形状矩阵
        self.d = jnp.zeros((2, 0))     # 中心位置
        self.a = jnp.zeros(0)           # 长轴参数

        # 存储目标位置
        # self.target_pos = [9, 0]
        self.target_pos = jnp.array([13.0, 0.0], dtype=jnp.float64)

        # 比例增益
        self.k_p = 1

        # 虚拟控制点与智能体距离
        self.r = 0.46

        self.deltaT = 0.033  # 默认初始值

        self.flag = 0

        # 车辆几何参数（可配置为参数）
        self.vehicle_length = 0.92    # 单位：米
        self.vehicle_width = 0.7     # 单位：米
        self.arrow_length = 0.5      # 方向箭头长度

        self.ctrl_timer = rospy.Timer(rospy.Duration(0.1), self._control_cb)

        # 订阅机器人状态
        self.__sub_curr_state = rospy.Subscriber(
            '/robot/dlio/odom_node/pose',
            PoseStamped,
            self.__curr_pose_cb,
            queue_size=10
        )

        # 订阅椭圆障碍物数据
        self.__sub_obs = rospy.Subscriber(
            '/local_map_pub/for_obs_track',
            Float32MultiArray,
            self.__obs_callback,
            queue_size=10
        )

        # 速度发布
        self.cmd_vel_pub = rospy.Publisher(
            '/cmd_vel', 
            Twist, 
            queue_size=1
        )

        self.__sub_lidar = rospy.Subscriber('/filtered_3d', PointCloud2, self.cloud_callback)

    def cloud_callback(self, msg):
        points_3d = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

        if points_3d is None or points_3d.size == 0:
            self.pointcloud = np.zeros((0,2))
            rospy.logwarn("没有接收到有效的点云数据，跳过处理")
            return  # 没有点云数据时跳过处理
        
        self.pointcloud = points_3d[:,:2] # 雷达坐标系下的点云坐标 
        # rospy.loginfo(f"三维点云数据为：{len(self.pointcloud)}")
        if self.pointcloud is not None:
            self.min_distance = self.pc_processor.process(self.pointcloud)
            # print(f"过滤后点云：{self.filter_pointcloud}")
            # rospy.loginfo(f"二维点云数据长度为：{(self.filter_pointcloud).shape}")
        else:
            rospy.loginfo("没有二维点云数据！")


    def signal_handler(self, sig, frame):
        print("\n检测到 Ctrl+C，正在保存数组...")
        end_time = time.time()
        print(f"run time is:{end_time - self.start_time}")
        vall = np.array(self.vall)
        avg_v = np.mean(vall)
        print(f"avg velocity is:{avg_v}")
        print(f"infeasible number is:{self.infeasible}")
        x = np.stack(self.all_points, axis=0)
        np.savez('/home/maslab1/L-CDF/src/sensor_cdf/scripts/saved_data/all_points_cdfbase.npz', X=x, allow_pickle=True)
        print("保存完毕，程序退出。")
        sys.exit(0)

    def __curr_pose_cb(self, msg):
        """ 机器人当前位置回调 """
        quax = msg.pose.orientation.x
        quay = msg.pose.orientation.y
        quaz = msg.pose.orientation.z
        quaw = msg.pose.orientation.w
        theta = np.arctan2(2 * (quaw * quaz + quax * quay), 1 - 2 * (quay**2 + quaz**2))
        self.current_pose_jax = jnp.array(
                [msg.pose.position.x, msg.pose.position.y], 
                dtype=jnp.float64
            )
        self.current_pose = [
                msg.pose.position.x,  # x
                msg.pose.position.y,  # y
                theta   # theta (朝向角)
            ]
        self.all_points.append([msg.pose.position.x, msg.pose.position.y])

    def _control_cb(self, event):
        if self.flag == 0:
            rospy.logwarn("开始控制！")
            self.flag = 1
        if len(self.current_pose) < 2:
            rospy.logwarn("当前位姿无效，跳过数据处理")
            return
        if (np.allclose(self.current_grad, np.zeros(2)) and 
            np.allclose(self.pred_grad, np.zeros(2))):
            rospy.loginfo("当前梯度为零向量，跳过本次障碍物处理")
            return
        with self.lock:
            current_grad = self.current_grad
            pred_grad = self.pred_grad
            current_C = self.C.copy()

        nominal_input = self.__calculate_nominal_input()

         
        try:
            from local_planner_cdf import cdf_control  # 动态导入避免循环依赖
            # 直接从内存读取预计算值
            u = cdf_control(
                current_grad=current_grad,
                pred_grad=pred_grad,
                dx=nominal_input,
                deltaT = self.deltaT
            )
            u1, u2 = u
            if u1 == 0 and u2 == 0:
                self.infeasible += 1
                rospy.loginfo("infeasible!")
            # rospy.loginfo("标称控制为: [%f, %f]", u[0], u[1])
            # 转换并发布控制指令
            v, w = self.__convert_to_diff_drive(u)
            if self.min_distance < 1 and self.min_distance != 0:
                self.vall.append(v)
            twist = Twist()
            twist.linear.x = float(v)
            twist.angular.z = float(w)
            self.cmd_vel_pub.publish(twist)
            
        except Exception as e:
            rospy.logerr(f"控制错误: {str(e)}")
            self.infeasible += 1

    def __obs_callback(self, msg):
        """ 椭圆障碍物回调 """
        with self.lock:
            if len(self.current_pose) < 2:
                rospy.logwarn("当前位姿无效，跳过数据处理")
                return
            if self.flag == 0:
                rospy.logwarn("还未开始控制！")
                return  # 或者直接退出回调
            # 解析椭圆参数并计算 C 和 d
            num_obstacles = len(msg.data) // 7
            obstacles = np.zeros((num_obstacles,5))
            robot_x, robot_y = self.current_pose[:2]
            C_list = []  # 临时存储 C
            d_list = []  # 临时存储 d
            a_list = []  # 临时存储长轴
            # 清理旧数据
            self.C = jnp.zeros((0, 2, 2))  # 形状矩阵
            self.d = jnp.zeros((2, 0))     # 中心位置
            self.a = jnp.zeros(0)           # 长轴参数
            # rospy.loginfo("障碍物数量:%d", num_obstacles)

            for i in range(num_obstacles):
                # 提取参数
                cx = msg.data[7 * i]
                cy = msg.data[7 * i + 1]
                a = msg.data[7 * i + 2]  # 长轴
                b = msg.data[7 * i + 3]  # 短轴
                theta = msg.data[7 * i + 4]  # 旋转角度
                obstacles[i, :] = [cx, cy, a, b, theta]

                # 计算机器人到椭圆圆心的距离
                distance = math.sqrt((robot_x - cx) ** 2 + (robot_y - cy) ** 2)

                # 判断距离是否大于三倍长轴
                if distance > 2.5: #* a:
                    continue  # 跳过该椭圆

                # 计算变换矩阵 C 和 d
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                C = np.array([
                    [a * cos_t, -b * sin_t],
                    [a * sin_t,  b * cos_t]
                ])
                d = np.array([cx, cy])

                # 添加到临时列表
                C_list.append(C)
                d_list.append(d)
                a_list.append(a)

            # 转换为JAX数组
            if C_list:
                self.C = jnp.stack(C_list, axis=0)
                self.d = jnp.column_stack(d_list)
                self.a = jnp.array(a_list)
            else:
                self.C = jnp.zeros((0, 2, 2))
                self.d = jnp.zeros((2, 0))
                self.a = jnp.zeros(0)

            # self.__log_debug_info()
        
            if self.C.size > 0:
                # 计算密度函数
                rho_func = self.calculate_density()
                rospy.loginfo("current density, %f, ", rho_func(robot_x, robot_y))
            else:
                rho_func = self.calculate_onedensity()
                rospy.loginfo("current density, %f, ", rho_func(robot_x, robot_y))
            # 将梯度计算移至线程池
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    density_grad, 
                    x = self.current_pose_jax,
                    density_func = rho_func
                )
                self.current_grad, self.pred_grad = future.result()
            
            rospy.loginfo("当前梯度:[%f, %f]", self.current_grad[0], self.current_grad[1])

            

            # if jnp.linalg.norm(jnp.array(self.current_pose[:2]) - self.target_pos) > 0.1:
            #     self._collect_data(obstacles)
            # else:
            #     print("到达目标位置！")
            
    def __calculate_nominal_input(self):
        """ Go-to-goal 标称控制器 """
        target_vector = self.target_pos - self.current_pose_jax  # 确保self.target_pos是JAX数组
        distance = jnp.linalg.norm(target_vector)

        return jnp.where(distance < 0.1, jnp.zeros(2), self.k_p * (target_vector / distance))

    @partial(jit, static_argnums=(0,))
    def _compute_psi_k(self, x, C_k, d_k, a_k):
        """JAX计算单个障碍物的psi_k"""
        inv_C_k = jnp.linalg.inv(C_k.astype(jnp.float64))
        x_minus_d = (x - d_k).astype(jnp.float64)
        
        # 计算c_k
        v = inv_C_k @ x_minus_d
        c_k = jnp.sum(v**2) - 1
        
        # 计算b_k
        # b_k = jnp.sum(x_minus_d**2) - (3 * a_k)**2
        b_k = jnp.sum(x_minus_d**2) - (3)**2
        
        # 处理分母为零的情况
        denominator = c_k - b_k
        safe_denominator = jnp.where(denominator == 0, 1e-6, denominator)
        m_k = c_k / safe_denominator
        
        # 限制m_k范围保证数值稳定
        safe_m_k = jnp.clip(m_k, 1e-6, 1-1e-6)
        
        # 计算psi_k
        exp_m = jnp.exp(-1 / safe_m_k)
        exp_1_m = jnp.exp(-1 / (1 - safe_m_k))
        psi_k = exp_m / (exp_m + exp_1_m + 1e-6)
        
        # 应用分段条件
        return jnp.where(
            c_k <= 0, 0.0,
            jnp.where(b_k <= 0, psi_k, 1.0)
        )

    def calculate_density(self):
    # 获取当前状态的快照
        current_C = self.C.copy()  # 显式拷贝
        current_d = self.d.copy()
        current_a = self.a.copy()
        target = self.target_pos.copy()
    
        @partial(jit, static_argnums=())
        def fresh_rho(x1, x2):
            x = jnp.array([x1, x2], dtype=jnp.float64)
            # 使用快照数据而非self引用
            def body_fun(k, psi_total):
                return psi_total * self._compute_psi_k(
                    x, current_C[k], current_d[:,k], current_a[k]
                )
            psi_total = jax.lax.fori_loop(0, len(current_a), body_fun, 1.0)
            # return 8500*psi_total / (jnp.sum((x - target)**2))
            return psi_total / (jnp.sum((x - target)**2))**(0.6)
    
        return fresh_rho  # 返回不依赖self的新函数
    
    def calculate_onedensity(self):
    # 获取当前状态的快照
        target = self.target_pos.copy()
    
        @partial(jit, static_argnums=())
        def fresh_rho(x1, x2):
            x = jnp.array([x1, x2], dtype=jnp.float64)
            return 1 / (jnp.sum((x - target)**2))**(0.6)
    
        return fresh_rho  # 返回不依赖self的新函数
    
    def __convert_to_diff_drive(self, u_control):
        """单积分器到差速驱动转换"""
        theta = jnp.array(self.current_pose[2], dtype=jnp.float64)
        u = jnp.array(u_control, dtype=jnp.float64).reshape(-1)
        
        A = jnp.array([
            [jnp.cos(theta), jnp.sin(theta)],
            [-1/self.r * jnp.sin(theta), 1/self.r * jnp.cos(theta)]
        ], dtype=jnp.float64)
        
        return (A @ u).astype(jnp.float64)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    tracker = EllipseTracker()
    rospy.loginfo("椭圆追踪节点已启动")
    tracker.run()