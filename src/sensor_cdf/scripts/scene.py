#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap
from cal_grad import density_grad
import sys
import signal
from lidar_measurement import batch_intersection_check, post_process

class EllipseTracker:
    def __init__(self):
        # 起始点
        self.start = jnp.array([0.0, 0.0], dtype=jnp.float64)

        # 数据收集容器
        self.collected_states = []      # 存储状态 (x, y)
        self.collected_gradients = []   # 存储梯度 (dx, dy)
        self.collected_pointcloud = []
        self.collected_cdfvalue = []
        self.obstacles = np.zeros((5,5))

        # 存储机器人当前位置 (x, y, theta)
        self.current_pose = [0.0, 0.0]
        self.current_pose_jax = jnp.array([0.0, 0.0], dtype=jnp.float64)

        # 存储密度函数梯度
        self.current_grad = jnp.array([0.0, 0.0], dtype=jnp.float64)
        self.pred_grad = jnp.array([0.0, 0.0], dtype=jnp.float64)

        # 存储参数改为JAX数组
        self.C = jnp.zeros((0, 2, 2))  # 形状矩阵
        self.d = jnp.zeros((2, 0))     # 中心位置
        self.C_all = jnp.zeros((0,2,2))
        self.d_all = jnp.zeros((2,0))


        # 存储目标位置
        self.target_pos = jnp.array([0.0, 0.0], dtype=jnp.float64)

        # 比例增益
        self.k_p = 1

        # 步长
        self.time_steps = 500

        self.deltaT = 0.05  # 时间步长

        # 点云数据
        self.pointcloud = np.zeros((200,2))

    def run(self):
        while len(self.collected_cdfvalue) < 100000:
            # 初始化数据
            self.start = jnp.array([0.0, 0.0], dtype=jnp.float64)
            self.obstacles = np.zeros((0,5))
            self.current_pose = [0.0, 0.0]  
            self.current_pose_jax = jnp.array([0.0, 0.0], dtype=jnp.float64)
            self.current_grad = jnp.array([0.0, 0.0], dtype=jnp.float64)
            self.pred_grad = jnp.array([0.0, 0.0], dtype=jnp.float64)
            self.C = jnp.zeros((0, 2, 2))  # 形状矩阵
            self.target_pos = jnp.array([9.0, 0.0], dtype=jnp.float64)

            # 生成起始点和终点
            self.start, self.target_pos = self._generate_valid_points()
            self.current_pose = self.start
            self.current_pose_jax = jnp.array(self.start, dtype=jnp.float64)
            print(f"起始点为：{self.start}，终点为：{self.target_pos}")

            # 处理障碍物
            num_obstacles = np.random.randint(4, 8)
            self.num_obstacle = num_obstacles

            #生成障碍物集合
            obstacles = self._generate_obstacles(num_obstacles, self.start, self.target_pos)
            self.obstacles = obstacles
            
            # 生成轨迹
            self._control_cb()

            if len(self.collected_cdfvalue) >= 100000:  # 判断是否已收集到足够的数据
                print(f"已收集到 100,000 个数据点，停止数据收集。")
                break

    def _generate_valid_points(self):
        """生成有效的起点和终点，确保最小距离"""
        while True:
            start = np.random.uniform(-11, 11, 2)
            end = np.random.uniform(-11, 11, 2)
            if (np.linalg.norm(start - end) > 10.0 and
                np.linalg.norm(start - end) < 15):
                return start.astype(np.float32), end.astype(np.float32)

    def _control_cb(self):
        for _ in range(self.time_steps):
            nominal_input = self.__calculate_nominal_input()

            current_pose = jnp.array(self.current_pose, dtype=jnp.float64)
            obstacles = jnp.array(self.obstacles, dtype=jnp.float64)

            self.__obs_callback(self.num_obstacle)
            
            result = batch_intersection_check(obstacles, current_pose)
            pointcloud = post_process(result)
            self.pointcloud = np.array(pointcloud)

            robot_x, robot_y = self.current_pose[:2]

            # 计算密度函数
            rho_func = self.calculate_density(self.C_all, self.d_all)
            rho_func_value = rho_func(robot_x, robot_y)
            if self.C.size > 0:
                # 梯度计算部分   
                rho_func = self.calculate_density(self.C, self.d)   
                self.current_grad, self.pred_grad = density_grad(
                    x = self.current_pose_jax,
                    density_func = rho_func,
                )
                
                from local_planner_cdf import cdf_control  # 动态导入避免循环依赖
                # 直接从内存读取预计算值
                u = cdf_control(
                    current_grad=self.current_grad,
                    pred_grad=self.pred_grad,
                    dx=nominal_input,
                    deltaT = self.deltaT
                )

                if np.allclose(u, np.zeros((2, 1)), atol=1e-6):
                    print("检测到零控制输入，跳过本次循环")
                    return
                
                self.current_pose = self.deltaT * np.array(u).flatten() + self.current_pose
                self.current_pose_jax = jnp.array(self.current_pose, dtype=jnp.float64)
            else:
                self.current_grad = np.array([0,0])
                self.pred_grad = np.array([0,0])
                self.current_pose = self.deltaT * np.array(nominal_input).flatten() + self.current_pose
                self.current_pose_jax = jnp.array(self.current_pose, dtype=jnp.float64)
            # 存储到容器
            if self.pointcloud.size > 0:
                self.collected_states.append(self.current_pose)
                self.collected_gradients.append(self.current_grad)  
                self.collected_pointcloud.append(self.pointcloud)
                self.collected_cdfvalue.append(rho_func_value)
            
            # 提前终止条件
            if jnp.linalg.norm(self.current_pose - self.target_pos) < 0.1:
                print(f"到达目标位置! 数据长度为:{len(self.collected_states)}")
                break
            else:
                if (self.current_grad[0] != 0 or self.current_grad[1] != 0):
                    print(f"当前梯度为:{self.current_grad}")
                    print(f"当前距离为:{jnp.linalg.norm(self.current_pose - self.target_pos)}")

    def _generate_obstacles(self, num_obstacles, start, end):
        """生成障碍物参数，确保不与起点终点重叠"""
        robotx, roboty = start[:2]
        targetx, targety = end[:2]
        obstacles = []
        for _ in range(num_obstacles):
            while True:
                # 随机生成椭圆参数
                cx, cy = np.random.uniform(-9.5, 9.5, 2)
                a = np.random.uniform(0.8, 1.5)
                b = np.random.uniform(0.5, 1.0)
                theta = np.random.uniform(0, 2*np.pi)
                if a < b:
                    temp = b
                    b = a
                    a = temp
                
                # 检查与起点终点的安全距离
                if (np.linalg.norm([cx - robotx, cy - roboty]) > 3 and
                    np.linalg.norm([cx - targetx, cy - targety]) > 3):
                    # 检查与已有障碍物的间距，确保大于2的距离
                    overlap = False
                    for obs in obstacles:
                        # 计算当前障碍物与已有障碍物之间的圆心间距
                        dist = np.linalg.norm([cx - obs[0], cy - obs[1]])
                        if dist < 3:
                            overlap = True
                            break
                    
                    if not overlap:
                        obstacles.append([cx, cy, a, b, theta])
                        break
                    
        return np.array(obstacles, dtype=np.float32)

    def __obs_callback(self, num_obstacles):
        """ 椭圆障碍物回调 """
        # 解析椭圆参数并计算 C 和 d
        C_list = []  # 临时存储 C
        d_list = []  # 临时存储 d
        Call_list = []
        dall_list = []
        # 清理旧数据
        self.C = jnp.zeros((0, 2, 2))  # 形状矩阵
        self.d = jnp.zeros((2, 0))     # 中心位置
        self.C_all = jnp.zeros((0,2,2))
        self.d_all = jnp.zeros((2,0))

        robot_x, robot_y = self.current_pose[:2]
        for i in range(num_obstacles):
            # 提取参数
            cx = self.obstacles[i, 0]
            cy = self.obstacles[i, 1]
            a = self.obstacles[i, 2]  # 长轴
            b = self.obstacles[i, 3]  # 短轴
            theta = self.obstacles[i, 4]  # 旋转角度

            # 计算机器人到椭圆圆心的距离
            distance = math.sqrt((robot_x - cx) ** 2 + (robot_y - cy) ** 2)

            # 计算变换矩阵 C 和 d
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            C = np.array([
                [a * cos_t, -b * sin_t],
                [a * sin_t,  b * cos_t]
            ])
            d = np.array([cx, cy])

            Call_list.append(C)
            dall_list.append(d)

            # 判断距离是否大于三倍长轴
            if distance > 3:
                continue  # 跳过该椭圆

            # 添加到临时列表
            C_list.append(C)
            d_list.append(d)

        self.C_all = jnp.stack(Call_list, axis=0)
        self.d_all = jnp.column_stack(dall_list)

        # 转换为JAX数组
        if C_list:
            self.C = jnp.stack(C_list, axis=0)
            self.d = jnp.column_stack(d_list)

        else:
            self.C = jnp.zeros((0, 2, 2))
            self.d = jnp.zeros((2, 0))

            
    def save_dataset(self):
        """ 保存数据集到.npz文件 """
        if len(self.collected_states) == 0:
            print("无数据可保存")
            return
        
        # 转换为NumPy数组
        states = np.stack(self.collected_states, axis=0)          # (N, 2)
        gradients = np.stack(self.collected_gradients, axis=0)    # (N, 2)
        cdf_values = np.array(self.collected_cdfvalue)            # (N, ) 保存CDF值

        # 保存点云数据为列表形式，保留原始的变长度
        # pointcloud = self.collected_pointcloud  # 直接使用列表而不进行堆叠
        pointcloud = np.empty(len(self.collected_pointcloud), dtype=object)
        for i, arr in enumerate(self.collected_pointcloud):
            pointcloud[i] = arr
        
        # 合并输入特征 X = [states, obstacles_flatten]
        N = states.shape[0]
        states_flatten = states.reshape(N, 2)
        X = states_flatten  # (N, 27)
        
        # 合并梯度和CDF，CDF存储在y中
        y = np.hstack([gradients, cdf_values[:, None]])  # (N, 3)，将CDF作为第三列加入y中
        
        # 保存为.npz文件
        # np.savez('/home/guo/MPC-D-CBF-main/output/collected_dataset.npz', X=X, y=y, z=pointcloud)
        np.savez('/home/ubuntu/gxf/model/lidar_lcdf_1.npz', X=X, y=y, z=pointcloud, allow_pickle=True)

        print(f'collected dataset saved with {len(X)} samples')


            
    def __calculate_nominal_input(self):
        """ Go-to-goal 标称控制器 """
        target_vector = self.target_pos - self.current_pose_jax  # 确保self.target_pos是JAX数组
        distance = jnp.linalg.norm(target_vector)

        return jnp.where(distance < 0.1, jnp.zeros(2), self.k_p * (target_vector / distance))

    @partial(jit, static_argnums=(0,))
    def _compute_psi_k(self, x, C_k, d_k):
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

    def calculate_density(self, current_C, current_d):
    # 获取当前状态的快照
        target = self.target_pos.copy()
    
        @partial(jit, static_argnums=())
        def fresh_rho(x1, x2):
            x = jnp.array([x1, x2], dtype=jnp.float64)
            # 使用快照数据而非self引用
            def body_fun(k, psi_total):
                return psi_total * self._compute_psi_k(
                    x, current_C[k], current_d[:,k]
                )
            psi_total = jax.lax.fori_loop(0, len(current_d), body_fun, 1.0)
            # return 8500*psi_total / (jnp.sum((x - target)**2))
            return 200*psi_total / jnp.sqrt(jnp.sum((x - target)**2))
    
        return fresh_rho  # 返回不依赖self的新函数

if __name__ == '__main__':
    tracker = EllipseTracker()
    # ========== 新增信号处理函数 ==========
    def signal_handler(sig, frame):
        print("\n捕获到 Ctrl+C，正在保存数据集...")
        tracker.save_dataset()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)  # 注册中断处理器
    
    try:
        tracker.run()
    finally:
        tracker.save_dataset()  # 确保即使异常退出也会保存