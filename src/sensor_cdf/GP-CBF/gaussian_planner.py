#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool, Float32MultiArray, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
import threading
import numpy as np
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
import tf
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import cvxpy as cp
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import random
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint
import signal 
import sys
import time


def distance_global(c1, c2):
    return np.sqrt((c1[0] - c2[0]) * (c1[0] - c2[0]) + (c1[1] - c2[1]) * (c1[1] - c2[1]))

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
        merged = self.merge_points(points_robot)
        
        return merged, self.min_distance

    def filter_and_sort_points(self, points, current_position):
        """基于距离筛选和排序"""
        if points.size == 0:
            return np.zeros((0, 2))
        
        # deltas = points - current_position
        distances = np.linalg.norm(points, axis=1)
        self.min_distance = np.min(distances)
        mask = distances < 5

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

class Local_Planner():
    def __init__(self):

        signal.signal(signal.SIGINT, self.signal_handler)
        self.start_time = time.time()
        self.min_distance = 0
        self.all_points = []
        self.vall = []
        self.infeasible = 0
        self.timeu= 0

        # 频率默认 20Hz
        self.z = 0
        self.numRays = 360
        self.N = 15
        self.d_max = 10
        self.d_sample = 0.1  # TODO?
        self.goal_state = np.zeros([self.N, 3])
        self.curr_state = None
        self.last_state = np.zeros(3)
        self.global_path = None
        self.curr_state_lock = threading.Lock()
        self.global_path_lock = threading.Lock()
        self.scan = None  # list(data.ranges)
        self.goal_point = np.array([15, 0])

        self.v = np.zeros(2)
        self.clf_enter = None
        self.mode = 0  # 0:goal-seeking 1:explore

        self.test = 0
        
        self.N_controller =0
        
        self.avg_points = 0
        self.avg_gp_synth =0
        self.avg_gp_update =0
        self.avg_controller =0

        self.kernel = RBF(length_scale=0.6, length_scale_bounds="fixed") + \
            WhiteKernel(noise_level=0.01, noise_level_bounds="fixed")
        # self.kernel = 1*RBF(length_scale=0.42) + \
        #     WhiteKernel(noise_level=0.01)

        # 注意这里y的均值和论文里的不一样，是0，还需要在源码中修改
        self.gpr = GaussianProcessRegressor(
            kernel=self.kernel, n_restarts_optimizer=0,  normalize_y=False)
        
        self.pc_processor = PointCloudProcessor(
            max_points=200,
            voxel_size=0.1
        )

        self.filter_pointcloud = np.zeros((0,2))
        self.__sub_lidar = rospy.Subscriber('/filtered_3d', PointCloud2, self.cloud_callback)

        self.__timer_replan = rospy.Timer(  # done
            rospy.Duration(0.05), self.__replan_cb)

        self.__sub_odom = rospy.Subscriber(  # done!
            '/robot/dlio/odom_node/pose', PoseStamped, self.__odom_cb, queue_size=1)
        
        self.vel_pub = rospy.Publisher(
            '/cmd_vel', Twist, queue_size=10)
        
        self.gaussian_cbf_pub = rospy.Publisher(
            '/cbf', Marker, queue_size=10)
        
        self.closest_point_pub = rospy.Publisher('closest_point_distance', Float32, queue_size=10)

    def signal_handler(self, sig, frame):
        print("\n检测到 Ctrl+C，正在保存数组...")
        end_time = time.time()
        print(f"run time is:{end_time - self.start_time}")
        vall = np.array(self.vall)
        avg_v = np.mean(vall)
        print(f"avg velocity is:{avg_v}")
        print(f"infeasible number is:{self.infeasible}")
        print(f"平均控制频率：{1/(self.avg_controller+self.avg_gp_synth)}")
        x = np.stack(self.all_points, axis=0)
        np.savez('/home/maslab1/L-CDF/src/sensor_cdf/scripts/saved_data/all_points_gpcbf.npz', X=x, allow_pickle=True)
        print("保存完毕，程序退出。")
        sys.exit(0)

    def select_point(self):

        pass

    def switch_mode(self):
        v_threshold = 0.005
        if self.mode == 0:  # current: goal-seeking
            if (self.vx**2+self.vy**2) < v_threshold:
                self.mode = 1

        else:
            pass

    def safe_boundary_unsafe_sample_creator(self, d_lidar, Xr):
        """
        Create safe and unsafe samples based on the lidar data
        input:
            d_lidar: lidar data, list of range data,length=360
            Xr: robot state ,numpy 2-D array (3,1)
            ns: number of samples in each ray
        return:
         shape: inputs: (n,2), output: (n,)


        """
        x_train = np.empty((0, 2), float)
        y_train = np.empty((0, 1), float)
        x_train = self.filter_pointcloud
        N = x_train.shape[0]
        y_train = -1*np.ones((N,1))
        rospy.loginfo(f"ytrain:{y_train.shape}")
        

        return x_train, y_train.ravel()

    def h(self, x_train, y_train, x):
        def frange(start, stop, step):
            while start < stop:
                yield start
                start += step

        # fit the gaussian model

        # rospy.loginfo("number of data points: %d" % x_train.shape[0])

        self.gpr.fit(x_train, y_train)
        state = x[0:2].reshape(1, -1)

        shift = 0.6
        _h = self.gpr.predict(state, return_cov=False)+shift

        if (self.test):  # visialize the gaussian cbf
            # x_min, x_max, x_step = x[0]-3.5, x[0]+3.5, 0.05
            # y_min, y_max, y_step = x[1]-3.5, x[1]+3.5, 0.05 0.02很好

            x_min, x_max, x_step = x[0]-2, x[0]+2, 0.02
            y_min, y_max, y_step = x[1]-2, x[1]+2, 0.02
            # x_min, x_max, x_step = x[0]-0.5, x[0]+0.5, 0.1
            # y_min, y_max, y_step = x[1]-0.5, x[1]+0.5, 0.1

            marker = Marker()
            marker.header.frame_id = 'os_sensor'
            marker.header.stamp = rospy.Time.now()
            marker.ns = 'gaussian_cbf'
            marker.id = 0
            marker.type = Marker.POINTS
            marker.action = Marker.ADD

            marker.scale.x = 0.01  # Point width
            marker.scale.y = 0.01  # Point height
            marker.color.a = 1.0  # Alpha
            marker.color.r = 0.0  # Red
            marker.color.g = 1.0  # Green
            marker.color.b = 0.0  # Blue

            points = []
            for x in frange(x_min, x_max, x_step):
                for y in frange(y_min, y_max, y_step):
                    state = np.array([x, y]).reshape(1, -1)
                    h = self.gpr.predict(state, return_cov=False)+shift

                    points.append([x, y, h[0]])

            z_min = min(points, key=lambda p: p[2])[2]
            z_max = max(points, key=lambda p: p[2])[2]
            rospy.loginfo("z_min: %.5f, z_max: %.5f,z_delta: %.5f" %
                          (z_min, z_max, z_max-z_min))

            # 确保 z_max - z_min = 1
            # z_max = z_min + 1
            marker.points = []
            marker.colors = []
            for (x, y, z) in points:
                point = Point(x=x, y=y, z=z)
                marker.points.append(point)

                # 归一化颜色
                normalized_z = (z - z_min) / (z_max - z_min)
                color = ColorRGBA()

                if np.abs(z) <= 0.015:
                    color.r = 0.0
                    color.g = 0.0
                    color.b = 1.0
                    color.a = 1.0
                else:
                    color.r = 1.0 - normalized_z
                    color.g = normalized_z
                    color.b = 0.0
                    color.a = 1.0

                marker.colors.append(color)

            self.gaussian_cbf_pub.publish(marker)

        # 现在求h对x的偏导数。应该是1*3的矩阵
        N = x_train.shape[0]
        K_inv = np.linalg.inv(self.gpr.kernel_.__call__(x_train))
        Y_T = y_train.reshape(1, -1)
        grad = np.zeros((1, 3))
        _grad = np.zeros((N, 2))

        l = self.gpr.kernel_.k1.length_scale

        for i in range(N):

            # 导数第二项
            kse = self.gpr.kernel_.k1.__call__(
                state, x_train[i, :].reshape(1, -1))
            p_3 = state-x_train[i, :].reshape(1, -1)  # bshape 1*2
            _grad[i, :] = -kse*p_3/(l**2)

        grad[0, 0:2] = Y_T@K_inv@_grad
        # h_dot=grad@(x.numpy().reshape(-1,1))
        self.h_value = _h
        grad = grad[0, 0:2]

        return _h, grad.reshape(1, -1)

    def u_reference(self, x):
        v_scaling = 1
        u = self.goal_point-x[0:2]
        return v_scaling*u/np.linalg.norm(u)

    def u(self):
        self.curr_state_lock.acquire()
        if self.curr_state is None:
            self.curr_state_lock.release()
            return

        if self.mode == 1:
            clf, grad = self.CLF()
            if clf <= self.clf_enter:
                self.mode = 0
                self.clf_enter = None
        self.mode = 0
        rospy.loginfo("1_current mode is %d" % self.mode)
        if self.mode == 0:  # for goal-seeking
            u = self.u_goal_seeking()
        else:
            # u = self.u_explore()
            u = self.u_explore_cvx()

        self.v = u
        # rospy.loginfo("3_current v is %.3f,%.3f" % (self.v[0], self.v[1]))
        u = self.diffeomorphism_transformation(u)
        if self.min_distance < 1 and self.min_distance != 0:
            self.vall.append(u[0])

        control_cmd = Twist()
        self.linear_speed = u[0]
        self.angular_speed = u[1]

        # rospy.loginfo("4_current Linear Speed: %.3f, Angular Speed: %.3f" %
        #               (self.linear_speed, self.angular_speed))

        control_cmd.linear.x = self.linear_speed
        control_cmd.angular.z = self.angular_speed
        # rospy.loginfo("Linear Speed: %.1f, Angular Speed: %.1f" %
        #               (self.linear_speed, self.angular_speed))

        # 停止运动
        if (self.test):
            control_cmd.linear.x = 0
            control_cmd.angular.z = 0

        # rospy.loginfo("运动距离: %.5f" % np.linalg.norm(self.last_state-self.curr_state))

        self.last_state = self.curr_state
        self.vel_pub.publish(control_cmd)

        # self.switch_mode()

        self.curr_state_lock.release()

    def u_explore_cvx(self):
        # 1. get observation

        x = self.curr_state
        measurement = self.scan

        start = rospy.Time.now()
        x_train, y_train = self.safe_boundary_unsafe_sample_creator(
            measurement, x)
        # rospy.loginfo("number of data points: %d" % x_train.shape[0])

        h, grad = self.h(x_train, y_train, x)
        rospy.loginfo("2_current_h: %.5f" % h)

        end1 = rospy.Time.now()
        rospy.loginfo("gaussian time cost: %.5f" % (end1-start).to_sec())

        g = np.eye(2)
        Lg_h = grad@g

        u = cp.Variable(2)
        r1 = cp.Variable(1)  # panalty term cbf
        r2 = cp.Variable(1)  # panalty term clf
        k1 = 2
        k2 = 10
        k3 = 1

        Q = (k1-k3)*np.eye(2)+k2*Lg_h.T@Lg_h
        p = -2*k1*self.v.reshape(1, -1)+2*k2*h*Lg_h

        objective_expression = cp.Minimize(
            cp.quad_form(u, Q)+p@u)

        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [-Lg_h[0, 0], -Lg_h[0, 1]]])
        b = np.array([0.2, 0.2, 0.2, 0.2, h[0]])

        constraints = [A @ u <= b]

        # 定义并解决优化问题
        prob = cp.Problem(objective_expression, constraints)
        prob.solve(verbose=False, solver=cp.OSQP)

        end2 = rospy.Time.now()
        rospy.loginfo("optimization time cost: %.5f" % (end2-end1).to_sec())
        # 如果优化问题不可解，则输出错误信息
        if prob.status != cp.OPTIMAL:
            rospy.logerr("优化问题不可解")
            rospy.loginfo("h: %.5f" % h)
            # rospy.loginfo("Lg_h: %.5f,%.5f" % (Lg_h[0, 0], Lg_h[0, 1]))
            return self.v

        # 输出优化结果
        # print("vx,vy最优值：", prob.value)
        # rospy.loginfo("vx,vy:%.5f,%.5f" %
        #               (res.x[0], res.x[1]))

        # clf, grad = self.CLF()

        # if clf <= self.clf_enter or grad@res.x > 0:
        # if clf <= self.clf_enter:
        #     self.mode = 0
        #     self.clf_enter = None

        return u.value

    def u_explore(self):
        # 1. get observation

        x = self.curr_state
        measurement = self.scan

        x_train, y_train = self.safe_boundary_unsafe_sample_creator(
            measurement, x)
        # rospy.loginfo("number of data points: %d" % x_train.shape[0])

        h, grad = self.h(x_train, y_train, x)
        rospy.loginfo("2_current_h: %.5f" % h)

        g = np.eye(2)
        Lg_h = grad@g

        if (0):

            u = cp.Variable(2)
            r1 = cp.Variable(1)  # panalty term cbf
            r2 = cp.Variable(1)  # panalty term clf
            k1 = 2
            k2 = 10
            k3 = 1

            Q = (k1-k3)*np.eye(2)+k2*Lg_h.T@Lg_h
            p = -2*k1*self.v.reshape(1, -1)+2*k2*h*Lg_h

            objective_expression = cp.Minimize(
                cp.quad_form(u, Q)+p@u)

            A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

            b = np.array([0.2, 0.2, 0.2, 0.2])

            constraints = [A @ u <= b]

            # 定义并解决优化问题
            prob = cp.Problem(objective_expression, constraints)
            prob.solve(verbose=True, solver=cp.QCQP)
            # 如果优化问题不可解，则输出错误信息
            if prob.status != cp.OPTIMAL:
                rospy.logerr("优化问题不可解")
                rospy.loginfo("h: %.5f" % h)
                rospy.loginfo("Lg_h: %.5f,%.5f" % (Lg_h[0, 0], Lg_h[0, 1]))
                return

        # 目标函数，变成+h?

        def objective(u, v, k1, k2, k3, Lg_h, h):
            return k1*np.sum((u - v)**2) + k2*((Lg_h @ u+1*h)**2) - k3*np.sum(u**2)

        # 约束条件
        bounds = [(-0.2, 0.2), (-0.2, 0.2)]

        linear_constraint = {'type': 'ineq', 'fun': lambda u: Lg_h @ u + 1*h}

        # 初始猜测
        u0 = self.v

        # 参数
        v = self.v  # 这里只是一个例子，你需要根据你的问题来设置v
        k1 = 1
        k2 = 10
        k3 = 1

        # 使用scipy.optimize.minimize函数来解决问题
        res = minimize(objective, u0, args=(
            v, k1, k2, k3, Lg_h, h), constraints=linear_constraint, bounds=bounds)

        # 输出优化结果
        # print("vx,vy最优值：", prob.value)
        # rospy.loginfo("vx,vy:%.5f,%.5f" %
        #               (res.x[0], res.x[1]))

        clf, grad = self.CLF()

        # if clf <= self.clf_enter or grad@res.x > 0:
        # if clf <= self.clf_enter:
        #     self.mode = 0
        #     self.clf_enter = None

        return res.x

    def solve_CBF_QP(self, x, measurement):

        # 1. get observation

        x_train, y_train = self.safe_boundary_unsafe_sample_creator(
            measurement, x)
        rospy.loginfo("number of data points: %d" % x_train.shape[0])

        h, grad = self.h(x_train, y_train, x)
        rospy.loginfo("h: %.2f" % h)

        g = np.eye(2)
        Lg_h = grad@g
        # 2. define the optimization problem
        u = cp.Variable(2)
        r = cp.Variable(1)  # panalty term

        # u
        u_ref = self.u_reference(x).reshape(u.shape)
        objective_expression = cp.Minimize(cp.sum_squares(u - u_ref)+1000*r)
        constraints = []

        # TODO 注意h函数有问题，所以k类函数项没有添加
        # constraints.append(Lg_h@u + self.h_alpha*h + r >= 0)
        constraints.append(Lg_h@u + 0.1*h+r >= 0)
        constraints.append(r >= 0)
        # constraints.append(u[0] >= -0.2)
        # constraints.append(u[0] <= 0.2)
        # constraints.append(u[1] >= -0.2)
        # constraints.append(u[1] <= 0.2)
        constraints.append(u >= -0.2)
        constraints.append(u <= 0.2)

        # 定义并解决优化问题
        prob = cp.Problem(objective_expression, constraints)
        prob.solve(verbose=False, solver=cp.ECOS)
        self.vx = u.value[0]
        self.vy = u.value[1]

        # 输出优化结果
        # print("最优值：", prob.value)
        rospy.loginfo("Optimal vx,xy: %.2f, %.2f" % (u.value[0], u.value[1]))

        u = self.diffeomorphism_transformation(u.value, x)
        return u

    def CLF(self):
        """
        x: numpy array(3,), current states,return the square of the distance to the goal point
        """

        v = 0.5*np.linalg.norm(self.curr_state[0:2]-self.goal_point)**2
        grad_v = np.array([self.curr_state[0]-self.goal_point[0],
                          self.curr_state[1]-self.goal_point[1]
                           ])

        return v, grad_v.reshape(1, -1)

    def u_goal_seeking(self):

        # 1. get observation
        x = self.curr_state
        measurement = self.scan

        

        # 统计时间

        x_train, y_train = self.safe_boundary_unsafe_sample_creator(
            measurement, x)
        # rospy.loginfo("number of data points: %d" % x_train.shape[0])

        # self.avg_points = (self.avg_points*(self.N_controller-1) + x_train.shape[0])/self.N_controller
        # rospy.loginfo("avg_points: %.5f" % self.avg_points)
        start= rospy.Time.now()
    
        h, grad = self.h(x_train, y_train, x)
        end1 = rospy.Time.now()
        if self.min_distance < 1 and self.min_distance != 0:
            self.N_controller += 1
            self.avg_gp_synth = (self.avg_gp_synth*(self.N_controller-1) + (end1-start).to_sec())/self.N_controller
        rospy.loginfo("avg_gp_synth: %.5f" % self.avg_gp_synth)
        # rospy.loginfo("2_current_h: %.5f" % h)


        g = np.eye(2)
        Lg_h = grad@g
        v, dv_dx = self.CLF()
        Lg_v = dv_dx@g
        # 2. define the optimization problem
        u = cp.Variable(2)
        r1 = cp.Variable(1)  # cbf panalty term
        r2 = cp.Variable(1)  # clf panalty term

        start = rospy.Time.now()

        # u
        u_ref = self.u_reference(x).reshape(u.shape)
        objective_expression = cp.Minimize(
            1*cp.sum_squares(u - u_ref)
            # 0.1*cp.sum_squares(u)
            + 1000*r1+r2)
        constraints = []

        # TODO 注意h函数有问题，所以k类函数项没有添加
        # constraints.append(Lg_h@u + self.h_alpha*h + r >= 0)
        constraints.append(Lg_h@u + 0.8*h+r1 >= 0)
        constraints.append(Lg_v@u+5*v+r2 >= 0)
        constraints.append(r1 >= 0)
        constraints.append(r2 >= 0)
        constraints.append(u >= -1)
        constraints.append(u <= 1)

        # 定义并解决优化问题
        prob = cp.Problem(objective_expression, constraints)
        prob.solve(verbose=False, solver=cp.ECOS)
        # prob.solve(verbose=True)
        if prob.status != cp.OPTIMAL:
            rospy.logerr("优化问题不可解")
            rospy.loginfo("h: %.5f" % h)
            self.infeasible += 1
            # rospy.loginfo("Lg_h: %.5f,%.5f" % (Lg_h[0], Lg_h[1]))
            self.mode = 1
            self.clf_enter, _ = self.CLF()

            return self.v

        end2 = rospy.Time.now()
        if self.min_distance < 1 and self.min_distance != 0:
            self.avg_controller = (self.avg_controller*(self.N_controller-1) + (end2-start).to_sec())/self.N_controller
        rospy.loginfo("avg_controller: %.5f" % self.avg_controller)
        # rospy.loginfo("optimization time cost: %.5f" % (end2-end1).to_sec())

        # 输出优化结果
        # print("最优值：", prob.value)
        # rospy.loginfo("vx,vy,R最优值:%.5f,%.5f,%.5f,%.5f" %
        #               (u.value[0], u.value[1], r1.value, r2.value))
        if (r2.value > 0.0000001 or np.linalg.norm(u.value) < 0.1 or np.linalg.norm(self.last_state-self.curr_state) < 0.05) and np.abs(h) < 0.15:
            if r2.value > 0.0000001:
                rospy.loginfo("CLF violation:%.5f" %
                              (r2.value))
            self.mode = 1
            rospy.loginfo("change mode to wandering")
            self.clf_enter, _ = self.CLF()
        return u.value

    def diffeomorphism_transformation(self, u):
        """Transform the control input from the u to v and w
        input: u: numpy array([2,])
            x: torch.Tensor([3,]), current states


        """
        v_limit = 1
        w_limit = 1
        theta = self.curr_state[2]
        L = 0.45

        v = u[0]*np.cos(theta) + u[1]*np.sin(theta)
        w = -u[0]*np.sin(theta)/L + \
            u[1]*np.cos(theta)/L

        if v > v_limit:
            v = v_limit
        if v < -v_limit:
            v = -v_limit
        if w > w_limit:
            w = w_limit
        if w < -w_limit:
            w = -w_limit

        _u = np.array([v, w])

        return _u

    def cloud_callback(self, msg):
        current_position = np.array(self.curr_state[:2])
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
            self.filter_pointcloud, self.min_distance = self.pc_processor.process(self.pointcloud, current_position)
            # print(f"过滤后点云：{self.filter_pointcloud}")
            # rospy.loginfo(f"二维点云数据长度为：{(self.filter_pointcloud).shape}")
        else:
            rospy.loginfo("没有二维点云数据！")

        

    def __replan_cb(self, event):
        if self.filter_pointcloud.size != 0:
            start = time.time()
            self.u()
            end = time.time()
            print(f"u的一次时长为：{end-start}")

    def __odom_cb(self, data):
        self.curr_state_lock.acquire()
        self.curr_state = np.zeros(3)
        self.curr_state[0] = data.pose.position.x
        self.curr_state[1] = data.pose.position.y
        quaternion = (
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w
        )
        self.all_points.append([self.curr_state[0], self.curr_state[1]])
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        self.curr_state[2] = yaw  # 弧度，（-pai， pai）

        self.curr_state_lock.release()


if __name__ == '__main__':
    rospy.init_node("nmpc_planner")
    nmpc_planner = Local_Planner()

    rospy.spin()
