#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

class PerfectLidarBridge:
    def __init__(self):
        rospy.init_node('perfect_lidar_bridge_node')

        # 1. 话题与坐标系配置
        self.input_topic = rospy.get_param("~input_topic", "/velodyne_points") # 接收仿真32线全量点云
        self.output_topic = rospy.get_param("~output_topic", "/densitynet_input_points") # 直喂给前端网络
        self.frame_id = "velodyne"

        # 2. 🔴 关键超参数配置：必须与您 DensityNet 训练时的设定完全一致！
        self.num_rays = rospy.get_param("~num_rays", 512)     # 🔴 请修改为您的网络训练时的射线数（如 512 或 180）
        self.max_range = rospy.get_param("~max_range", 3.0)   # 🔴 对应专家数据生成的探测距离（如 3.0m）
        self.range_min = 0.1

        # 3. 角度桶网格初始化
        self.bin_edges = jnp.linspace(-np.pi, np.pi, self.num_rays + 1) if 'jnp' in globals() else np.linspace(-np.pi, np.pi, self.num_rays + 1)
        self.ray_angles = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0 # 每个桶的中心角度

        # 订阅与发布
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.pc_callback, queue_size=1)
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=10)
        
        rospy.loginfo("【问题三完美修复节点】已启动！期望射线数: %d, 最大探测距离: %.1fm", self.num_rays, self.max_range)

    def pc_callback(self, msg):
        # 步骤 1: 极速读取全量 3D 点云坐标 (N, 3)
        pts_gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_3d = np.array([[p[0], p[1], p[2]] for p in pts_gen])
        
        if points_3d.size == 0: 
            return

        # 步骤 2: 【空间高度屏障截取】在雷达系下截取地面以上、雷达水平面附近的障碍物本体
        # 地面在 -0.5635m，选择 > -0.45m 完美踢除地面，选择 < 0.2m 踢除天花板或过高的无效噪点
        spatial_mask = (points_3d[:, 2] > -0.45) & (points_3d[:, 2] < 0.20)
        valid_3d_pts = points_3d[spatial_mask]
        
        if valid_3d_pts.size == 0: 
            return

        # 步骤 3: 【2D投射与极坐标转换】拍扁到 X-Y 平面
        x = valid_3d_pts[:, 0]
        y = valid_3d_pts[:, 1]
        distances_2d = np.hypot(x, y)
        angles_2d = np.arctan2(y, x)

        # 距离直通滤波
        dist_mask = (distances_2d >= self.range_min) & (distances_2d <= self.max_range)
        distances_2d = distances_2d[dist_mask]
        angles_2d = angles_2d[dist_mask]

        if distances_2d.size == 0: 
            return

        # 步骤 4: 【角度桶重采样】利用 NumPy 矢量化操作将异质点云强制重构成标准 2D Planar 射线
        # 计算每个点落在 512 个角度桶中的哪一个
        bin_indices = np.floor((angles_2d + np.pi) / (2 * np.pi) * self.num_rays).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_rays - 1)

        # 初始化 512 条射线的理想距离数组
        reconstructed_ranges = np.full(self.num_rays, np.inf)
        # 核心高频行：在每个角度桶内挑选距离最近的那个点（完美模拟 2D 激光雷达平面扫描）
        np.minimum.at(reconstructed_ranges, bin_indices, distances_2d)

        # 步骤 5: 【提取有效射线并还原二维坐标】
        valid_ray_mask = reconstructed_ranges < np.inf
        final_distances = reconstructed_ranges[valid_ray_mask]
        final_angles = self.ray_angles[valid_ray_mask]

        if final_distances.size == 0: 
            return

        # 重新解算出干净、无畸变的二维平面坐标 (N, 2)
        reconstructed_x = final_distances * np.cos(final_angles)
        reconstructed_y = final_distances * np.sin(final_angles)

        # 步骤 6: 🔴🔴【最核心对齐步骤】执行与训练集 post_process() 顺规一致的近到远距离排序！
        sorted_indices = np.argsort(final_distances)
        sorted_x = reconstructed_x[sorted_indices]
        sorted_y = reconstructed_y[sorted_indices]

        # 步骤 7: 打包并发布严格合规的 2D 拓扑点云给神经网络
        header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        # 补全 Z 轴为 0.0，使特征完美降维到 2D 平面
        output_3d = np.hstack((sorted_x.reshape(-1, 1), sorted_y.reshape(-1, 1), np.zeros((sorted_x.size, 1))))
        
        pc2_msg = point_cloud2.create_cloud(header, fields, output_3d)
        self.pub.publish(pc2_msg)

if __name__ == '__main__':
    try:
        bridge = PerfectLidarBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass