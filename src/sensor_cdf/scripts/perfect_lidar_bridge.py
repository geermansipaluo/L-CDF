#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

class PerfectLidarBridge:
    def __init__(self):
        rospy.init_node('perfect_lidar_bridge_node')

        self.input_topic = rospy.get_param("~input_topic", "/velodyne_points") 
        self.output_topic = rospy.get_param("~output_topic", "/densitynet_input_points") 
        self.frame_id = "velodyne"

        # 完全对齐新专家系统的雷达截断定义
        self.num_rays = rospy.get_param("~num_rays", 512)     
        self.max_range = rospy.get_param("~max_range", 3.0)   
        self.range_min = 0.1

        self.bin_edges = np.linspace(-np.pi, np.pi, self.num_rays + 1)
        self.ray_angles = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0 

        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.pc_callback, queue_size=1)
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=10)
        
        rospy.loginfo("【图网络对齐雷达桥接器】已全面换装启动！硬屏蔽截断上限: %.1fm", self.max_range)

    def pc_callback(self, msg):
        pts_gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_3d = np.array([[p[0], p[1], p[2]] for p in pts_gen])
        
        if points_3d.size == 0: 
            return

        spatial_mask = (points_3d[:, 2] > -0.45) & (points_3d[:, 2] < 0.20)
        valid_3d_pts = points_3d[spatial_mask]
        
        if valid_3d_pts.size == 0: 
            return

        x = valid_3d_pts[:, 0]
        y = valid_3d_pts[:, 1]
        distances_2d = np.hypot(x, y)
        angles_2d = np.arctan2(y, x)

        dist_mask = (distances_2d >= self.range_min) & (distances_2d <= self.max_range)
        distances_2d = distances_2d[dist_mask]
        angles_2d = angles_2d[dist_mask]

        if distances_2d.size == 0: 
            return

        bin_indices = np.floor((angles_2d + np.pi) / (2 * np.pi) * self.num_rays).astype(int)
        bin_indices = np.clip(bin_indices, 0, self.num_rays - 1)

        reconstructed_ranges = np.full(self.num_rays, np.inf)
        np.minimum.at(reconstructed_ranges, bin_indices, distances_2d)

        # 🔴【最关键改动】：严格同步训练集 min_t < 2.99 截断！
        # 以前这里是 reconstructed_ranges < np.inf。现在必须设为 < 2.99 
        # 这样射向虚空没有砸中任何东西、或者距离太远的射线在部署端会被彻底丢弃，绝不生成无效图节点
        valid_ray_mask = reconstructed_ranges < 2.99
        final_distances = reconstructed_ranges[valid_ray_mask]
        final_angles = self.ray_angles[valid_ray_mask]

        if final_distances.size == 0: 
            # 如果视野内极度空旷无点云，也要发布空消息维持控制回路的心跳频率
            header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
            fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1), PointField('z', 8, PointField.FLOAT32, 1)]
            pc2_msg = point_cloud2.create_cloud(header, fields, np.zeros((0, 3)))
            self.pub.publish(pc2_msg)
            return

        reconstructed_x = final_distances * np.cos(final_angles)
        reconstructed_y = final_distances * np.sin(final_angles)

        header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        output_3d = np.hstack((reconstructed_x.reshape(-1, 1), reconstructed_y.reshape(-1, 1), np.zeros((reconstructed_x.size, 1))))
        
        pc2_msg = point_cloud2.create_cloud(header, fields, output_3d)
        self.pub.publish(pc2_msg)

if __name__ == '__main__':
    try:
        bridge = PerfectLidarBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass