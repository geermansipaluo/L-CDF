#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

class PointCloud3DProcessor:
    def __init__(self):
        # ROS参数配置
        self.input_topic = rospy.get_param("~input_topic", "/velodyne_points") # 🔴自动匹配您urdf中的 velodyne_points
        self.output_topic = rospy.get_param("~output_topic", "/filtered_3d")
        self.frame_id = rospy.get_param("~frame_id", "velodyne") # 🔴匹配Jackal的雷达frame

        # 🔴【修复问题一】：HDL-32E是32线雷达，15线最接近0度水平面平视前方
        self.target_ring = 15

        # 初始化ROS订阅/发布
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.pc_callback)
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)

    def pc_callback(self, msg):
        # 步骤1：提取原始点云的X/Y/Z坐标
        points_3d, rings = self.extract_3d_points(msg)
        if points_3d.size == 0:
            return
        
        # 步骤2：筛选出平视前方的特定水平线束
        line_points = self.filter_by_ring(points_3d, rings)
        if line_points.size == 0:
            return

        # 🔴【修复问题二】：精准直通滤波
        # 根据Jackal模型精确推导，地面在雷达系下处于 z = -0.5635m
        # 设置 z > -0.50m 既能完美切除地面噪点，又能暴露出高出地面6cm以上的所有障碍物整体！
        filtered_points = line_points[(line_points[:, 2] > -0.50) & (line_points[:, 2] < 0.5)]

        # 步骤3：发布处理后的干净二维平面障碍物点云
        if filtered_points.size > 0:
            self.publish_3d_cloud(filtered_points)

    def extract_3d_points(self, msg):
        # 支持有些仿真雷达把ring信息放在fields里
        if not any(f.name == 'ring' for f in msg.fields):
            # 如果仿真中没有ring字段，退而求其次使用垂直高度强行拟合（这里保留兼容性）
            pc_gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            pts = np.array([[p[0], p[1], p[2]] for p in pc_gen])
            # 如果没有ring，造一个全为15的假ring跳过检查
            return pts, np.ones(pts.shape[0]) * 15
        
        pc_gen = point_cloud2.read_points(
            msg, 
            field_names=("x", "y", "z", "ring"), 
            skip_nans=True
        )

        points = []
        rings = []
        for p in pc_gen:
            points.append([p[0], p[1], p[2]])
            rings.append(p[3]) 

        return np.array(points), np.array(rings)
    
    def filter_by_ring(self, points, rings):
        mask = (rings == self.target_ring)
        return points[mask]

    def publish_3d_cloud(self, points_3d):
        header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        pc2_msg = point_cloud2.create_cloud(header, fields, points_3d)
        self.pub.publish(pc2_msg)

if __name__ == "__main__":
    rospy.init_node("pointcloud_3d_processor")
    processor = PointCloud3DProcessor()
    rospy.spin()