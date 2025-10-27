#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

class PointCloud3DProcessor:
    def __init__(self):
        # ROS参数配置
        self.input_topic = rospy.get_param("~input_topic", "/ouster/points")
        self.output_topic = rospy.get_param("~output_topic", "/filtered_3d")
        self.frame_id = rospy.get_param("~frame_id", "os_sensor")

        self.target_ring = 63

        # 初始化ROS订阅/发布
        self.sub = rospy.Subscriber(self.input_topic, PointCloud2, self.pc_callback)
        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)

    def pc_callback(self, msg):
        all_fields = [field.name for field in msg.fields]
        rospy.loginfo(f"当前点云字段: {all_fields}")

        # 步骤1：提取原始点云的X/Y/Z坐标（三维数组）
        points_3d, rings = self.extract_3d_points(msg)
        if points_3d.size == 0:
            rospy.logwarn("没有接收到有效的点云数据，跳过处理")
            return
        
        line63_points = self.filter_by_ring(points_3d, rings)
        if line63_points.size == 0:
            rospy.logwarn(f"第{self.target_ring}线无有效数据")
            return

        # 步骤2：直通滤波，保留z > 0.2的数据
        filtered_points = self.passthrough_filter(line63_points)

        # 步骤3：发布处理后的三维点云（兼容ROS工具链）
        self.publish_3d_cloud(filtered_points)

    def extract_3d_points(self, msg):
        if not any(f.name == 'ring' for f in msg.fields):
            rospy.logerr("点云中未检测到'ring'字段！")
            return np.empty((0,3)), np.array([])
        
        pc_gen = point_cloud2.read_points(
            msg, 
            field_names=("x", "y", "z", "ring", "t"), 
            skip_nans=True
        )

        points = []
        rings = []
        for p in pc_gen:
            points.append([p[0], p[1], p[2]])  # x,y,z
            rings.append(p[3]) 

        return np.array(points), np.array(rings)
    
    def filter_by_ring(self, points, rings):
        """筛选特定ring值的点"""
        mask = (rings == self.target_ring)
        return points[mask]

    def passthrough_filter(self, points_3d):
        """ 直通滤波：保留z > 0.2的数据 """
        # 筛选z值大于0.2的点
        return points_3d[points_3d[:, 2] > 0.0]

    def publish_3d_cloud(self, points_3d):
        """ 发布三维点云（兼容ROS PointCloud2格式） """
        header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        
        # 定义字段（x,y,z）
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        
        # 创建并发布消息
        pc2_msg = point_cloud2.create_cloud(header, fields, points_3d)
        self.pub.publish(pc2_msg)

if __name__ == "__main__":
    rospy.init_node("pointcloud_3d_processor")
    processor = PointCloud3DProcessor()
    rospy.spin()