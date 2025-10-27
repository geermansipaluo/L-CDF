#!/usr/bin/env python
import rospy
import math
import numpy as np
from sensor_msgs.msg import PointCloud2, LaserScan
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

class PointCloudToLaserScan:
    def __init__(self):
        rospy.init_node('pointcloud_to_laserscan_node')
        
        # 配置参数
        self.frame_id = "os_sensor"  # 固定坐标系
        self.range_min = rospy.get_param('~range_min', 0.1)   # 最小检测距离(m)
        self.range_max = rospy.get_param('~range_max', 10.0)  # 最大检测距离(m)
        self.angle_min = rospy.get_param('~angle_min', -math.pi)  # 起始角度(rad)
        self.angle_max = rospy.get_param('~angle_max', math.pi)   # 终止角度(rad)
        self.angle_increment = rospy.get_param('~angle_increment', 0.00174533)  # 角度增量(0.1度)
        
        # 创建LaserScan消息模板
        self.scan_msg = LaserScan()
        self.scan_msg.header = Header(frame_id=self.frame_id)
        self.scan_msg.angle_min = self.angle_min
        self.scan_msg.angle_max = self.angle_max
        self.scan_msg.angle_increment = self.angle_increment
        self.scan_msg.range_min = self.range_min
        self.scan_msg.range_max = self.range_max

        self.curr_state = np.zeros(3)
        
        # 初始化角度范围数组
        self.num_readings = int((self.angle_max - self.angle_min) / self.angle_increment)
        self.angles = [self.angle_min + i * self.angle_increment for i in range(self.num_readings)]
        
        # 发布器(20Hz)
        self.pub = rospy.Publisher('/Laserscan_Points', LaserScan, queue_size=10)
        
        # 订阅点云话题
        rospy.Subscriber('/filtered_3d', PointCloud2, self.cloud_callback)

        self.__sub_odom = rospy.Subscriber(  # done!
            '/robot/dlio/odom_node/pose', PoseStamped, self.__odom_cb, queue_size=1)
        
        # 计时器控制20Hz发布频率
        rospy.Timer(rospy.Duration(0.05), self. publish_scan)
        
        rospy.loginfo("PointCloud to LaserScan converter ready. Target frame: %s", self.frame_id)

    def __odom_cb(self, data):
        self.curr_state[0] = data.pose.position.x
        self.curr_state[1] = data.pose.position.y
        
    def cloud_callback(self, cloud_msg):
        """处理传入的点云数据"""
        # 提取点云中的XYZ点
        points = []
        for point in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            # 只处理有效范围内的点
            if not math.isnan(point[0]) and not math.isnan(point[1]) and not math.isnan(point[2]):
                distance = math.sqrt((point[0])**2 + (point[1])**2)
                if self.range_min <= distance <= self.range_max:
                    points.append(point)
        
        # 计算每个角度的最近距离
        ranges = [float('nan')] * self.num_readings  # 默认为NaN
        
        for x, y, z in points:
            angle = math.atan2(y, x)
            
            # 找到最近的索引
            diff = [abs(angle - a) for a in self.angles]
            min_idx = np.argmin(diff)
            
            # 计算距离
            dist = math.sqrt(x**2 + y**2)
            
            # 更新最近距离
            if math.isnan(ranges[min_idx]) or dist < ranges[min_idx]:
                ranges[min_idx] = dist
        
        # 更新扫描数据
        self.scan_msg.ranges = ranges
        self.scan_msg.header.stamp = rospy.Time.now()
        
    def publish_scan(self, event):
        """定时发布LaserScan数据(20Hz)"""
        if hasattr(self, 'scan_msg'):
            self.pub.publish(self.scan_msg)

if __name__ == '__main__':
    try:
        converter = PointCloudToLaserScan()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
