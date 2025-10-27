#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry

class OdomForwarder:
    def __init__(self):
        # 初始化发布器（队列大小设为1保持一致性）
        self.gazebo_pub = rospy.Publisher("/gazebo_pose", Odometry, queue_size=10)
        
        # 订阅原始话题（注意话题名称与C++版本一致）
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback)
        
    def callback(self, msg):
        """ 收到Odometry消息时的回调处理 """
        # 直接转发原始消息
        msg.header.frame_id = "world" 
        self.gazebo_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('odom_forward_py')
    OdomForwarder()
    rospy.spin()
