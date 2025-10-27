#!/usr/bin/env python3
# update_state.py

import rospy
import tf
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray

class StateUpdater:
    def __init__(self):
        rospy.init_node('update_state')
        
        # 坐标系配置参数
        self.target_frame = rospy.get_param('~target_frame', 'base_link')
        self.source_frame = rospy.get_param('~source_frame', 'world')
        
        # 初始化TF监听器
        self.tf_listener = tf.TransformListener()
        
        # 状态发布者
        self.state_pub = rospy.Publisher('/curr_state', Float32MultiArray, queue_size=10)
        
        # 定时器配置（100Hz更新）
        self.timer = rospy.Timer(rospy.Duration(0.01), self.update_callback)
        
        rospy.loginfo("State updater initialized for frame [%s] to [%s]", 
                     self.source_frame, self.target_frame)

    def quat_to_yaw(self, quat):
        """四元数转偏航角（简化版本）"""
        x, y, z, w = quat
        return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    def update_callback(self, event):
        """定时器回调函数"""
        try:
            # 获取最新坐标变换
            (trans, rot) = self.tf_listener.lookupTransform(
                self.source_frame, 
                self.target_frame, 
                rospy.Time(0)
            )
            
            # 创建并填充消息
            state_msg = Float32MultiArray()
            state_msg.data = [
                trans[0],   # X坐标
                trans[1],   # Y坐标
                self.quat_to_yaw(rot)  # 偏航角
            ]
            
            # 发布状态
            self.state_pub.publish(state_msg)

        except (tf.LookupException, 
                tf.ConnectivityException, 
                tf.ExtrapolationException) as e:
            rospy.logdebug("TF查询异常: %s", str(e))

if __name__ == '__main__':
    try:
        updater = StateUpdater()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("状态更新节点异常终止")
