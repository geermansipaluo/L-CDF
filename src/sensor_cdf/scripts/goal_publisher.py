#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

def publish_static_goal():
    pub = rospy.Publisher('/custom_goal', PoseStamped, queue_size=10, latch=True)
    goal = PoseStamped()
    goal.header.frame_id = "world"
    goal.pose.position.x = 9
    goal.pose.position.y = 0
    goal.pose.position.z = 0
    goal.pose.orientation.x = 0
    goal.pose.orientation.y = 0
    goal.pose.orientation.z = 0
    goal.pose.orientation.w = 1
    pub.publish(goal)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('goal_publisher')
    publish_static_goal()
