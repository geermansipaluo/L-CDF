#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include "moving_cylinder.hpp"

int main(int argc, char** argv) {
    ros::init(argc, argv, "movetest_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    // 声明可视化 Marker 发布器，以便在 Rviz 中实时肉眼观察往返效果
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::MarkerArray>("/visual_obstacles", 10);

    // 实例化两个完全解耦的动态柱体
    MovingCylinder obs1;
    MovingCylinder obs2;

    // 🟢 载入障碍物 1 的物理常数（让它在 X=4.5m 处横向在 Y=-2.5m 到 Y=2.5m 之间疯狂往返移动拦截小车）
    obs1.init(4.5, -2.0, 0.0,  0.0, 0.4,  4.0, 5.0,  -2.5, 2.5);
    
    // 🟢 载入障碍物 2 的物理常数（让它在 X=8.0m 处斜向在 Y=-1.5m 到 Y=1.5m 之间高频摆动）
    obs2.init(8.0, 1.5, 0.0,   0.1, 0.5,  7.5, 8.5,  -1.5, 1.5);

    double frequency = 30.0; // 30Hz 高频物理引擎刷新率
    ros::Rate rate(frequency);
    double dt = 1.0 / frequency;

    ROS_INFO("🔥【时空多边往复移动障碍物集群】成功上线运行！测试大闸开启...");

    while (ros::ok()) {
        // 更新两台障碍物的物理运动状态
        obs1.update(dt);
        obs2.update(dt);

        // 组装 Rviz 空间高维可视化 Marker 消息
        visualization_msgs::MarkerArray marker_array;
        
        // 组装障碍物 1 的圆柱体特征
        visualization_msgs::Marker m1;
        m1.header.frame_id = "odom"; // 确保与你自车的坐标系一致
        m1.header.stamp = ros::Time::now();
        m1.ns = "dynamic_obs"; m1.id = 1;
        m1.type = visualization_msgs::Marker::CYLINDER;
        m1.action = visualization_msgs::Marker::ADD;
        m1.pose = obs1.getPose();
        m1.scale.x = 0.6; m1.scale.y = 0.6; m1.scale.z = 1.0; // 直径 0.6m，高 1.0m 的圆柱
        m1.color.r = 1.0; m1.color.g = 0.0; m1.color.b = 0.0; m1.color.a = 0.8; // 亮红色
        marker_array.markers.push_back(m1);

        // 组装障碍物 2 的圆柱体特征
        visualization_msgs::Marker m2 = m1;
        m2.id = 2;
        m2.pose = obs2.getPose();
        m2.scale.x = 0.5; m2.scale.y = 0.5; m2.scale.z = 1.0;
        m2.color.r = 1.0; m2.color.g = 0.5; m2.color.b = 0.0; // 亮橙色
        marker_array.markers.push_back(m2);

        marker_pub.publish(marker_array);

        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}