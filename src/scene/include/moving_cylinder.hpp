#ifndef MOVING_CYLINDER_HPP
#define MOVING_CYLINDER_HPP

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <cmath>

class MovingCylinder {
public:
    double x, y, z;
    double vx, vy;
    double x_min, x_max; // X轴往返运动的硬几何边界
    double y_min, y_max; // Y轴往返运动的硬几何边界
    int x_direction;     // 1 代表正向，-1 代表反向
    int y_direction;     

    MovingCylinder() : x(0), y(0), z(0), vx(0), vy(0), 
                       x_min(-10), x_max(10), y_min(-10), y_max(10),
                       x_direction(1), y_direction(1) {}

    // 初始化障碍物的位置、基础速度、以及运动往返的区间限制
    void init(double start_x, double start_y, double start_z, 
              double speed_x, double speed_y,
              double min_x, double max_x, double min_y, double max_y) {
        x = start_x; y = start_y; z = start_z;
        vx = std::abs(speed_x); // 内部锁死为绝对值大小
        vy = std::abs(speed_y);
        x_min = min_x; x_max = max_x;
        y_min = min_y; y_max = max_y;
        x_direction = 1;
        y_direction = 1;
    }

    // 🟢 核心高频往复运动演进算子
    void update(double dt) {
        // 1. 根据当前的方向状态进行非线性时空演进
        x += x_direction * vx * dt;
        y += y_direction * vy * dt;

        // 2. 🟢 X轴硬边界在线碰撞审计与动态调头
        if (x >= x_max) {
            x = x_max;
            x_direction = -1; // 触及上限，立刻反向折返
        } else if (x <= x_min) {
            x = x_min;
            x_direction = 1;  // 触及下限，立刻正向冲刺
        }

        // 3. 🟢 Y轴硬边界在线碰撞审计与动态调头
        if (y >= y_max) {
            y = y_max;
            y_direction = -1;
        } else if (y <= y_min) {
            y = y_min;
            y_direction = 1;
        }
    }

    geometry_msgs::Pose getPose() const {
        geometry_msgs::Pose pose;
        pose.position.x = x;
        pose.position.y = y;
        pose.position.z = z;
        pose.orientation.w = 1.0;
        return pose;
    }
};

#endif // MOVING_CYLINDER_HPP