#!/usr/bin/env python3
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
jax.config.update("jax_enable_x64", True)  # 启用双精度
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse

@jit
def calculate_ellipse_intersections(ellipse_params, ray_start, ray_end):
    """
    修正后的椭圆-射线交点计算（正确的坐标变换）
    """
    h, k, a, b, theta = ellipse_params
    x0, y0 = ray_start
    dx = ray_end[0] - x0
    dy = ray_end[1] - y0

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    
    # 正确的坐标变换步骤
    # 1. 平移坐标系到椭圆中心
    translated_x = x0 - h
    translated_y = y0 - k
    
    # 2. 旋转坐标系对齐椭圆主轴
    rotated_x = translated_x * cos_theta + translated_y * sin_theta
    rotated_y = -translated_x * sin_theta + translated_y * cos_theta
    rotated_dx = dx * cos_theta + dy * sin_theta
    rotated_dy = -dx * sin_theta + dy * cos_theta
    
    # 3. 在标准椭圆坐标系中建立方程
    A = (rotated_dx**2) / a**2 + (rotated_dy**2) / b**2
    B = 2 * (rotated_x * rotated_dx) / a**2 + 2 * (rotated_y * rotated_dy) / b**2
    C = (rotated_x**2) / a**2 + (rotated_y**2) / b**2 - 1

    # 解方程并筛选有效解
    discriminant = B**2 - 4*A*C
    sqrt_disc = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    
    valid_mask = (discriminant >= 0) & (A != 0)
    t1 = jnp.where(valid_mask, (-B + sqrt_disc)/(2*A + 1e-12), jnp.inf)
    t2 = jnp.where(valid_mask, (-B - sqrt_disc)/(2*A + 1e-12), jnp.inf)
    
    # 坐标逆变换回原坐标系
    def transform_back(t):
        # 计算局部坐标
        local_x = rotated_x + t * rotated_dx
        local_y = rotated_y + t * rotated_dy
        
        # 逆旋转
        world_x = h + (local_x * cos_theta - local_y * sin_theta)
        world_y = k + (local_x * sin_theta + local_y * cos_theta)
        return world_x, world_y
    
    # 筛选有效t值
    t1_valid = jnp.where((t1 >= 0) & (t1 <= 1.0), t1, jnp.inf)
    t2_valid = jnp.where((t2 >= 0) & (t2 <= 1.0), t2, jnp.inf)
    min_t = jnp.minimum(t1_valid, t2_valid)
    
    # 计算最终坐标
    x_point, y_point = transform_back(min_t)
    distance = jnp.where(jnp.isfinite(min_t), 
                        jnp.hypot(x_point - x0, y_point - y0), 
                        jnp.inf)
    
    return jnp.array([distance, x_point, y_point])

@jit
def batch_intersection_check(obstacles, robot_state, num_rays=180, max_range=3):
    """
    JAX加速的批量交点检测 (返回每条射线最近交点)
    """
    # 生成射线参数矩阵
    angles = jnp.linspace(0, 2*jnp.pi, num_rays)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    rays_start = jnp.tile(jnp.array(robot_state), (num_rays, 1))
    rays_end = rays_start + directions * max_range

    # 处理每个椭圆对射线的交点
    def process_one_ellipse(ellipse):
        return vmap(calculate_ellipse_intersections, (None, 0, 0))(ellipse, rays_start, rays_end)
    
    # 获取所有椭圆的交点 (M, N, 3)
    all_intersections = vmap(process_one_ellipse)(obstacles)
    
    # 找到每条射线的最短距离交点
    min_distances = jnp.min(all_intersections[..., 0], axis=0)
    min_indices = jnp.argmin(all_intersections[..., 0], axis=0)
    
    # 收集最近交点坐标
    rays_range = jnp.arange(num_rays)
    closest_points = all_intersections[min_indices, rays_range, 1:]
    
    # 组合结果 [distance, x, y]
    result = jnp.column_stack([min_distances, closest_points])
    return jnp.where(result[:, 0:1] < jnp.inf, result, jnp.full_like(result, jnp.inf))

def post_process(results):
    """
    后处理：过滤无效点并按距离排序
    """
    valid_mask = jnp.isfinite(results[:, 0])
    valid_points = results[valid_mask]
    sorted_indices = jnp.argsort(valid_points[:, 0], stable=True)
    return jnp.array(valid_points[sorted_indices, 1:])

def visualize_scene(obstacles, agent_pos, intersection_points, max_range=3):
    """
    可视化场景：智能体、障碍物和相交点
    :param obstacles: 椭圆参数列表 [(h, k, a, b, theta_rad)]
    :param agent_pos: 智能体位置 (x, y)
    :param intersection_points: 相交点坐标数组 (N,2)
    :param max_range: 雷达探测半径
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制智能体位置
    ax.scatter(agent_pos[0], agent_pos[1], c='red', s=100, label='Agent', zorder=5)
    
    # 绘制雷达范围
    radar_circle = plt.Circle(agent_pos, max_range, 
                             color='gray', alpha=0.2, label='Radar Range')
    ax.add_patch(radar_circle)
    
    # 绘制障碍物椭圆
    for i, (h, k, a, b, theta_rad) in enumerate(obstacles):
        # 转换为matplotlib参数：角度单位为度，宽度为2a，高度为2b
        ellipse = Ellipse((h, k), 
                          width=2*a, 
                          height=2*b,
                          angle=np.degrees(theta_rad),
                          edgecolor='blue',
                          facecolor='none',
                          linestyle='--',
                          label=f'Obstacle {i+1}')
        ax.add_patch(ellipse)
    
    # 绘制相交点
    if len(intersection_points) > 0:
        ax.scatter(intersection_points[:,0], intersection_points[:,1],
                  c='green', s=50, marker='x', label='Intersections')
    
    # 设置坐标轴
    ax.set_aspect('equal')
    ax.set_xlim(agent_pos[0]-max_range*1.5, agent_pos[0]+max_range*1.5)
    ax.set_ylim(agent_pos[1]-max_range*1.5, agent_pos[1]+max_range*1.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.title("Obstacle Intersection Visualization")
    plt.show()

# 测试验证
if __name__ == "__main__":
    # 测试案例1：单个椭圆
    # obstacles = jnp.array([[3.0, 0.0, 1.0, 1.0, 0.0]])
    # robot_state = jnp.array([0.0, 0.0])
    # result = batch_intersection_check(obstacles, robot_state)
    
    # print(f"案例1结果：{final_result}")

    # # 测试案例2：无交点情况
    # obstacles = jnp.array([[10.0, 10.0, 1.0, 1.0, 0.0]])
    # robot_state = jnp.array([0.0, 0.0])
    # result = batch_intersection_check(obstacles, robot_state)
    
    # print(f"案例2结果：{result}")

    # # 测试案例3：多椭圆多交点
    obstacles = jnp.array([
        [3.0, 0.0, 1.0, 1.4, 3],
        [0.0, 3.0, 1.0, 1.0, 0.0]
    ])
    robot_state = jnp.array([0.0, 0.0])
    result = batch_intersection_check(obstacles, robot_state)
    final_result = post_process(result)
    print(final_result.shape)
    # 转换结果格式
    np_obstacles = np.array(obstacles)
    visualize_scene(np_obstacles, robot_state, final_result)
    # print(f"案例3结果：{result}")

    
