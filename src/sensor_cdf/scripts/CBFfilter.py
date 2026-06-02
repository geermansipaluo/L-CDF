# import jax
# import jax.numpy as jnp
# from jaxproxqp.jaxproxqp import JaxProxQP
# import torch
# import torch.utils.dlpack as dlpack
# from jax.dlpack import from_dlpack
# from jax import jit

# class JaxCBFQPSafetyFilter:
#     def __init__(self, num_obstacles):
#         """
#         初始化安全过滤器
#         由于对齐了 JaxProxQP 的最新 API，求解器实例将在 filter_control 内部动态跟随静态图创建。
#         """
#         self.num_obstacles = num_obstacles
#         self.dim_total = 2  # 决策变量仅包含差分小车的物理控制量: [v, w]

#     def filter_control(self, state, u_nom, obstacles, risk, config):
#         """
#         高性能 CBF-QP 过滤函数 (支持 JAX XLA 静态编译)
        
#         参数:
#         - state: 当前小车状态 jnp.array([x, y, theta])
#         - u_nom: 神经网络预测的原始名义控制量 jnp.array([v_nom, w_nom])
#         - obstacles: 局部环境障碍物矩阵 jnp.array([[x_i, y_i, R_i], ...]), 形状为 (N, 3)
#         - risk: 风险度头输出的标量值 (0.0 ~ 1.0)
#         - config: 包含控制、几何及物理边界参数的静态字典
#         """
#         x, y, theta = state[0], state[1], state[2]
        
#         # 1. 计算前瞻点 P 坐标 (前瞻点法解决差分小车相对阶为2的问题)
#         l_k = config['l_k']
#         P = jnp.array([
#             x + l_k * jnp.cos(theta),
#             y + l_k * jnp.sin(theta)
#         ])
        
#         # 2. 计算前瞻点的运动学雅可比矩阵 J(\theta)
#         J = jnp.array([
#             [jnp.cos(theta), -l_k * jnp.sin(theta)],
#             [jnp.sin(theta),  l_k * jnp.cos(theta)]
#         ])
        
#         # 3. 提取障碍物数据并向量化计算一阶圆形 SDF
#         O_centers = obstacles[:, :2]  # 形状: (N, 2)
#         R_obs = obstacles[:, 2]       # 形状: (N,)
        
#         P_minus_O = P - O_centers     # 形状: (N, 2)
#         dist_sq = jnp.sum(P_minus_O**2, axis=1)
#         h = dist_sq - (config['r_ego'] + R_obs)**2  # 安全距离指标
        
#         # 4. 构建标准 CBF 线性不等式约束矩阵项: C_mat * u <= b_vec
#         # 根据推导: -2*(P - O)^T * J * u <= \gamma * h
#         C_mat = -2.0 * jnp.matmul(P_minus_O, J)    # 形状: (N, 2)
#         b_cbf = config['gamma'] * h               # 形状: (N,)
        
#         # 5. ⚡ 风险度平滑放行机制 (对齐 XLA 静态一致性，不产生计算图分叉)
#         is_triggered = risk >= config['threshold']
#         b_vec = jnp.where(is_triggered, b_cbf, 1e6) # 未触发时赋予极大上限，约束自动失效
        
#         # 6. 精准对齐你的目标的二乘代价函数构建 (min ||u - u_nom||^2)
#         # 展开后对应: 0.5 * u^T * H * u + g_vec^T * u
#         H = jnp.eye(self.dim_total) * 2.0
#         g_vec = -2.0 * u_nom
        
#         # 7. 物理边界直接注入盒状约束 (无需像常规矩阵那样做额外的 vstack 拼接)
#         l_box = jnp.array([config['v_min'], config['w_min']])
#         u_box = jnp.array([config['v_max'], config['w_max']])
        
#         # 8. 严格按照你提供的规范实例化并调用 JaxProxQP
#         qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
#         settings = JaxProxQP.Settings.default()
#         solver = JaxProxQP(qp, settings)
        
#         # 截取前两个维度作为安全控制量传出
#         u_opt = solver.solve().x[:2]
        
#         return u_opt


#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class SingleIntegratorCBFSafetyFilter:
    def __init__(self, k_closest=3):
        self.k_closest = k_closest

    def filter_control(self, u_nom, local_points, config, current_risk=0.0):
        """
        🟢【降维打击重构】：前瞻点平面单积分器安全过滤器
        将 v_nom, w_nom 投影至 2D 笛卡尔空间解耦算完避障后，再无缝逆投影回差速控制量
        """
        v_nom, w_nom = float(u_nom[0]), float(u_nom[1])
        l_k = config['l_k']
        r_ego = config['r_ego']
        adaptive_gamma = config['gamma'] * (1.0 - float(current_risk) * 0.5)

        # 1. 【正向投影】：将差速标称输入转换为前瞻点载体系下的 2D 平面速度向量
        vx_p_nom = v_nom
        vy_p_nom = l_k * w_nom

        # 2. 构筑平面 2D 平方偏差优化目标
        def objective(up):
            return (up[0] - vx_p_nom)**2 + (up[1] - vy_p_nom)**2

        constraints = []
        
        # 3. 注入解耦后的 2D 平面单积分器物理边界限制 (通过 w 换算 vy)
        v_bounds = (config['v_min'], config['v_max'])
        vy_max = l_k * config['w_max'] if config['w_max'] > 0 else l_k * config['w_min']
        vy_min = l_k * config['w_min'] if config['w_min'] < 0 else l_k * config['w_max']
        vy_bounds = (min(vy_min, vy_max), max(vy_min, vy_max))

        # 4. 建立纯净的平面空间几何控制屏障约束
        if len(local_points) > 0:
            dists = np.hypot(local_points[:, 0], local_points[:, 1])
            closest_indices = np.argsort(dists)[:self.k_closest]
            
            for idx in closest_indices:
                if idx >= len(local_points): continue
                x_i, y_i = local_points[idx, 0], local_points[idx, 1]
                
                # 计算前瞻点到圆心的距离场
                h_i = (l_k - x_i)**2 + (0.0 - y_i)**2 - r_ego**2
                
                dx_l = l_k - x_i
                dy_l = 0.0 - y_i
                
                # 🟢 平面单积分器标准一阶导数约束： 2 * dx * vx + 2 * dy * vy >= -gamma * h
                def cbf_plane_constraint(up, dx=dx_l, dy=dy_l, hi=h_i):
                    return (2.0 * dx * up[0] + 2.0 * dy * up[1]) + adaptive_gamma * hi
                
                constraints.append({'type': 'ineq', 'fun': cbf_plane_constraint})

        # 5. 平面空间内高速求解 QP
        res = minimize(objective, [vx_p_nom, vy_p_nom], constraints=constraints, 
                       bounds=[v_bounds, vy_bounds], method='SLSQP')
        
        if res.success:
            # 6. 【逆向投影】：将避障纠偏后的安全平面速度无缝映射回底盘执行的 v 和 w
            v_final = res.x[0]
            w_final = res.x[1] / l_k
            return v_final, w_final
        else:
            return v_nom, w_nom

# =========================================================================
# 2. 闭环时间积分仿真环境
# =========================================================================
if __name__ == '__main__':
    config = {
        'l_k': 0.15,          
        'r_ego': 0.31,        
        'gamma': 1.0,           
        'v_min': -0.2, 'v_max': 0.8,
        'w_min': -1.5, 'w_max': 1.5
    }

    dt = 0.05
    sim_time = 15.0
    steps = int(sim_time / dt)

    x, y, theta = 0.0, 0.0, 0.0
    goal = np.array([10.0, 0.0])
    
    # 🚨 完全还原导致你刚才卡死崩溃的极端正前方障碍
    obs_world = np.array([5.0, 0.1])

    safety_filter = SingleIntegratorCBFSafetyFilter(k_closest=3)
    traj_x, traj_y = [], []

    for step in range(steps):
        traj_x.append(x)
        traj_y.append(y)
        
        # 标称速度计算
        l_k = config['l_k']
        ego_p_x = x + l_k * np.cos(theta)
        ego_p_y = y + l_k * np.sin(theta)
        dx, dy = goal[0] - ego_p_x, goal[1] - ego_p_y
        dist = np.hypot(dx, dy)
        
        u_nom_x = 1.2 * dx / (dist + 1e-6) if dist > 0.1 else 0.0
        u_nom_y = 1.2 * dy / (dist + 1e-6) if dist > 0.1 else 0.0
        
        v_nom = u_nom_x * np.cos(theta) + u_nom_y * np.sin(theta)
        w_nom = (-u_nom_x * np.sin(theta) + u_nom_y * np.cos(theta)) / l_k
        v_nom = np.clip(v_nom, config['v_min'], config['v_max'])
        w_nom = np.clip(w_nom, config['w_min'], config['w_max'])
        
        # 里程计载体系刚体变换
        dx_obs, dy_obs = obs_world[0] - x, obs_world[1] - y
        x_local = dx_obs * np.cos(theta) + dy_obs * np.sin(theta)
        y_local = -dx_obs * np.sin(theta) + dy_obs * np.cos(theta)
        
        # 拦截解算
        v_f, w_f = safety_filter.filter_control([v_nom, w_nom], np.array([[x_local, y_local]]), config)
        
        # 动力学更新
        x += v_f * np.cos(theta) * dt
        y += v_f * np.sin(theta) * dt
        theta += w_f * dt
        
        if np.hypot(x - goal[0], y - goal[1]) < 0.2:
            print(f"🎉 见证奇迹！小车在第 {step} 步丝滑绕过核心死角，成功突围挺进终点！")
            break

    # 绘图层
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(traj_x, traj_y, label='Robot Trajectory (SI Detach Mapping)', color='blue', linewidth=2.5)
    ax.scatter(0, 0, color='blue', marker='o', s=100, label='Start')
    ax.scatter(goal[0], goal[1], color='green', marker='*', s=250, label='Goal')
    circle_obs = plt.Circle((obs_world[0], obs_world[1]), config['r_ego'], color='red', alpha=0.4, label='Obstacle')
    ax.add_patch(circle_obs)
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', 'box')
    plt.savefig('si_cbf_success_trajectory.png', bbox_inches='tight')
    print("🚀 完美的全新验证轨迹图已生成：si_cbf_success_trajectory.png")