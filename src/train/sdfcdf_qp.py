#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from jaxproxqp.jaxproxqp import JaxProxQP

# 🟢 引入你之前编写的场景池大闸
from env import get_env_pool, convert_to_jax_tensors

# =========================================================================
# 1. 基于局部系变长点云最紧邻 SDF 的控制密度函数 (CDF)
# =========================================================================
@jit
def smooth_bump(c, b):
    denom = c - b
    denom = jnp.where(jnp.abs(denom) < 1e-6, 1e-6, denom)
    m_k = c / denom
    safe_mk = jnp.clip(m_k, 1e-5, 1.0 - 1e-5) 
    exp1 = jnp.exp(-1.0 / safe_mk)
    exp2 = jnp.exp(-1.0 / (1.0 - safe_mk))
    bump = exp1 / (exp1 + exp2)
    return jnp.where(c <= 0, 0.0, jnp.where(b >= 0, 1.0, bump))

@jit
def get_local_sdf_rho_and_psi(pos_p_local, target_local, local_pc):
    r_ego = 0.51 + 0.15
    sense_range = 3.0  
    num_pts = local_pc.shape[0]
    
    dists = jnp.sqrt(jnp.sum((local_pc - pos_p_local)**2, axis=1))
    min_dist = jnp.min(dists)
    
    c_val = min_dist - r_ego          
    b_val = min_dist - sense_range    
    
    psi_curr = smooth_bump(c_val, b_val)
    psi_curr = jnp.where(num_pts == 0, 1.0, psi_curr)
    
    V_x = jnp.sum((pos_p_local - target_local)**2)
    alpha = 0.5
    rho = psi_curr / (V_x ** alpha + 1e-6)
    return rho, psi_curr

# =========================================================================
# 2. 点云驱动型纯局部凸优化安全层求解类
# =========================================================================
class LocalSdfCdfPlanner:
    @partial(jit, static_argnums=(0,))
    def solve_agent_qp_local(self, ego_p_local, u_nom_local, local_pc, target_local):
        epsilon = 0.05
        inv_eps = 1.0 / epsilon

        def density_wrapper_local(p):
            rho, _ = get_local_sdf_rho_and_psi(p, target_local, local_pc)
            return rho
            
        grad_self = jax.grad(density_wrapper_local)(ego_p_local)
        drift_term = 0.0 

        norm_nom = jnp.linalg.norm(u_nom_local)
        dir_nom = jnp.where(norm_nom > 1e-5, u_nom_local / (norm_nom + 1e-8), jnp.array([1.0, 0.0]))
        z1_pos = ego_p_local + epsilon * dir_nom

        neg_grad = -grad_self
        norm_grad = jnp.linalg.norm(neg_grad)
        dir_safe = jnp.where(norm_grad > 1e-5, neg_grad / (norm_grad + 1e-8), jnp.array([0.0, 1.0])) 
        z2_pos = ego_p_local + epsilon * dir_safe

        v1 = dir_nom; v2_raw = dir_safe
        det_v = v1[0] * v2_raw[1] - v1[1] * v2_raw[0]
        is_independent = jnp.abs(det_v) > 1e-2 
        v2_ortho = jnp.array([-v1[1], v1[0]])
        v2 = jnp.where(is_independent, v2_raw, v2_ortho)

        V_mat = jnp.column_stack([v1, v2])
        W_mat = jnp.linalg.inv(V_mat)
        w1 = W_mat[0, :]; w2 = W_mat[1, :]
        
        rho_curr, psi_curr = get_local_sdf_rho_and_psi(ego_p_local, target_local, local_pc)
        rho_z1 = density_wrapper_local(z1_pos)
        rho_z2 = density_wrapper_local(z2_pos)

        dim_total = 6
        lambda_smooth = 25
        H = jnp.zeros((dim_total, dim_total))
        val_uu = 2.0 * (1.0 + 2.0 * lambda_smooth)
        val_zz = 2.0 * lambda_smooth
        H = H.at[0, 0].set(val_uu); H = H.at[1, 1].set(val_uu)
        H = H.at[2, 2].set(val_zz); H = H.at[3, 3].set(val_zz)
        H = H.at[4, 4].set(val_zz); H = H.at[5, 5].set(val_zz)
        
        val_uz = -2.0 * lambda_smooth
        H = H.at[0, 2].set(val_uz); H = H.at[2, 0].set(val_uz)
        H = H.at[1, 3].set(val_uz); H = H.at[3, 1].set(val_uz)
        H = H.at[0, 4].set(val_uz); H = H.at[4, 0].set(val_uz)
        H = H.at[1, 5].set(val_uz); H = H.at[5, 1].set(val_uz)

        g_vec = jnp.zeros(dim_total).at[0:2].set(-2.0 * u_nom_local)
        
        sum_w = w1 + w2
        coeff_u = (rho_curr * inv_eps) * sum_w
        coeff_z1 = -(rho_z1 * inv_eps) * w1
        coeff_z2 = -(rho_z2 * inv_eps) * w2

        C_mat = jnp.zeros((1, dim_total)).at[0, 0:2].set(coeff_u).at[0, 2:4].set(coeff_z1).at[0, 4:6].set(coeff_z2)
        b_vec = jnp.zeros(1).at[0].set(drift_term - rho_curr)
        
        limit = 1.2
        l_box = jnp.array([-limit] * 6); u_box = jnp.array([limit] * 6)
        
        qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
        settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, settings)
        u_opt = solver.solve().x[:2]
        
        return u_opt, psi_curr

# =========================================================================
# 3. 闭环时间积分仿真推演环境
# =========================================================================
if __name__ == '__main__':
    # 🟢 核心改动 1：拉起场景池，并让用户手动选择需要调试的环境 ID
    env_pool = get_env_pool()
    print(f"📊 当前环境池总计包含 {len(env_pool)} 个场景 (0 ~ {len(env_pool)-1})。")
    ENV_ID = 5
    print(f"\n⏳ 正在拉起 [场景 {ENV_ID}] 的纯局部载体系时空博弈仿真...")
    env_cfg = env_pool[ENV_ID]
    my_target = jnp.array(env_cfg['target'])
    meta_obstacles = env_cfg['meta_obstacles']
    
    R_EGO = 0.31
    ego_state = np.array([0.0, 0.0, 0.0]) # [x, y, theta]
    dt = 0.05
    total_steps = 2500
    L = 0.6

    planner = LocalSdfCdfPlanner()
    traj_x, traj_y, traj_residuals = [], [], []
    is_collided = False

    # 🟢 核心改动 2：从障碍物描述中动态释放出稠密密集的表面点云（用于喂给 CDF场）
    obs_pc_list = []
    t_edge = np.linspace(-1.0, 1.0, 150)
    angles = np.linspace(0, 2 * np.pi, 150)
    
    for obs in meta_obstacles:
        if obs['type'] == 'rect':
            c_x, c_y = obs['center'][0], obs['center'][1]
            a, b = obs['a'], obs['b']
            
            # 🟢 完美对齐：利用超椭圆参数方程，将 angles (0 ~ 2pi) 映射到 n=4 的圆角矩形表面
            # 参数方程：x = a * sgn(cos) * |cos|^(2/n), y = b * sgn(sin) * |sin|^(2/n)
            # 当 n=4 时，指数为 2/4 = 0.5
            cos_t = np.cos(angles)
            sin_t = np.sin(angles)
            
            ellip_x = c_x + a * np.sign(cos_t) * np.sqrt(np.abs(cos_t))
            ellip_y = c_y + b * np.sign(sin_t) * np.sqrt(np.abs(sin_t))
            
            rect_pc = np.column_stack([ellip_x, ellip_y])
            obs_pc_list.append(rect_pc)
        elif obs['type'] == 'circle':
            c_x, c_y = obs['center'][0], obs['center'][1]
            r = obs['r']
            circle_pts = np.column_stack([c_x + r * np.cos(angles), c_y + r * np.sin(angles)])
            obs_pc_list.append(circle_pts)
            
    obs_pc_world = jnp.vstack(obs_pc_list)

    # 主循环推演
    for step in range(total_steps):
        x, y, theta = float(ego_state[0]), float(ego_state[1]), float(ego_state[2])
        traj_x.append(x); traj_y.append(y)

        # 🟢 核心改动 3：动态场景级的精确物理硬碰撞审计大闸
        for obs in meta_obstacles:
            if obs['type'] == 'rect':
                closest_rect_x = np.clip(x, obs['center'][0] - obs['a'], obs['center'][0] + obs['a'])
                closest_rect_y = np.clip(y, obs['center'][1] - obs['b'], obs['center'][1] + obs['b'])
                if np.hypot(x - closest_rect_x, y - closest_rect_y) < R_EGO:
                    is_collided = True
            elif obs['type'] == 'circle':
                if np.hypot(x - obs['center'][0], y - obs['center'][1]) < (obs['r'] + R_EGO):
                    is_collided = True
                    
        if is_collided:
            print(f"💥 [碰撞熔断] 小车在场景 {ENV_ID} 的第 {step} 步发生刮蹭！已强制熔断。")
            break

        # 刚体坐标清洗投影至载体系
        dx_pc = obs_pc_world[:, 0] - x; dy_pc = obs_pc_world[:, 1] - y
        pc_local_x = dx_pc * np.cos(theta) + dy_pc * np.sin(theta)
        pc_local_y = -dx_pc * np.sin(theta) + dy_pc * np.cos(theta)
        current_pc_local = jnp.column_stack([pc_local_x, pc_local_y])

        # 目标点相对化
        dx_tg, dy_tg = float(my_target[0]) - x, float(my_target[1]) - y
        target_local_x = dx_tg * np.cos(theta) + dy_tg * np.sin(theta)
        target_local_y = -dx_tg * np.sin(theta) + dy_tg * np.cos(theta)
        target_local = jnp.array([target_local_x, target_local_y])

        ego_p_local = jnp.array([L, 0.0])
        target_vector_local = target_local - ego_p_local
        dist_local = jnp.linalg.norm(target_vector_local)
        u_nom_local = jnp.where(dist_local > 0.1, 1.0 * target_vector_local / (dist_local + 1e-6), jnp.zeros(2))

        # 前向拦截 QP 求解
        u_opt_local, _ = planner.solve_agent_qp_local(ego_p_local, u_nom_local, current_pc_local, target_local)

        residual = np.linalg.norm(np.array(u_opt_local) - np.array(u_nom_local))
        traj_residuals.append(residual)

        v = float(u_opt_local[0])         
        omega = float(u_opt_local[1] / L) 

        ego_state[0] += v * np.cos(theta) * dt
        ego_state[1] += v * np.sin(theta) * dt
        ego_state[2] += omega * dt

        if np.hypot(x - my_target[0], y - my_target[1]) < 0.43:
            print(f"🎉 场景 {ENV_ID} 顺利通关！于第 {step} 步平稳安全抵达终点刹车区！")
            break

    # =========================================================================
    # 4. Matplotlib 场景自适应高清分段渲染画图层
    # =========================================================================
    traj_x, traj_y = np.array(traj_x), np.array(traj_y)
    residuals = np.array(traj_residuals)
    
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # 🟢 核心改动 4：根据当前选定的场景配置，动态在图中还原障碍物外形
    for idx, obs in enumerate(meta_obstacles):
        if obs['type'] == 'rect':
            rect_patch = Rectangle(
                (obs['center'][0] - obs['a'], obs['center'][1] - obs['b']), 
                obs['a']*2, obs['b']*2, 
                fill=True, color='gray', alpha=0.3, hatch='//', 
                label='Static Rectangle' if idx == 0 else ""
            )
            ax.add_patch(rect_patch)
        elif obs['type'] == 'circle':
            circle_patch = Circle(
                (obs['center'][0], obs['center'][1]), obs['r'], 
                fill=True, color='dimgray', alpha=0.3, hatch='\\\\', 
                label='Static Circle' if idx == 0 else ""
            )
            ax.add_patch(circle_patch)
    
    # 动态分段渲染轨迹（红色代表避障介入纠偏，蓝色代表名义直驱）
    for i in range(len(traj_x) - 1):
        x_seg, y_seg = [traj_x[i], traj_x[i+1]], [traj_y[i], traj_y[i+1]]
        if i < len(residuals) and residuals[i] > 0.08:
            ax.plot(x_seg, y_seg, color='crimson', linewidth=3.5, zorder=3)
        else:
            ax.plot(x_seg, y_seg, color='royalblue', linewidth=2.0, zorder=2)

    # 绘制碰撞状态
    if is_collided:
        ego_crash_circle = Circle((traj_x[-1], traj_y[-1]), R_EGO, fill=False, color='red', linestyle='--', linewidth=2.0, label='Crash Outer Boundary')
        ax.add_patch(ego_crash_circle)
        ax.scatter(traj_x[-1], traj_y[-1], color='red', marker='X', s=150, zorder=5, label='Collision Point')

    # 起终点要素绘制
    ax.scatter(traj_x[0], traj_y[0], color='green', marker='o', s=120, zorder=5, label='Start (0,0)')
    ax.scatter(my_target[0], my_target[1], color='gold', marker='*', s=200, edgecolor='orange', zorder=5, label=f'Target ({my_target[0]},{my_target[1]})')

    # 细节美化
    ax.set_xlabel("World X (m)", fontsize=12)
    ax.set_ylabel("World Y (m)", fontsize=12)
    ax.set_title(f"Single-Scene Debugger (Scene ID: {ENV_ID}) - SDF-CDF-QP Validation Trajectory", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 画布显示边界自适应调节
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.0, 16.0)
    ax.set_ylim(-4.0, 4.0)
    
    # 处理图例重复去重
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # 🟢 核心改动 5：保存的文件名与场景 ID 强制绑定，方便针对性比对
    target_dir = "/home/guo/L-CDF/src/train/dataset_env"
    import os
    file_name = f'debug_trajectory_scene_{ENV_ID}.png'
    absolute_output_path = os.path.join(target_dir, file_name)
    output_fn = f'debug_trajectory_scene_{ENV_ID}.png'
    plt.tight_layout()
    plt.savefig(absolute_output_path, dpi=300)
    plt.close()