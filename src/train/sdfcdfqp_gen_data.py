#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# 🟢 引入外部多场景环境池大闸
from env import get_env_pool, convert_to_jax_tensors

# =========================================================================
# 1. 256线激光雷达物理射线模拟算子（增强支持多圆散射）
# =========================================================================
@partial(jit, static_argnums=(3, 4))
def simulate_parametric_lidar_256(pos_ego_world, theta_ego, meta_obs_tensors, num_rays=256, max_range=3.0):
    angles = jnp.linspace(-jnp.pi, jnp.pi, num_rays, endpoint=False) + theta_ego
    ray_dirs = jnp.column_stack([jnp.cos(angles), jnp.sin(angles)]) 

    num_steps = 200
    t_steps = jnp.linspace(0.0, max_range, num_steps) 
    pts_check_world = pos_ego_world[None, None, :] + t_steps[None, :, None] * ray_dirs[:, None, :]

    rect_c, rect_ab = meta_obs_tensors['rect_c'], meta_obs_tensors['rect_ab']
    
    in_rect = ((jnp.abs(pts_check_world[..., 0] - rect_c[0]) / rect_ab[0])**4 + 
               (jnp.abs(pts_check_world[..., 1] - rect_c[1]) / rect_ab[1])**4) <= 1.0
               
    # 动态支持 1-5 号圆形物体的碰撞散射检测
    in_circles = jnp.zeros(in_rect.shape, dtype=jnp.bool_)
    for i in range(1, 6):
        c_c = meta_obs_tensors.get(f'c{i}_c', jnp.array([99.0, 99.0]))
        c_r = meta_obs_tensors.get(f'c{i}_r', 1e-3)
        in_c = jnp.sum((pts_check_world - c_c[None, None, :])**2, axis=-1) <= c_r**2
        in_circles = in_circles | in_c

    hit_mask = in_rect | in_circles 

    t_hit = jnp.where(hit_mask, t_steps[None, :], max_range)
    min_t = jnp.min(t_hit, axis=-1) 

    valid_hit_mask = min_t < (max_range - 1e-3)
    
    local_hit_x = min_t * jnp.cos(jnp.linspace(-jnp.pi, jnp.pi, num_rays, endpoint=False))
    local_hit_y = min_t * jnp.sin(jnp.linspace(-jnp.pi, jnp.pi, num_rays, endpoint=False))
    local_pc_all = jnp.column_stack([local_hit_x, local_hit_y])

    return local_pc_all, valid_hit_mask

# =========================================================================
# 2. 控制密度场 (CDF) 核心算子
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
def get_local_sdf_rho_and_psi(pos_p_local, target_local, local_pc_valid):
    r_ego = 0.51 + 0.15
    sense_range = 3.0  
    
    dists = jnp.sqrt(jnp.sum((local_pc_valid - pos_p_local)**2, axis=1))
    min_dist = jnp.min(dists)
    
    c_val = min_dist - r_ego          
    b_val = min_dist - sense_range    
    psi_curr = smooth_bump(c_val, b_val)
    
    V_x = jnp.sum((pos_p_local - target_local)**2)
    alpha = 0.5
    rho = psi_curr / (V_x ** alpha + 1e-6)
    return rho, psi_curr

# =========================================================================
# 3. 6维参数化纯局部凸优化规划器
# =========================================================================
class LocalSdfCdfPlanner:
    @partial(jit, static_argnums=(0,))
    def _solve_qp_core_jit(self, ego_p_local, u_nom_local, local_pc_valid, target_local):
        epsilon = 0.05
        inv_eps = 1.0 / epsilon

        def density_wrapper_local(p):
            rho, _ = get_local_sdf_rho_and_psi(p, target_local, local_pc_valid)
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
        
        rho_curr, psi_curr = get_local_sdf_rho_and_psi(ego_p_local, target_local, local_pc_valid)
        rho_z1 = density_wrapper_local(z1_pos)
        rho_z2 = density_wrapper_local(z2_pos)

        dim_total = 6
        lambda_smooth = 25  # 维持你测试最好的参数 1
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
        
        from jaxproxqp.jaxproxqp import JaxProxQP
        qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
        settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, settings)
        sol_raw = solver.solve().x 
        
        return sol_raw, C_mat, b_vec

    def solve_agent_qp_local(self, ego_p_local, u_nom_local, local_pc_valid, target_local):
        sol_raw_j, C_mat_j, b_vec_j = self._solve_qp_core_jit(ego_p_local, u_nom_local, local_pc_valid, target_local)
        G_extracted = np.array(C_mat_j, dtype=np.float32).reshape(1, 6)
        h_extracted = np.array(b_vec_j, dtype=np.float32).reshape(1)
        return sol_raw_j, G_extracted, h_extracted

# =========================================================================
# 主批量场景迭代推演控制大闸
# =========================================================================
if __name__ == '__main__':
    R_EGO = 0.31  
    L = 0.6          
    dt = 0.05          
    total_steps = 2500 
    
    # 创建图片保存根目录
    img_dir = "dataset_png"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        
    planner = LocalSdfCdfPlanner()
    env_pool = get_env_pool()
    
    # 建立全局大合拢缓存区
    global_dataset_buffer = []

    print(f"============================================================")
    print(f"🚀 启动全场景泛化大闸！共计探测到 {len(env_pool)} 个异质多态环境...")
    print(f"============================================================\n")

    for env_id, env_cfg in enumerate(env_pool):
        print(f"⏳ 正在拉起 [场景 {env_id}/9] 运动学求解链条...")
        
        my_target = env_cfg['target']
        meta_obstacles = env_cfg['meta_obstacles']
        # 转换为JAX高速静态张量
        obs_tensors = convert_to_jax_tensors(meta_obstacles)
        
        ego_state = jnp.array([0.0, 0.0, 0.0]) # 起点严格锁定在 [0,0,0]
        episode_buffer = []
        is_episode_safe = True

        history_x = []
        history_y = []
        last_frame_lidar_world = None  

        last_executed_ux = 0.0
        last_executed_uy = 0.0
        
        for step in range(total_steps):
            x, y, theta = float(ego_state[0]), float(ego_state[1]), float(ego_state[2])
            history_x.append(x)
            history_y.append(y)

            # --- 2. 🟢【核心修正：散射256线物理激光雷达】 ---
            # 先发射雷达射线，因为我们要用雷达点阵来做 100% 精确的超椭圆硬碰撞审计
            pos_ego_world = jnp.array([x, y])
            local_pc_all, valid_hit_mask = simulate_parametric_lidar_256(
                pos_ego_world, theta, obs_tensors, num_rays=256, max_range=3.0
            )
            
            valid_mask_np = np.array(valid_hit_mask)
            current_pc_local = np.array(local_pc_all)[valid_mask_np] 

            # --- 1. 🟢【核心修正：硬碰撞安全审计大闸】 ---
            # 拒绝代数错配！如果雷达探测到的障碍物距离小于等于车体刚体半径，判定为碰撞
            if current_pc_local.shape[0] > 0:
                # 计算局部载体系下所有击中点到车中心的欧氏距离
                distances_to_ego = np.linalg.norm(current_pc_local, axis=1)
                if np.min(distances_to_ego) <= (R_EGO + 1e-3):
                    is_episode_safe = False
                    
            if not is_episode_safe:
                print(f"    ❌ [碰撞熔断] 场景 {env_id} 突发超椭圆边际挂蹭，本回合轨迹作废！")
                break

            if current_pc_local.shape[0] > 0:
                rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                last_frame_lidar_world = (rot_mat @ current_pc_local.T).T + np.array([x, y])
            else:
                last_frame_lidar_world = None

            is_empty = (current_pc_local.shape[0] == 0)

            # --- 3. 局部系相对导航目标结算 ---
            dx_tg = float(my_target[0]) - x; dy_tg = float(my_target[1]) - y
            target_local_x = dx_tg * np.cos(theta) + dy_tg * np.sin(theta)
            target_local_y = -dx_tg * np.sin(theta) + dy_tg * np.cos(theta)
            target_local = jnp.array([target_local_x, target_local_y])

            ego_p_local = jnp.array([L, 0.0])
            target_vector_local = target_local - ego_p_local
            dist_local = jnp.linalg.norm(target_vector_local)
            u_nom_local = jnp.where(dist_local > 0.1, 1.2 * target_vector_local / (dist_local + 1e-6), jnp.zeros(2))

            # --- 4. 门控过滤与松弛数据落盘 ---
            if is_empty:
                v_qp = float(u_nom_local[0]); omega_qp = float(u_nom_local[1] / L)
                last_executed_ux = u_nom_local[0]
                last_executed_uy = u_nom_local[1]
            else:
                current_pc_local_j = jnp.array(current_pc_local)
                sol_6d_raw, G_extracted, h_extracted = planner.solve_agent_qp_local(
                    ego_p_local, u_nom_local, current_pc_local_j, target_local
                )
                v_qp = float(sol_6d_raw[0]); omega_qp = float(sol_6d_raw[1])/L

                control_residual = np.linalg.norm(np.array(sol_6d_raw[:2]) - np.array(u_nom_local))
                should_record_frame = (control_residual > 0.05) or (np.random.rand() < 0.40)

                if should_record_frame:
                    X_frame = np.array([target_local_x, target_local_y, last_executed_ux, last_executed_uy], dtype=np.float32)
                    Y_frame = np.concatenate([
                        [float(sol_6d_raw[0])],
                        [float(sol_6d_raw[1])],   
                        G_extracted.flatten(), 
                        h_extracted.flatten()  
                    ]).astype(np.float32)
                    
                    episode_buffer.append({
                        'X': X_frame, 'Y': Y_frame, 'z': np.array(current_pc_local, dtype=np.float32)
                    })
                last_executed_ux = sol_6d_raw[0]
                last_executed_uy = sol_6d_raw[1]
            
            # --- 5. 单轴前推 ---
            new_x = x + v_qp * np.cos(theta) * dt
            new_y = y + v_qp * np.sin(theta) * dt
            new_theta = theta + omega_qp * dt
            ego_state = jnp.array([new_x, new_y, new_theta])

            if np.hypot(x - my_target[0], y - my_target[1]) < 0.4:
                print(f"    🎯 [胜利会师] 小车成功抵达场景 {env_id} 的终点刹车区！")
                break

        # --- 6. 场景级独立图纸固化及数据收拢 ---
        if is_episode_safe and len(episode_buffer) > 0:
            global_dataset_buffer.extend(episode_buffer)
            print(f"    📥 成功收拢场景 {env_id} 的有效专家数据: {len(episode_buffer)} 帧")

            # fig, ax = plt.subplots(figsize=(12, 6))
            
            # # 🟢【核心修正：Matplotlib 场景障碍物参数化圆角渲染】
            # angles_draw = np.linspace(0, 2*np.pi, 200)
            # for obs in meta_obstacles:
            #     if obs['type'] == 'rect':
            #         # 像素级还原 n=4 的超椭圆精细圆角轮廓
            #         cos_t = np.cos(angles_draw)
            #         sin_t = np.sin(angles_draw)
            #         ex = obs['center'][0] + obs['a'] * np.sign(cos_t) * np.sqrt(np.abs(cos_t))
            #         ey = obs['center'][1] + obs['b'] * np.sign(sin_t) * np.sqrt(np.abs(sin_t))
            #         ax.fill(ex, ey, facecolor='tomato', edgecolor='darkred', alpha=0.8, zorder=1)
            #     elif obs['type'] == 'circle':
            #         circle = Circle(
            #             (obs['center'][0], obs['center'][1]), obs['r'], 
            #             facecolor='coral', edgecolor='darkred', alpha=0.8, zorder=1
            #         )
            #         ax.add_patch(circle)

            # ax.plot(history_x, history_y, color='royalblue', linewidth=2.5, label='Ego Trajectory')
            # ax.scatter(history_x[0], history_y[0], color='green', marker='o', s=150, zorder=5, label='Start')
            # ax.scatter(my_target[0], my_target[1], color='gold', marker='*', s=200, zorder=5, edgecolor='orange', label='Target')

            # ego_final_circle = Circle(
            #     (history_x[-1], history_y[-1]), R_EGO, 
            #     facecolor='none', edgecolor='blue', linestyle='--', linewidth=1.5, label='Ego Size (R_ego)'
            # )
            # ax.add_patch(ego_final_circle)

            # if last_frame_lidar_world is not None:
            #     ax.scatter(
            #         last_frame_lidar_world[:, 0], last_frame_lidar_world[:, 1], 
            #         color='lime', s=2, alpha=0.6, zorder=4, label='LiDAR Hits (Last Frame)'
            #     )

            # ax.set_title(f"Scene {env_id} - Trajectory Evaluation & LiDAR Performance", fontsize=14, fontweight='bold')
            # ax.set_xlabel("World X (m)", fontsize=12)
            # ax.set_ylabel("World Y (m)", fontsize=12)
            # ax.grid(True, linestyle=':', alpha=0.6)
            # ax.set_aspect('equal', adjustable='box')
            
            # handles, labels = ax.get_legend_handles_labels()
            # by_label = dict(zip(labels, handles))
            # ax.legend(by_label.values(), by_label.keys(), loc='upper left')

            # plt.tight_layout()
            # output_plot_path = os.path.join(img_dir, f"trajectory_scene_{env_id}.png")
            # plt.savefig(output_plot_path, dpi=200)
            # plt.close()
            # print(f"    📊 场景图像可视化归档成功 -> {output_plot_path}\n")

    # 全链路落盘
    if len(global_dataset_buffer) > 0:
        print("\n" + "="*70)
        print(f"🎉 泛化大功告成！全链路成功收拢 {len(env_pool)} 个多态环境。")
        print(f"💾 正在固化跨场景混合数据集，总计包含 {len(global_dataset_buffer)} 帧高清晰专家控制对齐资产...")
        torch.save(global_dataset_buffer, "dataset_degenerate_test_10.pt")
        print("✅ 固化泛化训练文件已全面闭环落盘：dataset_degenerate_test_10.pt")
        print("="*70 + "\n")