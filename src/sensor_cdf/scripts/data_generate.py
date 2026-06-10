#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit
import torch
import os
from jaxproxqp.jaxproxqp import JaxProxQP

# =========================================================================
# 1. 基于车体系局部点云带符号距离场 (SDF) 的控制密度场算子
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
    """ 完全解耦大地图，只凭车体系下的离散点云 pos 求解全局单行 CDF 屏障 """
    r_ego = 0.31
    sense_range = 3.0  # 3.0m 的局部感应圆环外边际
    
    num_pts = local_pc.shape[0]
    
    # SDF高阶提炼：直接检索局部点云到前瞻点 P 的最近距离
    dists = jnp.sqrt(jnp.sum((local_pc - pos_p_local)**2, axis=1))
    min_dist = jnp.min(dists)
    
    c_val = min_dist - r_ego          # 硬防撞安全壁垒
    b_val = min_dist - sense_range    # 感知警报激活边际
    
    psi_curr = smooth_bump(c_val, b_val)
    psi_curr = jnp.where(num_pts == 0, 1.0, psi_curr)
    
    # 局部引力场：前瞻点到局部目标的平方距离
    V_x = jnp.sum((pos_p_local - target_local)**2)
    alpha = 0.5
    
    rho = psi_curr / (V_x ** alpha + 1e-6)
    return rho, psi_curr

# =========================================================================
# 2. 点云驱动型纯局部凸优化安全层求解类 (修合 Tracer 漏洞版)
# =========================================================================
class LocalSdfCdfPlanner:
    @partial(jit, static_argnums=(0,))
    def _solve_qp_core_jit(self, ego_p_local, u_nom_local, local_pc, target_local):
        """
        🟢【JAX 静态编译大核】：内部 100% 使用 jnp，杜绝任何物理内存和 Tracer 转换
        """
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

        # 组装 6 维紧凑型 ProxQP 凸优化
        dim_total = 6
        lambda_smooth = 25.0
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
        
        limit = 1.5
        l_box = jnp.array([-limit] * 6); u_box = jnp.array([limit] * 6)
        
        qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
        settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, settings)
        u_opt = solver.solve().x[:2]
        
        # 💡 全部在 JAX 的流形 Tracer 空间返回，坚决不在里面写任何 np.array()
        return u_opt, C_mat, b_vec

    def solve_agent_qp_local(self, ego_p_local, u_nom_local, local_pc, target_local):
        """
        外部解耦壳接口：负责调用上面的 JIT 核心，并在【退出编译域后】合法剥离系数
        """
        # 1. 运行 JIT 核心，吐出已经完成解算的确定性具体数据对象
        u_opt_j, C_mat_j, b_vec_j = self._solve_qp_core_jit(ego_p_local, u_nom_local, local_pc, target_local)
        
        # 2. 🟢【完美避坑】：此时数据已安全脱离编译追踪，可以极其畅快、安全地转化为通用 numpy 并重塑形状！
        G_extracted = np.array(C_mat_j, dtype=np.float32).reshape(1, 6)
        h_extracted = np.array(b_vec_j, dtype=np.float32).reshape(1)
        
        return u_opt_j, G_extracted, h_extracted

# =========================================================================
# 3. 物理级随机拓扑静态生成算子 (100% 静态，剔除动态威胁)
# =========================================================================
def generate_random_environment(r_ego):
    target_x = np.random.uniform(10.0, 13.0)
    my_target = np.array([target_x, 0.0])
    num_obstacles = np.random.randint(2, 7) 
    
    types = []
    for _ in range(num_obstacles):
        types.append(np.random.choice(['ellipse', 'rectangle']))
    if 'ellipse' not in types: types[0] = 'ellipse'
    if 'rectangle' not in types: types[1] = 'rectangle'
        
    all_C, all_d = [], []
    
    for i in range(num_obstacles):
        while True:
            a = np.random.uniform(0.1, 0.4)
            b = np.random.uniform(0.1, 0.4)
            theta = np.random.uniform(0.0, 2 * np.pi)
            n = 2.0 if types[i] == 'ellipse' else float(np.random.choice([4.0, 6.0, 8.0]))
            
            x_c = np.random.uniform(2.0, target_x - 2.0)
            y_c = np.random.uniform(-4.0, 4.0)
            r_bound = np.sqrt(a**2 + b**2)
            
            if np.sqrt(x_c**2 + y_c**2) < (r_bound + r_ego + 0.6): continue
            if np.sqrt((x_c - target_x)**2 + y_c**2) < (r_bound + r_ego + 0.6): continue
                
            overlap = False
            for j in range(len(all_d)):
                prev_d = all_d[j]
                prev_r_bound = np.sqrt(all_C[j][0]**2 + all_C[j][1]**2)
                dist = np.sqrt((x_c - prev_d[0])**2 + (y_c - prev_d[1])**2)
                if dist < (r_bound + prev_r_bound + 2 * r_ego + 0.2):
                    overlap = True
                    break
            if overlap: continue
                
            all_C.append([a, b, theta, n])
            all_d.append([x_c, y_c])
            break
            
    return jnp.array(all_C), jnp.array(all_d), my_target

# =========================================================================
# 4. 2D 激光雷达表面击中几何仿真算子
# =========================================================================
@jit
def simulate_lidar_local_raw(ego_state, all_C, all_d):
    num_rays = 512
    max_range = 3.0
    num_steps = 200  
    
    ego_pos = ego_state[:2]
    theta = ego_state[2]
    
    ray_angles_world = theta + jnp.linspace(-jnp.pi, jnp.pi, num_rays, endpoint=False)
    ray_dirs_world = jnp.column_stack([jnp.cos(ray_angles_world), jnp.sin(ray_angles_world)]) 
    
    t_steps = jnp.linspace(0.0, max_range, num_steps) 
    pts_world = ego_pos[None, None, :] + t_steps[None, :, None] * ray_dirs_world[:, None, :] 
    
    def inside_obstacle(p, C_obs, d_obs):
        dx = p[..., 0] - d_obs[0]
        dy = p[..., 1] - d_obs[1]
        cos_t = jnp.cos(C_obs[2])
        sin_t = jnp.sin(C_obs[2])
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        ellipse_val = (jnp.abs(x_rot) / C_obs[0]) ** C_obs[3] + (jnp.abs(y_rot) / C_obs[1]) ** C_obs[3]
        return ellipse_val <= 1.0

    is_inside = jax.vmap(inside_obstacle, in_axes=(None, 0, 0))(pts_world, all_C, all_d) 
    inside_any = jnp.any(is_inside, axis=0) 
    
    t_hit = jnp.where(inside_any, t_steps[None, :], max_range)
    min_t = jnp.min(t_hit, axis=-1) 
    
    ray_angles_local = jnp.linspace(-jnp.pi, jnp.pi, num_rays, endpoint=False)
    hit_pts_local = jnp.column_stack([
        min_t * jnp.cos(ray_angles_local),
        min_t * jnp.sin(ray_angles_local)
    ])
    return hit_pts_local, min_t

# =========================================================================
# 5. 自动化离线行为克隆数据集参数化封装系统
# =========================================================================
if __name__ == '__main__':
    R_EGO = 0.31  
    TARGET_DATA_LENGTH = 10000
    L = 0.15           
    dt = 0.05          
    total_steps = 1800 
    
    final_dataset = []
    total_episodes_run, successful_episodes, collision_discarded_episodes = 0, 0, 0

    planner = LocalSdfCdfPlanner()

    print("⏳ 正在进行纯局部 XLA 全局优化算子冷启动融合编译...")
    _all_C, _all_d, _my_target = generate_random_environment(R_EGO)
    _dummy_ego = jnp.array([0.0, 0.0, 0.0])
    _dummy_p = jnp.array([L, 0.0])
    _dummy_nom = jnp.array([0.0, 0.0])
    _dummy_pc = jnp.zeros((10, 2))
    _, _, _ = planner.solve_agent_qp_local(_dummy_p, _dummy_nom, _dummy_pc, _dummy_p)
    _, _ = simulate_lidar_local_raw(_dummy_ego, _all_C, _all_d) 
    print("✨ 算子离线参数化接口对齐成功！开启非对称门控一键清洗...\n")

    while len(final_dataset) < TARGET_DATA_LENGTH:
        total_episodes_run += 1
        all_C, all_d, my_target = generate_random_environment(R_EGO)
        
        ego_state = jnp.array([0.0, 0.0, 0.0])          
        episode_buffer = [] 
        is_episode_safe = True

        for step in range(total_steps):
            x, y, theta = float(ego_state[0]), float(ego_state[1]), float(ego_state[2])

            # --- 5.1 超椭圆硬碰撞全时在线审计 ---
            for i in range(all_C.shape[0]):
                a_obs, b_obs, th_obs, n_obs = all_C[i]
                x_c, y_c = all_d[i, 0], all_d[i, 1]
                
                dx_c = x - x_c
                dy_c = y - y_c
                x_rot_c = dx_c * np.cos(th_obs) + dy_c * np.sin(th_obs)
                y_rot_c = -dx_c * np.sin(th_obs) + dy_c * np.cos(th_obs)
                
                ellipse_val_c = (abs(x_rot_c) / (a_obs + R_EGO))**n_obs + (abs(y_rot_c) / (b_obs + R_EGO))**n_obs
                if ellipse_val_c <= 1.0:
                    is_episode_safe = False
                    break
            
            if not is_episode_safe:
                break 

            # --- 5.2 纯局部系车身相对投影变换 ---
            local_pc_all, min_t = simulate_lidar_local_raw(ego_state, all_C, all_d)
            valid_mask = np.array(min_t < 2.99)
            current_pc_local = np.array(local_pc_all)[valid_mask] 

            dx = float(my_target[0]) - x
            dy = float(my_target[1]) - y

            target_local_x = dx * np.cos(theta) + dy * np.sin(theta)
            target_local_y = -dx * np.sin(theta) + dy * np.cos(theta)
            target_local = jnp.array([target_local_x, target_local_y])

            ego_p_local = jnp.array([L, 0.0])
            target_vector_local = target_local - ego_p_local
            dist_local = jnp.linalg.norm(target_vector_local)
            u_nom_local = jnp.where(dist_local > 0.1, 1.2 * target_vector_local / (dist_local + 1e-6), jnp.zeros(2))

            is_empty = (current_pc_local.shape[0] == 0)

            # --- 5.4 门控硬过滤逻辑 ---
            should_record_frame = False
            if is_empty:
                v_qp = float(u_nom_local[0])
                omega_qp = float(u_nom_local[1] / L)
            else:
                u_opt_local, G_extracted, h_extracted = planner.solve_agent_qp_local(
                    ego_p_local, u_nom_local, current_pc_local, target_local
                )
                v_qp = float(u_opt_local[0])
                omega_qp = float(u_opt_local[1] / L)

                control_residual = np.linalg.norm(np.array(u_opt_local) - np.array(u_nom_local))
                should_record_frame = False
                if control_residual > 0.09:
                    should_record_frame = True # 大于阈值全部保存
                else:
                    should_record_frame = (np.random.rand() < 0.50) # 小于阈值只保存50%

                if should_record_frame:
                    X_frame = np.array([target_local_x, target_local_y, v_qp, omega_qp], dtype=np.float32)
                    Y_frame = np.concatenate([
                        u_opt_local,          # 6维
                        G_extracted.flatten(), # 6维
                        h_extracted.flatten()  # 1维
                    ]).astype(np.float32)
                    z_frame = np.array(current_pc_local, dtype=np.float32)

                    frame_sample = {
                        'X': X_frame,
                        'Y': Y_frame,
                        'z': z_frame
                    }
                    episode_buffer.append(frame_sample)

            # 动力学步进
            new_x = x + v_qp * np.cos(theta) * dt
            new_y = y + v_qp * np.sin(theta) * dt
            new_theta = theta + omega_qp * dt
            ego_state = jnp.array([new_x, new_y, new_theta])

            if np.hypot(x - my_target[0], y - my_target[1]) < 0.2:
                break

        # --- 5.5 轨迹级终审放行大闸 ---
        if is_episode_safe and len(episode_buffer) > 0:
            successful_episodes += 1
            final_dataset.extend(episode_buffer)
        else:
            collision_discarded_episodes += 1
        
        print(f"📊 收集进度: {len(final_dataset)}/{TARGET_DATA_LENGTH} 帧元组已固化 "
              f"({len(final_dataset)/TARGET_DATA_LENGTH*100:.1f}%) | "
              f"审计报告 -> 总运行战局: {total_episodes_run} [完美放行: {successful_episodes} | 整条硬删除: {collision_discarded_episodes}]")

    final_dataset = final_dataset[:TARGET_DATA_LENGTH]
    
    print("\n" + "="*60)
    print(f"💾 [可微参数化行为克隆库] 正在执行磁盘固化...")
    torch.save(final_dataset, "dataset_param_bc.pt")
    print(f"✅ 编译图和漏洞漏洞全部洗净！ {len(final_dataset)} 帧数据安全落盘于：dataset_param_bc.pt")
    print("="*60 + "\n")