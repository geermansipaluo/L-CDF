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
    r_ego = 0.46 + 0.15
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

def build_obstacle_surface_pointcloud(meta_obstacles, num_points_per_obs=150):
    """
    构造全局障碍物表面点云。
    这个点云只用于专家 CDF-QP 控制，不直接作为训练输入保存。
    """
    obs_pc_list = []
    angles = np.linspace(0, 2 * np.pi, num_points_per_obs)

    for obs in meta_obstacles:
        if obs['type'] == 'rect':
            c_x, c_y = obs['center'][0], obs['center'][1]
            a, b = obs['a'], obs['b']

            # 超椭圆边界，和 png 版本保持一致
            cos_t = np.cos(angles)
            sin_t = np.sin(angles)

            rect_x = c_x + a * np.sign(cos_t) * np.sqrt(np.abs(cos_t))
            rect_y = c_y + b * np.sign(sin_t) * np.sqrt(np.abs(sin_t))

            rect_pc = np.column_stack([rect_x, rect_y])
            obs_pc_list.append(rect_pc)

        elif obs['type'] == 'circle':
            c_x, c_y = obs['center'][0], obs['center'][1]
            r = obs['r']

            circle_pc = np.column_stack([
                c_x + r * np.cos(angles),
                c_y + r * np.sin(angles)
            ])

            obs_pc_list.append(circle_pc)

    if len(obs_pc_list) == 0:
        return np.zeros((1, 2), dtype=np.float32)

    return np.vstack(obs_pc_list).astype(np.float32)

# =========================================================================
# 主批量场景迭代推演控制大闸
# =========================================================================
if __name__ == '__main__':
    R_EGO = 0.31
    L = 0.4
    dt = 0.05
    total_steps = 2500

    # 每个环境生成多少条随机目标轨迹
    # 10个环境 × 8条 = 最多80条轨迹
    # 后面画 demonstrations = [5, 10, 20, 30, 40, 48] 足够用
    num_demos_per_env = 6

    # 随机目标范围
    target_x_min, target_x_max = 14.0, 16.0
    target_y_min, target_y_max = -2.0, 2.0

    # 固定随机种子，保证每次生成的数据可复现
    rng = np.random.default_rng(0)

    # 创建图片保存根目录
    img_dir = "dataset_png_random_target"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    planner = LocalSdfCdfPlanner()
    env_pool = get_env_pool()

    # ============================================================
    # 关键修改：
    # 这里不再保存为 global_dataset_buffer = []
    # 而是保存为 trajectory-level buffer
    #
    # 每个元素是一条完整轨迹：
    # {
    #   'env_id': int,
    #   'demo_id': int,
    #   'target': np.array([tx, ty]),
    #   'num_frames': int,
    #   'frames': [frame_0, frame_1, ...],
    #   'history_x': [...],
    #   'history_y': [...]
    # }
    # ============================================================
    global_trajectory_buffer = []

    print("=" * 70)
    print(f"🚀 启动随机目标轨迹级专家数据生成")
    print(f"   环境数量: {len(env_pool)}")
    print(f"   每个环境随机目标轨迹数: {num_demos_per_env}")
    print(f"   目标x范围: [{target_x_min}, {target_x_max}]")
    print(f"   目标y范围: [{target_y_min}, {target_y_max}]")
    print(f"   理论最大轨迹数: {len(env_pool) * num_demos_per_env}")
    print("=" * 70 + "\n")

    for env_id, env_cfg in enumerate(env_pool):
        print(f"\n==================== 场景 {env_id}/{len(env_pool)-1} ====================")

        meta_obstacles = env_cfg['meta_obstacles']

        # 转换为JAX高速静态张量
        # 障碍物保持不变，只随机目标
        obs_tensors = convert_to_jax_tensors(meta_obstacles)

        obs_pc_world_for_control = build_obstacle_surface_pointcloud(
            meta_obstacles,
            num_points_per_obs=150
        )

        for demo_id in range(num_demos_per_env):
            # ============================================================
            # 1. 为当前环境随机生成一个目标点
            # ============================================================
            my_target = np.array([
                rng.uniform(target_x_min, target_x_max),
                rng.uniform(target_y_min, target_y_max)
            ], dtype=np.float32)

            print(
                f"\n⏳ 正在生成轨迹: env={env_id}, demo={demo_id}, "
                f"target=({my_target[0]:.3f}, {my_target[1]:.3f})"
            )

            # 起点严格锁定在 [0, 0, 0]
            ego_state = jnp.array([0.0, 0.0, 0.0])

            # 当前轨迹的数据缓存
            episode_buffer = []

            # 当前轨迹状态标记
            is_episode_safe = True
            is_episode_success = False

            # 可视化与调试用轨迹历史
            history_x = []
            history_y = []
            last_frame_lidar_world = None

            # 上一帧执行控制，用于构造 X_frame
            last_executed_ux = 0.0
            last_executed_uy = 0.0

            for step in range(total_steps):
                x = float(ego_state[0])
                y = float(ego_state[1])
                theta = float(ego_state[2])

                # ========================================================
                # A. 控制器专用点云：不受 3m 限制，用全局障碍物表面点云
                # ========================================================
                dx_pc_ctrl = obs_pc_world_for_control[:, 0] - x
                dy_pc_ctrl = obs_pc_world_for_control[:, 1] - y

                control_pc_local_x = dx_pc_ctrl * np.cos(theta) + dy_pc_ctrl * np.sin(theta)
                control_pc_local_y = -dx_pc_ctrl * np.sin(theta) + dy_pc_ctrl * np.cos(theta)

                current_pc_control_local = np.column_stack([
                    control_pc_local_x,
                    control_pc_local_y
                ]).astype(np.float32)

                # 可选：只取距离车体最近的 K 个点，减少 QP/JAX 负担
                # 注意这里是控制用点云，不是保存到数据集的观测点云
                K_CONTROL = min(256, current_pc_control_local.shape[0])

                dist_to_ego_local = np.linalg.norm(
                    current_pc_control_local - np.array([L, 0.0], dtype=np.float32),
                    axis=1
                )

                nearest_idx = np.argpartition(dist_to_ego_local, K_CONTROL - 1)[:K_CONTROL]
                current_pc_control_local = current_pc_control_local[nearest_idx]

                history_x.append(x)
                history_y.append(y)

                # ========================================================
                # 2. 256线物理激光雷达模拟
                # ========================================================
                pos_ego_world = jnp.array([x, y])

                local_pc_all, valid_hit_mask = simulate_parametric_lidar_256(
                    pos_ego_world,
                    theta,
                    obs_tensors,
                    num_rays=256,
                    max_range=3.0
                )

                valid_mask_np = np.array(valid_hit_mask)
                current_pc_local = np.array(local_pc_all)[valid_mask_np]

                # ========================================================
                # 3. 硬碰撞安全审计
                # ========================================================
                if current_pc_local.shape[0] > 0:
                    distances_to_ego = np.linalg.norm(current_pc_local, axis=1)

                    if np.min(distances_to_ego) <= (R_EGO + 1e-3):
                        is_episode_safe = False
                        print(
                            f"    ❌ [碰撞熔断] env={env_id}, demo={demo_id}, "
                            f"step={step}, min_dist={np.min(distances_to_ego):.4f}"
                        )
                        break

                # 保存最后一帧雷达点的世界坐标，仅用于可视化
                if current_pc_local.shape[0] > 0:
                    rot_mat = np.array([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]
                    ])
                    last_frame_lidar_world = (
                        rot_mat @ current_pc_local.T
                    ).T + np.array([x, y])
                else:
                    last_frame_lidar_world = None

                is_empty = current_pc_local.shape[0] == 0

                # ========================================================
                # 4. 局部系相对目标
                # ========================================================
                dx_tg = float(my_target[0]) - x
                dy_tg = float(my_target[1]) - y

                target_local_x = dx_tg * np.cos(theta) + dy_tg * np.sin(theta)
                target_local_y = -dx_tg * np.sin(theta) + dy_tg * np.cos(theta)
                target_local = jnp.array([target_local_x, target_local_y])

                ego_p_local = jnp.array([L, 0.0])
                target_vector_local = target_local - ego_p_local
                dist_local = jnp.linalg.norm(target_vector_local)

                u_nom_local = jnp.where(
                    dist_local > 0.1,
                    1.2 * target_vector_local / (dist_local + 1e-6),
                    jnp.zeros(2)
                )

                # ========================================================
                # 5. 专家控制求解 + 数据记录
                # ========================================================
                if is_empty:
                    # v_qp = float(u_nom_local[0])
                    # omega_qp = float(u_nom_local[1] / L)

                    # last_executed_ux = float(u_nom_local[0])
                    # last_executed_uy = float(u_nom_local[1])
                    # ========================================================
                    # 无论 3m 内有没有观测点云，专家控制都始终走 CDF-QP
                    # ========================================================
                    current_pc_control_j = jnp.array(current_pc_control_local)

                    sol_6d_raw, G_extracted, h_extracted = planner.solve_agent_qp_local(
                        ego_p_local,
                        u_nom_local,
                        current_pc_control_j,
                        target_local
                    )

                    v_qp = float(sol_6d_raw[0])
                    omega_qp = float(sol_6d_raw[1]) / L
                    last_executed_ux = float(sol_6d_raw[0])
                    last_executed_uy = float(sol_6d_raw[1])

                else:
                    current_pc_local_j = jnp.array(current_pc_local)

                    sol_6d_raw, G_extracted, h_extracted = planner.solve_agent_qp_local(
                        ego_p_local,
                        u_nom_local,
                        current_pc_local_j,
                        target_local
                    )

                    v_qp = float(sol_6d_raw[0])
                    omega_qp = float(sol_6d_raw[1]) / L

                    # 与原代码一致：
                    # 避障控制变化明显时必记；
                    # 其他时候以一定概率记录，避免数据全是危险状态
                    control_residual = np.linalg.norm(
                        np.array(sol_6d_raw[:2]) - np.array(u_nom_local)
                    )

                    # should_record_frame = (
                    #     control_residual > 0.05
                    # ) or (
                    #     np.random.rand() < 0.40
                    # )
                    should_record_frame = True

                    if should_record_frame:
                        X_frame = np.array([
                            target_local_x,
                            target_local_y,
                            last_executed_ux,
                            last_executed_uy
                        ], dtype=np.float32)

                        Y_frame = np.concatenate([
                            [float(sol_6d_raw[0])],
                            [float(sol_6d_raw[1])],
                            G_extracted.flatten(),
                            h_extracted.flatten()
                        ]).astype(np.float32)

                        episode_buffer.append({
                            'X': X_frame,
                            'Y': Y_frame,
                            'z': np.array(current_pc_local, dtype=np.float32),

                            # 下面这些不是必须训练用，但后面调试很有用
                            'env_id': env_id,
                            'demo_id': demo_id,
                            'step': step,
                            'target': my_target.astype(np.float32),
                            'ego_state': np.array([x, y, theta], dtype=np.float32),
                            'control_residual': np.float32(control_residual)
                        })

                    last_executed_ux = float(sol_6d_raw[0])
                    last_executed_uy = float(sol_6d_raw[1])

                # ========================================================
                # 6. 单轴前推
                # ========================================================
                new_x = x + v_qp * np.cos(theta) * dt
                new_y = y + v_qp * np.sin(theta) * dt
                new_theta = theta + omega_qp * dt

                ego_state = jnp.array([new_x, new_y, new_theta])

                # ========================================================
                # 7. 到达目标判定
                # ========================================================
                dist_to_target = np.hypot(x - my_target[0], y - my_target[1])

                if dist_to_target < 0.44:
                    is_episode_success = True
                    print(
                        f"    🎯 [成功到达] env={env_id}, demo={demo_id}, "
                        f"step={step}, frames={len(episode_buffer)}, "
                        f"dist={dist_to_target:.3f}"
                    )
                    break

            # ============================================================
            # 8. 当前轨迹结束后，决定是否保存
            # ============================================================
            if is_episode_safe and is_episode_success and len(episode_buffer) > 0:
                trajectory_record = {
                    'env_id': env_id,
                    'demo_id': demo_id,
                    'target': my_target.astype(np.float32),
                    'num_frames': len(episode_buffer),
                    'frames': episode_buffer,
                    'history_x': np.array(history_x, dtype=np.float32),
                    'history_y': np.array(history_y, dtype=np.float32)
                }

                global_trajectory_buffer.append(trajectory_record)

                print(
                    f"    📥 [轨迹保存] env={env_id}, demo={demo_id}, "
                    f"target=({my_target[0]:.3f}, {my_target[1]:.3f}), "
                    f"frames={len(episode_buffer)}, "
                    f"当前总轨迹数={len(global_trajectory_buffer)}"
                )

            else:
                print(
                    f"    ⚠️ [轨迹丢弃] env={env_id}, demo={demo_id}, "
                    f"safe={is_episode_safe}, success={is_episode_success}, "
                    f"frames={len(episode_buffer)}"
                )

    # ============================================================
    # 10. 全部轨迹保存
    # ============================================================
    if len(global_trajectory_buffer) > 0:
        output_dataset_path = "dataset_trajectories.pt"

        print("\n" + "=" * 70)
        print(f"🎉 随机目标专家轨迹生成完成")
        print(f"   成功保存轨迹数: {len(global_trajectory_buffer)}")

        total_frames = sum(traj['num_frames'] for traj in global_trajectory_buffer)
        print(f"   总帧数: {total_frames}")

        torch.save(global_trajectory_buffer, output_dataset_path)

        print(f"💾 轨迹级数据集已保存: {output_dataset_path}")
        print("=" * 70 + "\n")

    else:
        print("\n" + "=" * 70)
        print("❌ 没有生成任何成功且安全的轨迹，未保存数据集。")
        print("=" * 70 + "\n")