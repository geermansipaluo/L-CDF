#!/usr/bin/env python3
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import jit
from jaxproxqp.jaxproxqp import JaxProxQP

# 🟢 引入外部多场景环境池大闸
from env import get_env_pool, convert_to_jax_tensors


# =========================================================================
# 1. 256线激光雷达物理射线模拟算子
#    这条链路只用于训练数据保存：3m 内有命中才保存，3m 外不保存。
# =========================================================================
@partial(jit, static_argnums=(3, 4))
def simulate_parametric_lidar_256(
    pos_ego_world,
    theta_ego,
    meta_obs_tensors,
    num_rays=256,
    max_range=3.0,
):
    angles_world = jnp.linspace(-jnp.pi, jnp.pi, num_rays, endpoint=False) + theta_ego
    ray_dirs_world = jnp.column_stack([jnp.cos(angles_world), jnp.sin(angles_world)])

    num_steps = 200
    t_steps = jnp.linspace(0.0, max_range, num_steps)
    pts_check_world = (
        pos_ego_world[None, None, :]
        + t_steps[None, :, None] * ray_dirs_world[:, None, :]
    )

    rect_c, rect_ab = meta_obs_tensors['rect_c'], meta_obs_tensors['rect_ab']

    in_rect = (
        (jnp.abs(pts_check_world[..., 0] - rect_c[0]) / rect_ab[0]) ** 4
        + (jnp.abs(pts_check_world[..., 1] - rect_c[1]) / rect_ab[1]) ** 4
    ) <= 1.0

    # 动态支持 1-5 号圆形物体的碰撞散射检测，保持原 datagenerate 逻辑。
    in_circles = jnp.zeros(in_rect.shape, dtype=jnp.bool_)
    for i in range(1, 6):
        c_c = meta_obs_tensors.get(f'c{i}_c', jnp.array([99.0, 99.0]))
        c_r = meta_obs_tensors.get(f'c{i}_r', 1e-3)
        in_c = jnp.sum((pts_check_world - c_c[None, None, :]) ** 2, axis=-1) <= c_r ** 2
        in_circles = in_circles | in_c

    hit_mask = in_rect | in_circles

    t_hit = jnp.where(hit_mask, t_steps[None, :], max_range)
    min_t = jnp.min(t_hit, axis=-1)

    valid_hit_mask = min_t < (max_range - 1e-3)

    # 注意：保存到数据集的是车体系/局部系 3m 雷达点云。
    angles_local = jnp.linspace(-jnp.pi, jnp.pi, num_rays, endpoint=False)
    local_hit_x = min_t * jnp.cos(angles_local)
    local_hit_y = min_t * jnp.sin(angles_local)
    local_pc_all = jnp.column_stack([local_hit_x, local_hit_y])

    return local_pc_all, valid_hit_mask


# =========================================================================
# 2. 基于局部系变长点云最紧邻 SDF 的控制密度函数 (CDF)
#    这部分和 png 版本的控制器逻辑对齐。
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
    r_ego = 0.46 + 0.15
    sense_range = 3.0
    num_pts = local_pc.shape[0]

    dists = jnp.sqrt(jnp.sum((local_pc - pos_p_local) ** 2, axis=1))
    min_dist = jnp.min(dists)

    c_val = min_dist - r_ego
    b_val = min_dist - sense_range

    psi_curr = smooth_bump(c_val, b_val)
    psi_curr = jnp.where(num_pts == 0, 1.0, psi_curr)

    V_x = jnp.sum((pos_p_local - target_local) ** 2)
    alpha = 0.5
    rho = psi_curr / (V_x ** alpha + 1e-6)
    return rho, psi_curr


# =========================================================================
# 3. 6维参数化纯局部凸优化规划器
#    控制动作、G、h 全部由“控制器专用表面点云”计算。
# =========================================================================
class LocalSdfCdfPlanner:
    @partial(jit, static_argnums=(0,))
    def _solve_qp_core_jit(self, ego_p_local, u_nom_local, local_pc_control, target_local):
        epsilon = 0.05
        inv_eps = 1.0 / epsilon

        def density_wrapper_local(p):
            rho, _ = get_local_sdf_rho_and_psi(p, target_local, local_pc_control)
            return rho

        grad_self = jax.grad(density_wrapper_local)(ego_p_local)
        drift_term = 0.0

        norm_nom = jnp.linalg.norm(u_nom_local)
        dir_nom = jnp.where(
            norm_nom > 1e-5,
            u_nom_local / (norm_nom + 1e-8),
            jnp.array([1.0, 0.0]),
        )
        z1_pos = ego_p_local + epsilon * dir_nom

        neg_grad = -grad_self
        norm_grad = jnp.linalg.norm(neg_grad)
        dir_safe = jnp.where(
            norm_grad > 1e-5,
            neg_grad / (norm_grad + 1e-8),
            jnp.array([0.0, 1.0]),
        )
        z2_pos = ego_p_local + epsilon * dir_safe

        v1 = dir_nom
        v2_raw = dir_safe
        det_v = v1[0] * v2_raw[1] - v1[1] * v2_raw[0]
        is_independent = jnp.abs(det_v) > 1e-2
        v2_ortho = jnp.array([-v1[1], v1[0]])
        v2 = jnp.where(is_independent, v2_raw, v2_ortho)

        V_mat = jnp.column_stack([v1, v2])
        W_mat = jnp.linalg.inv(V_mat)
        w1 = W_mat[0, :]
        w2 = W_mat[1, :]

        rho_curr, psi_curr = get_local_sdf_rho_and_psi(
            ego_p_local,
            target_local,
            local_pc_control,
        )
        rho_z1 = density_wrapper_local(z1_pos)
        rho_z2 = density_wrapper_local(z2_pos)

        dim_total = 6
        lambda_smooth = 25
        H = jnp.zeros((dim_total, dim_total))
        val_uu = 2.0 * (1.0 + 2.0 * lambda_smooth)
        val_zz = 2.0 * lambda_smooth
        H = H.at[0, 0].set(val_uu)
        H = H.at[1, 1].set(val_uu)
        H = H.at[2, 2].set(val_zz)
        H = H.at[3, 3].set(val_zz)
        H = H.at[4, 4].set(val_zz)
        H = H.at[5, 5].set(val_zz)

        val_uz = -2.0 * lambda_smooth
        H = H.at[0, 2].set(val_uz)
        H = H.at[2, 0].set(val_uz)
        H = H.at[1, 3].set(val_uz)
        H = H.at[3, 1].set(val_uz)
        H = H.at[0, 4].set(val_uz)
        H = H.at[4, 0].set(val_uz)
        H = H.at[1, 5].set(val_uz)
        H = H.at[5, 1].set(val_uz)

        g_vec = jnp.zeros(dim_total).at[0:2].set(-2.0 * u_nom_local)

        sum_w = w1 + w2
        coeff_u = (rho_curr * inv_eps) * sum_w
        coeff_z1 = -(rho_z1 * inv_eps) * w1
        coeff_z2 = -(rho_z2 * inv_eps) * w2

        C_mat = (
            jnp.zeros((1, dim_total))
            .at[0, 0:2].set(coeff_u)
            .at[0, 2:4].set(coeff_z1)
            .at[0, 4:6].set(coeff_z2)
        )
        b_vec = jnp.zeros(1).at[0].set(drift_term - rho_curr)

        limit = 1.2
        l_box = jnp.array([-limit] * 6)
        u_box = jnp.array([limit] * 6)

        qp = JaxProxQP.QPModel.create(
            H,
            g_vec,
            C_mat,
            b_vec,
            l_box=l_box,
            u_box=u_box,
        )
        settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, settings)
        sol_raw = solver.solve().x

        return sol_raw, C_mat, b_vec, psi_curr

    def solve_agent_qp_local(self, ego_p_local, u_nom_local, local_pc_control, target_local):
        sol_raw_j, C_mat_j, b_vec_j, psi_curr_j = self._solve_qp_core_jit(
            ego_p_local,
            u_nom_local,
            local_pc_control,
            target_local,
        )
        G_extracted = np.array(C_mat_j, dtype=np.float32).reshape(1, 6)
        h_extracted = np.array(b_vec_j, dtype=np.float32).reshape(1)
        psi_curr = float(np.array(psi_curr_j))
        return sol_raw_j, G_extracted, h_extracted, psi_curr


# =========================================================================
# 4. 控制器专用障碍物表面点云：和 png 版本对齐
# =========================================================================
def build_obstacle_surface_pointcloud(meta_obstacles, num_points_per_obs=150):
    """
    构造全局障碍物表面点云。

    注意：
    1. 这个点云只用于专家 CDF-QP 控制；
    2. 它不直接保存进训练数据；
    3. 保存到训练数据的 z 仍然来自 3m 雷达射线命中点。
    """
    obs_pc_list = []
    angles = np.linspace(0, 2 * np.pi, num_points_per_obs)

    for obs in meta_obstacles:
        if obs['type'] == 'rect':
            c_x, c_y = obs['center'][0], obs['center'][1]
            a, b = obs['a'], obs['b']

            # 和 png 版本保持一致：n=4 超椭圆/圆角矩形表面。
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
                c_y + r * np.sin(angles),
            ])
            obs_pc_list.append(circle_pc)

    if len(obs_pc_list) == 0:
        # 防止无障碍物场景下 jnp.min 空数组。放一个极远点，近似等价于无障碍物。
        return np.array([[1e6, 1e6]], dtype=np.float32)

    return np.vstack(obs_pc_list).astype(np.float32)


def transform_world_points_to_local(points_world, x, y, theta):
    """世界系点云 -> 车体系点云；和 png 版本里的投影公式一致。"""
    dx_pc = points_world[:, 0] - x
    dy_pc = points_world[:, 1] - y

    pc_local_x = dx_pc * np.cos(theta) + dy_pc * np.sin(theta)
    pc_local_y = -dx_pc * np.sin(theta) + dy_pc * np.cos(theta)

    return np.column_stack([pc_local_x, pc_local_y]).astype(np.float32)


def check_collision_by_meta_obstacles(x, y, meta_obstacles, r_ego):
    """和 png 版本一致的场景级硬碰撞审计。"""
    for obs in meta_obstacles:
        if obs['type'] == 'rect':
            closest_rect_x = np.clip(
                x,
                obs['center'][0] - obs['a'],
                obs['center'][0] + obs['a'],
            )
            closest_rect_y = np.clip(
                y,
                obs['center'][1] - obs['b'],
                obs['center'][1] + obs['b'],
            )
            if np.hypot(x - closest_rect_x, y - closest_rect_y) < r_ego:
                return True

        elif obs['type'] == 'circle':
            if np.hypot(x - obs['center'][0], y - obs['center'][1]) < (obs['r'] + r_ego):
                return True

    return False


# =========================================================================
# 5. 主批量场景迭代推演控制大闸
# =========================================================================
if __name__ == '__main__':
    R_EGO = 0.31
    L = 0.4
    dt = 0.05
    total_steps = 2500

    # 每个环境生成多少条随机目标轨迹
    # 10个环境 × 6条 = 最多60条轨迹，可按需要改大。
    num_demos_per_env = 6

    # 随机目标范围
    target_x_min, target_x_max = 14.0, 16.0
    target_y_min, target_y_max = -2.0, 2.0

    # 固定随机种子，保证每次生成的数据可复现
    rng = np.random.default_rng(0)

    # 创建图片保存根目录；保留原接口，当前脚本主要保存 .pt 轨迹数据。
    img_dir = 'dataset_png_random_target'
    os.makedirs(img_dir, exist_ok=True)

    output_dataset_path = 'dataset_trajectories.pt'

    planner = LocalSdfCdfPlanner()
    env_pool = get_env_pool()

    # ============================================================
    # trajectory-level buffer
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

    print('=' * 70)
    print('🚀 启动随机目标轨迹级专家数据生成')
    print(f'   环境数量: {len(env_pool)}')
    print(f'   每个环境随机目标轨迹数: {num_demos_per_env}')
    print(f'   目标x范围: [{target_x_min}, {target_x_max}]')
    print(f'   目标y范围: [{target_y_min}, {target_y_max}]')
    print(f'   理论最大轨迹数: {len(env_pool) * num_demos_per_env}')
    print('=' * 70 + '\n')

    for env_id, env_cfg in enumerate(env_pool):
        print(f'\n==================== 场景 {env_id}/{len(env_pool) - 1} ====================')

        meta_obstacles = env_cfg['meta_obstacles']

        # 保存链路用：JAX 雷达射线模拟仍然吃参数化障碍物。
        obs_tensors_for_lidar = convert_to_jax_tensors(meta_obstacles)

        # 控制链路用：和 png 版本一致，预先释放障碍物表面点云。
        obs_pc_world_for_control = build_obstacle_surface_pointcloud(
            meta_obstacles,
            num_points_per_obs=150,
        )

        for demo_id in range(num_demos_per_env):
            # ============================================================
            # 1. 为当前环境随机生成一个目标点
            # ============================================================
            my_target = np.array([
                rng.uniform(target_x_min, target_x_max),
                rng.uniform(target_y_min, target_y_max),
            ], dtype=np.float32)

            print(
                f'\n⏳ 正在生成轨迹: env={env_id}, demo={demo_id}, '
                f'target=({my_target[0]:.3f}, {my_target[1]:.3f})'
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

                history_x.append(x)
                history_y.append(y)

                # ========================================================
                # A. png 同款硬碰撞审计：不依赖 3m 雷达是否看见障碍物
                # ========================================================
                if check_collision_by_meta_obstacles(x, y, meta_obstacles, R_EGO):
                    is_episode_safe = False
                    print(
                        f'    ❌ [碰撞熔断] env={env_id}, demo={demo_id}, '
                        f'step={step}, pose=({x:.3f}, {y:.3f}, {theta:.3f})'
                    )
                    break

                # ========================================================
                # B. 控制器专用点云链路：障碍物表面点云 -> 当前车体系
                #    这条链路永远给 CDF-QP 用，不受 3m 保存规则影响。
                # ========================================================
                current_pc_control_local = transform_world_points_to_local(
                    obs_pc_world_for_control,
                    x,
                    y,
                    theta,
                )
                current_pc_control_j = jnp.array(current_pc_control_local)

                # ========================================================
                # C. 保存专用点云链路：256线 3m 雷达射线
                #    这条链路只决定是否保存 frame，以及 frame['z'] 保存什么。
                # ========================================================
                pos_ego_world = jnp.array([x, y])
                local_pc_all, valid_hit_mask = simulate_parametric_lidar_256(
                    pos_ego_world,
                    theta,
                    obs_tensors_for_lidar,
                    num_rays=256,
                    max_range=3.0,
                )

                valid_mask_np = np.array(valid_hit_mask)
                current_pc_save_local = np.array(local_pc_all, dtype=np.float32)[valid_mask_np]
                has_save_points = current_pc_save_local.shape[0] > 0

                # 仅用于后续可视化/调试，不影响控制和保存逻辑。
                if has_save_points:
                    rot_mat = np.array([
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)],
                    ])
                    last_frame_lidar_world = (
                        rot_mat @ current_pc_save_local.T
                    ).T + np.array([x, y])
                else:
                    last_frame_lidar_world = None

                # ========================================================
                # D. 局部系相对目标和名义控制
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
                    jnp.zeros(2),
                )

                # ========================================================
                # E. 专家控制求解：无论 3m 内有没有保存点，永远使用表面点云
                # ========================================================
                prev_executed_ux = last_executed_ux
                prev_executed_uy = last_executed_uy

                sol_6d_raw, G_extracted, h_extracted, psi_curr = planner.solve_agent_qp_local(
                    ego_p_local,
                    u_nom_local,
                    current_pc_control_j,
                    target_local,
                )

                v_qp = float(sol_6d_raw[0])
                omega_qp = float(sol_6d_raw[1]) / L

                control_residual = np.linalg.norm(
                    np.array(sol_6d_raw[:2]) - np.array(u_nom_local)
                )

                # ========================================================
                # F. 数据保存规则：3m 外不保存；3m 内全保存
                #    保存的 z 是 3m 雷达点云；Y 是表面点云专家控制标签。
                # ========================================================
                if has_save_points:
                    X_frame = np.array([
                        target_local_x,
                        target_local_y,
                        prev_executed_ux,
                        prev_executed_uy,
                    ], dtype=np.float32)

                    Y_frame = np.concatenate([
                        [float(sol_6d_raw[0])],
                        [float(sol_6d_raw[1])],
                        G_extracted.flatten(),
                        h_extracted.flatten(),
                    ]).astype(np.float32)

                    episode_buffer.append({
                        'X': X_frame,
                        'Y': Y_frame,
                        'z': np.array(current_pc_save_local, dtype=np.float32),

                        # 下面这些不是必须训练用，但后面调试很有用
                        'env_id': env_id,
                        'demo_id': demo_id,
                        'step': step,
                        'target': my_target.astype(np.float32),
                        'ego_state': np.array([x, y, theta], dtype=np.float32),
                        'control_residual': np.float32(control_residual),
                        'psi_control': np.float32(psi_curr),
                        'num_lidar_points': np.int32(current_pc_save_local.shape[0]),
                    })

                # 当前帧控制执行完以后，再更新 last_executed，供下一帧 X 使用。
                last_executed_ux = float(sol_6d_raw[0])
                last_executed_uy = float(sol_6d_raw[1])

                # ========================================================
                # G. 单轴前推
                # ========================================================
                new_x = x + v_qp * np.cos(theta) * dt
                new_y = y + v_qp * np.sin(theta) * dt
                new_theta = theta + omega_qp * dt

                ego_state = jnp.array([new_x, new_y, new_theta])

                # ========================================================
                # H. 到达目标判定
                # ========================================================
                dist_to_target = np.hypot(x - my_target[0], y - my_target[1])

                if dist_to_target < 0.44:
                    is_episode_success = True
                    print(
                        f'    🎯 [成功到达] env={env_id}, demo={demo_id}, '
                        f'step={step}, frames={len(episode_buffer)}, '
                        f'dist={dist_to_target:.3f}'
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
                    'history_y': np.array(history_y, dtype=np.float32),
                }

                global_trajectory_buffer.append(trajectory_record)

                print(
                    f'    📥 [轨迹保存] env={env_id}, demo={demo_id}, '
                    f'target=({my_target[0]:.3f}, {my_target[1]:.3f}), '
                    f'frames={len(episode_buffer)}, '
                    f'当前总轨迹数={len(global_trajectory_buffer)}'
                )

            else:
                print(
                    f'    ⚠️ [轨迹丢弃] env={env_id}, demo={demo_id}, '
                    f'safe={is_episode_safe}, success={is_episode_success}, '
                    f'frames={len(episode_buffer)}'
                )

    # ============================================================
    # 10. 全部轨迹保存
    # ============================================================
    if len(global_trajectory_buffer) > 0:
        print('\n' + '=' * 70)
        print('🎉 随机目标专家轨迹生成完成')
        print(f'   成功保存轨迹数: {len(global_trajectory_buffer)}')

        total_frames = sum(traj['num_frames'] for traj in global_trajectory_buffer)
        print(f'   总帧数: {total_frames}')

        torch.save(global_trajectory_buffer, output_dataset_path)

        print(f'💾 轨迹级数据集已保存: {output_dataset_path}')
        print('=' * 70 + '\n')

    else:
        print('\n' + '=' * 70)
        print('❌ 没有生成任何成功且安全的轨迹，未保存数据集。')
        print('=' * 70 + '\n')
