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
from env import get_env_pool


# =========================================================================
# 0. 障碍物参数化：把 env.py 里的 meta_obstacles 转成全局 CDF-QP 使用的 all_C/all_d/all_v
# =========================================================================
def _extract_obs_velocity(obs):
    """兼容不同 env 写法；静态障碍物默认速度为 0。"""
    if 'v' in obs:
        v = obs['v']
        return float(v[0]), float(v[1])
    if 'vel' in obs:
        v = obs['vel']
        return float(v[0]), float(v[1])
    return float(obs.get('vx', 0.0)), float(obs.get('vy', 0.0))


def meta_obstacles_to_global_cdf_tensors(meta_obstacles):
    """
    将场景池障碍物转成统一超椭圆参数形式：
        C = [a, b, theta, n]
        d = [cx, cy]
        v = [vx, vy]

    rect   -> n=4，和 png / lidar 中的圆角矩形超椭圆保持一致
    circle -> a=b=r, n=2
    ellipse-> n=2
    """
    all_C, all_d, all_v = [], [], []

    for obs in meta_obstacles:
        obs_type = obs.get('type', '')
        c_x, c_y = float(obs['center'][0]), float(obs['center'][1])
        theta = float(obs.get('theta', obs.get('yaw', 0.0)))
        vx, vy = _extract_obs_velocity(obs)

        if obs_type == 'rect':
            a = float(obs['a'])
            b = float(obs['b'])
            n = float(obs.get('n', 4.0))
        elif obs_type == 'circle':
            r = float(obs['r'])
            a = r
            b = r
            theta = 0.0
            n = 2.0
        elif obs_type == 'ellipse':
            a = float(obs['a'])
            b = float(obs['b'])
            n = float(obs.get('n', 2.0))
        else:
            raise ValueError(f"Unsupported obstacle type: {obs_type}")

        all_C.append([a, b, theta, n])
        all_d.append([c_x, c_y])
        all_v.append([vx, vy])

    if len(all_C) == 0:
        # 理论上 env_pool 不应该为空障碍物；这里给一个远处虚拟障碍物防止 vmap 空数组报错。
        all_C = [[0.1, 0.1, 0.0, 2.0]]
        all_d = [[999.0, 999.0]]
        all_v = [[0.0, 0.0]]

    return (
        jnp.array(all_C, dtype=jnp.float32),
        jnp.array(all_d, dtype=jnp.float32),
        jnp.array(all_v, dtype=jnp.float32),
    )


# =========================================================================
# 1. 3m 固定圆感知区域：参数化地图射线雷达，只用于保存 z，不参与专家控制
# =========================================================================
@partial(jit, static_argnums=(4, 5))
def simulate_global_map_lidar_256(pos_ego_world, theta_ego, all_C, all_d, num_rays=256, max_range=3.0):
    """
    固定 3m 圆形感知区域的 256 线射线模拟。
    返回：
        local_pc_all: 所有 ray 的局部命中点；未命中 ray 位于 max_range 圆上
        valid_hit_mask: True 表示 3m 内真的打到障碍物
    """
    ray_angles_local = jnp.linspace(-jnp.pi, jnp.pi, num_rays, endpoint=False)
    ray_angles_world = ray_angles_local + theta_ego
    ray_dirs_world = jnp.column_stack([jnp.cos(ray_angles_world), jnp.sin(ray_angles_world)])

    num_steps = 200
    t_steps = jnp.linspace(0.0, max_range, num_steps)
    pts_world = pos_ego_world[None, None, :] + t_steps[None, :, None] * ray_dirs_world[:, None, :]

    def inside_obstacle(p, C_obs, d_obs):
        dx = p[..., 0] - d_obs[0]
        dy = p[..., 1] - d_obs[1]
        cos_t = jnp.cos(C_obs[2])
        sin_t = jnp.sin(C_obs[2])
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        ellipse_val = (jnp.abs(x_rot) / C_obs[0]) ** C_obs[3] + (jnp.abs(y_rot) / C_obs[1]) ** C_obs[3]
        return ellipse_val <= 1.0

    # shape: [num_obs, num_rays, num_steps]
    is_inside_each = jax.vmap(inside_obstacle, in_axes=(None, 0, 0))(pts_world, all_C, all_d)
    inside_any = jnp.any(is_inside_each, axis=0)

    t_hit = jnp.where(inside_any, t_steps[None, :], max_range)
    min_t = jnp.min(t_hit, axis=-1)
    valid_hit_mask = min_t < (max_range - 1e-3)

    local_hit_x = min_t * jnp.cos(ray_angles_local)
    local_hit_y = min_t * jnp.sin(ray_angles_local)
    local_pc_all = jnp.column_stack([local_hit_x, local_hit_y])

    return local_pc_all, valid_hit_mask


# =========================================================================
# 2. 全局地图 CDF 核心算子：专家控制使用整张地图的参数化障碍物，不再吃点云
#    坐标检查版：密度查询变量在车体系，JAX 链式法则自动给出局部梯度。
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
def local_to_world_point(p_local, ego_pose_world):
    x, y, theta = ego_pose_world[0], ego_pose_world[1], ego_pose_world[2]
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([
        x + c * p_local[0] - s * p_local[1],
        y + s * p_local[0] + c * p_local[1],
    ])


@jit
def obstacle_bump_field(x_world, C_obs, d_obs, r_ego):
    """
    单个障碍物的 bump 安全场。

    坐标/单位约定：
    - x_world 是世界系二维查询点；
    - C_obs = [a, b, theta, n] 是世界地图中的超椭圆障碍物参数；
    - rect 默认 n=4，circle 为 a=b=r,n=2；
    - c_val 和 b_val 都使用“米”作为单位，避免把 ellipse_val-1 这种无量纲量
      和 3m 圆距离直接混进 smooth_bump。

    含义：
    - c_val = 查询点到膨胀超椭圆边界的径向 signed-distance-like 值；
      c_val <= 0 表示进入 ego 半径膨胀后的障碍物。
    - b_val = 查询点到障碍物中心的距离 - 3m；
      b_val >= 0 表示超出该障碍物的 3m 圆形作用域，psi=1。
    """
    a, b, theta, n = C_obs[0], C_obs[1], C_obs[2], C_obs[3]
    dx = x_world[0] - d_obs[0]
    dy = x_world[1] - d_obs[1]

    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    x_rot = dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t

    a_buf = a + r_ego
    b_buf = b + r_ego

    radial_dist = jnp.sqrt(x_rot ** 2 + y_rot ** 2)
    ux = x_rot / (radial_dist + 1e-8)
    uy = y_rot / (radial_dist + 1e-8)

    # 从障碍物中心沿当前方向打到超椭圆边界的半径。
    # 超椭圆边界：(|x|/a)^n + (|y|/b)^n = 1
    denom = (jnp.abs(ux) / a_buf) ** n + (jnp.abs(uy) / b_buf) ** n
    safe_denom = jnp.maximum(denom, 1e-12)
    boundary_radius_raw = safe_denom ** (-1.0 / n)
    boundary_radius = jnp.where(
        radial_dist > 1e-6,
        boundary_radius_raw,
        jnp.minimum(a_buf, b_buf),
    )

    # 碰撞边界：膨胀超椭圆；作用域边界：障碍物中心 3m 圆。
    # 两者都以米为单位，smooth_bump 的插值才是稳定的。
    c_val = radial_dist - boundary_radius
    b_val = radial_dist - 3.0
    return smooth_bump(c_val, b_val)


@jit
def get_global_map_density_from_local_query(p_local, target_world, all_C, all_d, r_ego, ego_pose_world):
    """
    控制器密度函数：
    - 查询变量 p_local 仍然在车体系中，这样 QP 输出保持 [ux_local, uy_local]
    - 但密度评估先把 p_local 投影到世界系，再对整张全局地图 all_C/all_d 求 CDF
    - 不做障碍物筛选；3m 感知/作用域由 obstacle_bump_field() 中的圆形 b_val 定义
    """
    p_world = local_to_world_point(p_local, ego_pose_world)

    def single_obs_density(C_obs, d_obs):
        return obstacle_bump_field(p_world, C_obs, d_obs, r_ego)

    psi_array = jax.vmap(single_obs_density)(all_C, all_d)
    psi_static = jnp.prod(psi_array)

    dist_target_sq = jnp.sum((p_world - target_world) ** 2)
    alpha = 0.5
    rho = psi_static / (dist_target_sq ** alpha + 1e-6)
    return rho, psi_static


@jit
def get_global_map_density_with_obs_centers(p_local, target_world, all_C, all_d, r_ego, ego_pose_world):
    rho, _ = get_global_map_density_from_local_query(p_local, target_world, all_C, all_d, r_ego, ego_pose_world)
    return rho


# =========================================================================
# 2.5 训练标签专用 SDF-CDF-QP：只用当前帧 3m 雷达射线最近交点构造局部 SDF-CDF
#     说明：
#     - 轨迹 rollout / 实际执行控制仍然由 GlobalMapCDFPlanner 给出；
#     - 只有保存数据时，才用 local_pc_save_local 重新计算 SDF-CDF 的 G/h；
#     - Y[:2] 默认保存全局 CDF-QP 实际执行的 expert u_ctrl=[v,L*omega]；
#       Y[2:] 保存与测试阶段 SDF-CDF-QP 一致的局部点云约束 G/h。
# =========================================================================
@jit
def get_local_sdf_rho_and_psi_for_label(pos_p_local, target_local, local_pc_valid, r_ego):
    """
    基于当前 3m 雷达命中点云的 nearest-point SDF-CDF。

    坐标约定：
    - pos_p_local: 车体系查询点，通常为控制点 [L, 0]
    - target_local: 目标在车体系下的坐标，原点是智能体中心
    - local_pc_valid: 3m 雷达实际命中点，车体系坐标
    - r_ego: CDF 膨胀半径

    这里和原始 SDF-CDF-QP 保持一致：
    c_val = min_dist - r_ego
    b_val = min_dist - 3.0
    二者都是米制距离，不再混合无量纲超椭圆值。
    """
    sense_range = 3.0

    dists = jnp.sqrt(jnp.sum((local_pc_valid - pos_p_local) ** 2, axis=1))
    min_dist = jnp.min(dists)

    c_val = min_dist - r_ego
    b_val = min_dist - sense_range
    psi_curr = smooth_bump(c_val, b_val)

    V_x = jnp.sum((pos_p_local - target_local) ** 2)
    alpha = 0.5
    rho = psi_curr / (V_x ** alpha + 1e-6)
    return rho, psi_curr


class LocalSdfCdfLabelPlanner:
    @partial(jit, static_argnums=(0,))
    def _solve_qp_core_jit(self, ego_p_local, u_nom_local, local_pc_valid, target_local, r_ego, qp_epsilon, qp_lambda_smooth, qp_limit):
        epsilon = qp_epsilon
        inv_eps = 1.0 / qp_epsilon

        def density_wrapper_local(p_local):
            rho, _ = get_local_sdf_rho_and_psi_for_label(
                p_local, target_local, local_pc_valid, r_ego
            )
            return rho

        grad_self = jax.grad(density_wrapper_local)(ego_p_local)
        drift_term = 0.0

        norm_nom = jnp.linalg.norm(u_nom_local)
        dir_nom = jnp.where(
            norm_nom > 1e-5,
            u_nom_local / (norm_nom + 1e-8),
            jnp.array([1.0, 0.0], dtype=jnp.float32),
        )
        z1_pos = ego_p_local + epsilon * dir_nom

        neg_grad = -grad_self
        norm_grad = jnp.linalg.norm(neg_grad)
        dir_safe = jnp.where(
            norm_grad > 1e-5,
            neg_grad / (norm_grad + 1e-8),
            jnp.array([0.0, 1.0], dtype=jnp.float32),
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

        rho_curr, psi_curr = get_local_sdf_rho_and_psi_for_label(
            ego_p_local, target_local, local_pc_valid, r_ego
        )
        rho_z1 = density_wrapper_local(z1_pos)
        rho_z2 = density_wrapper_local(z2_pos)

        dim_total = 6
        lambda_smooth = qp_lambda_smooth
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

        limit = qp_limit
        l_box = -limit * jnp.ones((dim_total,), dtype=jnp.float32)
        u_box =  limit * jnp.ones((dim_total,), dtype=jnp.float32)

        qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
        settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, settings)
        sol_raw = solver.solve().x

        return sol_raw, C_mat, b_vec, rho_curr, psi_curr

    def solve_agent_qp_local(self, ego_p_local, u_nom_local, local_pc_valid, target_local, r_ego, qp_epsilon, qp_lambda_smooth, qp_limit):
        sol_raw_j, C_mat_j, b_vec_j, rho_j, psi_j = self._solve_qp_core_jit(
            ego_p_local,
            u_nom_local,
            local_pc_valid,
            target_local,
            r_ego,
            qp_epsilon,
            qp_lambda_smooth,
            qp_limit,
        )
        G_extracted = np.array(C_mat_j, dtype=np.float32).reshape(1, 6)
        h_extracted = np.array(b_vec_j, dtype=np.float32).reshape(1)
        return sol_raw_j, G_extracted, h_extracted, float(rho_j), float(psi_j)


# =========================================================================
# 3. 全局地图 CDF-QP 专家规划器
# =========================================================================
class GlobalMapCDFPlanner:
    @partial(jit, static_argnums=(0,))
    def _solve_qp_core_jit(self, ego_pose_world, ego_p_local, u_nom_local, all_C, all_d, all_v, target_world, r_ego, qp_epsilon, qp_lambda_smooth, qp_limit):
        epsilon = qp_epsilon
        inv_eps = 1.0 / qp_epsilon

        def density_wrapper_local(p_local):
            rho, _ = get_global_map_density_from_local_query(
                p_local, target_world, all_C, all_d, r_ego, ego_pose_world
            )
            return rho

        def density_wrapper_obs(obs_centers):
            rho, _ = get_global_map_density_from_local_query(
                ego_p_local, target_world, all_C, obs_centers, r_ego, ego_pose_world
            )
            return rho

        # 对当前车体系控制点求梯度；梯度方向仍在局部坐标中。
        grad_self = jax.grad(density_wrapper_local)(ego_p_local)

        # 如果障碍物是动态的，保留 drift 项；静态场景 all_v=0 时该项自然为 0。
        grad_obs = jax.grad(density_wrapper_obs)(all_d)
        drift_term = jnp.sum(grad_obs * all_v)

        norm_nom = jnp.linalg.norm(u_nom_local)
        dir_nom = jnp.where(
            norm_nom > 1e-5,
            u_nom_local / (norm_nom + 1e-8),
            jnp.array([1.0, 0.0], dtype=jnp.float32),
        )
        z1_pos = ego_p_local + epsilon * dir_nom

        neg_grad = -grad_self
        norm_grad = jnp.linalg.norm(neg_grad)
        dir_safe = jnp.where(
            norm_grad > 1e-5,
            neg_grad / (norm_grad + 1e-8),
            jnp.array([0.0, 1.0], dtype=jnp.float32),
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

        rho_curr = density_wrapper_local(ego_p_local)
        rho_z1 = density_wrapper_local(z1_pos)
        rho_z2 = density_wrapper_local(z2_pos)

        dim_total = 6
        lambda_smooth = qp_lambda_smooth
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

        limit = qp_limit
        l_box = -limit * jnp.ones((dim_total,), dtype=jnp.float32)
        u_box =  limit * jnp.ones((dim_total,), dtype=jnp.float32)

        qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
        settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, settings)
        sol_raw = solver.solve().x

        rho_for_debug, psi_for_debug = get_global_map_density_from_local_query(
            ego_p_local, target_world, all_C, all_d, r_ego, ego_pose_world
        )

        return sol_raw, C_mat, b_vec, rho_for_debug, psi_for_debug

    def solve_agent_qp_local(self, ego_pose_world, ego_p_local, u_nom_local, all_C, all_d, all_v, target_world, r_ego, qp_epsilon, qp_lambda_smooth, qp_limit):
        sol_raw_j, C_mat_j, b_vec_j, rho_j, psi_j = self._solve_qp_core_jit(
            ego_pose_world, ego_p_local, u_nom_local, all_C, all_d, all_v, target_world, r_ego, qp_epsilon, qp_lambda_smooth, qp_limit
        )
        G_extracted = np.array(C_mat_j, dtype=np.float32).reshape(1, 6)
        h_extracted = np.array(b_vec_j, dtype=np.float32).reshape(1)
        return sol_raw_j, G_extracted, h_extracted, float(rho_j), float(psi_j)


def u_ctrl_local_to_vw_np(u_ctrl_local, L):
    """
    求解器输出的前两个量不是直接的 [v, omega]，而是控制点局部速度
        u_ctrl_local = [u_x, u_y] = [v, L * omega]
    这里统一做显式转换，避免在主循环里混淆坐标系/控制量。
    """
    u_arr = np.asarray(u_ctrl_local, dtype=np.float32).reshape(-1)
    v = float(u_arr[0])
    omega = float(u_arr[1]) / float(L)
    return v, omega


# =========================================================================
# 4. 物理硬碰撞审计：不用雷达命中点，直接查全局几何
# =========================================================================
def check_collision_world(x, y, meta_obstacles, r_ego):
    min_margin = np.inf
    collided = False

    for obs in meta_obstacles:
        obs_type = obs.get('type', '')
        c_x, c_y = float(obs['center'][0]), float(obs['center'][1])

        if obs_type == 'circle':
            r = float(obs['r'])
            dist = np.hypot(x - c_x, y - c_y)
            margin = dist - (r + r_ego)
            if margin <= 0.0:
                collided = True
            min_margin = min(min_margin, margin)

        elif obs_type in ('rect', 'ellipse'):
            a = float(obs['a']) + r_ego
            b = float(obs['b']) + r_ego
            theta = float(obs.get('theta', obs.get('yaw', 0.0)))
            n = float(obs.get('n', 4.0 if obs_type == 'rect' else 2.0))

            dx = x - c_x
            dy = y - c_y
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            x_rot = dx * cos_t + dy * sin_t
            y_rot = -dx * sin_t + dy * cos_t
            val = (abs(x_rot) / a) ** n + (abs(y_rot) / b) ** n
            margin_like = val - 1.0
            if margin_like <= 0.0:
                collided = True
            min_margin = min(min_margin, margin_like)

    return collided, float(min_margin)

# =========================================================================
# 5. 主批量场景迭代推演控制大闸
# =========================================================================
if __name__ == '__main__':
    # 碰撞审计半径：保持原代码
    R_EGO_COLLISION = 0.31

    # CDF 中使用的膨胀半径：只使用 ego 本体 CDF 半径，不再额外加 0.15。
    R_EGO_CDF = 0.56

    L = 0.33
    dt = 0.05
    total_steps = 2500

    # 固定 EPSILON：CDF 尺度修正后不再使用自适应 epsilon。
    EPSILON = 0.05
    LAMBDA_SMOOTH = 25.0
    QP_LIMIT = 1.2
    NOMINAL_SPEED = 1.2

    # X_frame 的后两维保存上一时刻 v, omega。
    SAVE_PREV_VW_IN_X = False

    # Y_frame 前两维动作标签默认保存“实际执行”的全局 CDF-QP teacher 控制。
    # 如果你后面想让动作标签也完全来自 SDF-CDF-QP，把这里改成 False。
    # 无论这个开关如何，Y_frame 后 7 维 G/h 都来自局部雷达点云 SDF-CDF-QP。
    SAVE_Y_ACTION_FROM_GLOBAL_TEACHER = True

    # 关键帧采样策略：
    # 1) 仍然只在 3m 雷达内存在命中点时才作为候选保存帧；
    # 2) residual > 0.1 判定为关键帧，关键帧全部保存；
    # 3) 非关键帧按 50% 概率保存，用于降低简单帧冗余。
    KEYFRAME_RESIDUAL_THRESHOLD = 0.1
    NON_KEYFRAME_KEEP_PROB = 0.5

    # 每个环境生成多少条目标轨迹：保持 6 条不变。
    # 第一阶段修复：每个环境强制包含 1 条终点 y=0 的直线目标轨迹，
    # 其余 num_demos_per_env-1 条仍然按原来的 y ∈ [-2,2] 随机采样。
    num_demos_per_env = 6
    force_one_y0_target_per_env = True
    forced_y0_demo_id = 0

    # 随机目标范围
    target_x_min, target_x_max = 14.0, 16.0
    target_y_min, target_y_max = -2.0, 2.0

    # 固定随机种子，保证每次生成的数据可复现
    rng = np.random.default_rng(0)

    # 输出数据集名字保持不变
    numrays=32
    output_dataset_path = f"dataset_trajectories_{numrays}.pt"

    control_planner = GlobalMapCDFPlanner()
    sdf_label_planner = LocalSdfCdfLabelPlanner()
    env_pool = get_env_pool()

    global_trajectory_buffer = []

    print("=" * 70)
    print("🚀 启动随机目标轨迹级专家数据生成：Global-Map CDF-QP Teacher")
    print(f"   环境数量: {len(env_pool)}")
    print(f"   每个环境目标轨迹数: {num_demos_per_env}")
    print(f"   每个环境强制 y=0 轨迹: {force_one_y0_target_per_env}, demo_id={forced_y0_demo_id}")
    print(f"   目标x范围: [{target_x_min}, {target_x_max}]")
    print(f"   随机目标y范围: [{target_y_min}, {target_y_max}]；强制轨迹 y=0.0")
    print(f"   控制专家: 全局参数化地图 CDF-QP + 固定 EPSILON")
    print(f"   训练约束标签: 3m 雷达点云 SDF-CDF-QP 的 G/h")
    print(f"   EPSILON={EPSILON}, lambda={LAMBDA_SMOOTH}, limit={QP_LIMIT}, L={L}, R_EGO_CDF={R_EGO_CDF}")
    print(f"   保存观测: 3m 固定圆 256线雷达命中点")
    print(f"   关键帧策略: residual > {KEYFRAME_RESIDUAL_THRESHOLD} 全保存；非关键帧按 {NON_KEYFRAME_KEEP_PROB:.0%} 概率保存")
    print(f"   输出文件: {output_dataset_path}")
    print(f"   理论最大轨迹数: {len(env_pool) * num_demos_per_env}")
    print("=" * 70 + "\n")

    for env_id, env_cfg in enumerate(env_pool):
        print(f"\n==================== 场景 {env_id}/{len(env_pool)-1} ====================")

        meta_obstacles = env_cfg['meta_obstacles']

        # 全局地图参数：控制器和雷达都使用同一份解析障碍物，但两条链路互不混用。
        all_C, all_d, all_v = meta_obstacles_to_global_cdf_tensors(meta_obstacles)

        for demo_id in range(num_demos_per_env):
            # ============================================================
            # 1. 为当前环境随机生成一个目标点
            # ============================================================
            if force_one_y0_target_per_env and demo_id == forced_y0_demo_id:
                # 每个环境固定保留一条终点严格在 y=0 的轨迹，用来覆盖正前方目标/同伦破局样本。
                my_target = np.array([
                    rng.uniform(target_x_min, target_x_max),
                    0.0,
                ], dtype=np.float32)
                target_sample_mode = 'forced_y0_per_env'
            else:
                my_target = np.array([
                    rng.uniform(target_x_min, target_x_max),
                    rng.uniform(target_y_min, target_y_max),
                ], dtype=np.float32)
                target_sample_mode = 'random_uniform_y'

            target_world_j = jnp.array(my_target, dtype=jnp.float32)

            print(
                f"\n⏳ 正在生成轨迹: env={env_id}, demo={demo_id}, "
                f"target=({my_target[0]:.3f}, {my_target[1]:.3f}), "
                f"mode={target_sample_mode}"
            )

            # 起点严格锁定在 [0, 0, 0]
            ego_state = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

            episode_buffer = []
            is_episode_safe = True
            is_episode_success = False
            history_x = []
            history_y = []

            # 当前轨迹的数据筛选统计
            candidate_save_frames = 0
            saved_key_frames = 0
            saved_non_key_frames = 0
            dropped_non_key_frames = 0

            # 上一帧执行控制，用于构造 X_frame。
            # 求解器原生控制量：u_ctrl=[v, L*omega]；
            # 差分/单轨模型控制量：vw=[v, omega]。
            last_executed_u_ctrl_x = 0.0
            last_executed_u_ctrl_y = 0.0
            last_executed_v = 0.0
            last_executed_w = 0.0

            for step in range(total_steps):
                x = float(ego_state[0])
                y = float(ego_state[1])
                theta = float(ego_state[2])

                history_x.append(x)
                history_y.append(y)

                # ========================================================
                # A. 全局几何硬碰撞审计
                # ========================================================
                collided, margin = check_collision_world(x, y, meta_obstacles, R_EGO_COLLISION)
                if collided:
                    is_episode_safe = False
                    print(
                        f"    ❌ [碰撞熔断] env={env_id}, demo={demo_id}, "
                        f"step={step}, margin={margin:.4f}"
                    )
                    break

                # ========================================================
                # B. 保存专用观测：3m 固定圆雷达点云
                # ========================================================
                pos_ego_world = jnp.array([x, y], dtype=jnp.float32)
                local_pc_all, valid_hit_mask = simulate_global_map_lidar_256(
                    pos_ego_world,
                    theta,
                    all_C,
                    all_d,
                    num_rays=numrays,
                    max_range=3.0,
                )

                valid_mask_np = np.array(valid_hit_mask)
                current_pc_save_local = np.array(local_pc_all, dtype=np.float32)[valid_mask_np]
                has_save_points = current_pc_save_local.shape[0] > 0

                # ========================================================
                # C. 局部系相对目标：保持原训练 X 格式
                # ========================================================
                dx_tg = float(my_target[0]) - x
                dy_tg = float(my_target[1]) - y

                target_local_x = dx_tg * np.cos(theta) + dy_tg * np.sin(theta)
                target_local_y = -dx_tg * np.sin(theta) + dy_tg * np.cos(theta)
                target_local = jnp.array([target_local_x, target_local_y], dtype=jnp.float32)

                # ego_p_local=[L,0] 是控制点，用来让求解器输出 u_ctrl=[v, L*omega]。
                # 但目标方向和到达距离按智能体中心计算，不再用 target_local-[L,0]。
                ego_p_local = jnp.array([L, 0.0], dtype=jnp.float32)
                dist_local = jnp.linalg.norm(target_local)

                u_nom_local = jnp.where(
                    dist_local > 0.1,
                    NOMINAL_SPEED * target_local / (dist_local + 1e-6),
                    jnp.zeros(2, dtype=jnp.float32),
                )

                # ========================================================
                # D. 专家控制：全局地图 CDF-QP + 固定 EPSILON
                # ========================================================
                sol_6d_raw, G_global_extracted, h_global_extracted, rho_debug, psi_debug = control_planner.solve_agent_qp_local(
                    ego_state,
                    ego_p_local,
                    u_nom_local,
                    all_C,
                    all_d,
                    all_v,
                    target_world_j,
                    R_EGO_CDF,
                    jnp.array(EPSILON, dtype=jnp.float32),
                    jnp.array(LAMBDA_SMOOTH, dtype=jnp.float32),
                    jnp.array(QP_LIMIT, dtype=jnp.float32),
                )

                u_ctrl_qp_np = np.array(sol_6d_raw[:2], dtype=np.float32)
                v_qp, omega_qp = u_ctrl_local_to_vw_np(u_ctrl_qp_np, L)

                control_residual = np.linalg.norm(
                    np.array(sol_6d_raw[:2], dtype=np.float32) - np.array(u_nom_local, dtype=np.float32)
                )

                # ========================================================
                # E. 数据保存：3m 外不保存；3m 内进入候选保存池
                #    候选帧再按 residual 做关键帧筛选：
                #    - control_residual > 0.1：关键帧，全部保存
                #    - 其他非关键帧：按 50% 概率保存
                # ========================================================
                should_save_frame = False
                is_key_frame = False
                frame_keep_reason = 'no_lidar_hit'

                if has_save_points:
                    candidate_save_frames += 1
                    is_key_frame = bool(control_residual > KEYFRAME_RESIDUAL_THRESHOLD)

                    if is_key_frame:
                        should_save_frame = True
                        frame_keep_reason = 'keyframe_residual_gt_threshold'
                        saved_key_frames += 1
                    else:
                        should_save_frame = bool(rng.random() < NON_KEYFRAME_KEEP_PROB)
                        if should_save_frame:
                            frame_keep_reason = 'non_keyframe_random_keep'
                            saved_non_key_frames += 1
                        else:
                            frame_keep_reason = 'non_keyframe_random_drop'
                            dropped_non_key_frames += 1

                if should_save_frame:
                    if SAVE_PREV_VW_IN_X:
                        # X = [target_x_local, target_y_local, last_v, last_omega]
                        # v/omega 是车体运动学控制量，本身不分世界系/局部系。
                        X_frame = np.array([
                            target_local_x,
                            target_local_y,
                            last_executed_u_ctrl_x,
                            last_executed_u_ctrl_y,
                        ], dtype=np.float32)
                        prev_control_repr = 'vw'
                    else:
                        # 兼容旧数据格式：X 后两维保存上一时刻 u_ctrl=[v, L*omega]
                        X_frame = np.array([
                            target_local_x,
                            target_local_y,
                            last_executed_u_ctrl_x,
                            last_executed_u_ctrl_y,
                        ], dtype=np.float32)
                        prev_control_repr = 'u_ctrl_local'

                    # 标签专用约束：用当前帧 3m 雷达命中点构造 SDF-CDF-QP，
                    # 只取它的 G/h。这样训练出来的约束标签和测试阶段的 SDF-CDF-QP 形式一致。
                    # 注意：轨迹控制仍然使用上面的全局 CDF-QP sol_6d_raw。
                    local_pc_label_j = jnp.array(current_pc_save_local, dtype=jnp.float32)
                    sol_sdf_label_raw, G_sdf_extracted, h_sdf_extracted, rho_sdf_debug, psi_sdf_debug = sdf_label_planner.solve_agent_qp_local(
                        ego_p_local,
                        u_nom_local,
                        local_pc_label_j,
                        target_local,
                        R_EGO_CDF,
                        jnp.array(EPSILON, dtype=jnp.float32),
                        jnp.array(LAMBDA_SMOOTH, dtype=jnp.float32),
                        jnp.array(QP_LIMIT, dtype=jnp.float32),
                    )

                    # Y[:2] 动作标签默认保存实际执行的全局 CDF-QP expert 控制 u_ctrl=[v,L*omega]；
                    # Y[2:] 永远保存 SDF-CDF-QP 基于局部雷达点云计算出的 G/h。
                    if SAVE_Y_ACTION_FROM_GLOBAL_TEACHER:
                        y_action_u0 = float(sol_6d_raw[0])
                        y_action_u1 = float(sol_6d_raw[1])
                        y_action_source = 'global_map_cdf_qp_executed'
                    else:
                        y_action_u0 = float(sol_sdf_label_raw[0])
                        y_action_u1 = float(sol_sdf_label_raw[1])
                        y_action_source = 'local_lidar_sdf_cdf_qp'

                    Y_frame = np.concatenate([
                        [y_action_u0],
                        [y_action_u1],
                        G_sdf_extracted.flatten(),
                        h_sdf_extracted.flatten(),
                    ]).astype(np.float32)

                    episode_buffer.append({
                        'X': X_frame,
                        'Y': Y_frame,
                        'z': np.array(current_pc_save_local, dtype=np.float32),

                        # 调试信息
                        'env_id': env_id,
                        'demo_id': demo_id,
                        'step': step,
                        'target': my_target.astype(np.float32),
                        'target_sample_mode': target_sample_mode,
                        'is_forced_y0_target': bool(target_sample_mode == 'forced_y0_per_env'),
                        'ego_state': np.array([x, y, theta], dtype=np.float32),
                        'control_residual': np.float32(control_residual),
                        'is_key_frame': bool(is_key_frame),
                        'keyframe_residual_threshold': np.float32(KEYFRAME_RESIDUAL_THRESHOLD),
                        'non_keyframe_keep_prob': np.float32(NON_KEYFRAME_KEEP_PROB),
                        'frame_keep_reason': frame_keep_reason,
                        'rho_global_cdf': np.float32(rho_debug),
                        'psi_global_cdf': np.float32(psi_debug),
                        'rho_sdf_label': np.float32(rho_sdf_debug),
                        'psi_sdf_label': np.float32(psi_sdf_debug),
                        'sdf_label_u_ctrl_local': np.array(sol_sdf_label_raw[:2], dtype=np.float32),
                        'G_h_source': 'local_lidar_sdf_cdf_qp',
                        'Y_action_source': y_action_source,
                        'control_source': 'global_map_cdf_qp',
                        'epsilon': np.float32(EPSILON),
                        'prev_control_repr': prev_control_repr,
                        'Y_control_repr': 'u_ctrl_local_[v_Lomega]',
                        'executed_vw': np.array([v_qp, omega_qp], dtype=np.float32),
                        'executed_u_ctrl_local': np.array([float(sol_6d_raw[0]), float(sol_6d_raw[1])], dtype=np.float32),
                    })

                # 更新上一帧执行控制。注意：即使当前帧 3m 外没保存，也必须更新，保证下一个 X_frame 合理。
                last_executed_u_ctrl_x = float(sol_6d_raw[0])
                last_executed_u_ctrl_y = float(sol_6d_raw[1])
                last_executed_v = v_qp
                last_executed_w = omega_qp

                # ========================================================
                # F. 单轴前推
                # ========================================================
                new_x = x + v_qp * np.cos(theta) * dt
                new_y = y + v_qp * np.sin(theta) * dt
                new_theta = theta + omega_qp * dt
                ego_state = jnp.array([new_x, new_y, new_theta], dtype=jnp.float32)

                # ========================================================
                # G. 到达目标判定
                # ========================================================
                dist_to_target = np.hypot(x - my_target[0], y - my_target[1])
                if dist_to_target < 0.44:
                    is_episode_success = True
                    print(
                        f"    🎯 [成功到达] env={env_id}, demo={demo_id}, "
                        f"step={step}, frames={len(episode_buffer)}, "
                        f"key={saved_key_frames}, non_key={saved_non_key_frames}, "
                        f"drop_non_key={dropped_non_key_frames}, "
                        f"candidates={candidate_save_frames}, "
                        f"dist={dist_to_target:.3f}"
                    )
                    break

            # ============================================================
            # 2. 当前轨迹结束后，决定是否保存
            # ============================================================
            if is_episode_safe and is_episode_success and len(episode_buffer) > 0:
                trajectory_record = {
                    'env_id': env_id,
                    'demo_id': demo_id,
                    'target': my_target.astype(np.float32),
                    'target_sample_mode': target_sample_mode,
                    'is_forced_y0_target': bool(target_sample_mode == 'forced_y0_per_env'),
                    'num_frames': len(episode_buffer),
                    'num_candidate_frames': int(candidate_save_frames),
                    'num_key_frames': int(saved_key_frames),
                    'num_non_key_frames': int(saved_non_key_frames),
                    'num_dropped_non_key_frames': int(dropped_non_key_frames),
                    'keyframe_residual_threshold': np.float32(KEYFRAME_RESIDUAL_THRESHOLD),
                    'non_keyframe_keep_prob': np.float32(NON_KEYFRAME_KEEP_PROB),
                    'frames': episode_buffer,
                    'history_x': np.array(history_x, dtype=np.float32),
                    'history_y': np.array(history_y, dtype=np.float32),
                }

                global_trajectory_buffer.append(trajectory_record)

                print(
                    f"    📥 [轨迹保存] env={env_id}, demo={demo_id}, "
                    f"target=({my_target[0]:.3f}, {my_target[1]:.3f}), "
                    f"frames={len(episode_buffer)}, "
                    f"key={saved_key_frames}, non_key={saved_non_key_frames}, "
                    f"drop_non_key={dropped_non_key_frames}, "
                    f"当前总轨迹数={len(global_trajectory_buffer)}"
                )

            else:
                print(
                    f"    ⚠️ [轨迹丢弃] env={env_id}, demo={demo_id}, "
                    f"safe={is_episode_safe}, success={is_episode_success}, "
                    f"frames={len(episode_buffer)}"
                )

    # ============================================================
    # 3. 全部轨迹保存：文件名保持不变
    # ============================================================
    if len(global_trajectory_buffer) > 0:
        print("\n" + "=" * 70)
        print("🎉 随机目标专家轨迹生成完成")
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