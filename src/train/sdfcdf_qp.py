#!/usr/bin/env python3
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
from jaxproxqp.jaxproxqp import JaxProxQP
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times']
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
from env import get_env_pool


# =========================================================================
# 0. 场景障碍物 -> 全局参数化 CDF 地图
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
    统一转成超椭圆参数：
        C = [a, b, theta, n]
        d = [cx, cy]
        v = [vx, vy]

    rect   -> n=4（和你 png 版本的圆角矩形超椭圆保持一致）
    circle -> a=b=r, n=2
    ellipse-> n=2（若场景里以后有这种类型，也能直接跑）
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
        all_C = [[0.1, 0.1, 0.0, 2.0]]
        all_d = [[999.0, 999.0]]
        all_v = [[0.0, 0.0]]

    return (
        jnp.array(all_C, dtype=jnp.float32),
        jnp.array(all_d, dtype=jnp.float32),
        jnp.array(all_v, dtype=jnp.float32),
    )


# =========================================================================
# 1. 全局参数化 CDF 核心函数
#    坐标检查版：密度查询变量在车体系，JAX 链式法则自动给出局部梯度。
#    单障碍物作用域是以障碍物中心为圆心的 3m 圆。
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
    ], dtype=jnp.float32)


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
    查询变量仍放在车体系 p_local，这样 QP 输出还是 [ux_local, uy_local]。
    但密度评估会先映射到世界系，再使用参数化地图计算 CDF。

    注意：这里不做任何障碍物筛选。
    整个地图中的所有障碍物都参与 CDF，单个障碍物的作用域已经在 obstacle_bump_field() 内
    通过“以该障碍物中心为圆心、半径 3m 的圆形 b_val”定义。
    """
    p_world = local_to_world_point(p_local, ego_pose_world)

    def single_obs_density(C_obs, d_obs):
        return obstacle_bump_field(p_world, C_obs, d_obs, r_ego)

    # 整张地图一次性构造 CDF：所有障碍物都参与乘积，不根据 ego 与障碍物距离做过滤。
    psi_array = jax.vmap(single_obs_density)(all_C, all_d)
    psi_static = jnp.prod(psi_array)

    V_x = jnp.sum((p_world - target_world) ** 2)
    alpha = 0.5
    rho = psi_static / (V_x ** alpha + 1e-6)
    return rho, psi_static


# =========================================================================
# 2. 全局地图 CDF-QP 规划器
# =========================================================================
class GlobalMapCDFPlanner:
    @partial(jit, static_argnums=(0,))
    def solve_agent_qp_local(self, ego_pose_world, ego_p_local, u_nom_local, all_C, all_d, all_v, target_world, r_ego, qp_epsilon, qp_lambda_smooth, qp_limit):
        epsilon = qp_epsilon
        inv_eps = 1.0 / qp_epsilon
        lambda_smooth = qp_lambda_smooth
        limit = qp_limit

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

        grad_self = jax.grad(density_wrapper_local)(ego_p_local)
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
        v2_ortho = jnp.array([-v1[1], v1[0]], dtype=jnp.float32)
        v2 = jnp.where(is_independent, v2_raw, v2_ortho)

        V_mat = jnp.column_stack([v1, v2])
        W_mat = jnp.linalg.inv(V_mat)
        w1 = W_mat[0, :]
        w2 = W_mat[1, :]

        rho_curr = density_wrapper_local(ego_p_local)
        rho_z1 = density_wrapper_local(z1_pos)
        rho_z2 = density_wrapper_local(z2_pos)

        dim_total = 6
        H = jnp.zeros((dim_total, dim_total), dtype=jnp.float32)
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

        g_vec = jnp.zeros(dim_total, dtype=jnp.float32).at[0:2].set(-2.0 * u_nom_local)

        sum_w = w1 + w2
        coeff_u = (rho_curr * inv_eps) * sum_w
        coeff_z1 = -(rho_z1 * inv_eps) * w1
        coeff_z2 = -(rho_z2 * inv_eps) * w2

        C_mat = (
            jnp.zeros((1, dim_total), dtype=jnp.float32)
            .at[0, 0:2].set(coeff_u)
            .at[0, 2:4].set(coeff_z1)
            .at[0, 4:6].set(coeff_z2)
        )
        b_vec = jnp.zeros(1, dtype=jnp.float32).at[0].set(drift_term - rho_curr)

        l_box = -limit * jnp.ones((dim_total,), dtype=jnp.float32)
        u_box =  limit * jnp.ones((dim_total,), dtype=jnp.float32)

        qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
        settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, settings)
        sol_raw = solver.solve().x

        rho_debug, psi_debug = get_global_map_density_from_local_query(
            ego_p_local, target_world, all_C, all_d, r_ego, ego_pose_world
        )

        return sol_raw, rho_debug, psi_debug


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
# 3. 硬碰撞审计
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
# 4.5 自适应 EPSILON：多候选 CDF-QP + 短时 rollout 打分选择
# =========================================================================
def signed_clearance_to_obstacles(px, py, meta_obstacles, r_ego):
    """
    返回 ego 圆心到所有障碍物的最小安全余量。
    > 0 安全，= 0 接触，< 0 碰撞。

    注意：这里用于短预测打分，不改变 CDF 构造。
    CDF 仍然对整张地图所有障碍物做乘积，不做障碍物过滤。
    """
    min_clearance = 1e6

    for obs in meta_obstacles:
        obs_type = obs.get('type', '')
        c_x, c_y = float(obs['center'][0]), float(obs['center'][1])

        if obs_type == 'circle':
            clearance = np.hypot(px - c_x, py - c_y) - float(obs['r']) - r_ego

        elif obs_type == 'rect':
            # 场景里的 rect 目前主要是轴对齐矩形；这里用矩形几何距离做 rollout 安全打分。
            # 若以后有旋转矩形，建议把点先旋回障碍物局部系后再计算 qx/qy。
            theta = float(obs.get('theta', obs.get('yaw', 0.0)))
            dx = px - c_x
            dy = py - c_y
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            x_rot = dx * cos_t + dy * sin_t
            y_rot = -dx * sin_t + dy * cos_t

            qx = abs(x_rot) - float(obs['a'])
            qy = abs(y_rot) - float(obs['b'])
            outside_dist = np.hypot(max(qx, 0.0), max(qy, 0.0))
            inside_dist = min(max(qx, qy), 0.0)
            clearance = outside_dist + inside_dist - r_ego

        elif obs_type == 'ellipse':
            # 对 ellipse 使用保守近似：中心距离减去最大半轴和 ego 半径。
            clearance = np.hypot(px - c_x, py - c_y) - max(float(obs['a']), float(obs['b'])) - r_ego

        else:
            continue

        min_clearance = min(min_clearance, clearance)

    return float(min_clearance)


def rollout_score_for_candidate(
    ego_state_np,
    u_local_np,
    u_nom_np,
    my_target_np,
    meta_obstacles,
    r_ego,
    L,
    dt,
    rollout_steps=12,
):
    """
    用候选控制短时间前推，选择更安全、更有进展、更不激进的 EPSILON。
    这只是候选选择器，不参与 CDF-QP 约束构造。
    """
    x = float(ego_state_np[0])
    y = float(ego_state_np[1])
    theta = float(ego_state_np[2])

    dist0 = np.hypot(x - my_target_np[0], y - my_target_np[1])
    min_clearance = 1e6

    v = float(u_local_np[0])
    omega = float(u_local_np[1] / L)

    for _ in range(rollout_steps):
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        clearance = signed_clearance_to_obstacles(x, y, meta_obstacles, r_ego)
        min_clearance = min(min_clearance, clearance)

        if clearance <= 0.0:
            return 1e6 + 1e4 * abs(clearance)

    dist_final = np.hypot(x - my_target_np[0], y - my_target_np[1])
    progress = dist0 - dist_final

    # clearance 越小，惩罚越大；加 0.08 防止数值爆炸。
    obstacle_penalty = 0.25 / max(min_clearance + 0.08, 1e-3)
    turn_penalty = 0.03 * abs(float(u_local_np[1]))
    residual_penalty = 0.04 * np.linalg.norm(u_local_np - u_nom_np)
    no_progress_penalty = 2.0 * max(-progress, 0.0)

    return float(dist_final + obstacle_penalty + turn_penalty + residual_penalty + no_progress_penalty)


# =========================================================================
# 4. 绘图辅助
# =========================================================================
def draw_obstacles(ax, meta_obstacles):
    for idx, obs in enumerate(meta_obstacles):
        obs_type = obs.get('type', '')

        if obs_type == 'rect':
            theta = float(obs.get('theta', obs.get('yaw', 0.0)))
            if abs(theta) < 1e-8:
                rect_patch = Rectangle(
                    (obs['center'][0] - obs['a'], obs['center'][1] - obs['b']),
                    obs['a'] * 2,
                    obs['b'] * 2,
                    fill=True,
                    color='gray',
                    alpha=0.3,
                    hatch='//',
                    label='Static obstacle' if idx == 0 else ""
                )
                ax.add_patch(rect_patch)
            else:
                ellipse_patch = Ellipse(
                    xy=(obs['center'][0], obs['center'][1]),
                    width=2 * obs['a'],
                    height=2 * obs['b'],
                    angle=np.degrees(theta),
                    fill=True,
                    color='gray',
                    alpha=0.3,
                    hatch='//',
                    label='Rotated Superellipse Approx.' if idx == 0 else ""
                )
                ax.add_patch(ellipse_patch)

        elif obs_type == 'circle':
            circle_patch = Circle(
                (obs['center'][0], obs['center'][1]),
                obs['r'],
                fill=True,
                color='dimgray',
                alpha=0.3,
                hatch='\\\\',
                label='Static obstacle' if idx == 0 else ""
            )
            ax.add_patch(circle_patch)

        elif obs_type == 'ellipse':
            ellipse_patch = Ellipse(
                xy=(obs['center'][0], obs['center'][1]),
                width=2 * obs['a'],
                height=2 * obs['b'],
                angle=np.degrees(float(obs.get('theta', obs.get('yaw', 0.0)))),
                fill=True,
                color='gray',
                alpha=0.3,
                hatch='//',
                label='Ellipse' if idx == 0 else ""
            )
            ax.add_patch(ellipse_patch)


# =========================================================================
# 5. 单场景调试主程序
# =========================================================================
if __name__ == '__main__':
    # ---------------------------------------------------------------------
    # 你平时只需要改这里
    # ---------------------------------------------------------------------
    ENV_ID = 0

    # 如果想直接测试不同目标位置，就把它改成 False，并手动指定 MANUAL_TARGET
    USE_ENV_TARGET = False
    MANUAL_TARGET = np.array([14.08, -0.81], dtype=np.float32)

    # QP / 轨迹参数：EPSILON 不再固定，每一步在候选里自动选择
    EPSILON_CANDIDATES = [0.05, 0.05]
    LAMBDA_SMOOTH = 25.0
    QP_LIMIT = 1.2
    NOMINAL_SPEED = 1.2
    ROLLOUT_STEPS = 0
    EPS_SWITCH_PENALTY = 0.08

    # 运动学参数
    R_EGO_COLLISION = 0.31
    R_EGO_CDF = 0.55
    L = 0.31
    dt = 0.05
    total_steps = 2500
    SUCCESS_RADIUS = 0.43

    # 输出路径和文件名保持不变
    target_dir = "/home/guo/L-CDF/src/train/dataset_env"

    # ---------------------------------------------------------------------
    env_pool = get_env_pool()
    print(f"📊 当前环境池总计包含 {len(env_pool)} 个场景 (0 ~ {len(env_pool)-1})。")
    print(f"\n⏳ 正在拉起 [场景 {ENV_ID}] 的 Global-Map CDF-QP 单场景仿真...")

    env_cfg = env_pool[ENV_ID]
    meta_obstacles = env_cfg['meta_obstacles']

    if USE_ENV_TARGET:
        my_target = np.array(env_cfg['target'], dtype=np.float32)
    else:
        my_target = MANUAL_TARGET.astype(np.float32)

    print(f"🎯 当前目标点: ({my_target[0]:.3f}, {my_target[1]:.3f})")
    print(f"⚙️  参数: eps_candidates={EPSILON_CANDIDATES}, lambda_smooth={LAMBDA_SMOOTH}, qp_limit={QP_LIMIT}, u_nom={NOMINAL_SPEED}")
    print("📐 坐标约定: QP 求解 u_ctrl_local=[v, L*omega]，轨迹积分前显式转换成 [v, omega]。")

    all_C, all_d, all_v = meta_obstacles_to_global_cdf_tensors(meta_obstacles)
    target_world_j = jnp.array(my_target, dtype=jnp.float32)

    planner = GlobalMapCDFPlanner()

    ego_state = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    traj_x, traj_y = [], []
    traj_residuals = []
    rho_hist = []
    psi_hist = []
    epsilon_hist = []
    is_collided = False
    is_success = False
    last_used_epsilon = EPSILON_CANDIDATES[1]

    for step in range(total_steps):
        x = float(ego_state[0])
        y = float(ego_state[1])
        theta = float(ego_state[2])

        traj_x.append(x)
        traj_y.append(y)

        # 1) 硬碰撞审计
        collided, margin = check_collision_world(x, y, meta_obstacles, R_EGO_COLLISION)
        if collided:
            is_collided = True
            print(f"💥 [碰撞熔断] 场景 {ENV_ID} 在第 {step} 步发生碰撞，margin={margin:.4f}。")
            break

        # 2) 目标点转换到局部系
        dx_tg = float(my_target[0]) - x
        dy_tg = float(my_target[1]) - y
        target_local_x = dx_tg * np.cos(theta) + dy_tg * np.sin(theta)
        target_local_y = -dx_tg * np.sin(theta) + dy_tg * np.cos(theta)
        target_local = jnp.array([target_local_x, target_local_y], dtype=jnp.float32)

        ego_p_local = jnp.array([L, 0.0], dtype=jnp.float32)
        dist_local = jnp.linalg.norm(target_local)
        u_nom_local = jnp.where(
            dist_local > 0.1,
            NOMINAL_SPEED * target_local / (dist_local + 1e-6),
            jnp.zeros(2, dtype=jnp.float32)
        )

        # 3) 全局参数化地图 CDF-QP 控制：多 EPSILON 候选 + 短预测选择
        best_score = 1e18
        best_eps = None
        sol_6d_raw = None
        rho_curr = None
        psi_curr = None

        ego_state_np = np.array([x, y, theta], dtype=np.float32)
        u_nom_np = np.array(u_nom_local, dtype=np.float32)

        for eps_candidate in EPSILON_CANDIDATES:
            cand_sol, cand_rho, cand_psi = planner.solve_agent_qp_local(
                ego_state,
                ego_p_local,
                u_nom_local,
                all_C,
                all_d,
                all_v,
                target_world_j,
                R_EGO_CDF,
                jnp.array(eps_candidate, dtype=jnp.float32),
                jnp.array(LAMBDA_SMOOTH, dtype=jnp.float32),
                jnp.array(QP_LIMIT, dtype=jnp.float32),
            )
            cand_u_np = np.array(cand_sol[:2], dtype=np.float32)
            score = rollout_score_for_candidate(
                ego_state_np=ego_state_np,
                u_local_np=cand_u_np,
                u_nom_np=u_nom_np,
                my_target_np=my_target,
                meta_obstacles=meta_obstacles,
                r_ego=R_EGO_COLLISION,
                L=L,
                dt=dt,
                rollout_steps=ROLLOUT_STEPS,
            )
            score += EPS_SWITCH_PENALTY * abs(eps_candidate - last_used_epsilon) / 0.05

            if score < best_score:
                best_score = score
                best_eps = eps_candidate
                sol_6d_raw = cand_sol
                rho_curr = cand_rho
                psi_curr = cand_psi

        used_epsilon = float(best_eps)
        last_used_epsilon = used_epsilon

        residual = np.linalg.norm(
            np.array(sol_6d_raw[:2], dtype=np.float32) - np.array(u_nom_local, dtype=np.float32)
        )
        traj_residuals.append(float(residual))
        rho_hist.append(float(rho_curr))
        psi_hist.append(float(psi_curr))
        epsilon_hist.append(used_epsilon)

        u_ctrl_np = np.array(sol_6d_raw[:2], dtype=np.float32)
        v, omega = u_ctrl_local_to_vw_np(u_ctrl_np, L)

        # 4) 单轨模型前推
        new_x = x + v * np.cos(theta) * dt
        new_y = y + v * np.sin(theta) * dt
        new_theta = theta + omega * dt
        ego_state = jnp.array([new_x, new_y, new_theta], dtype=jnp.float32)

        # 5) 成功判定
        dist_to_target = np.hypot(x - my_target[0], y - my_target[1])
        if dist_to_target < SUCCESS_RADIUS:
            is_success = True
            print(f"🎉 场景 {ENV_ID} 顺利通关！于第 {step} 步安全抵达终点。")
            break

    # =========================================================================
    # 6. 画图与保存（名字和路径保持不变）
    # =========================================================================
    traj_x = np.array(traj_x, dtype=np.float32)
    traj_y = np.array(traj_y, dtype=np.float32)
    residuals = np.array(traj_residuals, dtype=np.float32)
    if len(epsilon_hist) > 0:
        unique_eps, eps_counts = np.unique(np.array(epsilon_hist, dtype=np.float32), return_counts=True)
        print("📌 EPSILON 使用统计:", {float(k): int(v) for k, v in zip(unique_eps, eps_counts)})

    fig, ax = plt.subplots(figsize=(11, 6))

    draw_obstacles(ax, meta_obstacles)

    # 轨迹分段渲染：残差大代表 QP 明显介入避障
    for i in range(len(traj_x) - 1):
        x_seg = [traj_x[i], traj_x[i + 1]]
        y_seg = [traj_y[i], traj_y[i + 1]]
        if i < len(residuals) and residuals[i] > 5.1:
            ax.plot(x_seg, y_seg, color='crimson', linewidth=3.5, zorder=3)
        else:
            ax.plot(x_seg, y_seg, color='red', linewidth=2.0, zorder=2)

    if is_collided and len(traj_x) > 0:
        ego_crash_circle = Circle(
            (traj_x[-1], traj_y[-1]),
            R_EGO_COLLISION,
            fill=False,
            color='black',
            linestyle='--',
            linewidth=2.0,
            label='Crash Outer Boundary'
        )
        ax.add_patch(ego_crash_circle)
        ax.scatter(traj_x[-1], traj_y[-1], color='black', marker='X', s=150, zorder=5, label='Collision Point')

    if len(traj_x) > 0:
        ax.scatter(traj_x[0], traj_y[0], color='green', marker='o', s=120, zorder=5, label='Start (0,0)')
    ax.scatter(my_target[0], my_target[1], color='gold', marker='*', s=200, edgecolor='orange', zorder=5,
               label=f'Target ({my_target[0]:.2f},{my_target[1]:.2f})')

    title_status = 'Success' if is_success else ('Collision' if is_collided else 'Stopped')
    ax.set_xlabel("X (m)", fontsize=20)
    ax.set_ylabel("Y (m)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # ax.set_title(
    #     f"Single-Scene Debugger (Scene ID: {ENV_ID}) - Global CDF-QP Trajectory [{title_status}]\n"
    #     f"Target=({my_target[0]:.2f},{my_target[1]:.2f}), eps_adaptive={sorted(set(np.round(epsilon_hist, 3))) if len(epsilon_hist) > 0 else []}, lambda={LAMBDA_SMOOTH}, limit={QP_LIMIT}",
    #     fontsize=12,
    #     fontweight='bold'
    # )
    # ax.grid(False, linestyle=':', alpha=0.6)
    ax.grid(False)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.0, 16.0)
    ax.set_ylim(-2.0, 2.0)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    file_name = f'debug_trajectory_scene_{ENV_ID}.png'
    absolute_output_path = os.path.join(target_dir, file_name)

    plt.tight_layout()
    plt.savefig(
        absolute_output_path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.02,
    )
    plt.close()

    print(f"🖼️ 图片已保存到: {absolute_output_path}")
    print(f"📈 轨迹长度: {len(traj_x)} steps")
    if len(rho_hist) > 0:
        print(f"ρ范围: [{np.min(rho_hist):.4f}, {np.max(rho_hist):.4f}]")
        print(f"ψ范围: [{np.min(psi_hist):.4f}, {np.max(psi_hist):.4f}]")
