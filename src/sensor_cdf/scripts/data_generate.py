#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit
import matplotlib.pyplot as plt
from jaxproxqp.jaxproxqp import JaxProxQP

# =========================================================================
# 1. 核心底层数学函数定义 (原厂函数名与接口签名完全保全)
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
def inverse_bump_corridor(x, C, d_ellipse, wp, r_core):
    """
    结构纯正度 100% 保持。
    C 内部参数解包: [a, b, theta, n, is_obstacle]
    通过 is_obstacle 的值自动且平滑地在 GPU 内部切换外墙或内障碍物数学方程
    """
    x_vec = x.flatten()
    
    # 1. 直接解包显式几何特征 (末尾新增一位类型标识：0为外墙走廊，1为静态障碍物)
    a, b, theta, n, is_obstacle = C[0], C[1], C[2], C[3], C[4]
    x_c, y_c = d_ellipse[0], d_ellipse[1]
    
    # 2. 基础物理空间变换
    dx = x_vec[0] - x_c
    dy = x_vec[1] - y_c
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    x_rot =  dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    
    # 3. 🔴 保持一模一样的 Bump 骨架，通过 JAX 硬件分支条件计算各自的几何安全边界 c_val 和 b_val
    def corridor_branch(_):
        # 走廊类型：自车在内部，边界向内收缩
        ROBOT_RADIUS = 0.2
        a_buf = jnp.maximum(a - ROBOT_RADIUS, 0.1)
        b_buf = jnp.maximum(b - ROBOT_RADIUS, 0.1)
        ellipse_val = (jnp.abs(x_rot) / a_buf) ** n + (jnp.abs(y_rot) / b_buf) ** n
        c_val = 1.0 - ellipse_val
        b_val = r_core**2 - jnp.sum((x_vec - wp.flatten())**2)
        return c_val, b_val

    def obstacle_branch(_):
        # 静态障碍物类型：自车在外部，边界向外膨胀自车半径
        ROBOT_RADIUS = 0.2
        a_buf = a + ROBOT_RADIUS
        b_buf = b + ROBOT_RADIUS
        ellipse_val = (jnp.abs(x_rot) / a_buf) ** n + (jnp.abs(y_rot) / b_buf) ** n
        c_val = ellipse_val - 1.0   # 外部安全区域 (c_val > 0)
        b_val = ellipse_val - 1.5   # 远离感应缓冲区 (b_val > 0 则完全不受阻碍系数影响)
        return c_val, b_val

    # 运用 XLA 级无缝条件原语，确保微分链不断裂
    c_val, b_val = jax.lax.cond(is_obstacle > 0.5, obstacle_branch, corridor_branch, operand=None)
    
    # 4. 纯正的原厂标准 Bump 映射结算
    denom = c_val - b_val
    denom = jnp.where(jnp.abs(denom) < 1e-6, 1e-6, denom)
    m_k = c_val / denom
    
    safe_mk = jnp.clip(m_k, 1e-7, 1.0 - 1e-7)
    exp1 = jnp.exp(-1.0 / safe_mk)
    exp2 = jnp.exp(-1.0 / (1.0 - safe_mk))
    res = exp1 / (exp1 + exp2)
    
    return jnp.where(c_val <= 0, 0.0, jnp.where(b_val <= 0, 1.0, res))

@jit
def get_local_density(
    my_state, all_states, agent_idx, 
    all_C, all_d, all_wp, all_r, current_idx,   
    my_target, weights_corridor, safety_dist_sq, A
):
    curr_C = all_C[current_idx]
    curr_d = all_d[current_idx]
    curr_wp = all_wp[current_idx]
    R_CORE = 2.0  
    
    psi_curr = inverse_bump_corridor(my_state, curr_C, curr_d, curr_wp, R_CORE)
    
    def calc_combined_density(_):
        prev_idx = jnp.maximum(current_idx - 1, 0)
        prev_C = all_C[prev_idx]
        prev_d = all_d[prev_idx]
        prev_wp = all_wp[prev_idx]
        psi_prev = jax.lax.cond(
            current_idx > 0,
            lambda _: inverse_bump_corridor(my_state, prev_C, prev_d, prev_wp, R_CORE),
            lambda _: 0.0,
            operand=None
        )
        return weights_corridor[0] * psi_curr + weights_corridor[1] * psi_prev

    is_leader = (agent_idx == 0)
    is_safe_in_curr = psi_curr > 1e-4
    
    psi_static = jax.lax.cond(
        (is_leader | is_safe_in_curr),
        lambda _: psi_curr,        
        calc_combined_density,     
        operand=None
    )
    
    diff_ag = my_state - all_states 
    dist_ag_sq = jnp.sum(diff_ag**2, axis=1)
    
    R_MIN_SQ = 0.20 ** 2   
    R_SENSE_SQ = 1.50 ** 2 
    c_ag = dist_ag_sq - R_MIN_SQ
    b_ag = dist_ag_sq - R_SENSE_SQ
    
    psi_ag_all = smooth_bump(c_ag, b_ag)
    is_self = jnp.arange(all_states.shape[0]) == agent_idx

    psi_ag_safe = jnp.where(is_self, 1.0, psi_ag_all)
    psi_ag_safe = jnp.where(A > 0, psi_ag_safe, 1.0)
    psi_dynamic = jnp.prod(psi_ag_safe)
    
    dist_target_sq = jnp.sum((my_state - my_target)**2)
    V_x = dist_target_sq
    alpha = 0.5
    
    rho = (psi_static * psi_dynamic) / (V_x ** alpha + 1e-6)
    return rho

# =========================================================================
# 2. 核心类定义
# =========================================================================
class DynamicEnvCDFPlanner:
    @partial(jit, static_argnums=(0,))
    def solve_agent_qp(self, 
                       ego_state, ego_u_nom, 
                       obs_states, obs_velocities,
                       all_C, all_d, all_wp, all_r, current_idx, 
                       my_target, w_corridor):
        safety_dist_sq = 0.6 ** 2 
        epsilon = 0.1
        inv_eps = 1.0 / epsilon

        def density_wrapper_ego(pos_ego):
            combined_states = jnp.vstack([pos_ego.reshape(1, 2), obs_states])
            dummy_A = jnp.ones(combined_states.shape[0])
            return get_local_density(
                pos_ego, combined_states, 0,
                all_C, all_d, all_wp, all_r, current_idx, 
                my_target, w_corridor, safety_dist_sq, dummy_A
            )
            
        def density_wrapper_obs(pure_obs_states):
            combined_states = jnp.vstack([ego_state.reshape(1, 2), pure_obs_states])
            dummy_A = jnp.ones(combined_states.shape[0])
            return get_local_density(
                ego_state, combined_states, 0,
                all_C, all_d, all_wp, all_r, current_idx, 
                my_target, w_corridor, safety_dist_sq, dummy_A
            )

        grad_self = jax.grad(density_wrapper_ego)(ego_state)
        grad_obs = jax.grad(density_wrapper_obs)(obs_states)
        
        drift_term = jnp.sum(grad_obs * obs_velocities)

        norm_nom = jnp.linalg.norm(ego_u_nom)
        dir_nom = jnp.where(norm_nom > 1e-5, ego_u_nom / (norm_nom + 1e-8), jnp.array([1.0, 0.0]))
        z1_pos = ego_state + epsilon * dir_nom

        neg_grad = -grad_self
        norm_grad = jnp.linalg.norm(neg_grad)
        dir_safe = jnp.where(norm_grad > 1e-5, neg_grad / (norm_grad + 1e-8), jnp.array([0.0, 1.0])) 
        z2_pos = ego_state + epsilon * dir_safe

        v1 = dir_nom; v2_raw = dir_safe
        det_v = v1[0] * v2_raw[1] - v1[1] * v2_raw[0]
        is_independent = jnp.abs(det_v) > 1e-2 
        v2_ortho = jnp.array([-v1[1], v1[0]])
        v2 = jnp.where(is_independent, v2_raw, v2_ortho)

        V_mat = jnp.column_stack([v1, v2])
        W_mat = jnp.linalg.inv(V_mat)
        w1 = W_mat[0, :]; w2 = W_mat[1, :]
        
        rho_curr = density_wrapper_ego(ego_state)
        rho_z1 = density_wrapper_ego(z1_pos)
        rho_z2 = density_wrapper_ego(z2_pos)

        dim_total = 6
        lambda_smooth = 15.0
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

        g_vec = jnp.zeros(dim_total).at[0:2].set(-2.0 * ego_u_nom)
        
        sum_w = w1 + w2
        coeff_u = (rho_curr * inv_eps) * sum_w
        coeff_z1 = -(rho_z1 * inv_eps) * w1
        coeff_z2 = -(rho_z2 * inv_eps) * w2

        C_mat = jnp.zeros((1, dim_total)).at[0, 0:2].set(coeff_u).at[0, 2:4].set(coeff_z1).at[0, 4:6].set(coeff_z2)
        b_vec = jnp.zeros(1).at[0].set(drift_term - rho_curr)
        
        limit = 1.0
        l_box = jnp.array([-limit] * 6); u_box = jnp.array([limit] * 6)
        
        qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
        settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, settings)
        return solver.solve().x[:2]

# =========================================================================
# 3. 自动化闭环控制仿真循环（包含精确物理碰撞实时检测监测舱）
# =========================================================================
if __name__ == '__main__':
    print("⚡ [JAX-GPU 模式启动] 正在测试差分小车遭遇‘中央静态微型长方形障碍物’的闭环避障与碰撞监测...")

    # 🔴 载入您最新的测试障碍物尺寸：a=0.5m, b=0.2m, n=4.0
    all_C = jnp.array([[0.5, 0.2, 0.0, 4.0, 1.0]]) 
    all_d = jnp.array([[5.0, 0.0]])                 
    
    all_wp = jnp.array([[10.0, 0.0]])               
    all_r = jnp.array([0.1])
    current_idx = 0
    w_corridor = jnp.array([1.0, 0.0])

    # 智能体非完整运动学初始化
    ego_state = jnp.array([0.0, 0.0, 0.0])          
    my_target = jnp.array([10.0, 0.0])              
    
    # 动态迎面障碍物配置
    obs_states = jnp.array([[8.5, 0.15]])            
    obs_velocities = jnp.array([[-0.08, -0.01]])      

    dt = 0.1                                        
    total_steps = 1800                               # 扩展步数确保观测完整周期
    L = 0.15                                        # 前瞻点控制臂长

    # 设备内部非阻塞数组舱
    ego_hist = []
    obs_hist = []

    # 🔴 实时碰撞数据监测舱 (存储发生碰撞时小车中心的 [x, y] 坐标)
    static_collisions = []
    dynamic_collisions = []

    planner = DynamicEnvCDFPlanner()

    # 首次调用触发 XLA 全局冷编译
    print("⏳ 正在进行 XLA 全局算子融合编译...")
    dummy_p = jnp.array([0.0, 0.0])
    dummy_nom = jnp.array([0.0, 0.0])
    _ = planner.solve_agent_qp(dummy_p, dummy_nom, obs_states, obs_velocities, all_C, all_d, all_wp, all_r, current_idx, my_target, w_corridor)
    print("✨ 编译大功告成！小车开始向前推进并实时监测冲突。")

    import time
    t_start = time.time()

    for step in range(total_steps):
        ego_hist.append(ego_state)
        obs_hist.append(obs_states[0])

        # 转换为显式浮点数仅用于 Python 层的物理碰撞安全判定，绝不写回 GPU 变量，不阻塞大流程
        x, y, theta = float(ego_state[0]), float(ego_state[1]), float(ego_state[2])
        obs_x, obs_y = float(obs_states[0, 0]), float(obs_states[0, 1])

        # -----------------------------------------------------------------
        # 🔴 核心新增：物理碰撞检测器 (车体半径 R=0.2m)
        # -----------------------------------------------------------------
        ROBOT_R = 0.2
        obs_x_c, obs_y_c = 5.0, 0.0
        obs_a, obs_b = 0.5, 0.2
        
        # 计算小车中心到标准静态长方形障碍物边缘的最短距离
        dx_rect = max(abs(x - obs_x_c) - obs_a, 0.0)
        dy_rect = max(abs(y - obs_y_c) - obs_b, 0.0)
        dist_to_static_surface = (dx_rect**2 + dy_rect**2)**0.5
        
        # 判定1：若小车中心在矩形内，或者车体外壳触碰矩形边界，记录静态碰撞
        is_center_inside = (abs(x - obs_x_c) <= obs_a) and (abs(y - obs_y_c) <= obs_b)
        if is_center_inside or (dist_to_static_surface < ROBOT_R):
            static_collisions.append([x, y])

        # 判定2：监测与动态障碍物的物理圆形外壳冲突 (按R_MIN=0.2m死区判定)
        dist_to_dynamic = ((x - obs_x)**2 + (y - obs_y)**2)**0.5
        if dist_to_dynamic < ROBOT_R:
            dynamic_collisions.append([x, y])

        # -----------------------------------------------------------------
        # 标准控制推进流
        # -----------------------------------------------------------------
        ego_p = jnp.array([x + L * jnp.cos(theta), y + L * jnp.sin(theta)])
        target_vector = my_target - ego_p
        dist_to_goal = jnp.linalg.norm(target_vector)
        ego_u_nom = jnp.where(dist_to_goal > 0.1, 0.5 * target_vector / (dist_to_goal + 1e-6), jnp.zeros(2))

        u_opt = planner.solve_agent_qp(
            ego_p, ego_u_nom, obs_states, obs_velocities,
            all_C, all_d, all_wp, all_r, current_idx, my_target, w_corridor
        )

        v = u_opt[0] * jnp.cos(theta) + u_opt[1] * jnp.sin(theta)
        omega = (-u_opt[0] * jnp.sin(theta) + u_opt[1] * jnp.cos(theta)) / L

        new_x = x + v * jnp.cos(theta) * dt
        new_y = y + v * jnp.sin(theta) * dt
        new_theta = theta + omega * dt
        ego_state = jnp.array([new_x, new_y, new_theta])
        
        obs_states = obs_states + obs_velocities * dt

        if jnp.linalg.norm(my_target - ego_state[:2]) < 0.2:
            print(f"🎉 差分小车于第 {step} 步安全滑入终点控制圈！")
            break

    print(f"🚀 闭环全流程总体耗时: {time.time() - t_start:.4f} 秒！")
    print(f"📊 [冲突统计] 静态长方形碰撞状态数: {len(static_collisions)} 步 | 动态冲突状态数: {len(dynamic_collisions)} 步")

    # =========================================================================
    # 4. 渲染高清轨迹与冲突散点图
    # =========================================================================
    ego_hist_np = np.array(ego_hist)
    obs_hist_np = np.array(obs_hist)

    plt.figure(figsize=(11, 5))
    
    # 🔴 依据您最新的中心(5,0)与a=0.5, b=0.2，精确画出长方形障碍物边界
    # 左下角横坐标: 5.0 - 0.5 = 4.5; 纵坐标: 0.0 - 0.2 = -0.2. 宽=1.0, 高=0.4
    rect = plt.Rectangle((4.5, -0.2), 1.0, 0.4, fill=True, color='gray', alpha=0.3, hatch='//', label='Static Rectangle Obstacle (a=0.5, b=0.2)')
    plt.gca().add_patch(rect)

    # 绘制基础轨迹线
    plt.plot(ego_hist_np[:, 0], ego_hist_np[:, 1], 'b-', linewidth=2.5, label='Robot Center Path (Look-ahead)')
    plt.plot(obs_hist_np[:, 0], obs_hist_np[:, 1], 'r--', linewidth=1.5, label='Dynamic Obstacle Path')
    
    # 标注起点与终点
    plt.scatter(0.0, 0.0, color='blue', marker='o', s=100, label='Start (0,0)', zorder=4)
    plt.scatter(my_target[0], my_target[1], color='darkgreen', marker='X', s=150, label='Target (10,0)', zorder=5)

    # 🔴 核心绘制：如果发生碰撞，在图中使用明显的红色闪烁星号精确标记碰撞发生时的小车中心位置
    if len(static_collisions) > 0:
        sc_np = np.array(static_collisions)
        plt.scatter(sc_np[:, 0], sc_np[:, 1], color='crimson', marker='*', s=60, label='Static Collision Points', zorder=6)
    if len(dynamic_collisions) > 0:
        dc_np = np.array(dynamic_collisions)
        plt.scatter(dc_np[:, 0], dc_np[:, 1], color='darkorange', marker='x', s=50, label='Dynamic Collision Points', zorder=6)

    # 全自适应轨迹采样，均匀绘制 6 帧带车头箭头的物理蓝色外壳
    sample_indices = np.linspace(0, len(ego_hist_np) - 1, 6, dtype=int)
    for idx in sample_indices:
        th = ego_hist_np[idx, 2]
        plt.arrow(ego_hist_np[idx, 0], ego_hist_np[idx, 1], 0.18*np.cos(th), 0.18*np.sin(th), head_width=0.06, head_length=0.06, fc='darkblue', ec='darkblue', zorder=3)
        robot_shell = plt.Circle((ego_hist_np[idx, 0], ego_hist_np[idx, 1]), ROBOT_R, color='blue', fill=True, alpha=0.06, zorder=2)
        plt.gca().add_patch(robot_shell)

    plt.title("Expert Dataset Diagnosis: Look-ahead Trajectory with Collision Mapping", fontsize=11, fontweight='bold')
    plt.xlabel("World X-Coordinate (m)", fontsize=10)
    plt.ylabel("World Y-Coordinate (m)", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.xlim(-0.5, 10.5)
    plt.ylim(-2.0, 2.0)
    plt.legend(loc='upper right', shadow=True, fontsize=9)
    plt.show()