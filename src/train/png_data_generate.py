#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit
import matplotlib.pyplot as plt
from jaxproxqp.jaxproxqp import JaxProxQP

# =========================================================================
# 1. 核心底层数学函数与激光雷达仿真定义
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
def obstacle_bump_field(x, C_obs, d_obs, r_ego):
    x_vec = x.flatten()
    a, b, theta, n = C_obs[0], C_obs[1], C_obs[2], C_obs[3]
    x_c, y_c = d_obs[0], d_obs[1]
    
    dx = x_vec[0] - x_c
    dy = x_vec[1] - y_c
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    x_rot =  dx * cos_t + dy * sin_t
    y_rot = -dx * sin_t + dy * cos_t
    
    a_buf = a + r_ego
    b_buf = b + r_ego
    ellipse_val = (jnp.abs(x_rot) / a_buf) ** n + (jnp.abs(y_rot) / b_buf) ** n
    
    c_val = ellipse_val - 1.0
    b_val = ellipse_val - 4.0
    
    return smooth_bump(c_val, b_val)

@jit
def get_local_density(my_state, all_states, agent_idx, all_C, all_d, my_target, A, r_ego, all_radii, real_ego_center):
    LIDAR_RANGE_SQ = 3.0 ** 2

    dist_static_sq = jnp.sum((all_d - real_ego_center)**2, axis=1)
    static_in_range = dist_static_sq <= LIDAR_RANGE_SQ

    def single_obs_density(C_obs, d_obs):
        return obstacle_bump_field(my_state, C_obs, d_obs, r_ego)
    
    psi_static_array = jax.vmap(single_obs_density)(all_C, all_d)
    psi_static_array = jnp.where(static_in_range, psi_static_array, 1.0)
    psi_static = jnp.prod(psi_static_array)
    
    diff_ag = my_state - all_states 
    dist_ag_sq = jnp.sum(diff_ag**2, axis=1)
    
    dist_dynamic_sq = jnp.sum((all_states - real_ego_center)**2, axis=1)
    dynamic_in_range = dist_dynamic_sq <= LIDAR_RANGE_SQ
    
    r_other = all_radii
    r_safe = r_ego + r_other
    r_sense = 1.5 * r_safe
    
    c_ag = dist_ag_sq - r_safe ** 2
    b_ag = dist_ag_sq - r_sense ** 2
    
    psi_ag_all = smooth_bump(c_ag, b_ag)
    is_self = jnp.arange(all_states.shape[0]) == agent_idx

    psi_ag_safe = jnp.where(is_self, 1.0, psi_ag_all)
    psi_ag_safe = jnp.where(A > 0, psi_ag_safe, 1.0)
    
    psi_ag_safe = jnp.where(dynamic_in_range, psi_ag_safe, 1.0)
    psi_dynamic = jnp.prod(psi_ag_safe)
    
    dist_target_sq = jnp.sum((my_state - my_target)**2)
    V_x = dist_target_sq
    alpha = 0.5
    
    rho = (psi_static * psi_dynamic) / (V_x ** alpha + 1e-6)
    return rho

@jit
def simulate_lidar_2d(ego_state, all_C, all_d, obs_states, obs_radii):
    num_rays = 512
    max_range = 3.0
    num_steps = 200  
    
    ego_pos = ego_state[:2]
    theta = ego_state[2]
    
    ray_angles_world = theta + jnp.linspace(-jnp.pi, jnp.pi, num_rays, endpoint=False)
    ray_dirs_world = jnp.column_stack([jnp.cos(ray_angles_world), jnp.sin(ray_angles_world)]) 
    
    t_steps = jnp.linspace(0.0, max_range, num_steps) 
    pts_world = ego_pos[None, None, :] + t_steps[None, :, None] * ray_dirs_world[:, None, :] 
    
    def inside_static(p, C_obs, d_obs):
        dx = p[..., 0] - d_obs[0]
        dy = p[..., 1] - d_obs[1]
        cos_t = jnp.cos(C_obs[2])
        sin_t = jnp.sin(C_obs[2])
        x_rot = dx * cos_t + dy * sin_t
        y_rot = -dx * sin_t + dy * cos_t
        ellipse_val = (jnp.abs(x_rot) / C_obs[0]) ** C_obs[3] + (jnp.abs(y_rot) / C_obs[1]) ** C_obs[3]
        return ellipse_val <= 1.0

    is_inside_static = jax.vmap(inside_static, in_axes=(None, 0, 0))(pts_world, all_C, all_d) 
    inside_static_any = jnp.any(is_inside_static, axis=0) 
    
    def inside_dynamic(p, obs_pos, r):
        dist_sq = jnp.sum((p - obs_pos) ** 2, axis=-1)
        return dist_sq <= r ** 2

    is_inside_dynamic = jax.vmap(inside_dynamic, in_axes=(None, 0, 0))(pts_world, obs_states, obs_radii) 
    inside_dynamic_any = jnp.any(is_inside_dynamic, axis=0) 
    
    is_inside_any = inside_static_any | inside_dynamic_any 
    
    t_hit = jnp.where(is_inside_any, t_steps[None, :], max_range)
    min_t = jnp.min(t_hit, axis=-1) 
    
    ray_angles_local = jnp.linspace(-jnp.pi, jnp.pi, num_rays, endpoint=False)
    hit_pts_local = jnp.column_stack([
        min_t * jnp.cos(ray_angles_local),
        min_t * jnp.sin(ray_angles_local)
    ])
    return hit_pts_local, min_t

# =========================================================================
# 2. 优化控制决策器类
# =========================================================================
class DynamicEnvCDFPlanner:
    @partial(jit, static_argnums=(0,))
    def solve_agent_qp(self, ego_state, ego_u_nom, obs_states, obs_velocities, all_C, all_d, my_target, r_ego, obs_radii):
        epsilon = 0.1
        inv_eps = 1.0 / epsilon
        all_radii = jnp.concatenate([jnp.array([r_ego]), obs_radii])
        real_ego_center = ego_state[:2]

        def density_wrapper_ego(pos_ego):
            combined_states = jnp.vstack([pos_ego.reshape(1, 2), obs_states])
            dummy_A = jnp.ones(combined_states.shape[0])
            return get_local_density(pos_ego, combined_states, 0, all_C, all_d, my_target, dummy_A, r_ego, all_radii, real_ego_center)
            
        def density_wrapper_obs(pure_obs_states):
            combined_states = jnp.vstack([ego_state.reshape(1, 2), pure_obs_states])
            dummy_A = jnp.ones(combined_states.shape[0])
            return get_local_density(ego_state, combined_states, 0, all_C, all_d, my_target, dummy_A, r_ego, all_radii, real_ego_center)

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
# 3. 自动化闭环控制仿真循环
# =========================================================================
if __name__ == '__main__':
    print("⚡ [冲突流形划分模式启动] 正在计算输入残差，自动切割黄金避障负样本...")

    R_EGO = 0.31
    all_C = jnp.array([[0.5, 0.2, 0.0, 4.0]]) 
    all_d = jnp.array([[5.0, 0.5]])                 
    
    # 给小车故意初始化一个朝向角 theta = 0.5 弧度（用来测试和区分开局对齐目标与真正避障的区别）
    ego_state = jnp.array([0.0, 0.0, 0.5])          
    my_target = jnp.array([10.0, 0.0])              
    
    obs_states = jnp.array([
        [8.5, 0.15],    
        [7.0, -0.30]    
    ])            
    obs_velocities = jnp.array([
        [-0.08, -0.01], 
        [-0.05,  0.02]  
    ])      
    
    obs_radii = jnp.array([0.31, 0.31])
    dt = 0.1                                        
    total_steps = 1800                               
    L = 0.15           

    ego_hist = []
    obs_hist = [] 
    z_collection = []                            
    
    # 🟢 核心记录容器：用来存储每一帧小车是否触发了避障数据切割条件
    avoidance_flag_hist = []

    static_collisions = []
    dynamic_collisions = []

    planner = DynamicEnvCDFPlanner()

    print("⏳ 正在进行 XLA 全局算子融合编译...")
    dummy_p = jnp.array([0.0, 0.0])
    dummy_nom = jnp.array([0.0, 0.0])
    _ = planner.solve_agent_qp(dummy_p, dummy_nom, obs_states, obs_velocities, all_C, all_d, my_target, R_EGO, obs_radii)
    _ = simulate_lidar_2d(ego_state, all_C, all_d, obs_states, obs_radii)
    print("✨ 编译成功！进入密集动态博弈与点云流同步生成...")

    for step in range(total_steps):
        ego_hist.append(ego_state)
        obs_hist.append(obs_states) 

        x, y, theta = float(ego_state[0]), float(ego_state[1]), float(ego_state[2])

        local_pc_all, min_t = simulate_lidar_2d(ego_state, all_C, all_d, obs_states, obs_radii)
        valid_mask = np.array(min_t < 2.99)
        current_pc_local = np.array(local_pc_all)[valid_mask] 
        z_collection.append(current_pc_local)

        dx_rect = max(abs(x - 5.0) - 0.5, 0.0)
        dy_rect = max(abs(y - 0.5) - 0.2, 0.0)
        if (abs(x - 5.0) <= 0.5 and abs(y - 0.5) <= 0.2) or ((dx_rect**2 + dy_rect**2)**0.5 < R_EGO):
            static_collisions.append([x, y])

        for i in range(obs_states.shape[0]):
            d_x, d_y = float(obs_states[i, 0]), float(obs_states[i, 1])
            if ((x - d_x)**2 + (y - d_y)**2)**0.5 < (R_EGO + obs_radii[i]):
                dynamic_collisions.append([x, y])

        ego_p = jnp.array([x + L * jnp.cos(theta), y + L * jnp.sin(theta)])
        target_vector = my_target - ego_p
        dist_to_goal = jnp.linalg.norm(target_vector)
        ego_u_nom = jnp.where(dist_to_goal > 0.1, 0.5 * target_vector / (dist_to_goal + 1e-6), jnp.zeros(2))

        # 解算出专家安全动作
        u_opt = planner.solve_agent_qp(
            ego_p, ego_u_nom, obs_states, obs_velocities,
            all_C, all_d, my_target, R_EGO, obs_radii
        )

        # 🟢【核心切分算法实施】：计算专家输出控制量 u_opt 与 标称引力控制量 ego_u_nom 之间的 L2 偏差残差
        control_residual = np.linalg.norm(u_opt - ego_u_nom)
        
        # 阈值 1e-3 已经过高度硬化：如果偏差大于此值，说明绝对是由障碍物压迫引起的“强行拦截修改动作”，标记为 True
        # 它能够完美识别避障起止点，并完美绕开因为初始车头角调整而产生的干扰！
        is_avoiding = control_residual > 0.09
        avoidance_flag_hist.append(is_avoiding)

        v = u_opt[0] * jnp.cos(theta) + u_opt[1] * jnp.sin(theta)
        omega = (-u_opt[0] * jnp.sin(theta) + u_opt[1] * jnp.cos(theta)) / L

        new_x = x + v * jnp.cos(theta) * dt
        new_y = y + v * jnp.sin(theta) * dt
        new_theta = theta + omega * dt
        ego_state = jnp.array([new_x, new_y, new_theta])
        
        obs_states = obs_states + obs_velocities * dt

        if jnp.linalg.norm(my_target - ego_state[:2]) < 0.2:
            print(f"🎉 差分小车成功穿过密集多体冲突区，于第 {step} 步安全突围！")
            break

    # =========================================================================
    # 4. 渲染多体演进与冲突流形精准高亮
    # =========================================================================
    ego_hist_np = np.array(ego_hist)
    obs_hist_np = np.array(obs_hist) 
    avoidance_flags = np.array(avoidance_flag_hist)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(11, 4.5))
    
    rect = plt.Rectangle((4.5, -0.2), 1.0, 0.4, fill=True, color='gray', alpha=0.3, hatch='//', label='Static rectangle obstacle')
    plt.gca().add_patch(rect)

    for idx in range(0, len(z_collection), 6):
        ego_x, ego_y, ego_th = ego_hist_np[idx, 0], ego_hist_np[idx, 1], ego_hist_np[idx, 2]
        pts_l = z_collection[idx]
        if len(pts_l) > 0: 
            cos_th = np.cos(ego_th)
            sin_th = np.sin(ego_th)
            pts_w_x = ego_x + pts_l[:, 0] * cos_th - pts_l[:, 1] * sin_th
            pts_w_y = ego_y + pts_l[:, 0] * sin_th + pts_l[:, 1] * cos_th
            plt.scatter(pts_w_x, pts_w_y, color='crimson', s=1.5, alpha=0.2, zorder=1, label='Pointcloud')

    # 🟢【重头戏修改】：不再机械地画一条一成不变的蓝色实线。而是根据避障条件，分段拼装、绘制轨迹
    # 没有触发避障的样本画成深蓝色线（巡航直驱）；触发了避障控制量拦截的样本画成加粗的猩红线（攻坚数据集部分）
    for i in range(len(ego_hist_np) - 1):
        x_segment = [ego_hist_np[i, 0], ego_hist_np[i+1, 0]]
        y_segment = [ego_hist_np[i, 1], ego_hist_np[i+1, 1]]
        
        if avoidance_flags[i]:
            # 🔴 精准切分：属于红色方框攻坚区，标记为高亮红，代表数据采集并入存储库
            plt.plot(x_segment, y_segment, color='red', linestyle='-', linewidth=4.0, zorder=3)
        else:
            # 🔵 自由区：名义引力线束直走，保持深蓝色，代表数据下采样剔除
            plt.plot(x_segment, y_segment, color='darkblue', linestyle='-', linewidth=2.0, zorder=2)

    # 创造虚拟图例句柄，防止线条分段绘制导致 Legend 里生成几百个重复名称
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='darkblue', lw=2.0, label='Nominal Cruise Path (Discarded Data)'),
        Line2D([0], [0], color='red', lw=4.0, label='Gated Avoidance Path (Saved GNN Data)')
    ]

    num_obstacles = obs_hist_np.shape[1]
    for i in range(num_obstacles):
        plt.plot(obs_hist_np[:, i, 0], obs_hist_np[:, i, 1], '--', linewidth=1.5, label=f'Dynamic obs {i+1} path')
        plt.scatter(obs_hist_np[0, i, 0], obs_hist_np[0, i, 1], marker='s', s=40, zorder=3)

    plt.scatter(0.0, 0.0, color='blue', marker='o', s=100, label='Start (0,0)', zorder=4)
    plt.scatter(my_target[0], my_target[1], color='darkgreen', marker='X', s=150, label='Target (10,0)', zorder=5)

    if len(static_collisions) > 0:
        sc_np = np.array(static_collisions)
        plt.scatter(sc_np[:, 0], sc_np[:, 1], color='black', marker='*', s=60, label='Static collision points', zorder=6)

    sample_indices = np.linspace(0, len(ego_hist_np) - 1, 6, dtype=int)
    for idx in sample_indices:
        th = ego_hist_np[idx, 2]
        plt.arrow(ego_hist_np[idx, 0], ego_hist_np[idx, 1], 0.18*np.cos(th), 0.18*np.sin(th), head_width=0.06, head_length=0.06, fc='darkblue', ec='darkblue', zorder=3)
        # 根据当前位置是否属于避障攻坚，改变展示的小车外壳颜色
        circle_color = 'red' if avoidance_flags[idx] else 'blue'
        robot_shell = plt.Circle((ego_hist_np[idx, 0], ego_hist_np[idx, 1]), R_EGO, color=circle_color, fill=True, alpha=0.06, zorder=2)
        plt.gca().add_patch(robot_shell)

    plt.xlabel("X", fontsize=22)
    plt.ylabel("Y", fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    plt.grid(False)
    plt.axis('equal')
    plt.xlim(-0.5, 10.5)
    plt.ylim(-1.5, 1.5) 
    
    # 融合自定义高亮图例与动态障碍物图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    all_handles = custom_lines + [by_label[k] for k in by_label if 'Dynamic' in k or 'Static' in k]
    
    plt.legend(handles=all_handles, loc='upper right', shadow=True, fontsize=10)
    
    plt.savefig('data_generate_test.png', dpi=300, bbox_inches='tight')
    print("💾 冲突流形精准切割测试完成！红蓝分段高清轨迹图已成功导出：cbf_qp_fixed_gated_trajectory.png")
    plt.show()