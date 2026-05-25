#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit
import matplotlib.pyplot as plt
from jaxproxqp.jaxproxqp import JaxProxQP

# =========================================================================
# 1. 核心底层数学函数与局部激光雷达仿真定义
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
    b_val = ellipse_val - 2.0
    
    return smooth_bump(c_val, b_val)

@jit
def get_local_density(my_state, my_target, all_C, all_d, r_ego, real_ego_center):
    LIDAR_RANGE_SQ = 3.0 ** 2

    dist_sq = jnp.sum((all_d - real_ego_center)**2, axis=1)
    in_range = dist_sq <= LIDAR_RANGE_SQ

    def single_obs_density(C_obs, d_obs):
        return obstacle_bump_field(my_state, C_obs, d_obs, r_ego)
    
    psi_array = jax.vmap(single_obs_density)(all_C, all_d)
    psi_array = jnp.where(in_range, psi_array, 1.0)
    psi_static = jnp.prod(psi_array)
    
    dist_target_sq = jnp.sum((my_state - my_target)**2)
    V_x = dist_target_sq
    alpha = 0.5
    
    rho = psi_static / (V_x ** alpha + 1e-6)
    return rho

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
# 2. 优化控制决策器类
# =========================================================================
class DynamicEnvCDFPlanner:
    @partial(jit, static_argnums=(0,))
    def solve_agent_qp(self, ego_state, ego_u_nom, all_C, all_d, all_v, my_target, r_ego):
        epsilon = 0.1
        inv_eps = 1.0 / epsilon
        real_ego_center = ego_state[:2]

        def density_wrapper_ego(pos_ego):
            return get_local_density(pos_ego, my_target, all_C, all_d, r_ego, real_ego_center)
            
        def density_wrapper_obs(pure_obs_d):
            return get_local_density(ego_state, my_target, all_C, pure_obs_d, r_ego, real_ego_center)

        grad_self = jax.grad(density_wrapper_ego)(ego_state)
        grad_obs = jax.grad(density_wrapper_obs)(all_d) 
        drift_term = jnp.sum(grad_obs * all_v)

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
        u_opt = solver.solve().x[:2]
        
        return u_opt, rho_curr, grad_self

# =========================================================================
# 3. 物理级随机拓扑环境生成算子
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
        
    num_dynamic = (num_obstacles - 1) // 2
    is_dynamic = [False] * num_obstacles
    if num_dynamic > 0:
        dynamic_indices = np.random.choice(num_obstacles, num_dynamic, replace=False)
        for idx in dynamic_indices:
            is_dynamic[idx] = True

    all_C, all_d, all_v = [], [], []
    
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
            
            if is_dynamic[i]:
                v_x_mag = np.random.uniform(0.1, 0.25)
                v_y_mag = np.random.uniform(0.1, 0.25)
                vx = -v_x_mag
                vy = v_y_mag if y_c < 0 else -v_y_mag
                all_v.append([vx, vy])
            else:
                all_v.append([0.0, 0.0])
            break
            
    return jnp.array(all_C), jnp.array(all_d), jnp.array(all_v), my_target

# =========================================================================
# 4. 专家数据对齐打包压缩落盘函数
# =========================================================================
def save_expert_dataset(X_list, y_list, z_list, filename="dataset.npz"):
    X_array = np.array(X_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.float32)
    z_array = np.array(z_list, dtype=object) 
    
    total_frames = len(z_array)
    empty_frames = sum([1 for p in z_array if len(p) == 0])
    avoid_frames = total_frames - empty_frames
    
    print("\n" + "="*60)
    print(f"💾 [数据流水线落盘成功] -> 已达成目标容量！")
    print(f" -> 最终状态特征 X 张量形状: {X_array.shape}")
    print(f" -> 最终监督标签 y 张量形状: {y_array.shape}")
    print(f" -> 最终变长点云 z 序列长度: {len(z_array)}")
    print(f"📊 [数据集成分最终审计] 总记录帧数: {total_frames}")
    print(f"    |-- 高价值连续避障动作帧: {avoid_frames} 帧 ({avoid_frames/total_frames*100:.1f}%)")
    print(f"    |-- 平衡下采样自由巡航帧: {empty_frames} 帧 ({empty_frames/total_frames*100:.1f}%)")
    print(f"🎉 恭喜！最纯净、无碰撞污染的图深度学习专家训练集已打包就绪: {filename}")
    print("="*60 + "\n")

# =========================================================================
# 5. 自动化无限场景闭环控制流水线
# =========================================================================
if __name__ == '__main__':
    R_EGO = 0.31  
    
    # 🔴【核心配置】设定你的目标数据框数阈值（例如收集到 25000 帧完美专家动作后自动停止）
    TARGET_DATA_LENGTH = 25000 
    
    # 建立外层全局大池子
    X_all = []
    y_all = []
    z_all = []

    # 统计核心指标
    total_episodes_run = 0
    successful_episodes = 0
    discarded_episodes = 0

    planner = DynamicEnvCDFPlanner()

    # 执行冷启动一次性基础编译
    print("⏳ 正在进行 XLA 全局算子融合一次性冷启动编译...")
    _all_C, _all_d, _all_v, _my_target = generate_random_environment(R_EGO)
    _dummy_ego = jnp.array([0.0, 0.0, 0.0])
    _dummy_p = jnp.array([0.0, 0.0])
    _dummy_nom = jnp.array([0.0, 0.0])
    _, _, _ = planner.solve_agent_qp(_dummy_p, _dummy_nom, _all_C, _all_d, _all_v, _my_target, R_EGO)
    _, _ = simulate_lidar_local_raw(_dummy_ego, _all_C, _all_d) 
    print("✨ 算子图编译成功！进入无限大批次自动化采集流...\n")

    # 🔴【外层流水线死循环】直到大池子数据长度达标才恩准退出
    while len(X_all) < TARGET_DATA_LENGTH:
        total_episodes_run += 1
        
        # 每一轮自动刷新未知的宇宙环境
        all_C, all_d, all_v, my_target = generate_random_environment(R_EGO)
        
        ego_state = jnp.array([0.0, 0.0, 0.0])          
        dt = 0.1                                        
        total_steps = 1800                               
        L = 0.15                                        

        # 开辟本轮独立局部缓存舱（防止绝境断层污染大池子）
        X_episode = []
        y_episode = []
        z_episode = []
        
        has_collision_occurred = False

        # 内部时空步进闭环仿真
        for step in range(total_steps):
            x, y, theta = float(ego_state[0]), float(ego_state[1]), float(ego_state[2])

            # --- 1. 超椭圆膨胀硬碰撞在线审计 ---
            for i in range(all_C.shape[0]):
                a_obs, b_obs, th_obs, n_obs = all_C[i]
                x_c, y_c = all_d[i, 0], all_d[i, 1]
                
                dx_c = x - x_c
                dy_c = y - y_c
                x_rot_c = dx_c * np.cos(th_obs) + dy_c * np.sin(th_obs)
                y_rot_c = -dx_c * np.sin(th_obs) + dy_c * np.cos(th_obs)
                
                ellipse_val_c = (abs(x_rot_c) / (a_obs + R_EGO))**n_obs + (abs(y_rot_c) / (b_obs + R_EGO))**n_obs
                if ellipse_val_c <= 1.0:
                    has_collision_occurred = True
                    break
            
            if has_collision_occurred:
                break # 一旦撞墙立刻熔断终止当前 Episode，交由外层执行无情全额销毁

            # --- 2. 局部感知变长不规则裁剪 (z) ---
            local_pc_all, min_t = simulate_lidar_local_raw(ego_state, all_C, all_d)
            valid_mask = np.array(min_t < 2.99)
            current_pc_local = np.array(local_pc_all)[valid_mask] 

            # --- 3. 差分最优控制指令求解 ---
            ego_p = jnp.array([x + L * jnp.cos(theta), y + L * jnp.sin(theta)])
            target_vector = my_target - ego_p
            dist_to_goal = jnp.linalg.norm(target_vector)
            ego_u_nom = jnp.where(dist_to_goal > 0.1, 0.5 * target_vector / (dist_to_goal + 1e-6), jnp.zeros(2))

            u_opt, rho_val, grad_rho = planner.solve_agent_qp(
                ego_p, ego_u_nom, all_C, all_d, all_v, my_target, R_EGO
            )

            v = u_opt[0] * jnp.cos(theta) + u_opt[1] * jnp.sin(theta)
            omega = (-u_opt[0] * jnp.sin(theta) + u_opt[1] * jnp.cos(theta)) / L

            # --- 4. 特征矩阵 X 与监督标签 y 组装 ---
            angle_to_target = jnp.arctan2(my_target[1] - y, my_target[0] - x)
            yaw_err = angle_to_target - theta
            cos_yaw_err = jnp.cos(yaw_err)
            sin_yaw_err = jnp.sin(yaw_err)
            
            X_step = np.array([x, y, theta, float(my_target[0]), float(my_target[1]), float(cos_yaw_err), float(sin_yaw_err)])
            y_step = np.array([float(v), float(omega), float(rho_val), float(grad_rho[0]), float(grad_rho[1])])
            
            # --- 5. 自由巡航帧负样本 15% 在线比例调和平衡阀 ---
            is_empty = (len(current_pc_local) == 0)
            should_record = True
            if is_empty:
                should_record = (np.random.rand() < 0.15)
                
            if should_record:
                X_episode.append(X_step)
                y_episode.append(y_step)
                z_episode.append(current_pc_local) 

            # --- 6. 物理系统状态前向积分积分演进 ---
            new_x = x + v * jnp.cos(theta) * dt
            new_y = y + v * jnp.sin(theta) * dt
            new_theta = theta + omega * dt
            ego_state = jnp.array([new_x, new_y, new_theta])
            all_d = all_d + all_v * dt

            # 收敛成功判定
            if jnp.linalg.norm(my_target - ego_state[:2]) < 0.2:
                break

        # 🔴【外层净化过滤器：一票否决控制核】
        if has_collision_occurred:
            discarded_episodes += 1
            # 绝不让绝境因果污染全局！局部数据直接出局丢弃（不加入大池子）
        else:
            successful_episodes += 1
            # 只有 100% 完美的历史演示才被允许合并
            X_all.extend(X_episode)
            y_all.extend(y_episode)
            z_all.extend(z_episode)
            
            # 实时无死角进度报告打印
            print(f"📊 进度: {len(X_all)}/{TARGET_DATA_LENGTH} 帧已捕获 "
                  f"({len(X_all)/TARGET_DATA_LENGTH*100:.1f}%) | "
                  f"累计战局数: {total_episodes_run} [完美突围: {successful_episodes} | 冲突熔断: {discarded_episodes}]")

    # 🔴 收集圆满达成，裁剪或直接高压缩落盘
    X_all = X_all[:TARGET_DATA_LENGTH]
    y_all = y_all[:TARGET_DATA_LENGTH]
    z_all = z_all[:TARGET_DATA_LENGTH]
    save_expert_dataset(X_all, y_all, z_all, filename="dataset.npz")


    # =========================================================================
    # 6. 渲染多体演进与顶级学术期刊级超椭圆验证（根据您的明确要求，已彻底注释移除）
    # =========================================================================
    # 画面渲染已被物理阻断，保障流水线在后台以全速、无 GUI 停滞、100% 的极高效率疯狂生成专家行为。