#!/usr/bin/env python3
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit
import matplotlib.pyplot as plt
from jaxproxqp.jaxproxqp import JaxProxQP
import torch
from torch_geometric.data import HeteroData

# =========================================================================
# 1. 核心底层数学函数与局部激光雷达仿真定义 (保全 JAX 高速算子)
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
    
    physical_dist = jnp.sqrt(dx**2 + dy**2)
    b_val = (physical_dist / 3.0) - 1.0
    
    return smooth_bump(c_val, b_val)

@jit
def get_local_density_and_psi(my_state, my_target, all_C, all_d, r_ego, real_ego_center):
    LIDAR_RANGE_SQ = 3.0 ** 2

    dist_sq = jnp.sum((all_d - real_ego_center)**2, axis=1)
    in_range = dist_sq <= LIDAR_RANGE_SQ

    def single_obs_density(C_obs, d_obs):
        return obstacle_bump_field(my_state, C_obs, d_obs, r_ego)
    
    psi_array = jax.vmap(single_obs_density)(all_C, all_d)
    psi_array = jnp.where(in_range, psi_array, 1.0)
    
    psi_curr = jnp.prod(psi_array)
    
    dist_target_sq = jnp.sum((my_state - my_target)**2)
    V_x = dist_target_sq
    alpha = 0.5
    
    rho = psi_curr / (V_x ** alpha + 1e-6)
    return rho, psi_curr

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
            rho, _ = get_local_density_and_psi(pos_ego, my_target, all_C, all_d, r_ego, real_ego_center)
            return rho
            
        def density_wrapper_obs(pure_obs_d):
            rho, _ = get_local_density_and_psi(ego_state, my_target, all_C, pure_obs_d, r_ego, real_ego_center)
            return rho

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
        
        rho_curr, psi_curr = get_local_density_and_psi(ego_state, my_target, all_C, all_d, r_ego, real_ego_center)
        rho_z1, _ = get_local_density_and_psi(z1_pos, my_target, all_C, all_d, r_ego, real_ego_center)
        rho_z2, _ = get_local_density_and_psi(z2_pos, my_target, all_C, all_d, r_ego, real_ego_center)

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
        
        limit = 1.2
        l_box = jnp.array([-limit] * 6); u_box = jnp.array([limit] * 6)
        
        qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
        settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, settings)
        u_opt = solver.solve().x[:2]
        
        return u_opt, psi_curr

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
# 4. 🟢【全新重构】：PyG 异质图数据集固化保存系统
# =========================================================================
def save_pyg_hetero_dataset(pyg_data_list, filename="dataset.pt"):
    """
    完全对齐三实体（ego, point, goal）异质图规范的分布式落盘系统。
    将数据直接固化为 PyTorch 纯正的高压缩二进制分布式图集。
    """
    total_frames = len(pyg_data_list)
    empty_frames = sum([1 for data in pyg_data_list if data['point'].x.shape[0] == 0])
    avoid_frames = total_frames - empty_frames
    
    print("\n" + "="*60)
    print(f"💾 [图数据数据流落盘系统启动] -> 正在生成三实体统一图数据集...")
    
    # 执行 PyTorch 二进制序列化存储
    torch.save(pyg_data_list, filename)
    
    print(f" -> 成功将 {total_frames} 帧导航样本存储至：{filename}")
    print(f"📊 [图样本集成分最终审计]")
    print(f"    |-- 密集型高价值真实避障图帧: {avoid_frames} 帧 ({avoid_frames/total_frames*100:.1f}%)")
    print(f"    |-- 宽阔型平缓巡航无点云图帧: {empty_frames} 帧 ({empty_frames/total_frames*100:.1f}%)")
    print(f"💡 [提示] 在训练器端通过 `dataset = torch.load('{filename}')` 即可实现一行解包加载！")
    print("="*60 + "\n")

# =========================================================================
# 5. 自动化无限场景闭环控制流水线
# =========================================================================
if __name__ == '__main__':
    R_EGO = 0.31  
    
    # 学术论文级强鲁棒特征数据集标准体量调整（测试使用 10000 帧）
    TARGET_DATA_LENGTH = 10000
    
    pyg_dataset = []
    total_episodes_run, successful_episodes, discarded_episodes = 0, 0, 0

    planner = DynamicEnvCDFPlanner()

    print("⏳ 正在进行 XLA 全局算子融合一次性冷启动编译...")
    _all_C, _all_d, _all_v, _my_target = generate_random_environment(R_EGO)
    _dummy_ego = jnp.array([0.0, 0.0, 0.0])
    _dummy_p = jnp.array([0.0, 0.0])
    _dummy_nom = jnp.array([0.0, 0.0])
    _, _ = planner.solve_agent_qp(_dummy_p, _dummy_nom, _all_C, _all_d, _all_v, _my_target, R_EGO)
    _, _ = simulate_lidar_local_raw(_dummy_ego, _all_C, _all_d) 
    print("✨ 算子图编译成功！后台加速无图表分布式采集启动...\n")

    while len(pyg_dataset) < TARGET_DATA_LENGTH:
        total_episodes_run += 1
        all_C, all_d, all_v, my_target = generate_random_environment(R_EGO)
        
        ego_state = jnp.array([0.0, 0.0, 0.0])          
        dt = 0.05                                      
        total_steps = 1800                               
        L = 0.15                                        

        # 🟢 修复点 1：每个 Episode 开始前，正确、必须初始化轨迹临时缓冲区
        episode_pyg_buffer = []
        has_collision_occurred = False

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
                break 

            # --- 2. 局部感知变长不规则裁剪 ---
            local_pc_all, min_t = simulate_lidar_local_raw(ego_state, all_C, all_d)
            valid_mask = np.array(min_t < 2.99)
            current_pc_local = np.array(local_pc_all)[valid_mask] 

            # --- 3. 最优指导控制解算 ---
            ego_p = jnp.array([x + L * jnp.cos(theta), y + L * jnp.sin(theta)])
            target_vector = my_target - ego_p
            dist_to_goal = jnp.linalg.norm(target_vector)
            ego_u_nom = jnp.where(dist_to_goal > 0.1, 1.2 * target_vector / (dist_to_goal + 1e-6), jnp.zeros(2))

            u_opt, psi_curr_val = planner.solve_agent_qp(
                ego_p, ego_u_nom, all_C, all_d, all_v, my_target, R_EGO
            )

            v = u_opt[0] * jnp.cos(theta) + u_opt[1] * jnp.sin(theta)
            omega = (-u_opt[0] * jnp.sin(theta) + u_opt[1] * jnp.cos(theta)) / L

            # --- 4. 局部系转体特征组装层 —— 确保旋转与平移不变性 ---
            dx = float(my_target[0]) - x
            dy = float(my_target[1]) - y
            dist_to_goal_val = np.hypot(dx, dy)

            # 提取自车局部载体系下的目标相对位置坐标 (消除全局绝对坐标长距离冲刷隐患)
            # 严格遵照反向转体投影：
            target_local_x = dx * np.cos(theta) + dy * np.sin(theta)
            target_local_y = -dx * np.sin(theta) + dy * np.cos(theta)

            compressed_dist = np.tanh(dist_to_goal_val / 3.0)
            
            angle_to_target = jnp.arctan2(my_target[1] - y, my_target[0] - x)
            yaw_err = angle_to_target - theta
            cos_yaw_err = jnp.cos(yaw_err)
            sin_yaw_err = jnp.sin(yaw_err)

            v_nom_kin = ego_u_nom[0] * jnp.cos(theta) + ego_u_nom[1] * jnp.sin(theta)
            omega_nom_kin = (-ego_u_nom[0] * jnp.sin(theta) + ego_u_nom[1] * jnp.cos(theta)) / L
            
            # X 缩减为纯净的自车 4维固有特征向量：(x, y, theta, dist_to_goal)
            X_step = np.array([x, y, theta, dist_to_goal_val], dtype=np.float32)
            
            # y 固定为规整的多任务监督标签
            y_step = np.array([float(v), float(omega), float(psi_curr_val), float(v_nom_kin), float(omega_nom_kin)], dtype=np.float32)
            
            # --- 5. 巡航负样本平衡过滤阀与 🟢 三实体异质图组装 ---
            is_empty = (len(current_pc_local) == 0)
            should_record = True
            if is_empty:
                should_record = (np.random.rand() < 0.01)
                
            if should_record:
                # 声明高级异质图容器
                graph_sample = HeteroData()
                
                # 实体 1：自车节点特征 [1, 4] -> 保存当前纯净自车状态
                graph_sample['ego'].x = torch.tensor(X_step, dtype=torch.float32).unsqueeze(0)
                
                # 实体 2：目标点虚拟节点特征 [1, 2] -> 灌入自车局部系下的相对目标诱导坐标
                # 对应 GCBF+ 论文中多边感知架构对导航吸引力输入的表达
                graph_sample['goal'].x = torch.tensor([target_local_x, target_local_y], dtype=torch.float32).unsqueeze(0)
                
                # 实体 3：变长激光雷达击中点节点特征 [K, 2]
                if is_empty:
                    graph_sample['point'].x = torch.zeros((0, 2), dtype=torch.float32)
                    # 必须给一个明确的 [2, 0] 形状的空 Edge Index，强制 PyG 锁死异质图骨架
                    graph_sample['point', 'to', 'ego'].edge_index = torch.zeros((2, 0), dtype=torch.long)
                else:
                    graph_sample['point'].x = torch.tensor(current_pc_local, dtype=torch.float32)
                    num_points = graph_sample['point'].x.shape[0]
                    senders_p = torch.arange(num_points, dtype=torch.long)
                    receivers_p = torch.zeros(num_points, dtype=torch.long)
                    graph_sample['point', 'to', 'ego'].edge_index = torch.stack([senders_p, receivers_p], dim=0)
                    
                # 边编制 B：建立引力目标点指向自车的“引力导航消息流边” [2, 1]
                graph_sample['goal', 'to', 'ego'].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                
                # 挂载监督多任务多目标标签 [1, 5]
                graph_sample.y = torch.tensor(y_step, dtype=torch.float32).unsqueeze(0)
                
                # 推进当前 Episode 临时安全屋
                episode_pyg_buffer.append(graph_sample)

            # --- 6. 物理系统状态演进 ---
            new_x = x + v * jnp.cos(theta) * dt
            new_y = y + v * jnp.sin(theta) * dt
            new_theta = theta + omega * dt
            ego_state = jnp.array([new_x, new_y, new_theta])
            all_d = all_d + all_v * dt

            if jnp.linalg.norm(my_target - ego_state[:2]) < 0.2:
                break

        # =========================================================================
        # 🟢 修复点 2：洗数与数据汇流机制完全重构对接
        # =========================================================================
        if has_collision_occurred:
            discarded_episodes += 1
        else:
            successful_episodes += 1
            # 将通过碰撞在线审计审查后的纯净异质图列表推入全局长远存储中
            pyg_dataset.extend(episode_pyg_buffer)
            
            print(f"📊 收集进度: {len(pyg_dataset)}/{TARGET_DATA_LENGTH} 个统一异质图对象已固化 "
                  f"({len(pyg_dataset)/TARGET_DATA_LENGTH*100:.1f}%) | "
                  f"战局统计: {total_episodes_run} [完美: {successful_episodes} | 冲突熔断: {discarded_episodes}]")

    # 达标裁剪并真正触发硬盘二进制写入
    pyg_dataset = pyg_dataset[:TARGET_DATA_LENGTH]
    save_pyg_hetero_dataset(pyg_dataset, filename="dataset.pt")