#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from jaxproxqp.jaxproxqp import JaxProxQP  # 🔴 完美导入新版高级求解器

class DynamicEnvCDFPlanner:
    def __init__(self):
        pass

    def solve_agent_qp(self, 
                       ego_state, ego_u_nom, 
                       all_C, all_d, all_obs_v, all_wp, all_r, current_idx, 
                       my_target, w_corridor):
        """
        🔴 从头重构的单机动态环境 CDF-QP 求解器
        参数:
            ego_state: 自车当前位置坐标 [x, y]
            ego_u_nom: 自车当前标称控制输入 (Go-to-goal)
            all_C, all_d: 动态障碍物的形状矩阵与当前中心坐标矩阵 (shape: 2xM)
            all_obs_v: 动态障碍物的当前实时速度矩阵 (shape: 2xM)，用于计算外部漂移
        """
        safety_dist_sq = 0.6 ** 2 
        epsilon = 0.1
        inv_eps = 1.0 / epsilon

        # =========================================================================
        # 🔴 核心修改：建立针对“自车位置”与“动态障碍物位置”的专属密度包装器
        # =========================================================================
        
        # 1. 用于计算自车空间梯度的包装器 (障碍物位置固定)
        def density_wrapper_ego(pos_ego):
            dummy_states = jnp.array([pos_ego]) # 构建单机虚拟状态集
            return get_local_density(
                pos_ego, dummy_states, 0,
                all_C, all_d, all_wp, all_r, current_idx, 
                my_target, w_corridor, safety_dist_sq, None
            )
            
        # 2. 用于计算外部动态障碍物时空漂移梯度的包装器 (自车位置固定)
        def density_wrapper_obs(dynamic_d):
            dummy_states = jnp.array([ego_state])
            return get_local_density(
                ego_state, dummy_states, 0,
                all_C, dynamic_d, all_wp, all_r, current_idx, 
                my_target, w_corridor, safety_dist_sq, None
            )

        # =========================================================================
        # 🔴 数学对齐：利用 JAX 自动微分计算动态障碍物引发的纯外部漂移项
        # =========================================================================
        # 计算自车自身的空间特征梯度
        grad_self = jax.grad(density_wrapper_ego)(ego_state)
        
        # 计算密度函数对所有障碍物中心位置 [2, M] 的全量偏导数矩阵
        grad_obs = jax.grad(density_wrapper_obs)(all_d)
        
        # 🔴 核心物理量转换：偏导数点乘障碍物实时运动速度，得到环境动态演变引起的密度漂移
        drift_term = jnp.sum(grad_obs * all_obs_v)

        # --- Z1: 沿着标称控制方向 (Intention Look-ahead) ---
        norm_nom = jnp.linalg.norm(ego_u_nom)
        dir_nom = jnp.where(
            norm_nom > 1e-5, 
            ego_u_nom / (norm_nom + 1e-8), 
            jnp.array([1.0, 0.0])
        )
        z1_pos = ego_state + epsilon * dir_nom

        # --- Z2: 沿着负密度梯度方向 (Critical Safety Look-ahead) ---
        neg_grad = -grad_self
        norm_grad = jnp.linalg.norm(neg_grad)
        dir_safe = jnp.where(
            norm_grad > 1e-5,
            neg_grad / (norm_grad + 1e-8),
            jnp.array([0.0, 1.0]) 
        )
        z2_pos = ego_state + epsilon * dir_safe

        # 处理共线基底补全
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
        
        # 预测前瞻点处的实时密度值
        rho_curr = density_wrapper_ego(ego_state)
        rho_z1 = density_wrapper_ego(z1_pos)
        rho_z2 = density_wrapper_ego(z2_pos)
        
        # 计算前瞻点处的空间梯度
        grad_z1 = jax.grad(density_wrapper_ego)(z1_pos)
        grad_z2 = jax.grad(density_wrapper_ego)(z2_pos)

        # =========================================================================
        # 3. QP 问题二次型构建 (维度 = 6: [u, u_z1, u_z2])
        # =========================================================================
        dim_total = 6
        lambda_smooth = 10.0
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

        g_vec = jnp.zeros(dim_total)
        g_vec = g_vec.at[0:2].set(-2.0 * ego_u_nom)
        
        sum_w = w1 + w2
        coeff_u = (rho_curr * inv_eps) * sum_w
        coeff_z1 = -(rho_z1 * inv_eps) * w1      
        coeff_z2 = -(rho_z2 * inv_eps) * w2

        num_constrs = 1
        # 🔴 变量更名，防止与障碍物参数 all_C 发生同名命名空间污染
        C_mat = jnp.zeros((num_constrs, dim_total))
        b_vec = jnp.zeros(num_constrs)
        
        C_mat = C_mat.at[0, 0:2].set(coeff_u)
        C_mat = C_mat.at[0, 2:4].set(coeff_z1)
        C_mat = C_mat.at[0, 4:6].set(coeff_z2)
        
        # 🔴 将全新计算的障碍物时空漂移项完美植入 CDF 约束边界
        b_vec = b_vec.at[0].set(drift_term - rho_curr)
        
        # 边界约束极限
        limit = 1.0
        l_box = jnp.array([-limit] * 6)
        u_box = jnp.array([limit] * 6)
        
        # =========================================================================
        # 🔴 求解阶段：全面采用标准 JaxProxQP 矢量化流水线
        # =========================================================================
        qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
        settings = JaxProxQP.Settings.default()
        settings.max_iter = 100
        settings.eps_abs = 1e-4
        
        solver = JaxProxQP(qp, settings)
        
        # 返回前两个维度，即当前时刻作用于自车的最佳端到端单积分器控制量 u
        return solver.solve().x[:2]