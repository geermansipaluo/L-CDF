import jax
import jax.numpy as jnp
from jaxproxqp.jaxproxqp import JaxProxQP
import torch
import torch.utils.dlpack as dlpack
from jax.dlpack import from_dlpack
from jax import jit

class JaxCBFQPSafetyFilter:
    def __init__(self, num_obstacles):
        """
        初始化安全过滤器
        由于对齐了 JaxProxQP 的最新 API，求解器实例将在 filter_control 内部动态跟随静态图创建。
        """
        self.num_obstacles = num_obstacles
        self.dim_total = 2  # 决策变量仅包含差分小车的物理控制量: [v, w]

    def filter_control(self, state, u_nom, obstacles, risk, config):
        """
        高性能 CBF-QP 过滤函数 (支持 JAX XLA 静态编译)
        
        参数:
        - state: 当前小车状态 jnp.array([x, y, theta])
        - u_nom: 神经网络预测的原始名义控制量 jnp.array([v_nom, w_nom])
        - obstacles: 局部环境障碍物矩阵 jnp.array([[x_i, y_i, R_i], ...]), 形状为 (N, 3)
        - risk: 风险度头输出的标量值 (0.0 ~ 1.0)
        - config: 包含控制、几何及物理边界参数的静态字典
        """
        x, y, theta = state[0], state[1], state[2]
        
        # 1. 计算前瞻点 P 坐标 (前瞻点法解决差分小车相对阶为2的问题)
        l_k = config['l_k']
        P = jnp.array([
            x + l_k * jnp.cos(theta),
            y + l_k * jnp.sin(theta)
        ])
        
        # 2. 计算前瞻点的运动学雅可比矩阵 J(\theta)
        J = jnp.array([
            [jnp.cos(theta), -l_k * jnp.sin(theta)],
            [jnp.sin(theta),  l_k * jnp.cos(theta)]
        ])
        
        # 3. 提取障碍物数据并向量化计算一阶圆形 SDF
        O_centers = obstacles[:, :2]  # 形状: (N, 2)
        R_obs = obstacles[:, 2]       # 形状: (N,)
        
        P_minus_O = P - O_centers     # 形状: (N, 2)
        dist_sq = jnp.sum(P_minus_O**2, axis=1)
        h = dist_sq - (config['r_ego'] + R_obs)**2  # 安全距离指标
        
        # 4. 构建标准 CBF 线性不等式约束矩阵项: C_mat * u <= b_vec
        # 根据推导: -2*(P - O)^T * J * u <= \gamma * h
        C_mat = -2.0 * jnp.matmul(P_minus_O, J)    # 形状: (N, 2)
        b_cbf = config['gamma'] * h               # 形状: (N,)
        
        # 5. ⚡ 风险度平滑放行机制 (对齐 XLA 静态一致性，不产生计算图分叉)
        is_triggered = risk >= config['threshold']
        b_vec = jnp.where(is_triggered, b_cbf, 1e6) # 未触发时赋予极大上限，约束自动失效
        
        # 6. 精准对齐你的目标的二乘代价函数构建 (min ||u - u_nom||^2)
        # 展开后对应: 0.5 * u^T * H * u + g_vec^T * u
        H = jnp.eye(self.dim_total) * 2.0
        g_vec = -2.0 * u_nom
        
        # 7. 物理边界直接注入盒状约束 (无需像常规矩阵那样做额外的 vstack 拼接)
        l_box = jnp.array([config['v_min'], config['w_min']])
        u_box = jnp.array([config['v_max'], config['w_max']])
        
        # 8. 严格按照你提供的规范实例化并调用 JaxProxQP
        qp = JaxProxQP.QPModel.create(H, g_vec, C_mat, b_vec, l_box=l_box, u_box=u_box)
        settings = JaxProxQP.Settings.default()
        solver = JaxProxQP(qp, settings)
        
        # 截取前两个维度作为安全控制量传出
        u_opt = solver.solve().x[:2]
        
        return u_opt