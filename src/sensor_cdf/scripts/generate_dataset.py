#!/usr/bin/env python3
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import math
from cal_grad import density_grad
# export JAX_PLATFORM_NAME=cpu

# def generate_data(num_samples=1000, num_obstacles=1):
#     # 模拟椭圆障碍物参数 (cx, cy, a, b, theta)
#     # obstacles = np.random.rand(num_obstacles, 5) * 5
#     cx = 0.5
#     cy = 0.5
#     a = 1
#     b = 1
#     theta = 0
#     X = np.random.rand(num_samples, 2) * 20 - 10  # 状态空间采样
#     C_list = []  # 临时存储 C
#     d_list = []  # 临时存储 d
#     a_list = []

    # # 将障碍物参数转换
    # for obs in obstacles:
    #     cx, cy, a, b, theta = obs
    #     if a < b:
    #         temp = b
    #         b = a
    #         a = temp   
    # cos_t = math.cos(theta)
    # sin_t = math.sin(theta)
    # C = np.array([
    #     [a * cos_t, -b * sin_t],
    #     [a * sin_t,  b * cos_t]
    # ])
    # d = np.array([cx, cy])
    # C_list.append(C)
    # d_list.append(d)
    # a_list.append(a)

    # # 转换为jax运算方便计算
    # C_list = jnp.stack(C_list, axis=0)
    # d_list = jnp.column_stack(d_list)
    # a_list = jnp.array(a_list)
    
#     # 计算真值梯度 ∇ρ(x)
#     gradients = []
#     num = 1
#     for x in X:
#         rospy.loginfo("-----------for %d th ----------", num)
#         rho_func = calculate_density(C_list, d_list, a_list)  # 解析求导
#         x_jax = jnp.array(x)
#         current_grad, pred_grad = density_grad(
#                 x = x_jax,
#                 density_func = rho_func,
#         )
#         gradients.append(current_grad)
#         num = num+1
#     return X, np.array(gradients)

# generate_dataset.py
def generate_data(num_samples=4000, max_obstacles=5):
    X = []
    y = []
    obstacles_list = []
    
    for _ in range(num_samples):
        C_list = []  # 临时存储 C
        d_list = []  # 临时存储 d
        a_list = []
        # 随机生成环境配置
        # num_obs = np.random.randint(1, max_obstacles+1) 
        obstacles = np.random.rand(1, 5) * np.array([20, 8, 2, 2, np.pi]) - np.array([10, 4, 0, 0, 0])

            # 将障碍物参数转换
        for obs in obstacles:
            cx, cy, a, b, theta = obs
            if a < b:
                temp = b
                b = a
                a = temp   
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            C = np.array([
                [a * cos_t, -b * sin_t],
                [a * sin_t,  b * cos_t]
            ])
            d = np.array([cx, cy])
            C_list.append(C)
            d_list.append(d)
            a_list.append(a)

        # 转换为jax运算方便计算
        C_list = jnp.stack(C_list, axis=0)
        d_list = jnp.column_stack(d_list)
        a_list = jnp.array(a_list)
        
        # 随机采样状态点
        state = np.random.rand(2) * 20 - 10
        state[1] = state[1] * 0.5
        state_jax = jnp.array(state)
        
        # 计算真实梯度
        rho_func = calculate_density(C_list, d_list, a_list)
        grad, _ = density_grad(state_jax, rho_func)
        
        X.append(np.concatenate([state, obstacles.flatten()]))
        y.append(grad)
        obstacles_list.append(obstacles)
    
    # 填充障碍物数据为统一维度
    # max_len = max_obstacles * 5
    # X_padded = np.zeros((num_samples, 2 + max_len))
    # for i, x in enumerate(X):
    #     X_padded[i, :2] = x[:2]
    #     X_padded[i, 2:2+len(x[2:])] = x[2:]
    
    # return X_padded, np.array(y)
    return X, np.array(y)

@partial(jit, static_argnums=(0,))
def _compute_psi_k(x, C_k, d_k, a_k):
    """JAX计算单个障碍物的psi_k"""
    inv_C_k = jnp.linalg.inv(C_k.astype(jnp.float64))
    x_minus_d = (x - d_k).astype(jnp.float64)
    
    # 计算c_k
    v = inv_C_k @ x_minus_d
    c_k = jnp.sum(v**2) - 1
    
    # 计算b_k
    b_k = jnp.sum(x_minus_d**2) - (3)**2
    
    # 处理分母为零的情况
    denominator = c_k - b_k
    safe_denominator = jnp.where(denominator == 0, 1e-6, denominator)
    m_k = c_k / safe_denominator
    
    # 限制m_k范围保证数值稳定
    safe_m_k = jnp.clip(m_k, 1e-6, 1-1e-6)
    
    # 计算psi_k
    exp_m = jnp.exp(-1 / safe_m_k)
    exp_1_m = jnp.exp(-1 / (1 - safe_m_k))
    psi_k = exp_m / (exp_m + exp_1_m + 1e-6)
    
    # 应用分段条件
    return jnp.where(
        c_k <= 0, 0.0,
        jnp.where(b_k <= 0, psi_k, 1.0)
    )

def calculate_density(C_list, d_list, a_list):
    # 获取当前状态的快照
    target_pos = jnp.array([5.0, 0.0], dtype=jnp.float64)

    @partial(jit, static_argnums=())
    def fresh_rho(x1, x2):
        x = jnp.array([x1, x2], dtype=jnp.float64)
        # 使用快照数据而非self引用
        def body_fun(k, psi_total):
            return psi_total * _compute_psi_k(
                x, C_list[k], d_list[:,k], a_list[k]
            )
        psi_total = jax.lax.fori_loop(0, len(a_list), body_fun, 1.0)
        # return 8500*psi_total / (jnp.sum((x - target)**2))
        return 200*psi_total / jnp.sqrt(jnp.sum((x - target_pos)**2))

    return fresh_rho  # 返回不依赖self的新函数

# if __name__ == "__main__":
#     rospy.loginfo("正在生成训练数据集...")
#     X_train, y_train = generate_data()
#     rospy.loginfo(f"生成完成，数据集大小: {X_train.shape[0]} 样本")
    
#     # 保存数据集到文件
#     np.savez('gradient_dataset.npz', X=X_train, y=y_train)
#     rospy.loginfo("数据集已保存到 gradient_dataset.npz")
 