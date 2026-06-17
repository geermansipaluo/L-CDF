#!/usr/bin/env python3
import numpy as np
import jax.numpy as jnp

def get_env_pool():
    """
    配置10个几何多样性的测试场景池
    返回一个包含场景字典的列表
    """
    env_pool = []

    # ---------------------------------------------------------
    # Env 0: 原始经典混合障碍（中等难度）
    # ---------------------------------------------------------
    env_pool.append({
        'target': np.array([13.0, 0.0]),
        'meta_obstacles': [
            {'type': 'rect', 'center': np.array([5.0, 0.05]), 'a': 0.6, 'b': 0.2},
            {'type': 'circle', 'center': np.array([6.5, -0.5]), 'r': 0.5},
            {'type': 'circle', 'center': np.array([8.0, -2.5]), 'r': 0.5},
            {'type': 'circle', 'center': np.array([10.0, -0.5]), 'r': 0.25}
        ]
    })

    # ---------------------------------------------------------
    # Env 1: 走廊型窄门约束（高难度，考验CDF场的边缘逼近）
    # ---------------------------------------------------------
    env_pool.append({
        'target': np.array([14.5, 0.3]),
        'meta_obstacles': [
            {'type': 'rect', 'center': np.array([6.0, 2]), 'a': 0.5, 'b': 1.0},
            {'type': 'rect', 'center': np.array([6.0, -2]), 'a': 0.5, 'b': 1.0}, # 6米处一个宽1米的上下卡口
            {'type': 'circle', 'center': np.array([9.5, 0.0]), 'r': 0.4}
        ]
    })

    # ---------------------------------------------------------
    # Env 2: 纯净高速直道（极简难度，提供最大车速基准信号）
    # ---------------------------------------------------------
    env_pool.append({
        'target': np.array([10.0, -0.2]),
        'meta_obstacles': [
            {'type': 'circle', 'center': np.array([5.0, -0.0]), 'r': 0.3} # 障碍物远在天边，基本空旷
        ]
    })

    # ---------------------------------------------------------
    # Env 3: 横向交错S弯（高难度，考验大角度侧倾转向能力）
    # ---------------------------------------------------------
    env_pool.append({
        'target': np.array([13.5, -0.5]),
        'meta_obstacles': [
            {'type': 'circle', 'center': np.array([4.5, 0.8]), 'r': 0.6},   # 左边逼迫向下
            {'type': 'circle', 'center': np.array([7.5, -0.8]), 'r': 0.6},  # 右边逼迫向上
            {'type': 'circle', 'center': np.array([10.5, 0.8]), 'r': 0.5}   # 再逼迫向下
        ]
    })

    # ---------------------------------------------------------
    # Env 4: 密集小障碍乱石阵（极高难度，考验256线雷达超高分辨率）
    # ---------------------------------------------------------
    env_pool.append({
        'target': np.array([14.0, 0.5]),
        'meta_obstacles': [
            {'type': 'circle', 'center': np.array([4.0, 0.2]), 'r': 0.2},
            {'type': 'circle', 'center': np.array([5.5, -0.3]), 'r': 0.25},
            {'type': 'circle', 'center': np.array([7.0, 0.4]), 'r': 0.2},
            {'type': 'circle', 'center': np.array([8.5, -0.4]), 'r': 0.3},
            {'type': 'circle', 'center': np.array([10.5, 0.1]), 'r': 0.25}
        ]
    })

    # ---------------------------------------------------------
    # Env 5: 正前大方块拦路（中等难度，考验从两侧绕行的分流对称性）
    # ---------------------------------------------------------
    env_pool.append({
        'target': np.array([13.0, 0.8]),
        'meta_obstacles': [
            {'type': 'rect', 'center': np.array([6.5, 0.0]), 'a': 0.6, 'b': 0.6}, # 正前1.6x1.6m巨大方块
            {'type': 'circle', 'center': np.array([10.0, 1.2]), 'r': 0.3}
        ]
    })

    # ---------------------------------------------------------
    # Env 6: 倒V型漏斗阵（高难度，进入容易出来难）
    # ---------------------------------------------------------
    env_pool.append({
        'target': np.array([14.0, -0.8]),
        'meta_obstacles': [
            # {'type': 'rect', 'center': np.array([5.0, 0.8]), 'a': 0.5, 'b': 0.2},
            {'type': 'rect', 'center': np.array([5.0, -0.8]), 'a': 0.5, 'b': 0.2},
            {'type': 'circle', 'center': np.array([8.5, 0.0]), 'r': 0.6}
        ]
    })

    # ---------------------------------------------------------
    # Env 7: 经典斜向一字阵（中等难度）
    # ---------------------------------------------------------
    env_pool.append({
        'target': np.array([15.0, 0.0]),
        'meta_obstacles': [
            {'type': 'circle', 'center': np.array([4.0, -0.6]), 'r': 0.3},
            {'type': 'circle', 'center': np.array([6.5, 0.0]), 'r': 0.33},
            {'type': 'circle', 'center': np.array([9.0, 0.6]), 'r': 0.35}
        ]
    })

    # ---------------------------------------------------------
    # Env 8: 双侧夹击宽道（简单难度）
    # ---------------------------------------------------------
    env_pool.append({
        'target': np.array([14.2, 0.15]), # 略微偏移的非对称终点
        'meta_obstacles': [
            # 第一个大圆：卡在前半程偏上（逼迫小车从下方绕行）
            {'type': 'circle', 'center': np.array([6.5, 0.6]), 'r': 0.7},
            
            # 第二个大圆：卡在后半程偏下（逼迫小车绕过第一个后立刻向上拉起）
            {'type': 'circle', 'center': np.array([8.8, -0.6]), 'r': 0.7}
        ]
    })

    # ---------------------------------------------------------
    # Env 9: 终点前恶意设卡（中等难度，接近终点时的强减速逼近）
    # ---------------------------------------------------------
    env_pool.append({
        'target': np.array([13.0, -0.1]),
        'meta_obstacles': [
            {'type': 'circle', 'center': np.array([5.0, 0.0]), 'r': 0.4},
            {'type': 'rect', 'center': np.array([11.5, 0.3]), 'a': 0.3, 'b': 0.5} # 刚好卡在终点前
        ]
    })

    return env_pool

def convert_to_jax_tensors(meta_obstacles):
    """
    由于JAX需要静态或固定结构的dict，这里动态垫平单场景的Tensor转换
    """
    # 初始化一个远在天边的默认值，防止因为数量不一致导致JAX重新编译
    obs_tensors = {
        'rect_c': jnp.array([99.0, 99.0]), 'rect_ab': jnp.array([1e-3, 1e-3]),
        'c1_c': jnp.array([99.0, 99.0]), 'c1_r': 1e-3,
        'c2_c': jnp.array([99.0, 99.0]), 'c2_r': 1e-3,
        'c3_c': jnp.array([99.0, 99.0]), 'c3_r': 1e-3,
        'c4_c': jnp.array([99.0, 99.0]), 'c4_r': 1e-3,
        'c5_c': jnp.array([99.0, 99.0]), 'c4_r': 1e-3 # 扩展以支持最多5个圆
    }
    
    c_idx = 1
    for obs in meta_obstacles:
        if obs['type'] == 'rect':
            obs_tensors['rect_c'] = jnp.array(obs['center'])
            obs_tensors['rect_ab'] = jnp.array([obs['a'], obs['b']])
        elif obs['type'] == 'circle':
            obs_tensors[f'c{c_idx}_c'] = jnp.array(obs['center'])
            obs_tensors[f'c{c_idx}_r'] = float(obs['r'])
            c_idx += 1
            
    return obs_tensors