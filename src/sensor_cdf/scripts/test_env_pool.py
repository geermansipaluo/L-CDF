#!/usr/bin/env python3
import numpy as np
import jax.numpy as jnp

def get_env_pool():
    env_pool = []

    env_pool.append({
        'name': 'env0_original_mixed',
        'target': np.array([13.0, 0.0]),
        'meta_obstacles': [
            {'type': 'rect', 'center': np.array([5.0, 0.05]), 'a': 0.6, 'b': 0.2},
            {'type': 'circle', 'center': np.array([6.5, -0.5]), 'r': 0.5},
            {'type': 'circle', 'center': np.array([8.0, -2.5]), 'r': 0.5},
            {'type': 'circle', 'center': np.array([10.0, -0.5]), 'r': 0.25}
        ]
    })

    env_pool.append({
        'name': 'env1_corridor_gate',
        'target': np.array([14.5, 0.3]),
        'meta_obstacles': [
            {'type': 'rect', 'center': np.array([6.0, 1.5]), 'a': 0.5, 'b': 1.0},
            {'type': 'rect', 'center': np.array([6.0, -1.5]), 'a': 0.5, 'b': 1.0},
            {'type': 'circle', 'center': np.array([9.5, 0.0]), 'r': 0.4}
        ]
    })

    env_pool.append({
        'name': 'env3_s_curve',
        'target': np.array([13.5, -0.5]),
        'meta_obstacles': [
            {'type': 'circle', 'center': np.array([4.5, 0.8]), 'r': 0.6},
            {'type': 'circle', 'center': np.array([7.5, -0.8]), 'r': 0.6},
            {'type': 'circle', 'center': np.array([10.5, 0.8]), 'r': 0.5}
        ]
    })

    env_pool.append({
        'name': 'env4_dense_stones',
        'target': np.array([14.0, 0.5]),
        'meta_obstacles': [
            {'type': 'circle', 'center': np.array([4.0, 0.2]), 'r': 0.2},
            {'type': 'circle', 'center': np.array([5.5, -0.3]), 'r': 0.25},
            {'type': 'circle', 'center': np.array([7.0, 0.4]), 'r': 0.2},
            {'type': 'circle', 'center': np.array([8.5, -0.4]), 'r': 0.3},
            {'type': 'circle', 'center': np.array([10.5, 0.1]), 'r': 0.25}
        ]
    })

    env_pool.append({
        'name': 'env5_big_block_front',
        'target': np.array([13.0, 0.8]),
        'meta_obstacles': [
            {'type': 'rect', 'center': np.array([6.5, 0.0]), 'a': 0.8, 'b': 0.8},
            {'type': 'circle', 'center': np.array([10.0, 1.2]), 'r': 0.3}
        ]
    })

    return env_pool

def convert_to_jax_tensors(meta_obstacles):
    """
    将 meta_obstacles 转成 JAX tensor 格式。

    支持：
    - 最多 2 个矩形障碍物：rect1, rect2
    - 最多 5 个圆形障碍物：c1 ~ c5

    同时保留旧字段：
    - rect_c
    - rect_ab

    旧字段 rect_c / rect_ab 默认等于第一个矩形，
    这样旧代码如果还在用 rect_c / rect_ab，也不会立刻报错。
    """

    obs_tensors = {
        # 旧版兼容字段：默认代表第一个 rect
        'rect_c': jnp.array([99.0, 99.0]),
        'rect_ab': jnp.array([1e-3, 1e-3]),

        # 新版：支持两个 rect
        'rect1_c': jnp.array([99.0, 99.0]),
        'rect1_ab': jnp.array([1e-3, 1e-3]),

        'rect2_c': jnp.array([99.0, 99.0]),
        'rect2_ab': jnp.array([1e-3, 1e-3]),

        # circle，最多 5 个
        'c1_c': jnp.array([99.0, 99.0]), 'c1_r': 1e-3,
        'c2_c': jnp.array([99.0, 99.0]), 'c2_r': 1e-3,
        'c3_c': jnp.array([99.0, 99.0]), 'c3_r': 1e-3,
        'c4_c': jnp.array([99.0, 99.0]), 'c4_r': 1e-3,
        'c5_c': jnp.array([99.0, 99.0]), 'c5_r': 1e-3,
    }

    rect_idx = 1
    c_idx = 1

    for obs in meta_obstacles:
        if obs['type'] == 'rect':
            if rect_idx <= 2:
                center = jnp.array(obs['center'])
                ab = jnp.array([obs['a'], obs['b']])

                obs_tensors[f'rect{rect_idx}_c'] = center
                obs_tensors[f'rect{rect_idx}_ab'] = ab

                # 兼容旧代码：第一个 rect 同时写入 rect_c / rect_ab
                if rect_idx == 1:
                    obs_tensors['rect_c'] = center
                    obs_tensors['rect_ab'] = ab

                rect_idx += 1

        elif obs['type'] == 'circle':
            if c_idx <= 5:
                obs_tensors[f'c{c_idx}_c'] = jnp.array(obs['center'])
                obs_tensors[f'c{c_idx}_r'] = float(obs['r'])
                c_idx += 1

    return obs_tensors