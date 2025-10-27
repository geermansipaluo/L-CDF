#!/usr/bin/env python3
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial



# 类型转换回调函数
def to_numpy_callback(arr, dtype=np.float64):
    return np.array(arr, dtype=dtype)

@partial(jax.jit, static_argnames=('density_func',))
def compute_gradients(x, density_func):
    """向量化梯度计算"""
    grad_val = jax.grad(lambda x: density_func(x[0], x[1]))(x)
    return grad_val

@jax.jit
def predict_state(x, density_val, deltaT=0.05):
    """JIT编译的状态预测"""
    return x + deltaT * jnp.clip(density_val, 0.0, 1e3) * jnp.ones(2)

@partial(jax.jit, static_argnames=('density_func',))
def density_grad(x, density_func):
    """JAX兼容的CDF控制器"""
    # 参数预处理
    x = jnp.asarray(x).flatten().astype(jnp.float64)

    # ========== 梯度计算 ==========
    try:
        current_grad = compute_gradients(x, density_func)
        density_val = density_func(x[0], x[1])
        x_pred = predict_state(x, density_val)
        pred_grad = compute_gradients(x_pred, density_func)
        
        # 异步类型转换
        current_grad_np = jax.pure_callback(
            to_numpy_callback, 
            np.zeros(2, dtype=np.float64), 
            current_grad
        )
        pred_grad_np = jax.pure_callback(
            to_numpy_callback, 
            np.zeros(2, dtype=np.float64), 
            pred_grad
        )
        
    except Exception as e:
        print(f"计算异常: {str(e)}")
        return np.zeros((2, 1))

    

    return current_grad_np, pred_grad_np
