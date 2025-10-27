#!/usr/bin/env python3
import numpy as np
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

options['show_progress'] = False


def cdf_control(current_grad, pred_grad, dx, deltaT):
    # ========== QP问题构建 ==========
    H_diag = 2 * np.diag([1.0, 1.0, 0, 0, 1])
    H = sparse(matrix(H_diag))
    A_cdf = np.array([
        [-current_grad[0], -current_grad[1], 0, 0, 1],
        [0, 0, -pred_grad[0], 0, 1],
        [0, 0, 0, -pred_grad[1], 1],
        [1/deltaT, 1/deltaT, -1/deltaT, -1/deltaT, 0],
        [1, 0, 0, 0, 0], [-1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0], [0, -1, 0, 0, 0],
        [0, 0, 1, 0, 0], [0, 0, -1, 0, 0],
        [0, 0, 0, 1, 0], [0, 0, 0, -1, 0],
        [0, 0, 0, 0, 1], [0, 0, 0, 0, -1]
    ])
    
    b_cdf = np.array([
        0.0, 0.0, 0.0, 0.0, 
        1.0, 1.0, 1.0, 1.0,
        1e3, 1e3, 1e3, 1e3,
        1e3, -1e-3  # 收紧约束
    ])
    f_cdf = -2 * np.hstack([dx, np.zeros(3)])

    # ========== QP求解 ==========
    try:
        result = qp(
            H, matrix(f_cdf.T),
            matrix(A_cdf), matrix(b_cdf),
            options=options
        )
        u = np.array(result['x'])[:2]
    except Exception as e:
        print(f"QP失败: {str(e)}")
        return np.zeros((2,1))

    return u.reshape(2, 1)
