#!/usr/bin/env python3
import math
import contextlib
import io
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, knn_graph, GATConv
#  引入 BarrierNet 同款工业级可微参数化凸优化层
from qpth.qp import QPFunction, QPSolvers

class GeometricEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=512, k=10):
        super().__init__()
        self.k = k

        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            ),
            k=self.k,
            aggr='mean'
        )

        self.norm1 = nn.BatchNorm1d(hidden_dim)

        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=8,
            concat=False,
            add_self_loops=False
        )

        self.res_fc = nn.Identity()

        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            ),
            k=self.k,
            aggr='max'
        )

        self.downsample = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, points, node_features=None):
        """
        points.pos: [Total_Points, 2]
        node_features:
            None -> 使用点坐标 [x, y]
            非 None -> 使用拼接后的节点特征，例如 [x, y, state4]
        """
        pos = points.pos
        batch_idx = points.batch

        if node_features is None:
            x0 = pos
        else:
            x0 = node_features

        # GAT 的边仍然建议用几何坐标 pos 建图
        edge_index = knn_graph(pos, k=self.k, batch=batch_idx, loop=True)

        x1 = self.conv1(x0, batch_idx)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = F.leaky_relu(x2 + self.res_fc(x1), 0.2)

        x3 = self.conv3(x2, batch_idx)

        x_multi = torch.cat([x1, x2, x3], dim=1)
        x_multi = self.downsample(x_multi)

        x_global = global_max_pool(x_multi, batch_idx)
        return x_global


class DifferentiableSdfCdfSafetyLayer6D(nn.Module):
    """
    6D 升维参数化可微安全层。

    本版本主要做 4 件事：
    1) 默认和 JAX/ProxQP 专家对齐：启用 6D box constraints，关闭 G/h 行归一化；
    2) 暴露 qpth 求解器参数 maxIter / eps / notImprovedLim；
    3) 给 box constraints 增加极小 eps，缓解 qpth 内点法贴边数值问题；
    4) 增强 qpth 异常诊断和失败统计，避免只看到 TypeError 而不知道真实输入状态。

    注意：
    - qp_fail_mode="skip" 推荐用于训练：qpth 失败时抛出异常，由 trainer 跳过该 batch；
    - qp_fail_mode="raise" 用于严格排查：qpth 失败时直接抛错；
    - qp_fail_mode="fallback" 仅用于调试兼容：qpth 失败时返回 u_nom，不建议正式训练使用。
    """
    def __init__(
        self,
        lambda_smooth=1,
        qp_limit=1.2,
        use_qp_box_constraints=True,
        qp_jitter=1e-4,
        qp_normalize_constraints=False,
        qp_constraint_scale_floor=1.0,
        qp_box_eps=1e-4,
        qp_max_iter=50,
        qp_eps=1e-6,
        qp_not_improved_lim=10,
        qp_fail_mode="skip",
        qp_debug_max_print=20,
        qp_check_invalid_constraints=True,
        qp_invalid_g_norm_eps=1e-8,
        qp_invalid_h_eps=1e-6,
        qp_invalid_constraint_mode="warn",
        qp_invalid_debug_max_print=20,
        qp_sanitize_redundant_constraints=True,
        qp_redundant_constraint_h=1.0,
        qp_verify_solution=True,
        qp_solution_violation_tol=1e-3,
        qp_solution_debug_max_print=20,
        qp_suppress_qpth_warnings=True,
        learnable_lambda_smooth=False,
        lambda_smooth_min=0.1,
        lambda_smooth_max=80.0,
        lambda_reg_weight=1e-4,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # 可学习 lambda_smooth 参数化：
        #   lambda = lambda_min + (lambda_max - lambda_min) * sigmoid(lambda_raw)
        # 这样始终保证 lambda > 0，避免 Hessian 因 lambda 异常变坏。
        # 当 learnable_lambda_smooth=False 时，保持旧版固定 lambda 行为，
        # 且不向 state_dict 注入额外参数，便于加载旧模型。
        # ------------------------------------------------------------
        self.learnable_lambda_smooth = bool(learnable_lambda_smooth)
        self.lambda_smooth_init = float(lambda_smooth)
        self.lambda_smooth_min = float(lambda_smooth_min)
        self.lambda_smooth_max = float(lambda_smooth_max)
        self.lambda_reg_weight = float(lambda_reg_weight)

        if self.lambda_smooth_max <= self.lambda_smooth_min:
            raise ValueError(
                f"lambda_smooth_max must be larger than lambda_smooth_min, "
                f"got min={self.lambda_smooth_min}, max={self.lambda_smooth_max}"
            )

        lambda_init_clamped = min(
            max(self.lambda_smooth_init, self.lambda_smooth_min + 1e-6),
            self.lambda_smooth_max - 1e-6,
        )
        ratio = (
            (lambda_init_clamped - self.lambda_smooth_min)
            / (self.lambda_smooth_max - self.lambda_smooth_min)
        )
        ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
        raw_init = math.log(ratio / (1.0 - ratio))

        if self.learnable_lambda_smooth:
            self.lambda_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
            self.register_buffer(
                "lambda_prior",
                torch.tensor(lambda_init_clamped, dtype=torch.float32),
            )
        else:
            self.lambda_raw = None
            self.lambda_prior = None

        self.qp_limit = float(qp_limit)

        # 与 JAX/ProxQP 专家对齐的默认设置
        self.use_qp_box_constraints = bool(use_qp_box_constraints)
        self.qp_normalize_constraints = bool(qp_normalize_constraints)

        # 数值稳定参数
        self.qp_jitter = float(qp_jitter)
        self.qp_constraint_scale_floor = float(qp_constraint_scale_floor)
        self.qp_box_eps = float(qp_box_eps)

        # qpth 求解器参数
        self.qp_max_iter = int(qp_max_iter)
        self.qp_eps = float(qp_eps)
        self.qp_not_improved_lim = int(qp_not_improved_lim)

        # 失败处理模式：
        #   skip  : 推荐训练模式，抛出异常交给 trainer 跳过该 batch；
        #   raise : 严格调试模式，直接抛错终止；
        #   fallback: 兼容旧行为，失败时返回 u_nom，不建议正式训练。
        self.qp_fail_mode = str(qp_fail_mode).lower()
        if self.qp_fail_mode not in ("fallback", "raise", "skip"):
            raise ValueError(
                f"qp_fail_mode must be 'fallback', 'raise', or 'skip', got {qp_fail_mode}"
            )

        self.qp_debug_max_print = int(qp_debug_max_print)

        # G/h 异常约束诊断：
        # 1) ||G||≈0 且 h<0 会形成 0 <= negative 的不可行约束；
        # 2) 在 box 约束下，如果 h 小于 Gx 的理论最小值，也会不可行。
        self.qp_check_invalid_constraints = bool(qp_check_invalid_constraints)
        self.qp_invalid_g_norm_eps = float(qp_invalid_g_norm_eps)
        self.qp_invalid_h_eps = float(qp_invalid_h_eps)
        self.qp_invalid_constraint_mode = str(qp_invalid_constraint_mode).lower()
        if self.qp_invalid_constraint_mode not in ("warn", "raise", "ignore"):
            raise ValueError(
                "qp_invalid_constraint_mode must be one of "
                f"('warn', 'raise', 'ignore'), got {qp_invalid_constraint_mode}"
            )
        self.qp_invalid_debug_max_print = int(qp_invalid_debug_max_print)

        # qpth 的内点法需要严格正 slack。
        # 对于 G≈0 且 h≈0 的冗余约束 0*x<=0，数学上可行但没有严格内点，
        # 容易导致 qpth 返回 None / inaccurate solution。
        # 因此默认将这类约束改成 0*x<=1.0，相当于删除该冗余约束。
        self.qp_sanitize_redundant_constraints = bool(qp_sanitize_redundant_constraints)
        self.qp_redundant_constraint_h = float(qp_redundant_constraint_h)

        # qpth 可能返回 tensor 但同时警告 residual large。
        # 因此在返回前额外检查不等式约束违反量；
        # 若明显违反，则按 qpth failure 处理，让 trainer skip batch。
        self.qp_verify_solution = bool(qp_verify_solution)
        self.qp_solution_violation_tol = float(qp_solution_violation_tol)
        self.qp_solution_debug_max_print = int(qp_solution_debug_max_print)
        self.qp_suppress_qpth_warnings = bool(qp_suppress_qpth_warnings)

        # 统计量：用于 trainer 每个 epoch 打印 fail_rate
        self._qp_warning_count = 0
        self._qp_call_count = 0
        self._qp_fail_count = 0
        self._qp_invalid_warning_count = 0
        self._qp_invalid_zero_count = 0
        self._qp_infeasible_main_count = 0
        self._qp_redundant_zero_count = 0
        self._qp_bad_solution_count = 0
        self._qp_solution_warning_count = 0
        self._qp_last_max_violation = 0.0

    def lambda_smooth_value(self):
        """
        返回当前有效 lambda_smooth。
        - learnable=True: 返回有界可学习参数；
        - learnable=False: 返回固定初始化值张量。
        """
        if self.learnable_lambda_smooth:
            return (
                self.lambda_smooth_min
                + (self.lambda_smooth_max - self.lambda_smooth_min)
                * torch.sigmoid(self.lambda_raw)
            )

        # 固定 lambda 时也返回 Tensor，方便 forward 里统一处理 dtype/device。
        return torch.tensor(float(self.lambda_smooth_init), dtype=torch.float32)

    def lambda_regularization(self):
        """
        对可学习 lambda 做轻微正则，避免它无意义地贴到上下界。
        固定 lambda 时返回 0。
        """
        if not self.learnable_lambda_smooth:
            return torch.tensor(0.0)

        lambda_now = self.lambda_smooth_value()
        lambda_prior = self.lambda_prior.to(device=lambda_now.device, dtype=lambda_now.dtype)
        reg = torch.log(lambda_now / (lambda_prior + 1e-8)) ** 2
        return self.lambda_reg_weight * reg

    def reset_qp_stats(self):
        self._qp_call_count = 0
        self._qp_fail_count = 0
        self._qp_invalid_zero_count = 0
        self._qp_infeasible_main_count = 0
        self._qp_redundant_zero_count = 0
        self._qp_bad_solution_count = 0
        self._qp_last_max_violation = 0.0

    def get_qp_stats(self):
        fail_rate = self._qp_fail_count / max(self._qp_call_count, 1)
        return {
            "qp_call_count": self._qp_call_count,
            "qp_fail_count": self._qp_fail_count,
            "qp_fail_rate": fail_rate,
            "qp_warning_count": self._qp_warning_count,
            "qp_invalid_zero_count": self._qp_invalid_zero_count,
            "qp_infeasible_main_count": self._qp_infeasible_main_count,
            "qp_redundant_zero_count": self._qp_redundant_zero_count,
            "qp_bad_solution_count": self._qp_bad_solution_count,
            "qp_last_max_violation": self._qp_last_max_violation,
        }

    @staticmethod
    def _tensor_debug_string(name, tensor):
        if tensor is None:
            return f"{name}: None"
        if not torch.is_tensor(tensor):
            return f"{name}: non_tensor={type(tensor)}"
        info = (
            f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"device={tensor.device}, numel={tensor.numel()}"
        )
        if tensor.numel() == 0:
            return info + ", empty=True"
        with torch.no_grad():
            finite = torch.isfinite(tensor)
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            if finite.any():
                finite_tensor = tensor[finite]
                t_min = finite_tensor.min().item()
                t_max = finite_tensor.max().item()
                t_mean = finite_tensor.mean().item()
                info += f", finite_min={t_min:.6e}, finite_max={t_max:.6e}, finite_mean={t_mean:.6e}"
            else:
                info += ", no_finite_values=True"
            info += f", has_nan={has_nan}, has_inf={has_inf}"
        return info

    def _print_qp_failure_debug(self, e_qp, u_nom, P_in, g_in, G_all, h_all, G_main, h_main):
        print("\n" + "=" * 100)
        print(f"⚠️ [qpth 数值临界拦截] {type(e_qp).__name__}: {repr(e_qp)}")
        print(
            f"qp_config: box={self.use_qp_box_constraints}, normalize={self.qp_normalize_constraints}, "
            f"qp_limit={self.qp_limit}, box_eps={self.qp_box_eps}, jitter={self.qp_jitter}, "
            f"maxIter={self.qp_max_iter}, eps={self.qp_eps}, "
            f"notImprovedLim={self.qp_not_improved_lim}, fail_mode={self.qp_fail_mode}"
        )
        print(self._tensor_debug_string("u_nom", u_nom))
        print(self._tensor_debug_string("P_in", P_in))
        print(self._tensor_debug_string("g_in", g_in))
        print(self._tensor_debug_string("G_main", G_main))
        print(self._tensor_debug_string("h_main", h_main))
        print(self._tensor_debug_string("G_all", G_all))
        print(self._tensor_debug_string("h_all", h_all))

        # 额外打印当前 batch 中约束行范数，方便发现尺度异常
        try:
            with torch.no_grad():
                row_norm = torch.linalg.norm(G_all, dim=2)
                print(self._tensor_debug_string("G_all_row_norm", row_norm))
        except Exception as norm_e:
            print(f"G_all_row_norm debug failed: {repr(norm_e)}")

        print("=" * 100 + "\n")

    def _check_constraint_pathologies(self, G_main, h_main, stage="raw"):
        """
        排查会导致 qpth infeasible / NoneType 的异常主 CDF 约束。

        重点检测：
        1) ||G||≈0 且 h<0：
             这等价于 0 <= negative，必然不可行。
        2) 在 6D box x_i∈[-limit, limit] 下不可行：
             min_{box} Gx = -limit * sum(abs(G_i))
             如果 h < min_{box} Gx，则该约束在 box 内无解。
        """
        if (
            not self.qp_check_invalid_constraints
            or self.qp_invalid_constraint_mode == "ignore"
        ):
            return

        if G_main.numel() == 0 or h_main.numel() == 0:
            return

        with torch.no_grad():
            G_norm = torch.linalg.norm(G_main, dim=2)  # [B, M]
            h_view = h_main

            zero_bad = (
                (G_norm < self.qp_invalid_g_norm_eps)
                & (h_view < -self.qp_invalid_h_eps)
            )

            box_limit = float(self.qp_limit) + float(self.qp_box_eps)
            min_possible_Gx = -box_limit * torch.sum(torch.abs(G_main), dim=2)
            infeasible_main = h_view < (min_possible_Gx - self.qp_invalid_h_eps)

            zero_count = int(zero_bad.sum().item())
            infeasible_count = int(infeasible_main.sum().item())

            self._qp_invalid_zero_count += zero_count
            self._qp_infeasible_main_count += infeasible_count

            bad_any = zero_bad | infeasible_main
            bad_count = int(bad_any.sum().item())

            if bad_count <= 0:
                return

            self._qp_invalid_warning_count += 1

            if self._qp_invalid_warning_count <= self.qp_invalid_debug_max_print:
                print("\n" + "!" * 100)
                print(
                    f"❌ [QP INVALID CONSTRAINT DETECTED @ {stage}] "
                    f"bad_count={bad_count}, zero_bad={zero_count}, "
                    f"infeasible_main={infeasible_count}, mode={self.qp_invalid_constraint_mode}"
                )
                print(
                    f"thresholds: g_norm_eps={self.qp_invalid_g_norm_eps:.3e}, "
                    f"h_eps={self.qp_invalid_h_eps:.3e}, box_limit={box_limit:.6f}"
                )
                print(self._tensor_debug_string("G_main", G_main))
                print(self._tensor_debug_string("h_main", h_main))
                print(self._tensor_debug_string("G_main_norm", G_norm))
                print(self._tensor_debug_string("min_possible_Gx_under_box", min_possible_Gx))

                # 打印前 8 个异常位置，方便回查 batch 中的问题样本
                bad_idx = torch.nonzero(bad_any, as_tuple=False)[:8]
                examples = []
                for pair in bad_idx.detach().cpu().tolist():
                    b, m = int(pair[0]), int(pair[1])
                    examples.append(
                        {
                            "batch": b,
                            "constraint": m,
                            "G_norm": float(G_norm[b, m].detach().cpu().item()),
                            "h": float(h_view[b, m].detach().cpu().item()),
                            "min_possible_Gx": float(min_possible_Gx[b, m].detach().cpu().item()),
                            "zero_bad": bool(zero_bad[b, m].detach().cpu().item()),
                            "infeasible_main": bool(infeasible_main[b, m].detach().cpu().item()),
                        }
                    )
                print(f"bad_examples(first_8)={examples}")
                print("!" * 100 + "\n")

            elif self._qp_invalid_warning_count == self.qp_invalid_debug_max_print + 1:
                print("❌ [QP INVALID CONSTRAINT DETECTED] 后续详细异常约束诊断将静默统计，不再刷屏。")

        if self.qp_invalid_constraint_mode == "raise":
            raise RuntimeError(
                f"invalid_or_infeasible_cdf_constraints_detected: "
                f"zero_bad={zero_count}, infeasible_main={infeasible_count}"
            )

    def _sanitize_redundant_zero_constraints(self, G_main, h_main):
        """
        将 G≈0 且 h≈0 的冗余约束 0*x<=0 改成 0*x<=positive。

        原因：
        - 0*x<=0 数学上等价于无约束，但 slack 恒等于 0；
        - qpth 的 primal-dual interior point method 需要严格正 slack；
        - 这种冗余边界约束会导致 qpth 返回 None 或 inaccurate solution。

        注意：
        - 若 G≈0 且 h<0，这是真正不可行约束，不在这里修；
        - 这里只处理 h≈0 或 h>=0 的冗余零行。
        """
        if not self.qp_sanitize_redundant_constraints:
            return G_main, h_main

        if G_main.numel() == 0 or h_main.numel() == 0:
            return G_main, h_main

        G_norm = torch.linalg.norm(G_main, dim=2)  # [B, M]
        redundant_zero = (
            (G_norm < self.qp_invalid_g_norm_eps)
            & (h_main >= -self.qp_invalid_h_eps)
        )

        redundant_count = int(redundant_zero.detach().sum().item())
        if redundant_count <= 0:
            return G_main, h_main

        self._qp_redundant_zero_count += redundant_count

        # 避免 in-place 修改原输入张量导致 autograd/qpth 侧问题。
        h_sanitized = torch.where(
            redundant_zero,
            torch.full_like(h_main, float(self.qp_redundant_constraint_h)),
            h_main,
        )

        if self._qp_invalid_warning_count < self.qp_invalid_debug_max_print:
            print(
                f"⚠️ [QP SANITIZE] replaced redundant zero constraints "
                f"0*x<=0 by 0*x<={self.qp_redundant_constraint_h}; "
                f"count={redundant_count}"
            )

        return G_main, h_sanitized

    def _verify_qp_solution(self, sol_6d, G_all, h_all):
        """
        qpth 有时会返回 Tensor，但同时警告 residual large。
        这里至少检查 primal inequality feasibility:
            G_all x - h_all <= tol

        如果明显违反约束，则当作 qpth failure，让 trainer 在 skip 模式下跳过该 batch。
        """
        if not self.qp_verify_solution:
            return

        with torch.no_grad():
            lhs = torch.bmm(G_all, sol_6d.unsqueeze(-1)).squeeze(-1)
            violation = lhs - h_all
            max_violation = violation.max()
            max_violation_value = float(max_violation.detach().cpu().item())
            self._qp_last_max_violation = max_violation_value

            if max_violation_value <= self.qp_solution_violation_tol:
                return

            self._qp_bad_solution_count += 1
            self._qp_fail_count += 1
            self._qp_solution_warning_count += 1

            if self._qp_solution_warning_count <= self.qp_solution_debug_max_print:
                print("\n" + "#" * 100)
                print(
                    f"❌ [QP BAD SOLUTION] qpth returned a tensor but violates inequality constraints: "
                    f"max_violation={max_violation_value:.6e}, "
                    f"tol={self.qp_solution_violation_tol:.6e}, "
                    f"fail_mode={self.qp_fail_mode}"
                )
                print(self._tensor_debug_string("sol_6d", sol_6d))
                print(self._tensor_debug_string("ineq_violation=Gx-h", violation))
                print(self._tensor_debug_string("G_all", G_all))
                print(self._tensor_debug_string("h_all", h_all))
                bad_idx = torch.nonzero(violation > self.qp_solution_violation_tol, as_tuple=False)[:8]
                examples = []
                for pair in bad_idx.detach().cpu().tolist():
                    b, m = int(pair[0]), int(pair[1])
                    examples.append(
                        {
                            "batch": b,
                            "constraint": m,
                            "violation": float(violation[b, m].detach().cpu().item()),
                            "lhs": float(lhs[b, m].detach().cpu().item()),
                            "h": float(h_all[b, m].detach().cpu().item()),
                        }
                    )
                print(f"bad_solution_examples(first_8)={examples}")
                print("#" * 100 + "\n")
            elif self._qp_solution_warning_count == self.qp_solution_debug_max_print + 1:
                print("❌ [QP BAD SOLUTION] 后续 bad solution 详细诊断将静默统计，不再刷屏。")

        if self.qp_fail_mode in ("raise", "skip"):
            raise RuntimeError(
                f"qpth_failed_in_safety_layer: returned_solution_violates_constraints "
                f"max_violation={self._qp_last_max_violation:.6e}"
            )

    def forward(self, u_nom, G_cdf_6d, h_cdf):
        """
        u_nom: [B, 2]
        G_cdf_6d: [B, M, 6]，通常 M=1
        h_cdf: [B, M]
        """
        self._qp_call_count += 1

        batch_size = u_nom.shape[0]
        device = u_nom.device
        dtype = u_nom.dtype

        # 1. 重组 6x6 Hessian
        # 注意：这里必须保持 lambda_smooth 为 Tensor，不能 float(...)，
        # 否则 learnable lambda 的梯度会被截断。
        lambda_smooth = self.lambda_smooth_value().to(device=device, dtype=dtype)

        val_uu = 2.0 * (1.0 + 2.0 * lambda_smooth)
        val_zz = 2.0 * lambda_smooth
        val_uz = -2.0 * lambda_smooth

        P_in = torch.zeros(batch_size, 6, 6, device=device, dtype=dtype)

        P_in[:, 0, 0] = val_uu
        P_in[:, 1, 1] = val_uu
        P_in[:, 2, 2] = val_zz
        P_in[:, 3, 3] = val_zz
        P_in[:, 4, 4] = val_zz
        P_in[:, 5, 5] = val_zz

        P_in[:, 0, 2] = val_uz
        P_in[:, 2, 0] = val_uz
        P_in[:, 1, 3] = val_uz
        P_in[:, 3, 1] = val_uz
        P_in[:, 0, 4] = val_uz
        P_in[:, 4, 0] = val_uz
        P_in[:, 1, 5] = val_uz
        P_in[:, 5, 1] = val_uz

        if self.qp_jitter is not None and float(self.qp_jitter) > 0.0:
            eye6 = torch.eye(6, device=device, dtype=dtype).unsqueeze(0)
            P_in = P_in + float(self.qp_jitter) * eye6

        # 2. 线性项 g
        g_in = torch.zeros(batch_size, 6, device=device, dtype=dtype)
        g_in[:, 0:2] = -2.0 * u_nom

        # 3. 主 CDF 约束
        G_main = G_cdf_6d.to(device=device, dtype=dtype)
        h_main = h_cdf.to(device=device, dtype=dtype)

        # 防御 h shape 异常：允许 [B]，统一成 [B, 1]
        if h_main.dim() == 1:
            h_main = h_main.view(batch_size, 1)

        # 在任何归一化之前检查原始 G/h，方便直接定位专家数据中的异常约束。
        self._check_constraint_pathologies(G_main, h_main, stage="raw_before_normalization")

        # qpth 不喜欢 0*x<=0 这种无严格内点的冗余约束。
        # 它数学上等价于无约束，因此安全地改成 0*x<=positive。
        G_main, h_main = self._sanitize_redundant_zero_constraints(G_main, h_main)

        if self.qp_normalize_constraints:
            row_scale = torch.linalg.norm(G_main, dim=2, keepdim=True)
            row_scale = torch.clamp(row_scale, min=float(self.qp_constraint_scale_floor))
            G_main = G_main / row_scale
            h_main = h_main / row_scale.squeeze(-1)

        # 4. 与 JAX/ProxQP 对齐：默认加入 6D box constraints
        if self.use_qp_box_constraints:
            eye6 = torch.eye(6, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            box_G = torch.cat([eye6, -eye6], dim=1)

            # 内点法对贴边很敏感；用极小 eps 放宽 box。
            # 建议默认 1e-4，不要用 1e-2，否则会明显偏离 JAX 的 1.2 box。
            box_limit = float(self.qp_limit) + float(self.qp_box_eps)
            box_h = torch.full(
                (batch_size, 12),
                box_limit,
                device=device,
                dtype=dtype,
            )

            G_all = torch.cat([G_main, box_G], dim=1)
            h_all = torch.cat([h_main, box_h], dim=1)
        else:
            G_all = G_main
            h_all = h_main

        # 5. 空等式约束
        e = torch.empty(0, device=device, dtype=dtype)
        A = torch.empty(0, device=device, dtype=dtype)

        try:
            qp_solver = QPFunction(
                verbose=False,
                solver=QPSolvers.PDIPM_BATCHED,
                maxIter=self.qp_max_iter,
                eps=self.qp_eps,
                notImprovedLim=self.qp_not_improved_lim,
            )

            if self.qp_suppress_qpth_warnings:
                # qpth 有时会打印 residual warning，但我们后面会自己检查 feasibility。
                # 为了不刷屏，默认屏蔽 qpth 内部 stdout/stderr/warnings。
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        sol_6d = qp_solver(P_in, g_in, G_all, h_all, e, A)
            else:
                sol_6d = qp_solver(P_in, g_in, G_all, h_all, e, A)

            if torch.isnan(sol_6d).any() or torch.isinf(sol_6d).any():
                raise ValueError("qpth_nan_or_inf_detected")

            # qpth 可能返回 tensor 但 warning residual large；
            # 对这类情况至少检查 primal inequality violation。
            self._verify_qp_solution(sol_6d, G_all, h_all)

        except Exception as e_qp:
            self._qp_fail_count += 1
            self._qp_warning_count += 1

            if self._qp_warning_count <= self.qp_debug_max_print:
                self._print_qp_failure_debug(
                    e_qp=e_qp,
                    u_nom=u_nom,
                    P_in=P_in,
                    g_in=g_in,
                    G_all=G_all,
                    h_all=h_all,
                    G_main=G_main,
                    h_main=h_main,
                )
            elif self._qp_warning_count == self.qp_debug_max_print + 1:
                print("⚠️ [qpth 数值临界拦截] 后续同类详细诊断将静默统计，不再刷屏。")

            if self.qp_fail_mode in ("raise", "skip"):
                # 不再静默返回 u_nom。正式训练推荐 qp_fail_mode="skip"，
                # 由 trainer 捕获该异常并跳过当前 batch，避免污染梯度。
                raise RuntimeError(f"qpth_failed_in_safety_layer: {repr(e_qp)}") from e_qp

            # fallback 模式只用于兼容/临时调试：会污染这一批的 safety 监督，
            # 不建议正式训练使用。
            sol_6d = torch.zeros(batch_size, 6, device=device, dtype=dtype)
            sol_6d[:, 0:2] = u_nom

        return sol_6d[:, 0:2]

class UNet(nn.Module):
    def __init__(
        self,
        state_dim=4,
        hidden_dim=512,
        graph_k=10,
        lambda_smooth=25.0,
        qp_limit=1.2,
        use_qp_box_constraints=True,
        qp_jitter=1e-4,
        qp_normalize_constraints=False,
        qp_constraint_scale_floor=1.0,
        qp_box_eps=1e-4,
        qp_max_iter=50,
        qp_eps=1e-6,
        qp_not_improved_lim=10,
        qp_fail_mode="skip",
        qp_debug_max_print=20,
        qp_check_invalid_constraints=True,
        qp_invalid_g_norm_eps=1e-8,
        qp_invalid_h_eps=1e-6,
        qp_invalid_constraint_mode="warn",
        qp_invalid_debug_max_print=20,
        qp_sanitize_redundant_constraints=True,
        qp_redundant_constraint_h=1.0,
        qp_verify_solution=True,
        qp_solution_violation_tol=1e-3,
        qp_solution_debug_max_print=20,
        qp_suppress_qpth_warnings=True,
        learnable_lambda_smooth=False,
        lambda_smooth_min=0.1,
        lambda_smooth_max=80.0,
        lambda_reg_weight=1e-4,
        ablation="full",
    ):
        super().__init__()

        self.ablation = ablation
        self.use_dual_branch = ablation != "no_dual"
        self.use_safety_layer = ablation != "no_safety"

        if self.use_dual_branch:
            # 原始双分支：点云单独编码，状态单独编码
            self.geo_encoder = GeometricEncoder(
                in_dim=2,
                hidden_dim=hidden_dim,
                k=graph_k
            )

            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )

        else:
            # w/o 双分支：
            # 每个点拼接 4 维状态，所以 node feature = 2 + state_dim
            self.geo_encoder = GeometricEncoder(
                in_dim=2 + state_dim,
                hidden_dim=hidden_dim,
                k=graph_k
            )

            self.state_encoder = None

            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh()
        )

        if self.use_safety_layer:
            self.safety_layer = DifferentiableSdfCdfSafetyLayer6D(
                lambda_smooth=lambda_smooth,
                qp_limit=qp_limit,
                use_qp_box_constraints=use_qp_box_constraints,
                qp_jitter=qp_jitter,
                qp_normalize_constraints=qp_normalize_constraints,
                qp_constraint_scale_floor=qp_constraint_scale_floor,
                qp_box_eps=qp_box_eps,
                qp_max_iter=qp_max_iter,
                qp_eps=qp_eps,
                qp_not_improved_lim=qp_not_improved_lim,
                qp_fail_mode=qp_fail_mode,
                qp_debug_max_print=qp_debug_max_print,
                qp_check_invalid_constraints=qp_check_invalid_constraints,
                qp_invalid_g_norm_eps=qp_invalid_g_norm_eps,
                qp_invalid_h_eps=qp_invalid_h_eps,
                qp_invalid_constraint_mode=qp_invalid_constraint_mode,
                qp_invalid_debug_max_print=qp_invalid_debug_max_print,
                qp_sanitize_redundant_constraints=qp_sanitize_redundant_constraints,
                qp_redundant_constraint_h=qp_redundant_constraint_h,
                qp_verify_solution=qp_verify_solution,
                qp_solution_violation_tol=qp_solution_violation_tol,
                qp_solution_debug_max_print=qp_solution_debug_max_print,
                qp_suppress_qpth_warnings=qp_suppress_qpth_warnings,
                learnable_lambda_smooth=learnable_lambda_smooth,
                lambda_smooth_min=lambda_smooth_min,
                lambda_smooth_max=lambda_smooth_max,
                lambda_reg_weight=lambda_reg_weight,
            )
        else:
            self.safety_layer = None

    def forward(self, state, points, G_cdf, h_cdf):
        if self.use_dual_branch:
            geo_feat = self.geo_encoder(points)
            state_feat = self.state_encoder(state)
            fused = self.fusion(torch.cat([geo_feat, state_feat], dim=1))

        else:
            # state: [B, 4]
            # points.batch: [Total_Points]
            # state_per_point: [Total_Points, 4]
            state_per_point = state[points.batch]

            node_features = torch.cat(
                [points.pos, state_per_point],
                dim=1
            )

            geo_feat = self.geo_encoder(
                points,
                node_features=node_features
            )

            fused = self.fusion(geo_feat)

        u_nom = self.head(fused) * 1.2

        if self.use_safety_layer:
            u_safe = self.safety_layer(u_nom, G_cdf, h_cdf)
        else:
            # w/o safety layer：网络直接输出动作
            u_safe = u_nom

        # 保持 trainer 接口不变
        return u_safe, u_nom