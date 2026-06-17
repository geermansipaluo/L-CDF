#!/usr/bin/env python3
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, knn_graph, GATConv
from torch_geometric.utils import to_dense_batch
#  引入 BarrierNet 同款工业级可微参数化凸优化层
from qpth.qp import QPFunction, QPSolvers


class TimingMixin:
    def _init_timing(self, enable_timing_debug=False, timing_sync_cuda=True):
        self.enable_timing_debug = bool(enable_timing_debug)
        self.timing_sync_cuda = bool(timing_sync_cuda)
        self._timing_stats = {}

    def _timing_now(self, device=None):
        if self.enable_timing_debug and self.timing_sync_cuda and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device=device)
            except Exception:
                torch.cuda.synchronize()
        return time.perf_counter()

    def _timing_add(self, name, start_time, device=None):
        if not getattr(self, "enable_timing_debug", False):
            return
        if self.timing_sync_cuda and torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device=device)
            except Exception:
                torch.cuda.synchronize()
        dt = time.perf_counter() - start_time
        stat = self._timing_stats.setdefault(name, [0.0, 0])
        stat[0] += float(dt)
        stat[1] += 1

    def reset_timing_stats(self):
        self._timing_stats = {}

    def get_timing_report(self, prefix=""):
        report = {}
        for k, (total, count) in getattr(self, "_timing_stats", {}).items():
            report[prefix + k] = {
                "total": float(total),
                "count": int(count),
                "avg": float(total) / max(int(count), 1),
            }
        return report


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



class DifferentiableLocalSdfCdfConstraintLayer(TimingMixin, nn.Module):
    """
    向量化 PyTorch 局部 SDF-CDF 约束构造层。

    相比上一版逐样本 for-loop：
      - 使用 to_dense_batch 把 PyG 点云转成 [B, Nmax, 2]；
      - 一次性计算全 batch 的 min_dist / rho / z1 / z2；
      - 只调用一次 torch.autograd.grad 得到 grad rho(ego_p)；
      - 输出 G_cdf [B,1,6], h_cdf [B,1]。

    注意：点云最近点 min 本身在最近点切换处不可导，但 alpha/epsilon/rho_floor/margin
    仍然可通过 rho、z1/z2 和 QP loss 反传；这和原 JAX min-SDF 语义一致。
    """
    def __init__(
        self,
        l_k=0.33,
        r_ego=0.31,
        sense_range=3.0,
        alpha_init=0.25,
        alpha_min=0.10,
        alpha_max=0.80,
        epsilon_init=0.25,
        epsilon_min=0.05,
        epsilon_max=0.80,
        rho_floor_init=0.0,
        margin_init=0.0,
        learnable_alpha=True,
        learnable_epsilon=True,
        learnable_rho_floor=False,
        learnable_margin=False,
        valid_point_abs_max=50.0,
        padding_value=99.0,
        eps=1e-6,
        gh_reg_weight=0.0,
        enable_timing_debug=False,
        timing_sync_cuda=True,
    ):
        super().__init__()
        TimingMixin._init_timing(self, enable_timing_debug=enable_timing_debug, timing_sync_cuda=timing_sync_cuda)
        self.l_k = float(l_k)
        self.r_ego = float(r_ego)
        self.sense_range = float(sense_range)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_max = float(epsilon_max)
        self.valid_point_abs_max = float(valid_point_abs_max)
        self.padding_value = float(padding_value)
        self.eps = float(eps)
        self.gh_reg_weight = float(gh_reg_weight)

        if self.alpha_max <= self.alpha_min:
            raise ValueError("alpha_max must be larger than alpha_min")
        if self.epsilon_max <= self.epsilon_min:
            raise ValueError("epsilon_max must be larger than epsilon_min")

        def bounded_raw(init, lo, hi):
            init = min(max(float(init), float(lo) + 1e-6), float(hi) - 1e-6)
            ratio = (init - float(lo)) / (float(hi) - float(lo))
            ratio = min(max(ratio, 1e-6), 1.0 - 1e-6)
            return math.log(ratio / (1.0 - ratio))

        alpha_raw_init = bounded_raw(alpha_init, self.alpha_min, self.alpha_max)
        epsilon_raw_init = bounded_raw(epsilon_init, self.epsilon_min, self.epsilon_max)

        if learnable_alpha:
            self.alpha_raw = nn.Parameter(torch.tensor(alpha_raw_init, dtype=torch.float32))
        else:
            self.register_buffer("alpha_raw", torch.tensor(alpha_raw_init, dtype=torch.float32))

        if learnable_epsilon:
            self.epsilon_raw = nn.Parameter(torch.tensor(epsilon_raw_init, dtype=torch.float32))
        else:
            self.register_buffer("epsilon_raw", torch.tensor(epsilon_raw_init, dtype=torch.float32))

        def inv_softplus(x):
            x = max(float(x), 0.0)
            if x < 1e-10:
                return -20.0
            return math.log(math.exp(x) - 1.0)

        rho_floor_raw_init = inv_softplus(rho_floor_init)
        margin_raw_init = inv_softplus(margin_init)

        if learnable_rho_floor:
            self.rho_floor_raw = nn.Parameter(torch.tensor(rho_floor_raw_init, dtype=torch.float32))
        else:
            self.register_buffer("rho_floor_raw", torch.tensor(rho_floor_raw_init, dtype=torch.float32))

        if learnable_margin:
            self.margin_raw = nn.Parameter(torch.tensor(margin_raw_init, dtype=torch.float32))
        else:
            self.register_buffer("margin_raw", torch.tensor(margin_raw_init, dtype=torch.float32))

        self._last_G_cdf = None
        self._last_h_cdf = None
        self._last_alpha = None
        self._last_epsilon = None
        self._last_rho_floor = None
        self._last_margin = None

    def alpha_value(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.alpha_raw)

    def epsilon_value(self):
        return self.epsilon_min + (self.epsilon_max - self.epsilon_min) * torch.sigmoid(self.epsilon_raw)

    def rho_floor_value(self):
        return F.softplus(self.rho_floor_raw)

    def margin_value(self):
        return F.softplus(self.margin_raw)

    def get_param_dict(self):
        return {
            "cdf_alpha": float(self.alpha_value().detach().cpu().item()),
            "cdf_epsilon": float(self.epsilon_value().detach().cpu().item()),
            "cdf_rho_floor": float(self.rho_floor_value().detach().cpu().item()),
            "cdf_margin": float(self.margin_value().detach().cpu().item()),
        }

    def regularization(self):
        return self.alpha_value() * 0.0

    @staticmethod
    def _smooth_bump(c, b, eps=1e-6):
        denom = c - b
        denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)
        m_k = c / denom
        safe_mk = torch.clamp(m_k, 1e-5, 1.0 - 1e-5)
        exp1 = torch.exp(-1.0 / safe_mk)
        exp2 = torch.exp(-1.0 / (1.0 - safe_mk))
        bump = exp1 / (exp1 + exp2)
        return torch.where(c <= 0.0, torch.zeros_like(bump), torch.where(b >= 0.0, torch.ones_like(bump), bump))

    def _dense_points_and_mask(self, points, batch_size, device, dtype):
        """
        PyG sparse points -> dense [B,N,2] + valid_mask [B,N]。
        使用 torch_geometric.utils.to_dense_batch，避免 Python for-loop 按样本切点云。
        """
        pos = points.pos.to(device=device, dtype=dtype)
        batch_idx = points.batch.to(device=device)

        if pos.numel() == 0:
            pts_dense = torch.full((batch_size, 1, 2), self.padding_value, device=device, dtype=dtype)
            valid_mask = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
            return pts_dense, valid_mask

        pts_dense, mask = to_dense_batch(
            pos,
            batch_idx,
            batch_size=batch_size,
            fill_value=float(self.padding_value),
        )
        # pts_dense: [B,Nmax,2], mask: [B,Nmax]
        finite = torch.isfinite(pts_dense).all(dim=2)
        not_padding = ~(
            (torch.abs(pts_dense[..., 0] - self.padding_value) < 1e-4)
            & (torch.abs(pts_dense[..., 1] - self.padding_value) < 1e-4)
        )
        in_range = (
            (torch.abs(pts_dense[..., 0]) < self.valid_point_abs_max)
            & (torch.abs(pts_dense[..., 1]) < self.valid_point_abs_max)
        )
        valid_mask = mask & finite & not_padding & in_range
        return pts_dense, valid_mask

    def _rho_batch(self, p_local, target_local, pts_dense, valid_mask, alpha, rho_floor):
        """
        p_local:      [B,2]
        target_local: [B,2]
        pts_dense:    [B,N,2]
        valid_mask:   [B,N]
        return rho [B], psi [B]
        """
        B = p_local.shape[0]
        diff = pts_dense - p_local.unsqueeze(1)
        dists = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-12)

        large = torch.full_like(dists, 1e6)
        dists = torch.where(valid_mask, dists, large)
        min_dist = torch.min(dists, dim=1).values

        # 若某个样本没有有效点，min_dist 会是 1e6。此时 b>=0, psi=1，等价于无近障碍。
        c_val = min_dist - float(self.r_ego)
        b_val = min_dist - float(self.sense_range)
        psi = self._smooth_bump(c_val, b_val, eps=self.eps)

        V_x = torch.sum((p_local - target_local) ** 2, dim=1)
        rho = psi / (torch.pow(V_x + self.eps, alpha) + self.eps) + rho_floor
        return rho, psi

    def forward(self, state, points, u_nom):
        # 即便 validation 外层用了 torch.no_grad()，这里也必须启用梯度：
        # 需要一次 autograd.grad 计算 rho 对 ego 查询点的梯度。
        t_total = self._timing_now(device=u_nom.device)
        with torch.enable_grad():
            batch_size = state.shape[0]
            device = u_nom.device
            dtype = u_nom.dtype

            target_local = state[:, 0:2].to(device=device, dtype=dtype)

            t0 = self._timing_now(device=device)
            pts_dense, valid_mask = self._dense_points_and_mask(points, batch_size, device, dtype)
            self._timing_add("cdf/dense_points", t0, device=device)

            alpha = self.alpha_value().to(device=device, dtype=dtype)
            epsilon = self.epsilon_value().to(device=device, dtype=dtype)
            rho_floor = self.rho_floor_value().to(device=device, dtype=dtype)
            margin = self.margin_value().to(device=device, dtype=dtype)

            ego_p = torch.zeros(batch_size, 2, device=device, dtype=dtype)
            ego_p[:, 0] = float(self.l_k)
            ego_req = ego_p.detach().clone().requires_grad_(True)

            t0 = self._timing_now(device=device)
            rho_curr_for_grad, _ = self._rho_batch(
                ego_req,
                target_local,
                pts_dense,
                valid_mask,
                alpha,
                rho_floor,
            )
            grad_self = torch.autograd.grad(
                rho_curr_for_grad.sum(),
                ego_req,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )[0]
            self._timing_add("cdf/rho_grad", t0, device=device)

            t0 = self._timing_now(device=device)
            norm_nom = torch.linalg.norm(u_nom, dim=1, keepdim=True)
            default_nom = torch.zeros_like(u_nom)
            default_nom[:, 0] = 1.0
            dir_nom = torch.where(norm_nom > 1e-5, u_nom / (norm_nom + 1e-8), default_nom)
            z1_pos = ego_p + epsilon * dir_nom

            neg_grad = -grad_self
            norm_grad = torch.linalg.norm(neg_grad, dim=1, keepdim=True)
            default_safe = torch.zeros_like(u_nom)
            default_safe[:, 1] = 1.0
            dir_safe = torch.where(norm_grad > 1e-5, neg_grad / (norm_grad + 1e-8), default_safe)
            z2_pos = ego_p + epsilon * dir_safe

            v1 = dir_nom
            v2_raw = dir_safe
            det_v = v1[:, 0] * v2_raw[:, 1] - v1[:, 1] * v2_raw[:, 0]
            v2_ortho = torch.stack([-v1[:, 1], v1[:, 0]], dim=1)
            v2 = torch.where((torch.abs(det_v) > 1e-2).unsqueeze(1), v2_raw, v2_ortho)

            # W = inverse([v1 v2])，每个样本一个 2x2。显式公式比 torch.linalg.inv 更轻。
            a = v1[:, 0]
            c = v1[:, 1]
            b = v2[:, 0]
            d = v2[:, 1]
            det = a * d - b * c
            det_safe = torch.where(
                torch.abs(det) < 1e-6,
                torch.where(det >= 0.0, torch.full_like(det, 1e-6), torch.full_like(det, -1e-6)),
                det,
            )
            # V = [[a,b],[c,d]], inv(V) = 1/det [[d,-b],[-c,a]]
            w1 = torch.stack([d / det_safe, -b / det_safe], dim=1)
            w2 = torch.stack([-c / det_safe, a / det_safe], dim=1)
            self._timing_add("cdf/dirs_and_basis", t0, device=device)

            t0 = self._timing_now(device=device)
            rho_curr, _ = self._rho_batch(ego_p, target_local, pts_dense, valid_mask, alpha, rho_floor)
            rho_z1, _ = self._rho_batch(z1_pos, target_local, pts_dense, valid_mask, alpha, rho_floor)
            rho_z2, _ = self._rho_batch(z2_pos, target_local, pts_dense, valid_mask, alpha, rho_floor)
            self._timing_add("cdf/rho_curr_z1_z2", t0, device=device)

            t0 = self._timing_now(device=device)
            inv_eps = 1.0 / (epsilon + self.eps)
            sum_w = w1 + w2
            coeff_u = (rho_curr * inv_eps).unsqueeze(1) * sum_w
            coeff_z1 = -(rho_z1 * inv_eps).unsqueeze(1) * w1
            coeff_z2 = -(rho_z2 * inv_eps).unsqueeze(1) * w2

            G_cdf = torch.zeros(batch_size, 1, 6, device=device, dtype=dtype)
            G_cdf[:, 0, 0:2] = coeff_u
            G_cdf[:, 0, 2:4] = coeff_z1
            G_cdf[:, 0, 4:6] = coeff_z2

            h_cdf = (-rho_curr - margin).view(batch_size, 1)
            self._timing_add("cdf/build_Gh", t0, device=device)

        self._timing_add("cdf/total", t_total, device=u_nom.device)
        self._last_G_cdf = G_cdf
        self._last_h_cdf = h_cdf
        self._last_alpha = self.alpha_value().detach()
        self._last_epsilon = self.epsilon_value().detach()
        self._last_rho_floor = self.rho_floor_value().detach()
        self._last_margin = self.margin_value().detach()
        return G_cdf, h_cdf

class DifferentiableSdfCdfSafetyLayer6D(TimingMixin, nn.Module):
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
        learnable_lambda_smooth=False,
        lambda_smooth_min=0.1,
        lambda_smooth_max=80.0,
        lambda_reg_weight=1e-4,
        enable_timing_debug=False,
        timing_sync_cuda=True,
    ):
        super().__init__()
        TimingMixin._init_timing(self, enable_timing_debug=enable_timing_debug, timing_sync_cuda=timing_sync_cuda)

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

        # 统计量：用于 trainer 每个 epoch 打印 fail_rate
        self._qp_warning_count = 0
        self._qp_call_count = 0
        self._qp_fail_count = 0
        self._qp_invalid_warning_count = 0
        self._qp_invalid_zero_count = 0
        self._qp_infeasible_main_count = 0

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

    def get_qp_stats(self):
        fail_rate = self._qp_fail_count / max(self._qp_call_count, 1)
        return {
            "qp_call_count": self._qp_call_count,
            "qp_fail_count": self._qp_fail_count,
            "qp_fail_rate": fail_rate,
            "qp_warning_count": self._qp_warning_count,
            "qp_invalid_zero_count": self._qp_invalid_zero_count,
            "qp_infeasible_main_count": self._qp_infeasible_main_count,
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
        t_total = self._timing_now(device=device)
        t0 = self._timing_now(device=device)

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

        self._timing_add("safety/build_objective", t0, device=device)
        t0 = self._timing_now(device=device)

        # 3. 主 CDF 约束
        G_main = G_cdf_6d.to(device=device, dtype=dtype)
        h_main = h_cdf.to(device=device, dtype=dtype)

        # 防御 h shape 异常：允许 [B]，统一成 [B, 1]
        if h_main.dim() == 1:
            h_main = h_main.view(batch_size, 1)

        # 在任何归一化之前检查原始 G/h，方便直接定位专家数据中的异常约束。
        self._check_constraint_pathologies(G_main, h_main, stage="raw_before_normalization")

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

        self._timing_add("safety/build_constraints", t0, device=device)

        # 5. 空等式约束
        e = torch.empty(0, device=device, dtype=dtype)
        A = torch.empty(0, device=device, dtype=dtype)

        try:
            t0 = self._timing_now(device=device)
            sol_6d = QPFunction(
                verbose=False,
                solver=QPSolvers.PDIPM_BATCHED,
                maxIter=self.qp_max_iter,
                eps=self.qp_eps,
                notImprovedLim=self.qp_not_improved_lim,
            )(P_in, g_in, G_all, h_all, e, A)

            self._timing_add("safety/qpth_solve", t0, device=device)

            if torch.isnan(sol_6d).any() or torch.isinf(sol_6d).any():
                raise ValueError("qpth_nan_or_inf_detected")

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

        self._timing_add("safety/total", t_total, device=device)
        return sol_6d[:, 0:2]

class UNet(TimingMixin, nn.Module):
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
        learnable_lambda_smooth=False,
        lambda_smooth_min=0.1,
        lambda_smooth_max=80.0,
        lambda_reg_weight=1e-4,
        use_learned_cdf_constraints=False,
        cdf_l_k=0.33,
        cdf_r_ego=0.31,
        cdf_sense_range=3.0,
        cdf_alpha_init=0.25,
        cdf_alpha_min=0.10,
        cdf_alpha_max=0.80,
        cdf_epsilon_init=0.25,
        cdf_epsilon_min=0.05,
        cdf_epsilon_max=0.80,
        cdf_rho_floor_init=0.0,
        cdf_margin_init=0.0,
        learnable_cdf_alpha=True,
        learnable_cdf_epsilon=True,
        learnable_cdf_rho_floor=False,
        learnable_cdf_margin=False,
        cdf_valid_point_abs_max=50.0,
        cdf_padding_value=99.0,
        gh_loss_weight=0.0,
        enable_timing_debug=False,
        timing_sync_cuda=True,
        ablation="full",
    ):
        super().__init__()
        TimingMixin._init_timing(self, enable_timing_debug=enable_timing_debug, timing_sync_cuda=timing_sync_cuda)

        self.ablation = ablation
        self.use_dual_branch = ablation != "no_dual"
        self.use_safety_layer = ablation != "no_safety"
        self.use_learned_cdf_constraints = bool(use_learned_cdf_constraints)
        self.gh_loss_weight = float(gh_loss_weight)
        self._last_G_cdf_pred = None
        self._last_h_cdf_pred = None

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

        if self.use_learned_cdf_constraints and self.use_safety_layer:
            self.cdf_constraint_layer = DifferentiableLocalSdfCdfConstraintLayer(
                l_k=cdf_l_k,
                r_ego=cdf_r_ego,
                sense_range=cdf_sense_range,
                alpha_init=cdf_alpha_init,
                alpha_min=cdf_alpha_min,
                alpha_max=cdf_alpha_max,
                epsilon_init=cdf_epsilon_init,
                epsilon_min=cdf_epsilon_min,
                epsilon_max=cdf_epsilon_max,
                rho_floor_init=cdf_rho_floor_init,
                margin_init=cdf_margin_init,
                learnable_alpha=learnable_cdf_alpha,
                learnable_epsilon=learnable_cdf_epsilon,
                learnable_rho_floor=learnable_cdf_rho_floor,
                learnable_margin=learnable_cdf_margin,
                valid_point_abs_max=cdf_valid_point_abs_max,
                padding_value=cdf_padding_value,
                gh_reg_weight=gh_loss_weight,
                enable_timing_debug=enable_timing_debug,
                timing_sync_cuda=timing_sync_cuda,
            )
        else:
            self.cdf_constraint_layer = None

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
                learnable_lambda_smooth=learnable_lambda_smooth,
                lambda_smooth_min=lambda_smooth_min,
                lambda_smooth_max=lambda_smooth_max,
                lambda_reg_weight=lambda_reg_weight,
                enable_timing_debug=enable_timing_debug,
                timing_sync_cuda=timing_sync_cuda,
            )
        else:
            self.safety_layer = None

    def forward(self, state, points, G_cdf, h_cdf):
        device = state.device
        t_total = self._timing_now(device=device)
        t0 = self._timing_now(device=device)
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
        self._timing_add("model/network_forward", t0, device=device)

        if self.use_safety_layer:
            if self.use_learned_cdf_constraints:
                t0 = self._timing_now(device=device)
                G_used, h_used = self.cdf_constraint_layer(state, points, u_nom)
                self._timing_add("model/cdf_constraints", t0, device=device)
                self._last_G_cdf_pred = G_used
                self._last_h_cdf_pred = h_used
            else:
                G_used, h_used = G_cdf, h_cdf
                self._last_G_cdf_pred = None
                self._last_h_cdf_pred = None
            t0 = self._timing_now(device=device)
            u_safe = self.safety_layer(u_nom, G_used, h_used)
            self._timing_add("model/safety_layer", t0, device=device)
        else:
            # w/o safety layer：网络直接输出动作
            u_safe = u_nom
            self._last_G_cdf_pred = None
            self._last_h_cdf_pred = None

        self._timing_add("model/total_forward", t_total, device=device)

        # 保持 trainer 接口不变
        return u_safe, u_nom

    def reset_timing_stats(self):
        TimingMixin.reset_timing_stats(self)
        if self.cdf_constraint_layer is not None and hasattr(self.cdf_constraint_layer, "reset_timing_stats"):
            self.cdf_constraint_layer.reset_timing_stats()
        if self.safety_layer is not None and hasattr(self.safety_layer, "reset_timing_stats"):
            self.safety_layer.reset_timing_stats()

    def get_timing_report(self):
        report = TimingMixin.get_timing_report(self, prefix="")
        if self.cdf_constraint_layer is not None and hasattr(self.cdf_constraint_layer, "get_timing_report"):
            report.update(self.cdf_constraint_layer.get_timing_report(prefix=""))
        if self.safety_layer is not None and hasattr(self.safety_layer, "get_timing_report"):
            report.update(self.safety_layer.get_timing_report(prefix=""))
        return report

    def cdf_parameter_dict(self):
        if self.cdf_constraint_layer is None:
            return {}
        return self.cdf_constraint_layer.get_param_dict()