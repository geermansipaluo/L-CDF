#!/usr/bin/env python3
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
    包装成标准神经网络层的 6维升维参数化可微安全层
    它在前向调用 qpth 批量解算 6维凸优化，在反向自动完成 7x7 维的 KKT 齐次矩阵微分传导
    """
    def __init__(
        self,
        lambda_smooth=1,
        qp_limit=1.2,
        use_qp_box_constraints=False,
        qp_jitter=1e-4,
        qp_normalize_constraints=True,
        qp_constraint_scale_floor=1.0,
    ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.qp_limit = qp_limit
        self.use_qp_box_constraints = use_qp_box_constraints
        self.qp_jitter = qp_jitter
        self.qp_normalize_constraints = qp_normalize_constraints
        self.qp_constraint_scale_floor = qp_constraint_scale_floor
        self._qp_warning_count = 0
        
    def forward(self, u_nom, G_cdf_6d, h_cdf):
        """
        u_nom: UNet 吐出的狂野动作名义量 [Batch_Size, 2]
        G_cdf_6d: 数据集中切片出来的完整 6维约束矩阵系数 [Batch_Size, 1, 6]
        h_cdf: 约束势能上限 [Batch_Size, 1]
        """
        batch_size = u_nom.shape[0]
        device = u_nom.device

        #h_clipped = torch.clamp(h_cdf.view(batch_size), min=1e-5, max=1.0)
        #self.lambda_smooth = 1.0 + 30.0 * torch.exp(-15.0 * h_clipped)
        
        # 1. 【参数化升维代数重组】：在 PyTorch 内部完美重组专家 6x6 的正定代价矩阵 H
        val_uu = 2.0 * (1.0 + 2.0 * self.lambda_smooth)
        val_zz = 2.0 * self.lambda_smooth
        val_uz = -2.0 * self.lambda_smooth
        
        P_in = torch.zeros(batch_size, 6, 6, device=device)
        
        # 填充动作和松弛自相关的对角项
        P_in[:, 0, 0] = val_uu; P_in[:, 1, 1] = val_uu
        P_in[:, 2, 2] = val_zz; P_in[:, 3, 3] = val_zz
        P_in[:, 4, 4] = val_zz; P_in[:, 5, 5] = val_zz
        
        # 填充动作与松弛交叉互相关的非对角耦合项
        P_in[:, 0, 2] = val_uz; P_in[:, 2, 0] = val_uz
        P_in[:, 1, 3] = val_uz; P_in[:, 3, 1] = val_uz
        P_in[:, 0, 4] = val_uz; P_in[:, 4, 0] = val_uz
        P_in[:, 1, 5] = val_uz; P_in[:, 5, 1] = val_uz

        # qpth 的 PDIPM 对半正定/病态 Hessian 比 JaxProxQP 更敏感。
        # 理论 H 已正定，但加极小 jitter 可以显著减少 KKT 分解临界失败。
        if self.qp_jitter is not None and float(self.qp_jitter) > 0.0:
            P_in = P_in + float(self.qp_jitter) * torch.eye(6, device=device, dtype=u_nom.dtype).unsqueeze(0)
        
        # 2.  完美重组 6维线性项向量 g (前两位挂载名义驱动，后四位辅助位置补零)
        g_in = torch.zeros(batch_size, 6, device=device)
        g_in[:, 0:2] = -2.0 * u_nom
        
        # 3. 约束预处理。
        #    G/h 来自数据生成器的 SDF-CDF-QP。数值上，G 的量级可能因 rho/epsilon
        #    在不同样本间差很多；行归一化不改变可行域，但能明显改善 qpth 条件数。
        G_main = G_cdf_6d.to(device=device, dtype=u_nom.dtype)
        h_main = h_cdf.to(device=device, dtype=u_nom.dtype)
        if self.qp_normalize_constraints:
            row_scale = torch.linalg.norm(G_main, dim=2, keepdim=True)
            row_scale = torch.clamp(row_scale, min=float(self.qp_constraint_scale_floor))
            G_main = G_main / row_scale
            h_main = h_main / row_scale.squeeze(-1)

        if self.use_qp_box_constraints:
            # 可选：显式加入和 JAX/JaxProxQP 专家一致的 6D box 约束。
            # 默认关闭，因为 qpth 在动作刚好贴着 +/-1.2 边界时容易出现无严格内点/残差警告。
            eye6 = torch.eye(6, device=device, dtype=u_nom.dtype).unsqueeze(0).repeat(batch_size, 1, 1)
            box_G = torch.cat([eye6, -eye6], dim=1)
            box_h = torch.full(
                (batch_size, 12),
                float(self.qp_limit),
                device=device,
                dtype=u_nom.dtype,
            )
            G_all = torch.cat([G_main, box_G], dim=1)
            h_all = torch.cat([h_main, box_h], dim=1)
        else:
            G_all = G_main
            h_all = h_main

        # 4. 建立零维等式约束占位符
        e = torch.Tensor().to(device)
        A = torch.Tensor().to(device)
        
        try:
            # 采用具有迭代细化扩展的 PDIPM_BATCHED 求解器，增强数学鲁棒性
            sol_6d = QPFunction(verbose=False, solver=QPSolvers.PDIPM_BATCHED)(P_in, g_in, G_all, h_all, e, A)
            if torch.isnan(sol_6d).any() or torch.isinf(sol_6d).any():
                # 发现毒素，主动触发异常走防御兜底
                raise ValueError("qpth_nan_detected")
        except Exception as e_qp:
            # 如果遇到极端边界退化引发求解崩溃，通过标称直驱机制兜底。
            # 限制打印次数，避免日志被 qpth 临界样本刷屏。
            if self._qp_warning_count < 20:
                print(f"⚠️ [qpth 数值临界拦截] {type(e_qp).__name__}: 启用标称柔性降维防线.")
            elif self._qp_warning_count == 20:
                print("⚠️ [qpth 数值临界拦截] 后续同类警告将静默统计，不再刷屏。")
            self._qp_warning_count += 1
            sol_6d = torch.zeros(batch_size, 6, device=device, dtype=u_nom.dtype)
            sol_6d[:, 0:2] = u_nom

        u_safe = sol_6d[:, 0:2]
        # 5. 返回 6维完整最优状态（供 Loss 针对 6维进行全局全状态惩罚洗礼）
        return u_safe


class UNet(nn.Module):
    def __init__(
        self,
        state_dim=4,
        hidden_dim=512,
        graph_k=10,
        lambda_smooth=25.0,
        qp_limit=1.2,
        use_qp_box_constraints=False,
        qp_jitter=1e-4,
        qp_normalize_constraints=True,
        qp_constraint_scale_floor=1.0,
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
                qp_constraint_scale_floor=qp_constraint_scale_floor
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