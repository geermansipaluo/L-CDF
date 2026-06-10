#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, knn_graph, GATConv
# 🟢 引入 BarrierNet 同款工业级可微参数化凸优化层
from qpth.qp import QPFunction, QPSolvers

class GeometricEncoder(nn.Module):
    """ 多尺度点云图卷积编码器 —— 完美融汇变长局部空间拓扑 """
    def __init__(self, in_dim=2, hidden_dim=512, k=10):
        super().__init__()
        self.k = k
        
        # 第一层图卷积：Dynamic Edge Conv (局部邻域均值消息传递)
        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2*in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            ),
            k=self.k,
            aggr='mean'
        )
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        
        # 第二层图注意力卷积：GATConv 
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,  
            heads=8,                  
            #dropout=0.2,
            concat=False,             
            add_self_loops=False      
        )
        self.res_fc = nn.Identity()  
        
        # 第三层图卷积：Dynamic Edge Conv (提取极限边缘斥力特征)
        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            ),
            k=self.k,
            aggr='max'
        )
        
        # 特征降维融合层
        self.downsample = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            #nn.Dropout(0.3)
        )

    def forward(self, points):
        """
        points: 传入标准 PyG 同质图 Data 对象 (包含 points.pos 和 points.batch)
        """
        x = points.pos           # [Total_Points, 2] 局部坐标下的点云坐标
        batch_idx = points.batch # [Total_Points]
        
        # 1. 动态构建 KNN 图并进行第一层边缘卷积
        edge_index = knn_graph(x, k=self.k, batch=batch_idx, loop=True)
        x1 = self.conv1(x, batch_idx)
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        
        # 2. 第二层图注意力网络，带残差保护
        x2 = self.conv2(x1, edge_index)
        x2 = F.leaky_relu(x2 + self.res_fc(x1), 0.2)
        
        # 3. 第三层高动态特征提取
        x3 = self.conv3(x2, batch_idx)
        
        # 4. 级联多尺度空间几何特征
        x_multi = torch.cat([x1, x2, x3], dim=1)
        x_multi = self.downsample(x_multi)
        
        # 5. 全局最大池化：彻底干掉不规则变长输入，秒变固定维数几何特征
        x_global = global_max_pool(x_multi, batch_idx)  # [Batch_Size, hidden_dim]
        return x_global


class DifferentiableSdfCdfSafetyLayer6D(nn.Module):
    """
    包装成标准神经网络层的 6维升维参数化可微安全层
    它在前向调用 qpth 批量解算 6维凸优化，在反向自动完成 7x7 维的 KKT 齐次矩阵微分传导
    """
    def __init__(self, lambda_smooth=1):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        
    def forward(self, u_nom, G_cdf_6d, h_cdf):
        """
        u_nom: UNet 吐出的狂野动作名义量 [Batch_Size, 2]
        G_cdf_6d: 数据集中切片出来的完整 6维约束矩阵系数 [Batch_Size, 1, 6]
        h_cdf: 约束势能上限 [Batch_Size, 1]
        """
        batch_size = u_nom.shape[0]
        device = u_nom.device

        G_u = G_cdf_6d[:, :, 0:2] # [Batch_Size, 1, 2]
        # 计算名义量当前的约束违反度
        constraint_violation = torch.bmm(G_u, u_nom.unsqueeze(2)).squeeze(2) - h_cdf # [Batch_Size, 1]
        # 如果违反严重，将动作向内做一次解析衰减，极大地给 qpth 减压
        decay_factor = torch.where(constraint_violation > 1, 1.0 / (1.0 + constraint_violation), torch.ones_like(constraint_violation))
        u_nom_projected = u_nom * decay_factor

        # h_clipped = torch.clamp(h_cdf.view(batch_size), min=1e-5, max=1.0)
        # self.lambda_smooth = 1.0 + 30.0 * torch.exp(-15.0 * h_clipped)
        
        # 1. 🟢【参数化升维代数重组】：在 PyTorch 内部完美重组专家 6x6 的正定代价矩阵 H
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
        
        # 2. 🟢 完美重组 6维线性项向量 g (前两位挂载名义驱动，后四位辅助位置补零)
        g_in = torch.zeros(batch_size, 6, device=device)
        g_in[:, 0:2] = -2.0 * u_nom
        # g_in[:, 0:2] = -2.0 * u_nom_projected
        
        # 3. 建立零维等式约束占位符
        e = torch.Tensor().to(device)
        A = torch.Tensor().to(device)
        
        try:
            # 采用具有迭代细化扩展的 PDIPM_BATCHED 求解器，增强数学鲁棒性
            sol_6d = QPFunction(verbose=False, solver=QPSolvers.PDIPM_BATCHED)(P_in, g_in, G_cdf_6d, h_cdf, e, A)
        except Exception as e_qp:
            # 如果遇到极端边界退化引发求解崩溃，通过标称直驱机制进行无损防御兜底
            print(f"⚠️ [qpth 数值临界拦截] 捕获极端奇异发散，启用标称柔性降维防线.")
            sol_6d = torch.zeros(batch_size, 6, device=device)
            sol_6d[:, 0:2] = u_nom

        u_safe = sol_6d[:, 0:2]
        # 5. 返回 6维完整最优状态（供 Loss 针对 6维进行全局全状态惩罚洗礼）
        return u_safe


class UNet(nn.Module):
    """
    大合拢端到端参数化可微控制策略网络 (完全契合 6维辅助松弛松弛版)
    """
    def __init__(self, state_dim=4, hidden_dim=512):
        super().__init__()
        
        # 各分支编码器
        self.geo_encoder = GeometricEncoder(hidden_dim=hidden_dim, k=5)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            #nn.Dropout(0.2)
        )
        # self.state_encoder = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     #nn.Dropout(0.2)
        # )
        
        # 跨模态特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            #nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 控制策略输出头 (单任务直出 2 维标称动作量)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, 2),
            nn.Tanh()  # 输出连续物理空间的标称期望 (v_nom, w_nom)
        )
        # self.head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim//2),
        #     nn.SiLU(),
        #     nn.Linear(hidden_dim//2, 32),
        #     nn.SiLU(),
        #     nn.Linear(32,2)  # 输出连续物理空间的标称期望 (v_nom, w_nom)
        # )
        
        # 固锁可微安全阻尼过滤层
        self.safety_layer = DifferentiableSdfCdfSafetyLayer6D(lambda_smooth=25)

    def forward(self, state, points, G_cdf, h_cdf):
        """
        前向传播接口：
          训练端调用：pred_sol_6d = model(batch.state, batch, batch.G_cdf, batch.h_cdf)
        """
        # 1. 深度级联两路异质输入特征
        geo_feat = self.geo_encoder(points)      # [Batch_Size, hidden_dim]
        state_feat = self.state_encoder(state)    # [Batch_Size, hidden_dim]
        
        # 2. 空间几何特征与动态状态深度交融
        fused = self.fusion(torch.cat([geo_feat, state_feat], dim=1)) # [Batch_Size, hidden_dim*2]
        
        # 3. 策略网络原生直出的狂野名义控制量
        u_nom = self.head(fused)*1.2 # [Batch_Size, 2]
        
        # 4. 🚨 让动作名义量和空间 6维约束场强送入安全拦截大闸，前向求解 6维 QP，反向隐式传梯
        u_safe = self.safety_layer(u_nom, G_cdf, h_cdf) # [Batch_Size, 6]
        
        return u_safe