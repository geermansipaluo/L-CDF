#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, knn_graph, GATConv

class GeometricEncoder(nn.Module):
    """ 多尺度图卷积编码器 —— 提取未知的点云空间几何特征 """
    def __init__(self, in_dim=2, hidden_dim=256, k=6):
        super().__init__()
        self.k = k
        
        # 第一层图卷积：Dynamic Edge Conv (采用 mean 聚合局部邻域)
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
        
        # 第二层图注意力卷积：GATConv (利用多头注意力机制捕捉局部细节)
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,  # 保持输出维度一致以支持残差连接
            heads=8,                  # 8个注意力头
            dropout=0.2,
            concat=False,             # 平均多头特征
            add_self_loops=False      
        )
        self.res_fc = nn.Linear(hidden_dim, hidden_dim) if hidden_dim != hidden_dim else nn.Identity()  
        
        # 第三层图卷积：Dynamic Edge Conv (采用 max 聚合边缘特征)
        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            ),
            k=self.k,
            aggr='max'
        )
        
        # 特征多尺度降维融合层
        self.downsample = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3)
        )

    def forward(self, points, mask=None):
        """
        points: PyG Data 对象，包含 pos [num_nodes, 2] 和 batch [num_nodes]
        """
        x = points.pos  
        batch_idx = points.batch  
        
        # 1. 动态构建局部 KNN 图并进行第一层卷积
        edge_index = knn_graph(x, k=self.k, batch=batch_idx, loop=True)
        x1 = self.conv1(x=(x, x))
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        
        # 2. 第二层图注意力层，带残差连接
        x2 = self.conv2(x1, edge_index)
        x2 = F.leaky_relu(x2 + self.res_fc(x1), 0.2)
        
        # 3. 第三层全局关联提取
        x3 = self.conv3(x=x2, batch=batch_idx)
        
        # 4. 多尺度特征拼接与级联融合
        x_multi = torch.cat([x1, x2, x3], dim=1)
        x_multi = self.downsample(x_multi)
        
        # 5. 图全局最大池化，生成环境维度的固定维数几何特征
        x_global = global_max_pool(x_multi, batch_idx)  
        return x_global

class DensityNet(nn.Module):
    """
    DensityNet 核心纯端到端控制策略网络 (双头输出版)
    输入：
        state: 机器人局部相对状态特征 [Batch, state_dim=4] 
               -> (target_local_x, target_local_y, cos_yaw_err, sin_yaw_err)
        points: 局部激光点云图 (PyG Data 对象)
    输出：
        action: 连续控制指令 [Batch, 2] -> (v, omega)
        risk:   解耦的纯几何风险标量 [Batch, 1] -> \sigma \in [0, 1] (逼近 1.0 - \psi)
    """
    def __init__(self, state_dim=4, hidden_dim=256):
        super().__init__()
        # 点云几何图分支编码器 (保持高效的多尺度特征拓扑提取)
        self.geo_encoder = GeometricEncoder(hidden_dim=hidden_dim, k=10)
        
        # 机器人纯相对状态分支编码器 (接收移除绝对坐标污染后的 4 维纯净输入)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2)
        )
        
        # 跨模态高维特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 🔥【头分支 1】控制策略输出头 (直出 2 维连续动作量：线速度与角速度)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, 2)
        )

        # 🔥【头分支 2】物理信息指导的几何风险预测头 (输出 1 维风险标量 \sigma)
        # 关键改动：末尾采用 nn.Sigmoid() 严格限制输出范围在 [0, 1]，完美对应 1.0 - \psi 靶点
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, state, points):
        # 1. 分别提取异质输入的两路高维特征
        geo_feat = self.geo_encoder(points)
        state_feat = self.state_encoder(state)
        
        # 2. 拼接空间特征与动态状态并进行全连接深度融合
        fused = self.fusion(torch.cat([geo_feat, state_feat], dim=1))
        
        # 3. 双头解耦前向传播输出
        action = self.action_head(fused)  # 控制量 [batch, 2] -> (v, omega)
        risk = self.risk_head(fused)      # 风险度 [batch, 1] -> \sigma \in [0, 1]
        
        return action, risk