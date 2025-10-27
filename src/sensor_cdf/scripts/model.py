# 对应cdfnet收敛到0.15的版本 gradnet为1.5
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool, knn_graph, GATConv

class GeometricEncoder(nn.Module):
    """多尺度图卷积编码器"""
    def __init__(self, in_dim=2, hidden_dim=256, k=6):
        super().__init__()
        self.k = k
        
        # 第一层图卷积
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
        
        # 第二层图注意力卷积
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,  # 保持输出维度一致以支持残差
            heads=8,                  # 增加注意力头数
            dropout=0.2,
            concat=False,             # 输出维度不拼接，直接平均
            add_self_loops=False      # 避免重复连接
        )
        self.res_fc = nn.Linear(hidden_dim, hidden_dim) if hidden_dim != hidden_dim else nn.Identity()  
        
        # 第三层图卷积
        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2*hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1)
            ),
            k=self.k,
            aggr='max'
        )
        
        # 特征降维
        self.downsample = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3)
        )

    def forward(self, points, mask=None):
        """
        points: [batch_size, max_points, 2]
        mask: [batch_size, max_points] 
        """
        x = points.pos  # [num_nodes, 2]
        batch_idx = points.batch  # [num_nodes]
        
        # 第一层局部特征提取
        edge_index = knn_graph(x, k=self.k, batch=batch_idx,loop=True)

        x1 = self.conv1(x=(x, x))
        x1 = self.norm1(x1)
        x1 = F.relu(x1)
        
        # 第二层图注意力
        # x2 = F.leaky_relu(self.conv2(x1, edge_index), 0.2)
        x2 = self.conv2(x1, edge_index)
        x2 = F.leaky_relu(x2 + self.res_fc(x1), 0.2)
        
        # 第三层全局关联
        x3 = self.conv3(x=x2, batch=batch_idx)
        
        # 多尺度特征融合
        x_multi = torch.cat([x1, x2, x3], dim=1)
        x_multi = self.downsample(x_multi)
        
        # 全局池化
        x_global = global_max_pool(x_multi, batch_idx)  # [batch, hidden]
        
        return x_global

class GradNet(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=256):
        super().__init__()
        self.geo_encoder = GeometricEncoder(hidden_dim=hidden_dim, k=10)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            nn.Dropout(0.3),
            
            # 双重残差块
            # ResidualBlock(hidden_dim*4),
            # ResidualBlock(hidden_dim*4),
            
            # 渐进降维
            nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim*3),
                nn.LayerNorm(hidden_dim*3),
                nn.SiLU(),
                nn.Dropout(0.3),
                
                nn.Linear(hidden_dim*3, hidden_dim*2),
                nn.LayerNorm(hidden_dim*2),
                nn.SiLU(),
                
                nn.Linear(hidden_dim*2, hidden_dim),
            ),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim*2, 2)
        )

    def forward(self, state, points):
        geo_feat = self.geo_encoder(points)
        state_feat = self.state_encoder(state)
        fused = self.fusion(torch.cat([geo_feat, state_feat], dim=1))
        return self.head(fused)   # 输出梯度

class ResidualBlock(nn.Module):
    """残差块增强特征提取"""
    def __init__(self, features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, features),
            nn.LayerNorm(features),
            nn.GELU(),
            nn.Linear(features, features),
            nn.LayerNorm(features)
        )
    
    def forward(self, x):
        return x + self.block(x)

class CDFNet(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=256):
        super().__init__()
        self.geo_encoder = GeometricEncoder(hidden_dim=hidden_dim, k=10)
        # 残差状态编码
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*2),
            ResidualBlock(hidden_dim*2),
            ResidualBlock(hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim),  # 新增维度转换层
            nn.LayerNorm(hidden_dim)              # 调整LayerNorm维度
        )
        
        # 门控融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # CDF预测头
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Softplus(beta=0.3)  # 调整beta值
        )

    def forward(self, state, points):
        geo_feat = self.geo_encoder(points)
        state_feat = self.state_encoder(state)
        
        # 门控融合
        combined = torch.cat([geo_feat, state_feat, geo_feat*state_feat], dim=1)
        gate = self.fusion_gate(combined)
        fused = self.fusion(combined) * gate
        
        return self.head(fused)


