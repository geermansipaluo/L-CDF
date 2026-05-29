#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, global_max_pool

# =========================================================================
# 1. 基础多层前馈感知机算子 (MLP)
# =========================================================================
def build_mlp(in_dim, hidden_dims, out_dim, activation=nn.LeakyReLU, dropout=0.0):
    layers = []
    current_dim = in_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(nn.LayerNorm(h_dim))
        layers.append(activation(0.1))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = h_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)

# =========================================================================
# 2. 完全对齐最新异质图规范的 DensityNet 控制策略网络
# =========================================================================
class DensityNet(nn.Module):
    """
    DensityNet 核心多任务异质图控制网络。
    完全看齐 GCBF+ 论文架构，雷达点云消息与目标导航消息通过独立的图卷积分流聚拢，
    彻底从代数框架上消灭了 Batch_Size 与变长点云的维度不匹配硬伤。
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        
        # -----------------------------------------------------------------
        # 三实体独立高维空间投影编码器 (Entity Encoders)
        # -----------------------------------------------------------------
        # 自车 ego 状态输入 4维: [x, y, theta, dist_to_goal]
        self.ego_encoder = build_mlp(4, [hidden_dim], hidden_dim)
        # 导航引力节点 goal 相对输入 2维: [target_local_x, target_local_y]
        self.goal_encoder = build_mlp(2, [hidden_dim], hidden_dim)
        # 避障斥力节点 point 输入 2维: [pc_x, pc_y]
        self.point_encoder = build_mlp(2, [hidden_dim], hidden_dim)
        
        # -----------------------------------------------------------------
        # 🔴 论文同款：星形图双向消息异质图卷积 (Hetero Message Passing)
        # -----------------------------------------------------------------
        # 雷达点和引力靶点同时通过其 edge_index 关系，无缝向自车节点汇聚特征
        # PyG 在底层会自动处理 13015 到 256 的多对一消息合并，免去外界任何手动广播代码
        self.hetero_conv = HeteroConv({
            ('point', 'to', 'ego'): SAGEConv(in_channels=(hidden_dim, hidden_dim), 
                                             out_channels=hidden_dim, 
                                             aggr='mean'),
            ('goal', 'to', 'ego'): SAGEConv(in_channels=(hidden_dim, hidden_dim), 
                                            out_channels=hidden_dim, 
                                            aggr='mean')
        }, aggr='sum') # 将斥力场的障碍物特征与引力场的目标特征在自车节点处实施物理求和
        
        # 强力跨模态特征多层融合精炼
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        
        # -----------------------------------------------------------------
        # 解耦多任务直出头 (Decoupled Heads)
        # -----------------------------------------------------------------
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 2)  # 输出 (v, omega)
        )

        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 约束物理风险概率场强在 [0, 1] 空间内
        )

    def forward(self, batch):
        """
        前向计算流只接收一个完全自治、统一的大图 batch 对象。
        """
        # 1. 实体特征进行高维编码
        e_feat = self.ego_encoder(batch['ego'].x)      # [Batch_Size, hidden_dim]
        g_f = self.goal_encoder(batch['goal'].x)        # [Batch_Size, hidden_dim]
        
        # 防御性编程：兼容全图无障碍点云雷达数据流空值情况
        if batch['point'].x.shape[0] == 0:
            p_f = torch.zeros((0, e_feat.shape[1]), device=e_feat.device, dtype=e_feat.dtype)
        else:
            p_f = self.point_encoder(batch['point'].x)  # [Total_Points, hidden_dim]
        
        # 2. 激发图消息流流转
        h_dict = self.hetero_conv(
            x_dict={'ego': e_feat, 'goal': g_f, 'point': p_f},
            edge_index_dict=batch.edge_index_dict
        )
        
        # 提取汇聚后自车节点特征，此时维度在数学上全自动对齐自车批次大小: [Batch_Size, hidden_dim]
        fused_feat = self.fusion(h_dict['ego'])
        
        # 3. 双头多任务同步前向输出
        action = self.action_head(fused_feat)  # [Batch_Size, 2]
        risk = self.risk_head(fused_feat)      # [Batch_Size, 1]
        
        return action, risk