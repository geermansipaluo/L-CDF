#!/usr/bin/env python3
from argument import get_args
# 🔴【最高优先级执行】获取控制台与脚本形参并挂接系统环境变量，切断外部多余显卡竞争污染
args = get_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

import random
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader
from timm.optim import Lion
from torch.optim.lr_scheduler import OneCycleLR
import swanlab

# 引入局部相对特征化双头网络与训练流水线
from model import DensityNet
from trainer import GraphDataCollator, ModelSaver, train_epoch

def set_seed(seed):
    """ 🛡️ 注入确定性锚点锁，彻底消除跨设备、多批次间的随机对齐噪声，确保论文图表完美可复现 """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 牺牲微弱算力，换取严格的数学可重复验证 (Deterministic Control Layer)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(dataset_path):
    """ 纯局部感知映射读取层 """
    data = np.load(dataset_path, allow_pickle=True)
    X = data['X'] # 已在落盘前清洗为具有平移旋转不变性的 4 维局部状态
    y = data['y'] # 已在落盘前解耦出的 3 维平滑靶点 (v, omega, psi)
    z = data['z'] # 局部自车系不规则裁剪点云
    
    states = X[:, :4] 
    pointclouds = [np.array(pc) for pc in z]
    labels = y[:, :3]
    return (states, pointclouds), labels

def main():
    # 1. 建立确定性实验环境
    set_seed(args.seed)

    # 2. 初始化分布式 NCCL 通信原语进程组
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # 3. 跨计算节点分布式数据加载与配额计算
    (states, pointclouds), labels = load_data(args.dataset_path)
    dataset = list(zip(states, pointclouds, labels))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, seed=args.seed
    )
    
    # 严格按照总 batch_size 平均分摊至当前各独占显卡
    per_gpu_batch = args.batch_size // world_size
    train_loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        collate_fn=GraphDataCollator(args.max_points),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 4. 初始化端到端双头网络架构并接入 DDP 分布式双向传播体系
    model = DensityNet(state_dim=4, hidden_dim=512).to(device)
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # 5. 配置多任务损失约束器与Lion高阶进化优化算子
    criterion_ctrl = nn.HuberLoss(delta=1.0) # 控制分支使用 HuberLoss 隔离专家突变毛刺
    criterion_risk = nn.MSELoss()            # 风险预测分支使用有界的标准 MSELoss
    
    optimizer = Lion(
        model.parameters(),
        lr=args.learning_rate * 0.5, # 配合多卡大规模 batch 调和学习率底座
        weight_decay=args.weight_decay,
        betas=(0.95, 0.98)
    )

    # 6. 全局动态学习率一阶余弦退火退火配置 (OneCycleLR 调度)
    total_steps = args.num_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=args.learning_rate * 3, 
        total_steps=total_steps, 
        pct_start=0.3
    )

    # 7. 原子落盘保存件初始化
    saver = ModelSaver(model, args.save_path, rank)

    # 8. 🔥【云端大看板握手】仅在主进程建立 SwanLab 监控闭环
    if rank == 0:
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_run_name,
            # 将 args 内的所有控制台、算法超参数整体打包同步云端，实现全面的超参快照备份
            config=vars(args) 
        )
        print(f"🚀 SwanLab 云端可视化分析工作空间已成功握手对接！项目名: {args.swanlab_project}")

    # 9. 开启大规模无限逼近自动化闭环训练
    for epoch in range(args.num_epochs):
        train_epoch(
            model=model, train_loader=train_loader, optimizer=optimizer,
            criterion_ctrl=criterion_ctrl, criterion_risk=criterion_risk,
            saver=saver, args=args, scheduler=scheduler, device=device,
            rank=rank, epoch=epoch
        )

    # 10. 正常破产熔断/训练圆满结束清理工作
    saver.cleanup_save()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()