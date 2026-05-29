#!/usr/bin/env python3
from argument import get_args
import os

# 🔴【最高优先级拦截】立刻截获并锁定训练所绑定的单张显卡，杜绝外部抢点竞争
args = get_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

import random
import torch
import torch.nn as nn
import numpy as np
import torch_geometric

# 🟢【核心修改 1】：淘汰标准 DataLoader，改用 PyG 官方专门处理异质图的专用加载器
from torch_geometric.loader import DataLoader

from timm.optim import Lion
from torch.optim.lr_scheduler import OneCycleLR
import swanlab

# 锁死安全推流凭证
swanlab.login(api_key="qxLMN0eaBQlXUoiWGKXux")

from model import DensityNet
from trainer import ModelSaver, train_epoch

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 🟢【核心修改 2】：重构数据解包函数，完美对接新版 HeteroData 固化数据流
def load_pyg_dataset(dataset_path):
    print(f"⏳ 正在从硬盘加载二进制 PyG 异质图数据集: {dataset_path} ...")
    # 直接一行解包由 torch.save 固化的 HeteroData 列表
    torch.serialization.add_safe_globals([
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.hetero_data.HeteroData
    ])
    dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
    print(f"✅ 数据集加载成功！总计包含 {len(dataset)} 帧三实体（ego, point, goal）统一图样本。")
    return dataset

def main():
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ 纯单卡开发环境激活成功 -> 当前物理显卡独占绑定卡槽: {device}")

    # 🟢【核心修改 3】：加载异质图集，并使用 PyG 专用 DataLoader 自动接管打包
    dataset = load_pyg_dataset(args.dataset_path)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,  
        num_workers=4, 
        pin_memory=True, 
        persistent_workers=True
        # 💡 注意：原先臃肿、耗CPU的 GraphDataCollator 彻底退出历史舞台，PyG 会自动处理异质图批编制
    )

    # 🟢【核心修改 4】：由于 model 内部已变成统一图架构，不再需要传入割裂的 state_dim 参数
    model = DensityNet(hidden_dim=args.hidden_dim).to(device)
    
    # 动作控制分支损失（维持 HuberLoss 配置）
    criterion_ctrl = nn.HuberLoss(delta=1.0)
    # 💡 注意：根据你最新的 trainer.py 逻辑，criterion_risk (nn.MSELoss) 已经在 train_epoch 内部实例化，此处剔除
    
    optimizer = Lion(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.95, 0.98))
    scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate*3, total_steps=args.num_epochs*len(train_loader), pct_start=0.3)

    # 🔴【核心路径对齐】使用你在 argument 中动态组装好的高隔离度实验文件夹
    saver = ModelSaver(model, args.experiment_dir)

    # 🔴【核心 SwanLab 看板对齐】建立多任务流远程初始化连接
    if args.swanlab:
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_exp_name,
            config=vars(args),
            mode="cloud"  # 开启云端大画布实时画线
        )
        print(f"🚀 SwanLab 规范化大看板初始化成功！实验名称: {args.swanlab_exp_name}")

    # 🟢【核心修改 5】：历史最佳资产监视字典，其 Key 严格对应你最新 trainer.py 内部的 "best_risk_loss"
    best_tracker = {"best_risk_loss": float("inf")}

    # 开启自动化单卡流训练
    try:
        for epoch in range(args.num_epochs):
            # 🟢【核心修改 6】：传参完全对齐你最新重构的、包含时序差分前向演进的 train_epoch 函数规范
            has_interrupted = train_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer,
                criterion_ctrl=criterion_ctrl, 
                saver=saver, 
                args=args, 
                scheduler=scheduler, 
                device=device, 
                epoch=epoch, 
                best_tracker=best_tracker
            )

            if has_interrupted:
                print("🛑 检测到内部大循环已成功截断，准备体面退场。")
                break
    except KeyboardInterrupt:
        print("\n[控制台介入] 用户手动干预终止，退出前向训练沙盒...")
    finally:
        saver.save_checkpoint("model_final_exit")
        if args.swanlab:
            swanlab.finish()
        print("🎉 实验结束，数据流清理圆满收官。")

if __name__ == "__main__":
    main()