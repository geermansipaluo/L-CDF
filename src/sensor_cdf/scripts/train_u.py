#!/usr/bin/env python3
import os
# 根据你的实际GPU环境配置，保持原样
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
import signal
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
# 🔥 修改：引入我们上一轮重构的双头 DensityNet
from model import DensityNet 
from torch_geometric.data import Batch, Data
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from timm.optim import Lion
import atexit
import tempfile

# 自定义数据collator处理变长点云
class GraphDataCollator:
    def __init__(self, max_points=200):
        self.max_points = max_points
        
    def __call__(self, batch):
        states, graphs, labels = [], [], []
        
        # 🔥 修改：对应专家系统3维标签 (v, omega, psi)
        for batch_idx, (state, pc, label) in enumerate(batch):
            pos = torch.tensor(pc[:self.max_points], dtype=torch.float32)
            data = Data(
                pos=pos,
                batch=torch.full((len(pos),), batch_idx, dtype=torch.long) 
            )
            
            states.append(state)
            graphs.append(data)
            labels.append(label)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            Batch.from_data_list(graphs),
            torch.tensor(np.array(labels), dtype=torch.float32)
        )

def load_data(dataset_path='/home/ubuntu/gxf/model/dateset.npz'):
    data = np.load(dataset_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    z = data['z']
    
    # 🔥【硬核修复】消除绝对坐标污染，直接读取整条 4 维局部状态
    # X 包含: [target_local_x, target_local_y, cos_yaw_err, sin_yaw_err]
    states = X[:, :4] 
    
    pointclouds = [np.array(pc) for pc in z]
    
    # 🔥【新增】保留专家的 3 维标签，包含: [v, omega, psi]
    labels = y[:, :3]
    
    return (states, pointclouds), labels

class ModelSaver:
    def __init__(self, model, save_path, rank):
        self.model = model
        self.save_path = save_path
        self.rank = rank
        self.should_save = False
        self.save_lock = False
        
        if rank == 0:
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            atexit.register(self.cleanup_save)

        if self.rank != 0:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def signal_handler(self, sig, frame):
        print("\n捕获退出信号，启动安全保存...")
        self.should_save = True

    def safe_save(self):
        if self.rank != 0 or self.save_lock:
            return
            
        try:
            self.save_lock = True
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
                state_dict = {
                    k: v.cpu().detach().clone()
                    for k, v in self.model.module.state_dict().items()
                }
                
                torch.save(state_dict, tmp_path)
                os.replace(tmp_path, self.save_path)
                print(f"模型安全保存到 {self.save_path}")
                
        except Exception as e:
            print(f"保存失败: {str(e)}")
        finally:
            self.save_lock = False
            self.should_save = False
        if self.rank == 0:
            try:
                test = torch.load(self.save_path, map_location='cpu')
                print(f"验证成功: {len(test)} 个参数")
            except:
                print("⚠️ 模型文件损坏")

    def cleanup_save(self):
        if self.rank == 0 and not self.should_save:
            self.safe_save()

# 🔥 修改：重构多任务训练核心回路
def train(model, train_loader, optimizer, criterion_ctrl, criterion_risk, 
         saver, lambda_risk=1.0, scheduler=None, num_epochs=10, 
         device='cuda', rank=0, writer=None):
    model.train()
    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        if saver.should_save:
            saver.safe_save()

        total_loss = 0.0
        total_loss_ctrl = 0.0
        total_loss_risk = 0.0
        train_loader.sampler.set_epoch(epoch)
        
        for states, graph_batch, labels in train_loader:
            states = states.to(device, non_blocking=True)
            graph_batch = graph_batch.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # 1. 解耦专家标签
                true_action = labels[:, :2]  # [v, omega]
                true_psi = labels[:, 2:3]    # 纯几何安全准入度 \psi
                
                # 2. 🔥 构造线性互补防御靶点: 风险 = 1.0 - 安全准入度
                risk_target = 1.0 - true_psi
                
                # 3. 前向传播（双头同时输出）
                pred_action, pred_risk = model(states, graph_batch)
                
                # 4. 计算解耦的多任务损失
                loss_ctrl = criterion_ctrl(pred_action, true_action)   # 动作回归 (Huber Loss)
                loss_risk = criterion_risk(pred_risk, risk_target)     # 风险回归 (MSE Loss)
                
                # 5. 复合总损失
                loss = loss_ctrl + lambda_risk * loss_risk
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            total_loss_ctrl += loss_ctrl.item()
            total_loss_risk += loss_risk.item()

        # 主进程处理日志和保存
        if rank == 0:
            num_batches = len(train_loader)
            avg_loss = total_loss / num_batches
            avg_loss_ctrl = total_loss_ctrl / num_batches
            avg_loss_risk = total_loss_risk / num_batches
            lr = optimizer.param_groups[0]['lr']
            
            if scheduler:
                scheduler.step()

            # 🔥 极其细致的 TensorBoard 多维度风险监控指标
            if writer:
                writer.add_scalar('Loss/Total_Loss', avg_loss, epoch)
                writer.add_scalar('Loss/Action_Huber_Loss', avg_loss_ctrl, epoch)
                writer.add_scalar('Loss/Risk_MSE_Loss', avg_loss_risk, epoch)
                writer.add_scalar('Train/Learning_Rate', lr, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Total Loss: {avg_loss:.4f} | "
                  f"Action Loss: {avg_loss_ctrl:.4f} | "
                  f"Risk Loss: {avg_loss_risk:.4f} | LR: {lr:.6f}")

            if (epoch + 1) % 100 == 0:
                saver.safe_save()

def main():
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    config = {
        "batch_size": 1024,
        "num_epochs": 500000,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "max_points": 200,
        "lambda_risk": 0.5  # 🔥 物理风险 Head 的任务损失平衡权重超参数
    }

    # 数据加载 (全局部化干净特征与标签)
    (states, pointclouds), labels = load_data()
    dataset = list(zip(states, pointclouds, labels))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"] // 4,
        sampler=train_sampler,
        collate_fn=GraphDataCollator(config["max_points"]),
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 🔥【架构对接】初始化重构后的 4 维状态输入双头 DensityNet
    model = DensityNet(state_dim=4, hidden_dim=512).to(device)
    
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )

    optimizer = Lion(
        model.parameters(),
        lr=config["learning_rate"] * 0.5,  
        weight_decay=config["weight_decay"],
        betas=(0.95, 0.98)
    )

    # 🔥【多任务损失函数对齐】
    criterion_ctrl = nn.HuberLoss(delta=1.0) # 控制指令头：Huber Loss 抵抗飞跃噪点
    criterion_risk = nn.MSELoss()            # 风险预测头：MSE Loss 稳定逼近 [0,1] 空间

    scheduler = OneCycleLR(
        optimizer, 
        max_lr=config["learning_rate"]*3, 
        total_steps=config["num_epochs"]*len(train_loader), 
        pct_start=0.3
    )

    save_path = "/home/ubuntu/gxf/densitynet_dual_head_model.pt"
    saver = ModelSaver(model, save_path, rank)

    writer = SummaryWriter(log_dir='/home/ubuntu/gxf/graph_cdf_experiment') if rank == 0 else None

    # 训练闭环启动
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion_ctrl=criterion_ctrl,
        criterion_risk=criterion_risk,
        saver=saver,
        lambda_risk=config['lambda_risk'],
        scheduler=scheduler,
        num_epochs=config["num_epochs"],
        device=device,
        rank=rank,
        writer=writer
    )

    saver.cleanup_save()
    if writer:
        writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()