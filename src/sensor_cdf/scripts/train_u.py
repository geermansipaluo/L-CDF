#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
import signal
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from model import GradNet, CDFNet
from torch_geometric.data import Batch, Data
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau
from timm.optim import Lion
import atexit
import tempfile

# 自定义数据collator处理变长点云
class GraphDataCollator:
    def __init__(self, max_points=200):
        self.max_points = max_points
        
    def __call__(self, batch):
        states, graphs, u_control = [], [], []
        
        for batch_idx, (state, pc, u) in enumerate(batch):
            pos = torch.tensor(pc[:self.max_points], dtype=torch.float32)
            data = Data(
                pos=pos,
                batch=torch.full((len(pos),), batch_idx, dtype=torch.long) 
            )
            
            states.append(state)
            graphs.append(data)
            u_control.append(u)
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            Batch.from_data_list(graphs),
            torch.tensor(np.array(u_control), dtype=torch.float32)
        )

def load_data(dataset_path='/home/ubuntu/gxf/model/test_oneenv_u.npz'):
    data = np.load(dataset_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    z = data['z']
    
    states = X[:, :2]
    pointclouds = [np.array(pc) for pc in z]
    u = y[:, :2]
    
    return (states, pointclouds), u

# 修改后的信号处理和安全保存机制
class ModelSaver:
    def __init__(self, model, save_path, rank):
        self.model = model
        self.save_path = save_path
        self.rank = rank
        self.should_save = False
        self.save_lock = False
        
        # 注册信号和退出处理
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
        """线程安全的模型保存方法"""
        if self.rank != 0 or self.save_lock:
            return
            
        try:
            self.save_lock = True
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
                # 获取CPU状态字典
                state_dict = {
                    k: v.cpu().detach().clone()
                    for k, v in self.model.module.state_dict().items()
                }
                
                # 先保存到临时文件
                torch.save(state_dict, tmp_path)
                
                # 原子操作替换原文件
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
        """程序正常退出时保存"""
        if self.rank == 0 and not self.should_save:
            self.safe_save()

def train(model, train_loader, optimizer, criterion, saver, 
         task_type='u', scheduler=None, num_epochs=10, 
         device='cuda', rank=0, writer=None):
    model.train()
    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        # 检查是否需要保存
        if saver.should_save:
            saver.safe_save()

        total_loss = 0.0
        train_loader.sampler.set_epoch(epoch)  # 确保shuffle正确
        
        for states, graph_batch, u in train_loader:
            states = states.to(device, non_blocking=True)
            graph_batch = graph_batch.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                labels = u.to(device)
                pred = model(states, graph_batch)
                loss = criterion(pred, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()

        # 主进程处理日志和保存
        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            lr = optimizer.param_groups[0]['lr']
            
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_loss)
                else:
                    scheduler.step()

            if writer:
                writer.add_scalar('Loss/train', avg_loss, epoch)
                writer.add_scalar('Learning Rate', lr, epoch)

            print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")

            # 定期保存
            if (epoch + 1) % 100 == 0:
                saver.safe_save()

def main():
    # 分布式初始化
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    # 配置参数
    config = {
        "batch_size": 1024,
        "num_epochs": 500000,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "max_points": 200,
        "task_type": 'u'
    }

    # 数据加载
    (states, pointclouds), u = load_data()
    dataset = list(zip(states, pointclouds, u))
    
    # 分布式数据加载
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

    # 模型初始化
    model = GradNet(hidden_dim=512).to(device)
    
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )

    # 优化器和损失函数

    optimizer = Lion(
        model.parameters(),
        lr=config["learning_rate"] * 0.5,  
        weight_decay=config["weight_decay"],
        betas=(0.95, 0.98)
    )

    
    criterion = nn.HuberLoss(delta=1.0)
    
    # 学习率调度
    scheduler = (
        OneCycleLR(optimizer, max_lr=config["learning_rate"]*3, total_steps=config["num_epochs"]*len(train_loader), pct_start=0.3)
    )

    # 模型保存设置
    save_path = ("/home/ubuntu/gxf/lidar_u_model.pt" )
    
    saver = ModelSaver(model, save_path, rank)

    # TensorBoard（仅主进程）
    writer = SummaryWriter(log_dir='/home/ubuntu/gxf/graph_cdf_experiment') if rank == 0 else None

    # 训练循环
    train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        saver=saver,
        task_type=config['task_type'],
        scheduler=scheduler,
        num_epochs=config["num_epochs"],
        device=device,
        rank=rank,
        writer=writer
    )

    # 最终保存和清理
    saver.cleanup_save()
    if writer:
        writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
