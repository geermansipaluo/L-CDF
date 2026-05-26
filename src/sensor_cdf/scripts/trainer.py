#!/usr/bin/env python3
import os
import signal
import tempfile
import atexit
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Batch, Data
import swanlab

class GraphDataCollator:
    """ 高效异步处理变长点云的图聚合整理器 """
    def __init__(self, max_points=200):
        self.max_points = max_points
        
    def __call__(self, batch):
        states, graphs, labels = [], [], []
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

class ModelSaver:
    """ 线程安全的多卡原子级模型落盘固化系统 """
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
        else:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def signal_handler(self, sig, frame):
        print("\n[信号触发展开] 捕获系统退出信号，启动安全保存屏障...")
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
                print(f"🎉 核心物理权重已安全原子同步替换至: {self.save_path}")
        except Exception as e:
            print(f"⚠️ 保存故障: {str(e)}")
        finally:
            self.save_lock = False
            self.should_save = False

    def cleanup_save(self):
        if self.rank == 0 and not self.should_save:
            self.safe_save()

def train_epoch(model, train_loader, optimizer, criterion_ctrl, criterion_risk, 
                saver, args, scheduler, device, rank, epoch):
    """ 单个完整 Epoch 的物理信息多任务多卡并行梯度解耦迭代回路 """
    model.train()
    scaler = torch.amp.GradScaler()
    
    total_loss = 0.0
    total_loss_ctrl = 0.0
    total_loss_risk = 0.0
    
    train_loader.sampler.set_epoch(epoch) # 保证多卡联合打乱的均匀性
    
    for states, graph_batch, labels in train_loader:
        # 检查是否由于中断信号产生安全熔断落盘请求
        if saver.should_save:
            saver.safe_save()

        states = states.to(device, non_blocking=True)
        graph_batch = graph_batch.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            # 解耦监督数据项
            true_action = labels[:, :2] # (v, omega)
            true_psi = labels[:, 2:3]   # 纯几何安全度 \psi
            
            # 建立物理指导的线性互补防御靶点: 风险 \sigma = 1.0 - \psi
            risk_target = 1.0 - true_psi
            
            # 前向多回路同步直出
            pred_action, pred_risk = model(states, graph_batch)
            
            # 计算多任务损失函数
            loss_ctrl = criterion_ctrl(pred_action, true_action)
            loss_risk = criterion_risk(pred_risk, risk_target)
            
            # 结合梯度平流层参数进行任务协同
            loss = loss_ctrl + args.lambda_risk * loss_risk
        
        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0) # 梯度剪裁稳定网络边界
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # 步进学习率调度器（OneCycleLR 是单步更新激活的）
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        total_loss_ctrl += loss_ctrl.item()
        total_loss_risk += loss_risk.item()

    # 仅由主计算进程（Rank 0）处理统计看板与阶段固化
    if rank == 0:
        num_batches = len(train_loader)
        avg_loss = total_loss / num_batches
        avg_loss_ctrl = total_loss_ctrl / num_batches
        avg_loss_risk = total_loss_risk / num_batches
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1}/{args.num_epochs}] | Total Loss: {avg_loss:.4f} | "
              f"Action Huber: {avg_loss_ctrl:.4f} | Risk MSE: {avg_loss_risk:.4f} | LR: {lr:.6f}")

        # 🔥【升级核心】利用 SwanLab 云端动态流式图表接口替换旧版 TensorBoard
        swanlab.log({
            "Loss/Total_Combined": avg_loss,
            "Loss/Action_Imitation_Huber": avg_loss_ctrl,
            "Loss/Safety_Complementary_MSE": avg_loss_risk,
            "Optimization/Adaptive_Learning_Rate": lr
        }, step=epoch)

        if (epoch + 1) % 100 == 0:
            saver.safe_save()