# online_learning.py
import torch
import numpy as np
from collections import deque
import threading
import time
import torch.optim as optim
import torch.nn as nn
import os

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()
    
    def add(self, state, obstacles, gradient):
        with self.lock:
            self.buffer.append((
                torch.tensor(state, dtype=torch.float32),
                torch.tensor(obstacles, dtype=torch.float32),
                torch.tensor(gradient, dtype=torch.float32)
            ))

class OnlineTrainer:
    def __init__(self, model_path, save_path, buffer_size=10000,  old_data_loader=None):
        # 加载基础模型
        self.model = torch.load(model_path, map_location='cpu')
        self.model.train()

        # 加载旧参数和Fisher矩阵
        ewc_params_path = save_path + '_ewc_params.pt'
        if os.path.exists(ewc_params_path):
            ewc_params = torch.load(ewc_params_path)
            self.old_params = ewc_params['old_params']
            self.fisher_matrix = ewc_params['fisher_matrix']
            print("已加载EWC参数")
        else:
            self.old_params = {name: param.detach().clone() for name, param in self.model.named_parameters()}
            self.fisher_matrix = {}
            print("初始化空EWC参数")
        
        # 初始化训练组件
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5)
        self.criterion = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_size)
        self.last_save = 0
        
        # 配置保存参数
        self.save_path = save_path
        self.save_interval = 5  # 10分钟
        self.last_save = 0
        
    def add_experience(self, state, obstacles, gradient):
        """ 添加经验数据 """
        self.buffer.add(state, obstacles, gradient)

    def compute_fisher_matrix(self, data_loader):
        """计算Fisher信息矩阵"""
        self.model.eval()
        fisher_matrix = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        
        for batch in data_loader:
            states, obstacles, gradients = batch
            self.model.zero_grad()
            outputs = self.model(states, obstacles)
            loss = self.criterion(outputs, gradients)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_matrix[name] += param.grad.pow(2).mean(dim=0)  # 近似Fisher信息
        
        # 取平均值
        for name in fisher_matrix:
            self.fisher_matrix[name] = fisher_matrix[name] / len(data_loader)
        print("Fisher矩阵已更新") 
    
    def train_step(self):
        """ 执行单步训练 """
        # print("执行单步训练")
        with self.buffer.lock:
            all_data = self.buffer.buffer
        if not all_data:
            print("无数据")
            return None
            
        states, obstacles, gradients = zip(*all_data)
        self.optimizer.zero_grad()
        
        # 转换数据格式
        states = torch.stack(states)
        obstacles = torch.stack(obstacles)
        gradients = torch.stack(gradients)
        
        # 前向传播
        outputs = self.model(states, obstacles)
        loss = self.criterion(outputs, gradients)
        # print(f"loss: {loss.item()}")

        # 计算EWC正则项
        ewc_loss = 0.0
        lambda_ewc = 1e3  # 调节系数
        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                ewc_loss += (self.fisher_matrix[name] * (param - self.old_params[name])**2).sum()
        
        total_loss = loss + lambda_ewc * ewc_loss

        
        # 反向传播
        # loss.backward()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 自动保存检查点
        if time.time() - self.last_save > self.save_interval:
            self.save_model()
            self.last_save = time.time()
            print("已保存")
            
        return total_loss.item()
    
    def save_model(self):
        """保存模型和EWC参数"""
        torch.save(self.model, self.save_path)
        ewc_params = {
            'old_params': self.old_params,
            'fisher_matrix': self.fisher_matrix
        }
        torch.save(ewc_params, self.save_path + '_ewc_params.pt')
        print("模型和EWC参数已保存")
    
    def get_inference_model(self):
        """ 返回用于推理的模型副本 """
        # 深拷贝模型结构
        model_copy = type(self.model)()  
        # 深拷贝参数
        model_copy.load_state_dict(self.model.state_dict())  
        return model_copy.eval()
