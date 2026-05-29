#!/usr/bin/env python3
import argparse
import os
from datetime import datetime

def get_args():
    parser = argparse.ArgumentParser(description="DensityNet Dual-Head Single-GPU Refactored Training Arguments")
    
    # 1. 基础命名空间参数 (对应你的新要求)
    parser.add_argument("--scenario", type=str, default="UnknownGym",
                        help="Scenario or environment context identifier.")
    parser.add_argument("--alg", type=str, default="DensityNet",
                        help="Algorithm classification identifier.")
    parser.add_argument("--exp_name", type=str, default="DualHead_PINN",
                        help="Core tag or feature identifier for current unique test.")
    parser.add_argument("--save_dir", type=str, default="/home/ubuntu/Desktop/gxf/LCDF/results",
                        help="Root path for experimental tracking and weight storage.")

    # 2. 数据路径与显卡指定
    parser.add_argument("--dataset_path", type=str, default="/home/ubuntu/Desktop/gxf/LCDF/train/model/dataset.pt",
                        help="Path to the compressed expert trajectory dataset.")
    parser.add_argument("--cuda_device", type=str, default="0",
                        help="Target single GPU device index for training.")

    # 3. 核心控制与风险多任务超参数
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size utilized for training on the single GPU.")
    parser.add_argument("--num_epochs", type=int, default=500000,
                        help="Maximum training epochs for total optimization loop.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Base learning rate for Lion optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="L2 regularization scale factor.")
    parser.add_argument("--max_points", type=int, default=200,
                        help="Maximum LiDAR points pruned for spatial graph construction.")
    parser.add_argument("--lambda_risk", type=float, default=1,
                        help="Loss balance coefficient for decoupled geometric risk head.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Anchored random seed to enforce deterministic behavior.")
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=512)

    # 4. 📊 SwanLab 开关与工作区设置
    parser.add_argument("--swanlab", action="store_true", default=True,
                        help="Toggle switch to active SwanLab real-time remote logger.")
    parser.add_argument("--swanlab_project", type=str, default="DensityNet-Safe-Navigation",
                        help="SwanLab project workspace name.")
    parser.add_argument("--swanlab_exp_name", type=str, default=None,
                        help="Experiment runtime unique string name. Generated auto if None.")

    # 周期保存控制项 (单卡无 worker 概念，以周期 Epoch 替代 Episode)
    parser.add_argument("--num_epoch_save", type=int, default=100,
                        help="Frequency of checkpoint evaluation and synchronization.")

    args = parser.parse_args()

    # 🔴【核心规范化修剪 1】 自动化构建高度可追溯的云端 SwanLab 实验运行名称
    if args.swanlab and args.swanlab_exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.swanlab_exp_name = f"{args.scenario}_{args.alg}_{timestamp}"
        print(f"-> SwanLab Experiment Name Compiled: {args.swanlab_exp_name}")

    # 🔴【核心规范化修剪 2】 自动化构建硬件隔离的高阶物理保存路径目录 (带精确秒级时间戳)
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    args.experiment_dir = f"{args.save_dir}/{args.scenario}-{args.alg}-{args.exp_name}-{current_time_str}"
    
    return args