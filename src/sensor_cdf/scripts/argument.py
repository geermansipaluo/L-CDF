#!/usr/bin/env python3
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="DensityNet Dual-Head Physics-Informed Training Ecosystem")
    
    # 1. 路径与系统配置文件
    parser.add_argument("--dataset_path", type=str, default="/home/ubuntu/gxf/model/dateset.npz",
                        help="Path to the compressed expert trajectory dataset.")
    parser.add_argument("--save_path", type=str, default="/home/ubuntu/gxf/densitynet_dual_head_model.pt",
                        help="Safe fallback save trajectory for compiled weights.")
    parser.add_argument("--cuda_devices", type=str, default="1,2,3,4",
                        help="Visible CUDA devices for data-parallel group synthesis.")

    # 2. 核心控制与风险超参数
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Global batch size accumulated across all computing nodes.")
    parser.add_argument("--num_epochs", type=int, default=500000,
                        help="Maximum training epochs for full coverage optimization.")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Base learning rate for Lion optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="L2 regularization scale factor.")
    parser.add_argument("--max_points", type=int, default=200,
                        help="Maximum LiDAR points pruned for spatial graph construction.")
    parser.add_argument("--lambda_risk", type=float, default=0.5,
                        help="Loss balance coefficient for decoupled geometric risk head.")

    # 3. 🛡️ 确定性可复现随机种子
    parser.add_argument("--seed", type=int, default=42,
                        help="Anchored random seed to enforce deterministic behavior.")

    # 4. 📊 SwanLab 云端画布看板配置
    parser.add_argument("--swanlab_project", type=str, default="DensityNet-Safe-Navigation",
                        help="SwanLab project workspace name.")
    parser.add_argument("--swanlab_run_name", type=str, default="DualHead_PINN_Optimization",
                        help="Unique name identifier for current operational run.")

    return parser.parse_args()