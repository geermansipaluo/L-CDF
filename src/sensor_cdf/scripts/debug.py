# fix_model.py
import torch

# 加载损坏模型
broken_model = torch.load(
    "/home/guo/MPC-D-CBF-main/src/sensor_cdf/scripts/saved_models/lidar_grad_model.pt",
    map_location='cpu'
)

# 重新保存为state_dict
torch.save(broken_model.state_dict(), 
          "/home/guo/MPC-D-CBF-main/src/sensor_cdf/scripts/saved_models/fixed_lidar_grad_model.pt")
