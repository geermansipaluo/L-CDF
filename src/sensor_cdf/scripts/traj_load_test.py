import torch
import matplotlib.pyplot as plt

# --------------------------
# 修改这里为你的轨迹文件路径
trajectory_path = "/home/guo/L-CDF/densitynet_trajectory.pt"
# --------------------------

# 读取轨迹
all_runs_trajectories = torch.load(trajectory_path)

# 如果只跑一轮，取第一条轨迹
trajectory = all_runs_trajectories[0]  # [(x, y, theta, time), ...]

# 提取 x, y 坐标
x = [point[0] for point in trajectory]
y = [point[1] for point in trajectory]

# 绘制轨迹
plt.figure(figsize=(8, 6))
plt.plot(x, y, '-o', markersize=2, label='Trajectory')
plt.scatter([x[0]], [y[0]], c='green', marker='o', label='Start')
plt.scatter([x[-1]], [y[-1]], c='red', marker='x', label='Goal')
plt.title("Single Run Trajectory")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()