import numpy as np

def remove_empty_pointclouds(dataset_path='/home/ubuntu/gxf/model/lidar_lcdf_1.npz', new_dataset_path='/home/ubuntu/gxf/model/lidar_lcdf_filtered.npz'):
    # 加载原始数据集
    data = np.load(dataset_path, allow_pickle=True)
    X = data['X']  # 输入特征
    y = data['y']  # 标签（梯度 + CDF）
    z = data['z']  # 点云数据(列表形式，每一个元素长度不等，但都是二维的)
    
    print(f"老数据集已保存到 {dataset_path}，共包含 {len(X)} 条有效数据。")

    # 过滤掉空点云数据
    filtered_indices = []
    for i, pc in enumerate(z):
        if pc.size > 0:
            filtered_indices.append(i)
    
    # 根据这些索引过滤掉所有相关的数据
    X_filtered = X[filtered_indices]
    y_filtered = y[filtered_indices]
    z_filtered = [z[i] for i in filtered_indices]  # 仅保留非空点云数据
    pointcloud = np.empty(len(z_filtered), dtype=object)
    for i, arr in enumerate(z_filtered):
        pointcloud[i] = arr
    
    # 保存过滤后的数据集到新的文件
    np.savez(new_dataset_path, X=X_filtered, y=y_filtered, z=pointcloud, allow_pickle=True)
    
    print(f"新数据集已保存到 {new_dataset_path}，共包含 {len(X_filtered)} 条有效数据。")

# 调用函数，删除空点云数据
remove_empty_pointclouds()
