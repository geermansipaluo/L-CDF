#!/usr/bin/env python3
import numpy as np

def merge_datasets(num_batches=1):
    merged_X, merged_y, merged_z = [], [], []
    
    for i in range(1, num_batches+1):
        data = np.load(f'/home/ubuntu/gxf/lidar_lcdf_{i}.npz', allow_pickle=True)
        # data = np.load(f'output/train_{i}.npz')
        merged_X.append(data['X'])
        merged_y.append(data['y'])
        merged_z.extend(data['z'].tolist())
    
    X = np.concatenate(merged_X)
    y = np.concatenate(merged_y)
    z = np.array(merged_z, dtype=object)
    
    np.savez('/home/ubuntu/lidar_lcdf_dataset.npz', X=X, y=y, z=z, allow_pickle=True)
    print(f'Merged dataset saved with {len(X)} samples')

if __name__ == "__main__":
    merge_datasets(num_batches=4)