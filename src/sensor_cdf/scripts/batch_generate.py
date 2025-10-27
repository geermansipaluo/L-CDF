#!/usr/bin/env python3
import numpy as np
from generate_dataset import generate_data  # 从原代码导入生成函数

def save_in_batches(total_samples=40, batch_size=10):
    # for i in range(1, total_samples//batch_size + 1):
    #     X, y = generate_data(num_samples=batch_size)
    #     np.savez(f'output/data_batch_{i}.npz', X=X, y=y)
    #     print(f'Batch {i} saved with {batch_size} samples')
    
    X, y = generate_data(num_samples=batch_size)
    np.savez(f'output/data_batch_3.npz', X=X, y=y)
    print(f'Batch 3 saved with {batch_size} samples')

if __name__ == "__main__":
    save_in_batches(total_samples=1000, batch_size=1000)