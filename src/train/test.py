import torch


def check_pt_data(file_path):
    try:
        # 使用 torch.load 读取 .pt 文件
        # map_location='cpu' 可以防止在没有 GPU 的机器上读取 CUDA 张量时报错
        pt_data = torch.load(file_path, map_location="cpu", weights_only=False)

        print(f"成功读取文件: {file_path}\n")
        print("文件中包含的数据量/结构如下：")
        print("-" * 50)

        # 情况 1：如果 .pt 文件本身就是一个独立的 PyTorch 张量 (Tensor)
        if isinstance(pt_data, torch.Tensor):
            print(f"该文件包含一个独立的张量 | 数据形状/量: {pt_data.shape}")

        # 情况 2：如果 .pt 文件是一个字典 (Dict)，例如保存了多个变量或模型权重
        elif isinstance(pt_data, dict):
            for key, value in pt_data.items():
                if isinstance(value, torch.Tensor):
                    print(
                        f"键名(Key): {key:<15} | 数据形状/量: {value.shape}"
                    )
                else:
                    print(f"键名(Key): {key:<15} | 类型: {type(value)}")

        # 情况 3：如果 .pt 文件是一个列表 (List) 或元组 (Tuple)
        elif isinstance(pt_data, (list, tuple)):
            print(f"该文件包含一个列表/元组，长度为: {len(pt_data)}")
            for idx, item in enumerate(pt_data):
                if isinstance(item, torch.Tensor):
                    print(
                        f"  索引 [{idx}]: 元素是张量 | 数据形状/量: {item.shape}"
                    )
                else:
                    print(f"  索引 [{idx}]: 类型: {type(item)}")

        # 情况 4：其他未知的数据类型
        else:
            print(f"文件包含的数据类型为: {type(pt_data)}")

        print("-" * 50)

    except Exception as e:
        print(f"读取文件时出错: {e}")


# 文件路径
file_path = "/home/guo/L-CDF/dataset_trajectories.pt"

# 执行脚本
check_pt_data(file_path)