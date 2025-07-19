"""
Flow Matching 配置文件

定义不同的训练配置，方便快速实验和复现结果。
"""

import argparse
from typing import Dict, Any


# 预定义配置
CONFIGS = {
    "quick_test": {
        "class_idx": 2,  # bird
        "batch_size": 32,
        "epochs": 20,
        "lr": 2e-4,
        "channels": [32, 64, 128],
        "time_emb_dim": 64,
        "use_ema": False,
        "save_interval": 5,
        "sample_interval": 5,
        "description": "快速测试配置，小模型，短时间训练"
    },
    
    "small_model": {
        "class_idx": 2,
        "batch_size": 64,
        "epochs": 50,
        "lr": 1e-4,
        "channels": [64, 128, 256],
        "time_emb_dim": 128,
        "use_ema": True,
        "ema_decay": 0.999,
        "save_interval": 10,
        "sample_interval": 5,
        "description": "小型模型配置，适合快速实验"
    },
    
    "standard": {
        "class_idx": 2,
        "batch_size": 64,
        "epochs": 100,
        "lr": 1e-4,
        "weight_decay": 1e-6,
        "channels": [64, 128, 256, 512],
        "time_emb_dim": 128,
        "use_ema": True,
        "ema_decay": 0.9999,
        "save_interval": 10,
        "sample_interval": 5,
        "description": "标准配置，平衡质量和训练时间"
    },
    
    "high_quality": {
        "class_idx": 2,
        "batch_size": 32,  # 更小的batch size，因为模型更大
        "epochs": 200,
        "lr": 5e-5,  # 更小的学习率
        "weight_decay": 1e-6,
        "channels": [128, 256, 512, 768],
        "time_emb_dim": 256,
        "use_ema": True,
        "ema_decay": 0.9999,
        "save_interval": 20,
        "sample_interval": 10,
        "description": "高质量配置，大模型，长时间训练"
    },
    
    "animals": {
        "class_idx": 3,  # cat
        "batch_size": 64,
        "epochs": 100,
        "lr": 1e-4,
        "channels": [64, 128, 256, 512],
        "time_emb_dim": 128,
        "use_ema": True,
        "save_interval": 10,
        "sample_interval": 5,
        "description": "动物类别（猫）专用配置"
    },
    
    "vehicles": {
        "class_idx": 1,  # automobile
        "batch_size": 64,
        "epochs": 100,
        "lr": 1e-4,
        "channels": [64, 128, 256, 512],
        "time_emb_dim": 128,
        "use_ema": True,
        "save_interval": 10,
        "sample_interval": 5,
        "description": "交通工具（汽车）专用配置"
    }
}

# 生成配置
GENERATION_CONFIGS = {
    "fast": {
        "num_steps": 25,
        "method": "euler",
        "description": "快速生成，较低质量"
    },
    
    "balanced": {
        "num_steps": 50,
        "method": "euler",
        "description": "平衡速度和质量"
    },
    
    "quality": {
        "num_steps": 100,
        "method": "euler",
        "description": "高质量生成"
    },
    
    "best": {
        "num_steps": 100,
        "method": "rk4",
        "description": "最佳质量，最慢速度"
    }
}


def get_config(config_name: str) -> Dict[str, Any]:
    """获取预定义配置"""
    if config_name not in CONFIGS:
        available_configs = list(CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available_configs}")
    
    config = CONFIGS[config_name].copy()
    return config


def get_generation_config(config_name: str) -> Dict[str, Any]:
    """获取生成配置"""
    if config_name not in GENERATION_CONFIGS:
        available_configs = list(GENERATION_CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available_configs}")
    
    config = GENERATION_CONFIGS[config_name].copy()
    return config


def list_configs():
    """列出所有可用配置"""
    print("=" * 50)
    print("Available Training Configurations:")
    print("=" * 50)
    
    for name, config in CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config.get('description', 'No description')}")
        print(f"  Class: {config['class_idx']} (CIFAR-10)")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Model channels: {config['channels']}")
        print(f"  Use EMA: {config.get('use_ema', False)}")
    
    print("\n" + "=" * 50)
    print("Available Generation Configurations:")
    print("=" * 50)
    
    for name, config in GENERATION_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config.get('description', 'No description')}")
        print(f"  Steps: {config['num_steps']}")
        print(f"  Method: {config['method']}")


def create_train_args_from_config(config_name: str) -> argparse.Namespace:
    """从配置创建训练参数"""
    config = get_config(config_name)
    
    # 创建ArgumentParser来获取默认值
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_idx', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--time_emb_dim', type=int, default=128)
    parser.add_argument('--channels', nargs='+', type=int, default=[64, 128, 256, 512])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--sample_interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    
    # 解析空参数以获得默认值
    args = parser.parse_args([])
    
    # 更新配置
    for key, value in config.items():
        if key != 'description':
            setattr(args, key, value)
    
    return args


def create_generate_args_from_config(generation_config: str, checkpoint_path: str) -> argparse.Namespace:
    """从配置创建生成参数"""
    config = get_generation_config(generation_config)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--time_emb_dim', type=int, default=128)
    parser.add_argument('--channels', nargs='+', type=int, default=[64, 128, 256, 512])
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--method', type=str, default='euler')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='generated_images')
    parser.add_argument('--nrow', type=int, default=8)
    parser.add_argument('--show_images', action='store_true')
    parser.add_argument('--save_individual', action='store_true')
    
    args = parser.parse_args([])
    args.checkpoint = checkpoint_path
    
    # 更新配置
    for key, value in config.items():
        if key != 'description':
            setattr(args, key, value)
    
    return args


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_configs()
    else:
        list_configs()
        
        print("\n" + "=" * 50)
        print("Usage Examples:")
        print("=" * 50)
        print("python config.py list                    # 显示所有配置")
        print("python train_with_config.py quick_test   # 使用quick_test配置训练")
        print("python train_with_config.py standard     # 使用standard配置训练") 