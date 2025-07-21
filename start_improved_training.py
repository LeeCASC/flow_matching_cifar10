#!/usr/bin/env python3
"""
便捷启动脚本 - 改进的Flow Matching训练

提供多种预配置的训练选项，解决损失函数记录问题并延长训练时间
"""

import subprocess
import sys
import argparse
import os

def get_gpu_memory():
    """获取GPU内存信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            memory_mb = int(result.stdout.strip())
            return memory_mb / 1024  # 转换为GB
    except:
        pass
    return 8  # 默认假设8GB

def create_training_configs():
    """创建不同的训练配置"""
    gpu_memory = get_gpu_memory()
    
    configs = {
        "快速测试": {
            "description": "快速测试改进的损失记录功能 (1小时)",
            "args": [
                "--epochs", "50",
                "--batch_size", "32",
                "--lr", "2e-4",
                "--use_ema",
                "--model_size", "small",
                "--log_interval", "5",
                "--eval_interval", "5"
            ],
            "estimated_time": "1小时"
        },
        
        "标准训练": {
            "description": "标准长时间训练，增强损失记录 (20-40小时)",
            "args": [
                "--epochs", "500",
                "--batch_size", "32",
                "--lr", "2e-4",
                "--use_ema",
                "--model_size", "standard",
                "--log_interval", "10",
                "--eval_interval", "5",
                "--auto_resume"
            ],
            "estimated_time": "20-40小时"
        },
        
        "高质量训练": {
            "description": "大模型长时间训练，最佳质量 (40-80小时)",
            "args": [
                "--epochs", "800",
                "--batch_size", "24" if gpu_memory >= 12 else "16",
                "--lr", "1.5e-4",
                "--use_ema",
                "--model_size", "large",
                "--log_interval", "10",
                "--eval_interval", "5",
                "--auto_resume"
            ],
            "estimated_time": "40-80小时"
        },
        
        "高分辨率": {
            "description": "128x128高分辨率训练 (需要12GB+显存)",
            "args": [
                "--epochs", "600",
                "--image_size", "128",
                "--batch_size", "16" if gpu_memory >= 16 else "8",
                "--lr", "1e-4",
                "--use_ema",
                "--model_size", "large",
                "--log_interval", "10",
                "--eval_interval", "10",
                "--auto_resume"
            ],
            "estimated_time": "50-100小时",
            "min_gpu_memory": 12
        },
        
        "从检查点继续": {
            "description": "从现有检查点继续训练",
            "args": [
                "--epochs", "500",
                "--batch_size", "32",
                "--lr", "1e-4",
                "--use_ema",
                "--model_size", "standard",
                "--auto_resume",
                "--log_interval", "10"
            ],
            "estimated_time": "根据剩余epochs"
        }
    }
    
    return configs

def main():
    parser = argparse.ArgumentParser(description="改进的Flow Matching训练启动器")
    parser.add_argument('--config', type=str, choices=['快速测试', '标准训练', '高质量训练', '高分辨率', '从检查点继续'],
                       help='选择训练配置')
    parser.add_argument('--list', action='store_true', help='列出所有可用配置')
    parser.add_argument('--gpu_check', action='store_true', help='检查GPU信息')
    
    args = parser.parse_args()
    
    if args.gpu_check:
        gpu_memory = get_gpu_memory()
        print(f"🔍 GPU内存: {gpu_memory:.1f}GB")
        return
    
    configs = create_training_configs()
    gpu_memory = get_gpu_memory()
    
    if args.list or not args.config:
        print("=" * 80)
        print("🚀 改进的Flow Matching训练配置选项")
        print("=" * 80)
        print(f"💻 检测到GPU内存: {gpu_memory:.1f}GB")
        print()
        
        for name, config in configs.items():
            print(f"📋 {name}:")
            print(f"   描述: {config['description']}")
            print(f"   预计时间: {config['estimated_time']}")
            if 'min_gpu_memory' in config:
                print(f"   最低显存要求: {config['min_gpu_memory']}GB")
                if gpu_memory < config['min_gpu_memory']:
                    print(f"   ⚠️  当前显存不足，建议选择其他配置")
            print(f"   启动命令: python start_improved_training.py --config {name}")
            print("-" * 60)
        
        print("\n💡 使用示例:")
        print("python start_improved_training.py --config 标准训练")
        print("python start_improved_training.py --config 快速测试")
        return
    
    if args.config not in configs:
        print(f"❌ 未知配置: {args.config}")
        print("使用 --list 查看所有可用配置")
        return
    
    config = configs[args.config]
    
    # 检查显存要求
    if 'min_gpu_memory' in config and gpu_memory < config['min_gpu_memory']:
        print(f"⚠️  警告: {args.config} 需要至少 {config['min_gpu_memory']}GB 显存")
        print(f"当前检测到 {gpu_memory:.1f}GB，可能会出现内存不足错误")
        response = input("是否继续？(y/N): ")
        if response.lower() != 'y':
            print("训练已取消")
            return
    
    print("=" * 80)
    print(f"🚀 启动配置: {args.config}")
    print("=" * 80)
    print(f"📝 描述: {config['description']}")
    print(f"⏱️  预计时间: {config['estimated_time']}")
    print(f"💾 GPU内存: {gpu_memory:.1f}GB")
    print()
    
    # 构建完整的训练命令
    cmd = ["python", "train_imagenet_improved.py"] + config['args']
    
    print(f"🔧 执行命令:")
    print(" ".join(cmd))
    print()
    
    # 显示关键改进点
    print("🔥 主要改进:")
    print("✅ 修复损失函数记录问题 - 每个batch都记录损失")
    print("✅ 增加移动平均损失监控 - 更稳定的损失趋势")
    print("✅ 延长训练时间 - 500个epochs获得更好收敛")
    print("✅ 自动恢复功能 - 训练中断后可自动继续")
    print("✅ 更详细的训练统计 - 梯度范数、参数统计等")
    print("✅ 异常处理 - 自动跳过NaN损失，提高训练稳定性")
    print()
    
    print("📊 Tensorboard监控:")
    print("tensorboard --logdir logs_imagenet_improved")
    print()
    
    response = input("开始训练？(Y/n): ")
    if response.lower() in ['', 'y', 'yes']:
        print("🚀 开始训练...")
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n⚠️  训练被用户中断")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ 训练失败: {e}")
    else:
        print("训练已取消")

if __name__ == "__main__":
    main() 