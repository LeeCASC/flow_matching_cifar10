#!/usr/bin/env python3
"""
恢复训练脚本
支持从不同的checkpoint文件恢复训练
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="恢复COCO文本条件Flow Matching训练")
    parser.add_argument("--checkpoint", type=str, 
                       default="/home/chenxiaoyu/projects/flow_matching_cifar10/checkpoints_coco_conditional/best.pth",
                       help="Checkpoint文件路径")
    parser.add_argument("--epochs", type=int, default=800, help="总训练epoch数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--image_size", type=int, default=128, help="图像大小")
    parser.add_argument("--model_size", type=str, default="standard", choices=["small", "standard", "large"])
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="使用的GPU ID")
    parser.add_argument("--ddp_backend", type=str, default="gloo", choices=["nccl", "gloo"])
    
    args = parser.parse_args()
    
    # 检查checkpoint文件
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        print("请检查文件路径是否正确")
        return 1
    
    print(f"🔄 从checkpoint恢复训练: {args.checkpoint}")
    print(f"📊 训练参数:")
    print(f"   - 总epoch数: {args.epochs}")
    print(f"   - 批次大小: {args.batch_size}")
    print(f"   - 图像大小: {args.image_size}")
    print(f"   - 模型大小: {args.model_size}")
    print(f"   - 学习率: {args.lr}")
    print(f"   - 使用GPU: {args.gpus}")
    
    # 构建训练命令
    cmd = [
        "CUDA_VISIBLE_DEVICES=" + args.gpus,
        "torchrun",
        f"--nproc_per_node={len(args.gpus.split(','))}",
        "train_coco_conditional.py",
        "--distributed",
        f"--ddp_backend", args.ddp_backend,
        "--coco_dir", "/data/chenxiaoyu/coco2017",
        "--batch_size", str(args.batch_size),
        "--image_size", str(args.image_size),
        "--model_size", args.model_size,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--use_ema",
        "--save_validation",
        "--eval_interval", "1",
        "--samples_dir", "samples_coco_validation",
        "--checkpoint_dir", "checkpoints_coco_conditional",
        "--log_dir", "logs_coco_conditional",
        "--resume", args.checkpoint
    ]
    
    print(f"\n🚀 执行命令:")
    print(" ".join(cmd))
    print("\n" + "="*70)
    
    # 执行训练
    try:
        result = subprocess.run(" ".join(cmd), shell=True, check=True)
        print("\n✅ 恢复训练完成！")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\n⚠️ 训练被用户中断")
        return 1

if __name__ == "__main__":
    sys.exit(main())



