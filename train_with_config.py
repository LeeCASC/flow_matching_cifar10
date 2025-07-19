#!/usr/bin/env python3
"""
使用预定义配置的训练脚本

使用方法:
python train_with_config.py quick_test      # 使用快速测试配置
python train_with_config.py standard        # 使用标准配置
python train_with_config.py high_quality    # 使用高质量配置
"""

import sys
import argparse
from config import get_config, create_train_args_from_config, list_configs

# 导入训练模块
from train import main as train_main
from train import (
    train_epoch, generate_samples, 
    get_cifar10_dataloader, get_device, print_model_info,
    create_checkpoint_dir, save_checkpoint, load_checkpoint,
    save_images, EMAModel, get_cifar10_class_names
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np

from flow_matching import UNet, FlowMatching


def train_with_config(config_name: str, resume_checkpoint: str = None):
    """使用预定义配置训练模型"""
    
    # 获取配置
    try:
        config = get_config(config_name)
        args = create_train_args_from_config(config_name)
    except ValueError as e:
        print(f"Error: {e}")
        list_configs()
        return
    
    # 如果提供了resume参数，设置它
    if resume_checkpoint:
        args.resume = resume_checkpoint
    
    print("=" * 60)
    print(f"Training with config: {config_name}")
    print("=" * 60)
    print(f"Description: {config.get('description', 'No description')}")
    print(f"Configuration details:")
    for key, value in config.items():
        if key != 'description':
            print(f"  {key}: {value}")
    print("=" * 60)
    
    # 执行训练
    train_main_with_args(args)


def train_main_with_args(args):
    """使用给定参数执行训练（复制并修改train.py的main函数）"""
    
    # 打印配置
    class_names = get_cifar10_class_names()
    print(f"Dataset: CIFAR-10 class '{class_names[args.class_idx]}' (index: {args.class_idx})")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Use EMA: {args.use_ema}")
    print(f"Model channels: {args.channels}")
    print("=" * 60)
    
    # 获取设备
    device = get_device()
    
    # 创建目录
    checkpoint_dir = create_checkpoint_dir(args.checkpoint_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # 创建数据加载器
    print("\nLoading data...")
    dataloader = get_cifar10_dataloader(
        class_idx=args.class_idx,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=True
    )
    
    # 创建模型
    print("\nCreating model...")
    velocity_network = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=args.time_emb_dim,
        channels=tuple(args.channels)
    )
    
    model = FlowMatching(velocity_network)
    model = model.to(device)
    
    # 打印模型信息
    print_model_info(model)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # EMA模型
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(model, decay=args.ema_decay)
        print(f"Using EMA with decay: {args.ema_decay}")
    
    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
    
    print(f"\nStarting training from epoch {start_epoch}...")
    print("=" * 60)
    
    # 训练循环
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        # 训练一个epoch
        avg_loss = train_epoch(
            model, dataloader, optimizer, device, 
            epoch, writer, ema_model
        )
        
        # 更新学习率
        scheduler.step()
        
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, 
                          os.path.join(checkpoint_dir, 'best'))
        
        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_dir)
        
        # 生成样本
        if (epoch + 1) % args.sample_interval == 0:
            print("Generating samples...")
            
            # 使用原始模型生成
            samples = generate_samples(model, device, num_samples=16)
            save_images(samples, f'samples/epoch_{epoch}_original.png', nrow=4)
            
            # 如果使用EMA，也生成EMA模型的样本
            if ema_model is not None:
                # 临时应用EMA参数
                original_params = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        original_params[name] = param.data.clone()
                        param.data.copy_(ema_model.shadow[name])
                
                samples_ema = generate_samples(model, device, num_samples=16)
                save_images(samples_ema, f'samples/epoch_{epoch}_ema.png', nrow=4)
                
                # 恢复原始参数
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.data.copy_(original_params[name])
            
            # 添加样本到tensorboard
            writer.add_images('Generated_Samples', (samples + 1) / 2, epoch)
    
    # 保存最终模型
    save_checkpoint(model, optimizer, args.epochs - 1, best_loss, 
                   os.path.join(checkpoint_dir, 'final'))
    
    print("=" * 60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"Samples saved in: samples/")
    print("=" * 60)
    
    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Flow Matching with predefined configs')
    parser.add_argument('config', type=str, nargs='?', default=None,
                       help='Configuration name (see --list for available configs)')
    parser.add_argument('--list', action='store_true', 
                       help='List available configurations')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    if args.list or args.config is None:
        list_configs()
        if args.config is None:
            print("\n" + "=" * 60)
            print("Please specify a configuration name:")
            print("python train_with_config.py <config_name>")
            print("\nExample:")
            print("python train_with_config.py quick_test")
        return
    
    train_with_config(args.config, args.resume)


if __name__ == "__main__":
    main() 