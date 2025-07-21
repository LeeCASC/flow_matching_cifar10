#!/usr/bin/env python3
"""
ImageNet猫咪数据Flow Matching训练脚本

支持更高分辨率、每个epoch保存结果、完整tensorboard记录
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime

from flow_matching_diffusers import FlowMatching
from diffusers import UNet2DModel
from utils_imagenet import (
    get_imagenet_dataloader,
    get_device,
    create_checkpoint_dir,
    save_checkpoint,
    load_checkpoint,
    save_images_imagenet,
    show_images_imagenet,
    denormalize_imagenet,
    EMAModel,
    estimate_imagenet_cat_size
)


def create_imagenet_unet(image_size: int = 64, model_size: str = "standard") -> FlowMatching:
    """为ImageNet创建合适的UNet配置"""
    
    if model_size == "small":
        return FlowMatching(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(128, 256, 384),
            layers_per_block=2,
            attention_head_dim=8,
            norm_num_groups=32,
        )
    elif model_size == "standard":
        return FlowMatching(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(128, 256, 512, 768),
            layers_per_block=2,
            attention_head_dim=16,
            norm_num_groups=32,
        )
    elif model_size == "large":
        return FlowMatching(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(256, 512, 768, 1024),
            layers_per_block=3,
            attention_head_dim=32,
            norm_num_groups=32,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ImageNet猫咪Flow Matching训练')
    
    # 数据相关
    parser.add_argument('--data_dir', type=str, default='./data/imagenet',
                       help='ImageNet数据目录 (默认: ./data/imagenet)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--image_size', type=int, default=64,
                       help='图片尺寸 (默认: 64, 推荐64/128)')
    parser.add_argument('--use_domestic_only', action='store_true',
                       help='只使用家猫类别 (推荐)')
    
    # 模型相关
    parser.add_argument('--model_size', type=str, default='standard',
                       choices=['small', 'standard', 'large'],
                       help='模型大小 (默认: standard)')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数 (默认: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率 (默认: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='权重衰减 (默认: 1e-6)')
    parser.add_argument('--use_ema', action='store_true',
                       help='使用指数移动平均')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                       help='EMA衰减率 (默认: 0.9999)')
    
    # 保存和记录
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_imagenet',
                       help='检查点目录 (默认: checkpoints_imagenet)')
    parser.add_argument('--log_dir', type=str, default='logs_imagenet',
                       help='Tensorboard日志目录 (默认: logs_imagenet)')
    parser.add_argument('--samples_dir', type=str, default='samples_imagenet',
                       help='生成样本目录 (默认: samples_imagenet)')
    parser.add_argument('--save_every_epoch', action='store_true',
                       help='每个epoch都保存检查点')
    parser.add_argument('--sample_every_epoch', action='store_true', 
                       help='每个epoch都生成样本 (默认开启)')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='每次生成的样本数量 (默认: 16)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, epoch, writer, ema_model=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        
        # 计算损失
        loss = model.compute_loss(data)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 更新EMA
        if ema_model is not None:
            ema_model.update()
        
        # 更新统计
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg': f'{avg_loss:.4f}',
            'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'CPU'
        })
        
        # 详细的tensorboard记录
        global_step = epoch * num_batches + batch_idx
        writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
        writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
        
        # 记录梯度信息 (每50个batch)
        if batch_idx % 50 == 0:
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            writer.add_scalar('Training/Gradient_Norm', total_norm, global_step)
        
        # 记录GPU内存使用 (每100个batch)
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            writer.add_scalar('System/GPU_Memory_Allocated_GB', memory_allocated, global_step)
            writer.add_scalar('System/GPU_Memory_Reserved_GB', memory_reserved, global_step)
    
    avg_loss = total_loss / num_batches
    writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch)
    
    return avg_loss


def generate_samples(model, device, num_samples=16, image_size=64, num_steps=100, method='euler'):
    """生成样本"""
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(
            shape=(num_samples, 3, image_size, image_size),
            num_steps=num_steps,
            method=method
        )
    
    return samples


def save_epoch_samples(model, device, epoch, samples_dir, image_size, ema_model=None, num_samples=16):
    """保存每个epoch的生成样本"""
    
    # 生成原始模型样本
    print(f"生成第{epoch+1}个epoch的样本...")
    samples = generate_samples(model, device, num_samples, image_size, num_steps=50)
    
    # 保存原始模型样本
    original_path = os.path.join(samples_dir, f'epoch_{epoch:03d}_original.png')
    save_images_imagenet(samples, original_path, nrow=4)
    
    samples_for_tb = samples.clone()  # 用于tensorboard
    
    # 如果有EMA模型，也生成EMA样本
    if ema_model is not None:
        # 临时切换到EMA参数
        original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_params[name] = param.data.clone()
                param.data.copy_(ema_model.shadow[name])
        
        # 生成EMA样本
        samples_ema = generate_samples(model, device, num_samples, image_size, num_steps=50)
        ema_path = os.path.join(samples_dir, f'epoch_{epoch:03d}_ema.png')
        save_images_imagenet(samples_ema, ema_path, nrow=4)
        
        # 恢复原始参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(original_params[name])
        
        return samples_for_tb, samples_ema
    
    return samples_for_tb, None


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🐱 ImageNet猫咪Flow Matching训练")
    print("=" * 60)
    
    # 显示数据集大小估算
    estimate_imagenet_cat_size()
    
    print(f"\n📊 训练配置:")
    print(f"数据目录: {args.data_dir}")
    print(f"图片尺寸: {args.image_size}x{args.image_size}")
    print(f"模型大小: {args.model_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"使用EMA: {args.use_ema}")
    print(f"只用家猫: {args.use_domestic_only}")
    print("=" * 60)
    
    # 获取设备
    device = get_device()
    
    # 创建目录
    checkpoint_dir = create_checkpoint_dir(args.checkpoint_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    
    # 保存训练配置
    config_path = os.path.join(args.log_dir, 'training_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"ImageNet猫咪Flow Matching训练配置\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据目录: {args.data_dir}\n")
        f.write(f"图片尺寸: {args.image_size}x{args.image_size}\n")
        f.write(f"模型大小: {args.model_size}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"使用EMA: {args.use_ema}\n")
        f.write(f"只用家猫: {args.use_domestic_only}\n")
    
    # 创建数据加载器
    print("🔄 加载数据...")
    try:
        dataloader = get_imagenet_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            use_domestic_only=args.use_domestic_only,
            train=True
        )
        print(f"✅ 成功加载 {len(dataloader.dataset)} 张猫咪图片")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        print("请确保ImageNet数据已下载，或运行 download_imagenet_cats.py")
        return
    
    # 创建模型
    print(f"\n🤖 创建{args.model_size}模型...")
    model = create_imagenet_unet(args.image_size, args.model_size)
    model = model.to(device)
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,}")
    print(f"模型大小: {total_params * 4 / 1024**2:.1f} MB")
    
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
        print(f"✅ 使用EMA，衰减率: {args.ema_decay}")
    
    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        print(f"📂 从检查点恢复: {args.resume}")
        start_epoch, best_loss = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
    
    print(f"\n🚀 开始训练 (从第{start_epoch+1}个epoch开始)...")
    print("=" * 60)
    
    # 训练循环
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # 训练一个epoch
        avg_loss = train_epoch(
            model, dataloader, optimizer, device,
            epoch, writer, ema_model
        )
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        
        print(f"\n📊 Epoch {epoch+1}/{args.epochs} 完成:")
        print(f"   平均损失: {avg_loss:.4f}")
        print(f"   学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   本epoch用时: {epoch_time/60:.1f}分钟")
        print(f"   总训练时间: {total_time/3600:.1f}小时")
        
        # 记录时间到tensorboard
        writer.add_scalar('System/Epoch_Time_Minutes', epoch_time/60, epoch)
        writer.add_scalar('System/Total_Time_Hours', total_time/3600, epoch)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = os.path.join(checkpoint_dir, 'best')
            os.makedirs(best_dir, exist_ok=True)  # 确保目录存在
            save_checkpoint(model, optimizer, epoch, avg_loss, best_dir)
            print(f"   💎 保存最佳模型 (损失: {best_loss:.4f})")
        
        # 每个epoch保存检查点
        if args.save_every_epoch:
            save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_dir)
        
        # 每个epoch生成样本
        if args.sample_every_epoch or (epoch + 1) % 5 == 0:
            print(f"   🎨 生成第{epoch+1}个epoch的样本...")
            
            samples_orig, samples_ema = save_epoch_samples(
                model, device, epoch, args.samples_dir, args.image_size,
                ema_model, args.num_samples
            )
            
            # 添加到tensorboard
            samples_for_tb = denormalize_imagenet(samples_orig)
            writer.add_images('Samples/Original', torch.clamp(samples_for_tb, 0, 1), epoch)
            
            if samples_ema is not None:
                samples_ema_for_tb = denormalize_imagenet(samples_ema)
                writer.add_images('Samples/EMA', torch.clamp(samples_ema_for_tb, 0, 1), epoch)
        
        print("-" * 60)
    
    # 训练完成
    total_training_time = time.time() - training_start_time
    
    # 保存最终模型
    final_dir = os.path.join(checkpoint_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)  # 确保目录存在
    save_checkpoint(model, optimizer, args.epochs - 1, best_loss, final_dir)
    
    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print("=" * 60)
    print(f"📊 训练统计:")
    print(f"   总轮数: {args.epochs}")
    print(f"   最佳损失: {best_loss:.4f}")
    print(f"   总训练时间: {total_training_time/3600:.1f}小时")
    print(f"   平均每epoch: {total_training_time/args.epochs/60:.1f}分钟")
    print(f"\n📁 输出文件:")
    print(f"   检查点: {checkpoint_dir}/")
    print(f"   生成样本: {args.samples_dir}/")
    print(f"   训练日志: {args.log_dir}/")
    print(f"\n🔍 查看训练过程:")
    print(f"   tensorboard --logdir {args.log_dir}")
    print("=" * 60)
    
    writer.close()


if __name__ == "__main__":
    main() 