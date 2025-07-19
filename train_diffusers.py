import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from tqdm import tqdm
import numpy as np

from flow_matching_diffusers import (
    create_small_unet_for_cifar10,
    create_unet_for_cifar10, 
    create_large_unet_for_cifar10
)
from utils import (
    get_cifar10_dataloader, 
    get_device, 
    create_checkpoint_dir,
    save_checkpoint,
    load_checkpoint,
    save_images,
    EMAModel,
    get_cifar10_class_names
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Flow Matching model with Diffusers UNet on CIFAR-10')
    
    # 数据相关
    parser.add_argument('--class_idx', type=int, default=2, 
                       help='CIFAR-10 class index (0-9, default: 2 for bird)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--image_size', type=int, default=32,
                       help='Image size (default: 32)')
    
    # 模型相关
    parser.add_argument('--model_size', type=str, default='standard', 
                       choices=['small', 'standard', 'large'],
                       help='Model size (default: standard)')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay (default: 1e-6)')
    parser.add_argument('--use_ema', action='store_true',
                       help='Use exponential moving average')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                       help='EMA decay rate (default: 0.9999)')
    
    # 保存和日志
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_diffusers',
                       help='Directory to save checkpoints (default: checkpoints_diffusers)')
    parser.add_argument('--log_dir', type=str, default='logs_diffusers',
                       help='Directory for tensorboard logs (default: logs_diffusers)')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--sample_interval', type=int, default=5,
                       help='Generate samples every N epochs (default: 5)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint')
    
    return parser.parse_args()


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, device, epoch, writer, ema_model=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
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
            'Avg Loss': f'{avg_loss:.4f}'
        })
        
        # 记录到tensorboard
        global_step = epoch * num_batches + batch_idx
        writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
        writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
        
        # 记录梯度信息
        if batch_idx % 100 == 0:  # 每100个batch记录一次梯度信息
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            writer.add_scalar('Training/Gradient_Norm', total_norm, global_step)
    
    avg_loss = total_loss / num_batches
    writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch)
    
    return avg_loss


def generate_samples(model, device, num_samples=16, num_steps=100, method='euler'):
    """生成样本"""
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(
            shape=(num_samples, 3, 32, 32),
            num_steps=num_steps,
            method=method
        )
    
    return samples


def main():
    args = parse_args()
    
    # 打印配置
    print("=" * 50)
    print("Flow Matching Training with Diffusers UNet")
    print("=" * 50)
    class_names = get_cifar10_class_names()
    print(f"Dataset: CIFAR-10 class '{class_names[args.class_idx]}' (index: {args.class_idx})")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Use EMA: {args.use_ema}")
    print(f"Model size: {args.model_size}")
    print("=" * 50)
    
    # 获取设备
    device = get_device()
    
    # 创建目录
    checkpoint_dir = create_checkpoint_dir(args.checkpoint_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('samples_diffusers', exist_ok=True)
    
    # 创建数据加载器
    print("\nLoading data...")
    dataloader = get_cifar10_dataloader(
        class_idx=args.class_idx,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train=True
    )
    
    # 创建模型
    print(f"\nCreating {args.model_size} model with Diffusers UNet...")
    
    if args.model_size == 'small':
        model = create_small_unet_for_cifar10()
    elif args.model_size == 'standard':
        model = create_unet_for_cifar10()
    elif args.model_size == 'large':
        model = create_large_unet_for_cifar10()
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = count_parameters(model)
    print(f"Model has {total_params:,} trainable parameters")
    
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
    print("=" * 50)
    
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
        
        # 记录更多训练信息到tensorboard
        writer.add_scalar('Training/GPU_Memory_Allocated', 
                         torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0, epoch)
        
        # 每个epoch都生成少量样本用于tensorboard记录
        if epoch % max(1, args.sample_interval // 2) == 0:  # 更频繁地生成用于记录
            print("Generating samples for monitoring...")
            
            # 使用原始模型生成少量样本用于tensorboard
            quick_samples = generate_samples(model, device, num_samples=8, num_steps=25, method='euler')
            writer.add_images('Quick_Samples/Original', (quick_samples + 1) / 2, epoch)
            
            # 如果使用EMA，也生成EMA样本用于tensorboard
            if ema_model is not None:
                # 临时应用EMA参数
                original_params = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        original_params[name] = param.data.clone()
                        param.data.copy_(ema_model.shadow[name])
                
                quick_samples_ema = generate_samples(model, device, num_samples=8, num_steps=25, method='euler')
                writer.add_images('Quick_Samples/EMA', (quick_samples_ema + 1) / 2, epoch)
                
                # 恢复原始参数
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.data.copy_(original_params[name])
        
        # 生成高质量样本保存到文件
        if (epoch + 1) % args.sample_interval == 0:
            print("Generating high-quality samples...")
            
            # 使用原始模型生成
            samples = generate_samples(model, device, num_samples=16)
            save_images(samples, f'samples_diffusers/epoch_{epoch}_original.png', nrow=4)
            writer.add_images('High_Quality_Samples/Original', (samples + 1) / 2, epoch)
            
            # 如果使用EMA，也生成EMA模型的样本
            if ema_model is not None:
                # 临时应用EMA参数
                original_params = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        original_params[name] = param.data.clone()
                        param.data.copy_(ema_model.shadow[name])
                
                samples_ema = generate_samples(model, device, num_samples=16)
                save_images(samples_ema, f'samples_diffusers/epoch_{epoch}_ema.png', nrow=4)
                writer.add_images('High_Quality_Samples/EMA', (samples_ema + 1) / 2, epoch)
                
                # 恢复原始参数
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.data.copy_(original_params[name])
    
    # 保存最终模型
    save_checkpoint(model, optimizer, args.epochs - 1, best_loss, 
                   os.path.join(checkpoint_dir, 'final'))
    
    print("=" * 50)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"Samples saved in: samples_diffusers/")
    print("=" * 50)
    
    writer.close()


if __name__ == "__main__":
    main() 