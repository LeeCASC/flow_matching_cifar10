#!/usr/bin/env python3
"""
Flow Matching Demo Script with Diffusers UNet

这个脚本演示如何使用diffusers库中的UNet2DModel来训练Flow Matching模型。
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchvision.utils

from flow_matching_diffusers import create_small_unet_for_cifar10, create_unet_for_cifar10
from utils import (
    get_cifar10_dataloader,
    get_device,
    save_images,
    show_images,
    get_cifar10_class_names
)


def quick_demo_diffusers():
    """使用diffusers UNet的快速演示"""
    
    print("=" * 60)
    print("Flow Matching with Diffusers UNet Demo")
    print("=" * 60)
    
    # 配置参数
    config = {
        'class_idx': 2,  # bird class
        'batch_size': 32,
        'epochs': 10,
        'lr': 2e-4,
        'num_generate': 16,
        'ode_steps': 50
    }
    
    class_names = get_cifar10_class_names()
    print(f"Training on CIFAR-10 class: {class_names[config['class_idx']]}")
    print(f"Training epochs: {config['epochs']}")
    print(f"Using Diffusers UNet2DModel")
    print("=" * 60)
    
    # 获取设备
    device = get_device()
    
    # 准备数据
    print("Loading data...")
    dataloader = get_cifar10_dataloader(
        class_idx=config['class_idx'],
        batch_size=config['batch_size'],
        train=True
    )
    print(f"Dataset size: {len(dataloader.dataset)} images")
    
    # 显示一些真实图像
    print("\nShowing some real images from the dataset:")
    real_batch = next(iter(dataloader))[0][:16]
    show_images(real_batch, "Real Images", nrow=4, figsize=(8, 8))
    
    # 创建模型 - 使用小型配置进行演示
    print("\nCreating model with Diffusers UNet...")
    model = create_small_unet_for_cifar10()
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    print("\nStarting training...")
    print("=" * 60)
    
    # 训练循环
    losses = []
    model.train()
    
    for epoch in range(config['epochs']):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            
            # 前向传播
            loss = model.compute_loss(data)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
        
        # 每几个epoch生成一些样本
        if (epoch + 1) % 3 == 0:
            print(f"Generating samples at epoch {epoch+1}...")
            model.eval()
            with torch.no_grad():
                samples = model.sample(
                    shape=(config['num_generate'], 3, 32, 32),
                    num_steps=config['ode_steps'],
                    method='euler'
                )
            show_images(samples, f"Generated Images (Epoch {epoch+1})", nrow=4, figsize=(8, 8))
            model.train()
    
    print("=" * 60)
    print("Training completed!")
    
    # 显示训练损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # 最终生成
    print("\nGenerating final samples...")
    model.eval()
    
    # 比较不同步数的效果
    step_configs = [25, 50, 100]
    fig, axes = plt.subplots(1, len(step_configs), figsize=(15, 5))
    
    for i, steps in enumerate(step_configs):
        print(f"Generating with {steps} steps...")
        with torch.no_grad():
            samples = model.sample(
                shape=(16, 3, 32, 32),
                num_steps=steps,
                method='euler'
            )
        
        # 转换为可显示格式
        grid = torch.clamp((samples + 1) / 2, 0, 1)
        grid = torchvision.utils.make_grid(grid, nrow=4, normalize=False)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        
        axes[i].imshow(grid)
        axes[i].set_title(f'{steps} ODE Steps')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Generated Images with Different ODE Steps', y=1.02)
    plt.show()
    
    # 保存一些样本
    print("\nSaving generated samples...")
    final_samples = model.sample(
        shape=(64, 3, 32, 32),
        num_steps=100,
        method='euler'
    )
    save_images(final_samples, 'demo_diffusers_generated_samples.png', nrow=8)
    
    print("=" * 60)
    print("Demo completed!")
    print("Benefits of using Diffusers UNet:")
    print("1. More stable and well-tested implementation")
    print("2. Better architecture design with proper skip connections")
    print("3. Optimized attention mechanisms")
    print("4. No dimension mismatch issues")
    print("5. Professional-grade normalization and activation choices")
    print("=" * 60)


def compare_model_sizes():
    """比较不同大小的模型"""
    print("=" * 60)
    print("Model Size Comparison")
    print("=" * 60)
    
    from flow_matching_diffusers import create_small_unet_for_cifar10, create_unet_for_cifar10, create_large_unet_for_cifar10
    
    models = {
        "Small": create_small_unet_for_cifar10(),
        "Standard": create_unet_for_cifar10(), 
        "Large": create_large_unet_for_cifar10()
    }
    
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name} Model: {total_params:,} parameters")
        
        # 估算内存使用
        param_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per float32
        print(f"  Estimated size: {param_size_mb:.1f} MB")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Flow Matching Demo with Diffusers')
    parser.add_argument('--mode', type=str, default='demo', 
                       choices=['demo', 'compare'],
                       help='Demo mode (default: demo)')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        quick_demo_diffusers()
    elif args.mode == 'compare':
        compare_model_sizes() 