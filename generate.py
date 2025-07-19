import torch
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from flow_matching import UNet, FlowMatching
from utils import (
    get_device, 
    load_checkpoint, 
    save_images, 
    show_images,
    get_cifar10_class_names
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Generate images using trained Flow Matching model')
    
    # 模型相关
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--time_emb_dim', type=int, default=128,
                       help='Time embedding dimension (default: 128)')
    parser.add_argument('--channels', nargs='+', type=int, default=[64, 128, 256, 512],
                       help='U-Net channels (default: [64, 128, 256, 512])')
    
    # 生成相关
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate (default: 64)')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of ODE steps (default: 100)')
    parser.add_argument('--method', type=str, default='euler', choices=['euler', 'rk4'],
                       help='ODE solver method (default: euler)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Generation batch size (default: 16)')
    
    # 输出相关
    parser.add_argument('--output_dir', type=str, default='generated_images',
                       help='Output directory for generated images (default: generated_images)')
    parser.add_argument('--nrow', type=int, default=8,
                       help='Number of images per row in grid (default: 8)')
    parser.add_argument('--show_images', action='store_true',
                       help='Show generated images using matplotlib')
    parser.add_argument('--save_individual', action='store_true',
                       help='Save individual images in addition to grid')
    
    return parser.parse_args()


def generate_batch(model, device, batch_size, num_steps, method):
    """生成一批样本"""
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(
            shape=(batch_size, 3, 32, 32),
            num_steps=num_steps,
            method=method
        )
    
    return samples


def main():
    args = parse_args()
    
    print("=" * 50)
    print("Flow Matching Image Generation")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of samples: {args.num_samples}")
    print(f"ODE steps: {args.num_steps}")
    print(f"Method: {args.method}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 50)
    
    # 检查检查点文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found!")
        return
    
    # 获取设备
    device = get_device()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建模型
    print("Creating model...")
    velocity_network = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=args.time_emb_dim,
        channels=tuple(args.channels)
    )
    
    model = FlowMatching(velocity_network)
    model = model.to(device)
    
    # 加载检查点
    print(f"Loading checkpoint from {args.checkpoint}...")
    try:
        epoch, loss = load_checkpoint(args.checkpoint, model)
        print(f"Model loaded successfully (trained for {epoch + 1} epochs, final loss: {loss:.4f})")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # 生成图像
    print(f"\nGenerating {args.num_samples} images...")
    print(f"Using {args.method} method with {args.num_steps} steps")
    
    all_samples = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(args.batch_size, args.num_samples - batch_idx * args.batch_size)
        
        print(f"Generating batch {batch_idx + 1}/{num_batches} ({current_batch_size} samples)...")
        
        samples = generate_batch(model, device, current_batch_size, args.num_steps, args.method)
        all_samples.append(samples)
    
    # 合并所有样本
    all_samples = torch.cat(all_samples, dim=0)
    print(f"Generated {all_samples.shape[0]} samples")
    
    # 生成时间戳用于文件命名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存网格图像
    grid_filename = os.path.join(args.output_dir, f'generated_grid_{timestamp}.png')
    save_images(all_samples, grid_filename, nrow=args.nrow)
    
    # 保存单独的图像
    if args.save_individual:
        individual_dir = os.path.join(args.output_dir, f'individual_{timestamp}')
        os.makedirs(individual_dir, exist_ok=True)
        
        print(f"Saving individual images to {individual_dir}...")
        for i in range(all_samples.shape[0]):
            individual_filename = os.path.join(individual_dir, f'sample_{i:03d}.png')
            save_images(all_samples[i:i+1], individual_filename, nrow=1)
    
    # 显示图像（如果请求）
    if args.show_images:
        print("Displaying generated images...")
        show_images(all_samples, title=f"Generated Images ({args.method}, {args.num_steps} steps)", 
                   nrow=args.nrow, figsize=(12, 12))
    
    # 保存生成参数信息
    info_filename = os.path.join(args.output_dir, f'generation_info_{timestamp}.txt')
    with open(info_filename, 'w') as f:
        f.write("Flow Matching Image Generation Info\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Number of samples: {args.num_samples}\n")
        f.write(f"ODE steps: {args.num_steps}\n")
        f.write(f"Method: {args.method}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Model channels: {args.channels}\n")
        f.write(f"Time embedding dim: {args.time_emb_dim}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Generation time: {timestamp}\n")
        if 'epoch' in locals():
            f.write(f"Model epoch: {epoch + 1}\n")
            f.write(f"Model loss: {loss:.4f}\n")
    
    print(f"\nGeneration info saved to: {info_filename}")
    print("=" * 50)
    print("Image generation completed!")
    print(f"Grid image saved to: {grid_filename}")
    if args.save_individual:
        print(f"Individual images saved to: {individual_dir}")
    print("=" * 50)


def generate_comparison():
    """生成不同步数和方法的比较图像"""
    parser = argparse.ArgumentParser(description='Generate comparison images with different settings')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='comparison', help='Output directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found!")
        return
    
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建模型
    velocity_network = UNet(in_channels=3, out_channels=3)
    model = FlowMatching(velocity_network)
    model = model.to(device)
    load_checkpoint(args.checkpoint, model)
    
    # 不同的配置
    configs = [
        {'steps': 50, 'method': 'euler'},
        {'steps': 100, 'method': 'euler'},
        {'steps': 50, 'method': 'rk4'},
        {'steps': 100, 'method': 'rk4'},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, config in enumerate(configs):
        print(f"Generating with {config['steps']} steps, {config['method']} method...")
        samples = generate_batch(model, device, 16, config['steps'], config['method'])
        
        # 显示样本网格
        grid = torchvision.utils.make_grid((samples + 1) / 2, nrow=4, normalize=False)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        
        axes[i].imshow(grid)
        axes[i].set_title(f"{config['method'].upper()}, {config['steps']} steps")
        axes[i].axis('off')
        
        # 保存单独的图像
        save_images(samples, os.path.join(args.output_dir, 
                   f"{config['method']}_{config['steps']}_steps.png"), nrow=4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison images saved in: {args.output_dir}")


if __name__ == "__main__":
    main() 