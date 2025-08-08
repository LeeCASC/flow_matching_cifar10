#!/usr/bin/env python3
"""
改进的ImageNet猫咪数据Flow Matching训练脚本

增强功能：
- 更详细的损失函数记录
- 延长训练时间
- 更频繁的监控和保存
- 训练稳定性改进
- 自动恢复训练功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import json
import logging
from pathlib import Path

from flow_matching_diffusers import FlowMatching
from diffusers import UNet2DModel
from utils_imagenet import (
    get_imagenet_dataloader,
    get_device,
    create_checkpoint_dir,
    save_checkpoint,
    save_best_model_only,
    load_checkpoint,
    save_images_imagenet,
    show_images_imagenet,
    denormalize_imagenet,
    EMAModel,
    estimate_imagenet_cat_size
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(description='改进的ImageNet猫咪Flow Matching训练')
    
    # 数据相关
    parser.add_argument('--data_dir', type=str, default='./data/oxford_pets',
                       help='ImageNet数据目录 (默认: ./data/oxford_pets)')
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
    
    # 训练相关 - 延长训练时间
    parser.add_argument('--epochs', type=int, default=500,  # 增加到500个epochs
                       help='训练轮数 (默认: 500, 更长训练时间)')
    parser.add_argument('--lr', type=float, default=2e-4,  # 稍微提高学习率
                       help='学习率 (默认: 2e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减 (默认: 1e-5)')
    parser.add_argument('--use_ema', action='store_true', default=True,  # 默认启用EMA
                       help='使用指数移动平均 (默认启用)')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                       help='EMA衰减率 (默认: 0.9999)')
    
    # 保存和记录 - 更频繁的监控
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_imagenet_improved',
                       help='检查点目录 (默认: checkpoints_imagenet_improved)')
    parser.add_argument('--log_dir', type=str, default='logs_imagenet_improved',
                       help='Tensorboard日志目录 (默认: logs_imagenet_improved)')
    parser.add_argument('--samples_dir', type=str, default='samples_imagenet_improved',
                       help='生成样本目录 (默认: samples_imagenet_improved)')
    parser.add_argument('--save_every_epoch', action='store_true', default=False,
                       help='每个epoch都保存检查点 (默认禁用，只保存最佳模型)')
    parser.add_argument('--sample_every_epoch', action='store_true', default=True,
                       help='每个epoch都生成样本 (默认启用)')
    parser.add_argument('--num_samples', type=int, default=16,
                       help='每次生成的样本数量 (默认: 16)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--auto_resume', action='store_true', default=True,
                       help='自动从最新检查点恢复训练')
    
    # 监控相关
    parser.add_argument('--log_interval', type=int, default=10,
                       help='损失记录间隔 (每N个batch记录一次)')
    parser.add_argument('--eval_interval', type=int, default=5,
                       help='评估间隔 (每N个epoch评估一次)')
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, device, epoch, writer, ema_model=None, log_interval=10):
    """训练一个epoch - 增强版本"""
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    # 检查数据加载器是否为空
    if num_batches == 0:
        logger.warning("数据加载器为空，返回默认损失值")
        return float('inf'), float('inf')
    
    # 用于计算移动平均损失
    loss_history = []
    valid_batches = 0  # 记录有效的batch数量
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, (data, _) in enumerate(pbar):
        data = data.to(device)
        
        # 计算损失
        loss = model.compute_loss(data)
        
        # 检查损失是否为NaN或无穷大
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"检测到异常损失值: {loss.item()}, 跳过此batch")
            continue
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 更新EMA
        if ema_model is not None:
            ema_model.update()
        
        # 更新统计
        loss_value = loss.item()
        total_loss += loss_value
        loss_history.append(loss_value)
        valid_batches += 1  # 增加有效batch计数
        
        # 保持最近100个损失值用于移动平均
        if len(loss_history) > 100:
            loss_history.pop(0)
        
        # 使用有效batch数量计算平均损失
        avg_loss = total_loss / max(1, valid_batches)
        moving_avg_loss = np.mean(loss_history[-20:]) if loss_history else float('inf')  # 最近20个batch的移动平均
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss_value:.4f}',
            'Avg': f'{avg_loss:.4f}',
            'MA': f'{moving_avg_loss:.4f}',
            'GradNorm': f'{grad_norm:.3f}',
            'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'CPU'
        })
        
        # 详细的tensorboard记录 - 更频繁
        global_step = epoch * num_batches + batch_idx
        
        # 每个batch都记录损失
        writer.add_scalar('Training/Batch_Loss', loss_value, global_step)
        writer.add_scalar('Training/Moving_Avg_Loss', moving_avg_loss, global_step)
        writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('Training/Gradient_Norm', grad_norm, global_step)
        
        # 每log_interval个batch记录详细信息
        if batch_idx % log_interval == 0:
            # 记录损失统计
            writer.add_scalar('Training/Loss_Std', np.std(loss_history[-20:]), global_step)
            writer.add_scalar('Training/Loss_Min', np.min(loss_history[-20:]), global_step)
            writer.add_scalar('Training/Loss_Max', np.max(loss_history[-20:]), global_step)
            
            # 记录模型参数统计
            param_norms = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param_norm = param.data.norm().item()
                    param_norms.append(param_norm)
                    writer.add_scalar(f'Parameters/{name}_norm', param_norm, global_step)
            
            writer.add_scalar('Parameters/Total_Norm', np.mean(param_norms), global_step)
        
        # 每50个batch记录GPU内存使用
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_cached = torch.cuda.memory_cached() / 1024**3
            
            writer.add_scalar('System/GPU_Memory_Allocated_GB', memory_allocated, global_step)
            writer.add_scalar('System/GPU_Memory_Reserved_GB', memory_reserved, global_step)
            writer.add_scalar('System/GPU_Memory_Cached_GB', memory_cached, global_step)
            
            # 记录GPU利用率信息
            if hasattr(torch.cuda, 'utilization'):
                gpu_util = torch.cuda.utilization()
                writer.add_scalar('System/GPU_Utilization_Percent', gpu_util, global_step)
    
    # Epoch结束时的统计
    if valid_batches == 0:
        logger.warning("本epoch没有有效的batch，返回默认损失值")
        avg_loss = float('inf')
        final_moving_avg = float('inf')
    else:
        avg_loss = total_loss / valid_batches
        final_moving_avg = np.mean(loss_history[-50:]) if len(loss_history) >= 50 else avg_loss
    
    writer.add_scalar('Training/Epoch_Loss', avg_loss, epoch)
    writer.add_scalar('Training/Epoch_Moving_Avg_Loss', final_moving_avg, epoch)
    
    logger.info(f"Epoch {epoch+1} 完成 - 平均损失: {avg_loss:.4f}, 移动平均: {final_moving_avg:.4f}, 有效batches: {valid_batches}/{num_batches}")
    
    return avg_loss, final_moving_avg


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
    logger.info(f"生成第{epoch+1}个epoch的样本...")
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


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点文件，优先查找best.unet"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # 优先查找最佳模型
    best_model_path = checkpoint_dir / 'best.unet'
    if best_model_path.exists():
        return str(best_model_path)
    
    # 如果没有最佳模型，查找其他检查点文件
    checkpoint_files = list(checkpoint_dir.glob('**/checkpoint_epoch_*.pth'))
    if not checkpoint_files:
        return None
    
    # 按epoch编号排序，返回最新的
    def extract_epoch(path):
        try:
            return int(path.stem.split('_')[-1])
        except:
            return -1
    
    latest_checkpoint = max(checkpoint_files, key=extract_epoch)
    return str(latest_checkpoint)


def save_training_state(epoch, loss, best_loss, training_time, log_dir):
    """保存训练状态信息"""
    state = {
        'epoch': epoch,
        'loss': loss,
        'best_loss': best_loss,
        'training_time_hours': training_time / 3600,
        'timestamp': datetime.now().isoformat()
    }
    
    state_path = os.path.join(log_dir, 'training_state.json')
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🚀 改进的ImageNet猫咪Flow Matching训练")
    print("=" * 70)
    
    # 显示数据集大小估算
    estimate_imagenet_cat_size()
    
    print(f"\n📊 训练配置:")
    print(f"数据目录: {args.data_dir}")
    print(f"图片尺寸: {args.image_size}x{args.image_size}")
    print(f"模型大小: {args.model_size}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs} (更长训练时间)")
    print(f"学习率: {args.lr}")
    print(f"使用EMA: {args.use_ema}")
    print(f"只用家猫: {args.use_domestic_only}")
    print(f"自动恢复: {args.auto_resume}")
    print(f"磁盘节省模式: 只保存最佳模型 (best.unet)")
    print("=" * 70)
    
    # 获取设备
    device = get_device()
    
    # 创建目录
    checkpoint_dir = create_checkpoint_dir(args.checkpoint_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    
    # 保存训练配置
    config_path = os.path.join(args.log_dir, 'training_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"改进的ImageNet猫咪Flow Matching训练配置\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据目录: {args.data_dir}\n")
        f.write(f"图片尺寸: {args.image_size}x{args.image_size}\n")
        f.write(f"模型大小: {args.model_size}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"使用EMA: {args.use_ema}\n")
        f.write(f"只用家猫: {args.use_domestic_only}\n")
        f.write(f"自动恢复: {args.auto_resume}\n")
    
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
        logger.error(f"❌ 数据加载失败: {e}")
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
    
    # 学习率调度器 - 使用余弦退火，更适合长时间训练
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # 每50个epoch重启一次
        T_mult=2,  # 重启间隔倍数
        eta_min=args.lr * 0.001
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
    
    # 自动恢复或手动指定检查点
    resume_path = args.resume
    if args.auto_resume and not resume_path:
        resume_path = find_latest_checkpoint(args.checkpoint_dir)
    
    if resume_path:
        print(f"📂 从检查点恢复: {resume_path}")
        try:
            start_epoch, best_loss = load_checkpoint(resume_path, model, optimizer)
            start_epoch += 1
            print(f"✅ 成功恢复到第{start_epoch}个epoch，最佳损失: {best_loss:.4f}")
        except Exception as e:
            logger.error(f"❌ 检查点加载失败: {e}")
            print("将从头开始训练...")
            start_epoch = 0
            best_loss = float('inf')
    
    print(f"\n🚀 开始训练 (从第{start_epoch+1}个epoch开始)...")
    print(f"📈 预计总训练时间: {args.epochs * 2:.0f}-{args.epochs * 4:.0f}小时")
    print("=" * 70)
    
    # 训练循环
    training_start_time = time.time()
    
    # 初始化变量，避免UnboundLocalError
    avg_loss = float('inf')
    moving_avg_loss = float('inf')
    epoch = start_epoch
    
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start_time = time.time()
            
            try:
                # 训练一个epoch
                avg_loss, moving_avg_loss = train_epoch(
                    model, dataloader, optimizer, device,
                    epoch, writer, ema_model, args.log_interval
                )
                
                # 更新学习率
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                epoch_time = time.time() - epoch_start_time
                total_time = time.time() - training_start_time
                
                print(f"\n📊 Epoch {epoch+1}/{args.epochs} 完成:")
                print(f"   平均损失: {avg_loss:.4f}")
                print(f"   移动平均损失: {moving_avg_loss:.4f}")
                print(f"   学习率: {current_lr:.6f}")
                print(f"   本epoch用时: {epoch_time/60:.1f}分钟")
                print(f"   总训练时间: {total_time/3600:.1f}小时")
                print(f"   预计剩余时间: {(total_time/(epoch-start_epoch+1))*(args.epochs-epoch-1)/3600:.1f}小时")
                
                # 记录时间到tensorboard
                writer.add_scalar('System/Epoch_Time_Minutes', epoch_time/60, epoch)
                writer.add_scalar('System/Total_Time_Hours', total_time/3600, epoch)
                writer.add_scalar('Training/Current_Learning_Rate', current_lr, epoch)
                
                # 只保存最佳模型为best.unet
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_best_model_only(model, optimizer, epoch, avg_loss, checkpoint_dir)
                
                # 可选：每个epoch保存(不推荐，占用大量磁盘空间)
                if args.save_every_epoch and (epoch + 1) % 10 == 0:
                    save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_dir)
                    print(f"   💾 保存定期检查点: epoch_{epoch}")
                
                # 保存训练状态
                save_training_state(epoch, avg_loss, best_loss, total_time, args.log_dir)
                
                # 每个epoch生成样本
                if args.sample_every_epoch or (epoch + 1) % args.eval_interval == 0:
                    print(f"   🎨 生成第{epoch+1}个epoch的样本...")
                    
                    try:
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
                        
                        print(f"   ✅ 样本已保存")
                    except Exception as e:
                        logger.error(f"   ❌ 样本生成失败: {e}")
                
                print("-" * 70)
                
                # 每50个epoch强制刷新tensorboard
                if (epoch + 1) % 50 == 0:
                    writer.flush()
                    
            except Exception as epoch_error:
                logger.error(f"Epoch {epoch+1} 训练失败: {epoch_error}")
                print(f"❌ Epoch {epoch+1} 训练失败，跳过到下一个epoch")
                # 如果单个epoch失败，继续训练下一个epoch
                continue
                
    except KeyboardInterrupt:
        print(f"\n⚠️  训练被用户中断 (Epoch {epoch+1})")
        print(f"💾 保存当前状态...")
        try:
            total_time = time.time() - training_start_time
            save_best_model_only(model, optimizer, epoch, avg_loss, checkpoint_dir)
            save_training_state(epoch, avg_loss, best_loss, total_time, args.log_dir)
            print(f"✅ 状态已保存，可以使用 --auto_resume 继续训练")
        except Exception as save_error:
            logger.error(f"保存状态失败: {save_error}")
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误 (Epoch {epoch+1}): {e}")
        print(f"💾 保存当前状态...")
        try:
            total_time = time.time() - training_start_time
            save_best_model_only(model, optimizer, epoch, avg_loss, checkpoint_dir)
            save_training_state(epoch, avg_loss, best_loss, total_time, args.log_dir)
            print(f"✅ 状态已保存")
        except Exception as save_error:
            logger.error(f"保存状态失败: {save_error}")
        raise
    
    # 训练完成
    total_training_time = time.time() - training_start_time
    
    # 最终模型已经在best.unet中，不需重复保存
    print(f"🏆 最佳模型已保存为: {checkpoint_dir}/best.unet")
    
    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print("=" * 70)
    print(f"📊 训练统计:")
    print(f"   总轮数: {args.epochs}")
    print(f"   最佳损失: {best_loss:.4f}")
    print(f"   总训练时间: {total_training_time/3600:.1f}小时")
    print(f"   平均每epoch: {total_training_time/args.epochs/60:.1f}分钟")
    print(f"\n📁 输出文件:")
    print(f"   最佳模型: {checkpoint_dir}/best.unet")
    print(f"   生成样本: {args.samples_dir}/")
    print(f"   训练日志: {args.log_dir}/")
    print(f"\n🔍 查看训练过程:")
    print(f"   tensorboard --logdir {args.log_dir}")
    print("=" * 70)
    
    writer.close()


if __name__ == "__main__":
    main() 