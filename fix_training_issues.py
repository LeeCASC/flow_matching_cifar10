#!/usr/bin/env python3
"""
训练问题诊断和修复脚本

诊断并修复常见的Flow Matching训练问题：
1. 损失函数不记录
2. Tensorboard显示异常
3. 训练进程冲突
4. 检查点损坏
"""

import os
import subprocess
import json
import glob
from pathlib import Path
import argparse

def check_running_processes():
    """检查正在运行的训练进程"""
    print("🔍 检查正在运行的训练进程...")
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        train_processes = []
        for line in lines:
            if 'train_imagenet' in line and 'python' in line:
                parts = line.split()
                if len(parts) >= 2:
                    pid = parts[1]
                    train_processes.append((pid, line))
        
        if train_processes:
            print(f"⚠️  发现 {len(train_processes)} 个训练进程:")
            for pid, line in train_processes:
                print(f"   PID {pid}: {line.split()[-1]}")
            return train_processes
        else:
            print("✅ 没有发现正在运行的训练进程")
            return []
    except Exception as e:
        print(f"❌ 检查进程失败: {e}")
        return []

def kill_training_processes(processes):
    """终止训练进程"""
    if not processes:
        return
    
    print(f"\n🛑 终止 {len(processes)} 个训练进程...")
    for pid, _ in processes:
        try:
            subprocess.run(['kill', pid], check=True)
            print(f"✅ 已终止进程 {pid}")
        except subprocess.CalledProcessError:
            try:
                subprocess.run(['kill', '-9', pid], check=True)
                print(f"✅ 强制终止进程 {pid}")
            except subprocess.CalledProcessError:
                print(f"❌ 无法终止进程 {pid}")

def check_tensorboard_logs():
    """检查Tensorboard日志文件"""
    print("\n🔍 检查Tensorboard日志...")
    
    log_dirs = ['logs_imagenet', 'logs_imagenet_improved']
    issues = []
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            print(f"📁 检查目录: {log_dir}")
            
            # 检查事件文件
            event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
            print(f"   发现 {len(event_files)} 个事件文件")
            
            # 检查文件大小
            for event_file in event_files:
                size = os.path.getsize(event_file)
                if size < 1000:  # 小于1KB可能有问题
                    issues.append(f"事件文件过小: {event_file} ({size} bytes)")
                else:
                    print(f"   ✅ {os.path.basename(event_file)}: {size/1024:.1f}KB")
            
            # 检查配置文件
            config_file = os.path.join(log_dir, 'training_config.txt')
            if os.path.exists(config_file):
                print(f"   ✅ 找到配置文件: {config_file}")
            else:
                issues.append(f"缺少配置文件: {config_file}")
    
    if issues:
        print("⚠️  发现以下问题:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ Tensorboard日志文件正常")
    
    return issues

def check_checkpoints():
    """检查检查点文件"""
    print("\n🔍 检查检查点文件...")
    
    checkpoint_dirs = ['checkpoints_imagenet', 'checkpoints_imagenet_improved']
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            print(f"📁 检查目录: {checkpoint_dir}")
            
            # 检查best目录
            best_dir = os.path.join(checkpoint_dir, 'best')
            if os.path.exists(best_dir):
                checkpoints = glob.glob(os.path.join(best_dir, 'checkpoint_epoch_*.pth'))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    epoch = latest_checkpoint.split('_')[-1].split('.')[0]
                    size = os.path.getsize(latest_checkpoint)
                    print(f"   ✅ 最新检查点: epoch_{epoch} ({size/1024/1024:.1f}MB)")
                else:
                    print(f"   ⚠️  没有找到检查点文件")
            else:
                print(f"   ⚠️  没有找到best目录")

def clean_corrupted_logs():
    """清理损坏的日志文件"""
    print("\n🧹 清理损坏的日志文件...")
    
    log_dirs = ['logs_imagenet', 'logs_imagenet_improved']
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
            
            for event_file in event_files:
                size = os.path.getsize(event_file)
                if size < 500:  # 小于500字节认为是损坏的
                    print(f"   🗑️  删除损坏文件: {os.path.basename(event_file)}")
                    os.remove(event_file)

def create_new_training_session():
    """创建新的训练会话"""
    print("\n🆕 创建新的训练会话...")
    
    # 创建新的日志目录
    new_log_dir = f"logs_imagenet_fixed"
    new_checkpoint_dir = f"checkpoints_imagenet_fixed"
    new_samples_dir = f"samples_imagenet_fixed"
    
    os.makedirs(new_log_dir, exist_ok=True)
    os.makedirs(new_checkpoint_dir, exist_ok=True)
    os.makedirs(new_samples_dir, exist_ok=True)
    
    print(f"✅ 创建新目录:")
    print(f"   日志: {new_log_dir}")
    print(f"   检查点: {new_checkpoint_dir}")
    print(f"   样本: {new_samples_dir}")
    
    return new_log_dir, new_checkpoint_dir, new_samples_dir

def generate_fixed_training_command(log_dir, checkpoint_dir, samples_dir):
    """生成修复后的训练命令"""
    cmd = [
        "python", "train_imagenet_improved.py",
        "--epochs", "500",
        "--batch_size", "32",
        "--lr", "2e-4",
        "--use_ema",
        "--model_size", "standard",
        "--log_dir", log_dir,
        "--checkpoint_dir", checkpoint_dir,
        "--samples_dir", samples_dir,
        "--log_interval", "1",  # 每个batch都记录
        "--eval_interval", "5",
        "--auto_resume"
    ]
    
    return cmd

def main():
    parser = argparse.ArgumentParser(description="训练问题诊断和修复")
    parser.add_argument('--diagnose', action='store_true', help='只诊断问题，不修复')
    parser.add_argument('--fix', action='store_true', help='诊断并修复问题')
    parser.add_argument('--kill_processes', action='store_true', help='终止所有训练进程')
    parser.add_argument('--clean_logs', action='store_true', help='清理损坏的日志')
    parser.add_argument('--new_session', action='store_true', help='创建新的训练会话')
    
    args = parser.parse_args()
    
    if not any([args.diagnose, args.fix, args.kill_processes, args.clean_logs, args.new_session]):
        # 默认执行完整诊断和修复
        args.fix = True
    
    print("=" * 70)
    print("🔧 Flow Matching训练问题诊断和修复工具")
    print("=" * 70)
    
    # 检查运行中的进程
    processes = check_running_processes()
    
    if args.kill_processes or (args.fix and processes):
        if processes:
            response = input("\n是否终止所有训练进程？(y/N): ")
            if response.lower() == 'y':
                kill_training_processes(processes)
    
    # 检查日志文件
    log_issues = check_tensorboard_logs()
    
    # 检查检查点
    check_checkpoints()
    
    if args.clean_logs or (args.fix and log_issues):
        if log_issues:
            response = input("\n是否清理损坏的日志文件？(y/N): ")
            if response.lower() == 'y':
                clean_corrupted_logs()
    
    if args.new_session or args.fix:
        response = input("\n是否创建新的训练会话？(Y/n): ")
        if response.lower() in ['', 'y', 'yes']:
            log_dir, checkpoint_dir, samples_dir = create_new_training_session()
            
            # 生成修复后的训练命令
            cmd = generate_fixed_training_command(log_dir, checkpoint_dir, samples_dir)
            
            print(f"\n🚀 修复后的训练命令:")
            print(" ".join(cmd))
            
            print(f"\n📊 新的Tensorboard命令:")
            print(f"tensorboard --logdir {log_dir}")
            
            print(f"\n🔥 主要修复:")
            print("✅ 使用新的日志目录，避免冲突")
            print("✅ 每个batch都记录损失 (--log_interval 1)")
            print("✅ 启用自动恢复功能")
            print("✅ 使用改进的训练脚本")
            
            start_response = input("\n立即开始修复后的训练？(Y/n): ")
            if start_response.lower() in ['', 'y', 'yes']:
                print("🚀 开始训练...")
                try:
                    subprocess.run(cmd)
                except KeyboardInterrupt:
                    print("\n⚠️  训练被用户中断")
                except Exception as e:
                    print(f"\n❌ 训练失败: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 诊断和修复完成！")
    print("=" * 70)

if __name__ == "__main__":
    main() 