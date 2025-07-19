# ImageNet猫咪Flow Matching训练指南

## 📊 数据集大小估算

### 🐱 ImageNet猫咪数据
- **家猫类别**: 5个品种 (虎斑、波斯、暹罗等)
- **所有猫科**: 13个品种 (包括狮子、老虎、豹子等)
- **图片数量**: 约5,000-15,000张
- **数据大小**: 1-3GB (取决于分辨率)
- **推荐分辨率**: 64x64 或 128x128

## 🚀 快速开始

### 1. 数据准备 (3种方法)

#### 方法A: Oxford Pet数据集 (推荐)
```bash
# 自动下载高质量的猫咪数据 (~800MB)
python download_imagenet_cats.py --method oxford

# 数据将保存到: ./data/oxford_pets/cats/
```

#### 方法B: 演示数据
```bash
# 创建合成演示数据，快速测试
python download_imagenet_cats.py --method demo

# 数据将保存到: ./data/demo_cats/
```

#### 方法C: 官方ImageNet
```bash
# 查看官方ImageNet下载说明
python download_imagenet_cats.py --method info

# 需要手动注册和下载
```

### 2. 快速演示
```bash
# 10分钟体验 (使用合成数据)
python demo_imagenet.py
```

### 3. 开始训练

#### 基础训练 (推荐新手)
```bash
# 使用Oxford数据，64x64分辨率
python train_imagenet.py \
    --data_dir ./data/oxford_pets \
    --image_size 64 \
    --model_size small \
    --epochs 100 \
    --batch_size 32 \
    --use_ema \
    --sample_every_epoch
```

#### 高质量训练
```bash
# 128x128分辨率，大模型，长时间训练
python train_imagenet.py \
    --data_dir ./data/oxford_pets \
    --image_size 128 \
    --model_size large \
    --epochs 300 \
    --batch_size 16 \
    --use_ema \
    --sample_every_epoch \
    --save_every_epoch
```

## 📈 Tensorboard监控

```bash
# 启动tensorboard
tensorboard --logdir logs_imagenet

# 在浏览器中查看: http://localhost:6006
```

### 监控内容
- **Training/Batch_Loss**: 实时损失变化
- **Training/Epoch_Loss**: 每个epoch平均损失  
- **Training/Learning_Rate**: 学习率调度
- **Training/Gradient_Norm**: 梯度范数监控
- **System/GPU_Memory**: GPU内存使用
- **System/Epoch_Time**: 每个epoch训练时间
- **Samples/Original**: 原始模型生成样本
- **Samples/EMA**: EMA模型生成样本

## ⚙️ 训练参数说明

### 核心参数
```bash
--data_dir          # 数据目录
--image_size        # 图片尺寸 (32/64/128)
--model_size        # 模型大小 (small/standard/large)  
--epochs            # 训练轮数
--batch_size        # 批次大小
--lr                # 学习率 (默认1e-4)
--use_ema           # 使用指数移动平均
--use_domestic_only # 只使用家猫类别
```

### 输出控制
```bash
--sample_every_epoch    # 每个epoch保存样本
--save_every_epoch      # 每个epoch保存检查点
--num_samples          # 每次生成的样本数量
--samples_dir          # 样本保存目录
--checkpoint_dir       # 检查点目录
--log_dir              # Tensorboard日志目录
```

## 💻 硬件配置建议

### 根据GPU内存选择配置

| GPU内存 | 图片尺寸 | 模型大小 | 批次大小 | 预期训练时间 |
|---------|----------|----------|----------|--------------|
| 4-6GB   | 64x64   | small    | 16       | 4-6小时      |
| 8-12GB  | 64x64   | standard | 32       | 6-10小时     |
| 12GB+   | 128x128 | large    | 16       | 12-20小时    |
| 16GB+   | 128x128 | large    | 32       | 10-15小时    |

### 推荐配置组合

#### 入门配置 (4GB GPU)
```bash
python train_imagenet.py \
    --image_size 64 \
    --model_size small \
    --batch_size 16 \
    --epochs 100
```

#### 标准配置 (8GB GPU)  
```bash
python train_imagenet.py \
    --image_size 64 \
    --model_size standard \
    --batch_size 32 \
    --epochs 200 \
    --use_ema
```

#### 高端配置 (16GB+ GPU)
```bash
python train_imagenet.py \
    --image_size 128 \
    --model_size large \
    --batch_size 32 \
    --epochs 300 \
    --use_ema \
    --sample_every_epoch
```

## 📁 输出文件结构

训练完成后会产生以下文件：

```
checkpoints_imagenet/
├── best/                    # 最佳模型
│   └── checkpoint_epoch_*.pth
├── final/                   # 最终模型  
│   └── checkpoint_epoch_*.pth
└── checkpoint_epoch_*.pth   # 定期保存的检查点

logs_imagenet/
├── events.out.tfevents.*    # Tensorboard事件文件
└── training_config.txt      # 训练配置记录

samples_imagenet/
├── epoch_001_original.png   # 每个epoch的原始模型样本
├── epoch_001_ema.png        # 每个epoch的EMA模型样本
├── epoch_002_original.png
└── ...

demo_samples_imagenet/       # 演示样本
└── final_cats.png
```

## 🎨 生成新图像

训练完成后，可以生成新的猫咪图像：

```bash
# 使用最佳模型生成图像
python generate.py \
    --checkpoint checkpoints_imagenet/best/checkpoint_epoch_*.pth \
    --num_samples 64 \
    --num_steps 100 \
    --method rk4 \
    --show_images
```

## 📈 训练进度示例

### 典型的训练曲线
- **前20 epochs**: 损失快速下降 (10.0 → 5.0)
- **20-100 epochs**: 稳定收敛 (5.0 → 2.0)  
- **100+ epochs**: 精细调优 (2.0 → 1.5)

### 生成质量提升
- **Epoch 1-10**: 随机噪声
- **Epoch 20-50**: 模糊的猫咪形状
- **Epoch 50-100**: 清晰的猫咪特征
- **Epoch 100+**: 高质量猫咪图像

## 🔧 故障排除

### 常见错误及解决方案

#### 1. CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方法:**
- 减少batch_size: `--batch_size 16`
- 使用更小模型: `--model_size small`
- 降低分辨率: `--image_size 64`

#### 2. 数据加载失败
```
FileNotFoundError: 未找到任何图片数据
```
**解决方法:**
- 检查数据目录: `ls ./data/oxford_pets/cats/`
- 重新下载数据: `python download_imagenet_cats.py --method oxford`

#### 3. 训练速度太慢
**优化方法:**
- 增加num_workers: 在代码中修改为 `num_workers=4`
- 使用更大batch_size (如果内存允许)
- 确保数据在SSD上而不是机械硬盘

#### 4. 生成质量不好
**改进方法:**
- 使用EMA: `--use_ema`
- 延长训练时间: `--epochs 300`
- 使用更大模型: `--model_size large`
- 增加ODE步数: `--num_steps 150`

## 🎯 进阶技巧

### 1. 超参数调优
- **学习率**: 从1e-4开始，如果收敛慢可以尝试2e-4
- **EMA衰减率**: 0.9999 (标准) 或 0.999 (快速更新)
- **梯度裁剪**: 默认1.0，如果训练不稳定可以降到0.5

### 2. 数据增强
代码已包含合适的数据增强：
- 随机水平翻转
- 随机旋转 (±10度)
- 颜色抖动
- ImageNet标准化

### 3. 模型架构优化
- **小模型**: 适合快速实验和资源受限的情况
- **标准模型**: 平衡质量和训练时间
- **大模型**: 最佳质量，需要更多计算资源

### 4. 训练策略
- **阶段性训练**: 先用小模型验证，再用大模型精细训练
- **检查点恢复**: 使用`--resume`从中断处继续训练
- **定期验证**: 观察生成样本判断训练进展

## 📚 相关资源

- **Flow Matching论文**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **Diffusers文档**: [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)
- **Oxford Pet数据集**: [Visual Geometry Group](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## 🎉 结语

ImageNet猫咪Flow Matching训练相比CIFAR-10有以下优势：
1. **更高分辨率**: 64x64或128x128 vs 32x32
2. **更真实数据**: 真实照片 vs 低分辨率图像  
3. **更好效果**: 更清晰、更逼真的生成结果
4. **更大挑战**: 更适合验证模型能力

通过本指南，你可以从零开始训练出高质量的猫咪图像生成模型！ 