# Flow Matching for Image Generation

这是一个基于PyTorch的Flow Matching实现，用于无条件图像生成。Flow Matching是一种新兴的生成模型技术，通过学习从噪声分布到数据分布的向量场来生成高质量图像。

## 📋 特性

- **完整的Flow Matching实现**：包含U-Net架构的向量场网络
- **两种UNet实现**：
  - 🌟 **Diffusers UNet**（推荐）：使用HuggingFace diffusers库中经过充分测试的UNet2DModel
  - **自定义UNet**：从头实现的U-Net架构，用于学习理解
- **CIFAR-10支持**：可以选择特定类别进行训练
- **多种ODE求解器**：支持Euler和RK4方法
- **指数移动平均(EMA)**：提高生成质量
- **Tensorboard日志**：实时监控训练过程
- **灵活的生成选项**：支持不同步数和批次大小
- **演示脚本**：快速理解Flow Matching原理

## 🚀 快速开始

### 1. 安装依赖

#### 方法A：自动安装GPU版本（推荐）
```bash
python install_gpu.py
```

#### 方法B：手动安装
```bash
pip install -r requirements.txt
```

**注意**: 如果要使用GPU加速，请参考 [GPU训练指南](GPU_TRAINING_GUIDE.md)

### 2. 一键启动（超级简单！）

```bash
# 自动检测GPU配置并启动训练
python quick_start.py

# 快速演示（10分钟体验）
python quick_start.py --demo

# 指定训练类别
python quick_start.py --class_idx 3  # 训练猫的图片
```

### 3. 手动启动

#### 快速演示
```bash
python demo_diffusers.py
```

这将训练一个小型模型10个epoch并生成样本图像。

### 3. 完整训练

#### 使用Diffusers UNet（推荐，更稳定）
```bash
# 训练CIFAR-10的鸟类图像（类别2）
python train_diffusers.py --class_idx 2 --epochs 100 --batch_size 64 --use_ema --model_size standard

# 快速测试小模型
python train_diffusers.py --class_idx 2 --epochs 50 --model_size small

# 高质量大模型
python train_diffusers.py --class_idx 2 --epochs 200 --model_size large --use_ema
```

#### 使用自定义UNet
```bash
# 训练CIFAR-10的鸟类图像（类别2）
python train.py --class_idx 2 --epochs 100 --batch_size 64 --use_ema

# 训练其他类别，例如猫（类别3）
python train.py --class_idx 3 --epochs 100 --batch_size 64 --use_ema
```

### 4. 生成图像

使用训练好的模型生成图像：

```bash
python generate.py --checkpoint checkpoints_diffusers/best/checkpoint_epoch_*.pth --num_samples 64 --show_images
```

## 📊 CIFAR-10 类别

| 索引 | 类别名称 | 索引 | 类别名称 |
|------|----------|------|----------|
| 0    | airplane | 5    | dog      |
| 1    | automobile | 6  | frog     |
| 2    | bird     | 7    | horse    |
| 3    | cat      | 8    | ship     |
| 4    | deer     | 9    | truck    |

## 🛠️ 详细使用说明

### 训练参数

```bash
python train.py --help
```

主要参数：
- `--class_idx`: CIFAR-10类别索引 (0-9)
- `--batch_size`: 批次大小 (默认: 64)
- `--epochs`: 训练轮数 (默认: 100)
- `--lr`: 学习率 (默认: 1e-4)
- `--use_ema`: 使用指数移动平均
- `--channels`: U-Net通道数 (默认: [64, 128, 256, 512])

### 生成参数

```bash
python generate.py --help
```

主要参数：
- `--checkpoint`: 模型检查点路径
- `--num_samples`: 生成样本数量 (默认: 64)
- `--num_steps`: ODE求解步数 (默认: 100)
- `--method`: 求解方法 (euler/rk4)
- `--show_images`: 显示生成的图像

## 🔬 Flow Matching 原理

Flow Matching是一种基于连续归一化流的生成模型，其核心思想是：

1. **定义路径**: 从噪声分布 x₀ 到数据分布 x₁ 的插值路径
2. **学习向量场**: 训练网络预测路径上每个点的切向量
3. **ODE生成**: 使用数值求解器从噪声生成数据

### 数学公式

插值路径：`x_t = (1-t) * x₀ + t * x₁`

目标向量场：`v_t = x₁ - x₀`

损失函数：`L = E[||v_θ(x_t, t) - (x₁ - x₀)||²]`

生成过程：`dx/dt = v_θ(x, t)`，从 t=0 到 t=1

## 📈 训练监控

使用Tensorboard实时监控训练过程：

```bash
# 启动tensorboard
tensorboard --logdir logs_diffusers

# 在浏览器中打开 http://localhost:6006
```

### 🔍 监控内容包括：

#### 训练指标
- **Training/Batch_Loss**: 每个batch的损失值
- **Training/Epoch_Loss**: 每个epoch的平均损失  
- **Training/Learning_Rate**: 学习率变化曲线
- **Training/Gradient_Norm**: 梯度范数（防止梯度爆炸）
- **Training/GPU_Memory_Allocated**: GPU内存使用情况

#### 生成样本可视化
- **Quick_Samples**: 每2-3个epoch的快速预览样本
- **High_Quality_Samples**: 高质量生成样本展示
- **Original vs EMA**: 对比原始模型和EMA模型的生成效果

### 📊 训练进度实时跟踪
- 损失函数收敛曲线
- 每个epoch生成的图像质量变化
- 模型训练稳定性指标

## 🎯 项目结构

```
flow_matching/
├── quick_start.py            # 🚀 一键启动脚本（推荐入口）
├── flow_matching_diffusers.py # Flow Matching核心模型
├── utils.py                  # 工具函数和数据加载器
├── train_diffusers.py        # GPU训练脚本
├── generate.py               # 图像生成脚本
├── demo_diffusers.py         # 快速演示脚本
├── install_gpu.py            # GPU环境自动安装脚本
├── requirements.txt          # 依赖包列表
├── README.md                # 项目说明
└── GPU_TRAINING_GUIDE.md    # GPU训练详细指南
```

## ⚡ 为什么推荐使用Diffusers UNet？

使用HuggingFace diffusers库中的UNet2DModel有以下优势：

1. **稳定性更好**：经过大量测试和优化，避免了自定义实现中的各种维度不匹配问题
2. **性能更优**：使用了更高效的注意力机制和归一化技术
3. **架构更合理**：专业设计的skip connections和残差结构
4. **配置灵活**：支持多种不同大小的模型配置
5. **社区支持**：作为开源社区的标准实现，有持续的维护和更新

对比自定义UNet：
- **Diffusers UNet**: 生产级质量，稳定可靠 🌟
- **自定义UNet**: 教学目的，帮助理解架构细节

## 💡 使用技巧

### 训练技巧
- 使用EMA可以显著提高生成质量
- 梯度裁剪有助于训练稳定性
- 余弦学习率调度通常效果更好
- 较大的批次大小有助于稳定训练

### 生成技巧
- 更多ODE步数通常产生更好的结果，但速度更慢
- RK4方法比Euler方法更准确但计算成本更高
- EMA模型通常生成质量更好

### 性能优化
- 使用GPU加速训练和生成
- 调整批次大小以充分利用GPU内存
- 使用较小的模型进行快速原型设计

## 📝 示例输出

训练过程中会生成以下文件：

```
checkpoints/           # 模型检查点
├── best/             # 最佳模型
├── final/            # 最终模型
└── checkpoint_epoch_*.pth

logs/                 # Tensorboard日志
samples/              # 训练期间生成的样本
generated_images/     # 最终生成的图像
```

## 🔧 自定义扩展

### 修改模型架构

编辑 `flow_matching.py` 中的 UNet 类：

```python
# 使用不同的通道配置
model = UNet(channels=(32, 64, 128, 256))

# 调整时间嵌入维度
model = UNet(time_emb_dim=256)
```

### 使用其他数据集

修改 `utils.py` 中的数据加载函数，或创建新的数据加载器。

### 实验不同的路径

Flow Matching支持不同的插值路径，可以在 `flow_matching.py` 的 `sample_path` 方法中修改。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

MIT License

## 📚 参考文献

1. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
2. [Conditional Flow Matching](https://arxiv.org/abs/2302.00482)
3. [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)

---

**注意**: 这是一个教学和研究用途的实现。对于生产环境，可能需要进一步的优化和调试。 