import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
import os


def get_cifar10_dataloader(class_idx: int, batch_size: int = 32, 
                          image_size: int = 32, train: bool = True) -> DataLoader:
    """
    获取CIFAR-10特定类别的数据加载器
    
    Args:
        class_idx: CIFAR-10类别索引 (0-9)
        batch_size: 批次大小
        image_size: 图片尺寸
        train: 是否使用训练集
    
    Returns:
        DataLoader对象
    """
    # CIFAR-10类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"Loading CIFAR-10 class: {class_names[class_idx]} (index: {class_idx})")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
    ])
    
    # 加载完整数据集
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=train, 
        download=True, 
        transform=transform
    )
    
    # 筛选特定类别
    indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
    subset = Subset(dataset, indices)
    
    print(f"Found {len(subset)} images for class {class_names[class_idx]}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader


def save_images(images: torch.Tensor, filename: str, nrow: int = 8):
    """保存图片网格"""
    # 反归一化
    images = (images + 1) / 2  # 从[-1,1]转换到[0,1]
    images = torch.clamp(images, 0, 1)
    
    # 保存图片
    torchvision.utils.save_image(images, filename, nrow=nrow, normalize=False)
    print(f"Saved images to {filename}")


def show_images(images: torch.Tensor, title: str = "", nrow: int = 8, figsize: Tuple[int, int] = (12, 6)):
    """显示图片网格"""
    # 反归一化
    images = (images + 1) / 2  # 从[-1,1]转换到[0,1]
    images = torch.clamp(images, 0, 1)
    
    # 转换为numpy
    grid = torchvision.utils.make_grid(images, nrow=nrow, normalize=False)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=figsize)
    plt.imshow(grid)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_checkpoint_dir(checkpoint_dir: str = 'checkpoints') -> str:
    """创建检查点目录"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, checkpoint_dir: str):
    """保存检查点"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer = None) -> Tuple[int, float]:
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}, epoch: {epoch}, loss: {loss:.4f}")
    
    return epoch, loss


class EMAModel:
    """指数移动平均模型"""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        
        # 初始化影子参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        """更新EMA参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
                
    def apply_shadow(self):
        """应用影子参数到模型"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
                
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])


def get_device() -> torch.device:
    """获取可用设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def print_model_info(model: torch.nn.Module):
    """打印模型信息"""
    total_params = count_parameters(model)
    print(f"Model has {total_params:,} trainable parameters")
    
    # 估算模型大小（MB）
    param_size = total_params * 4  # 假设每个参数是float32 (4 bytes)
    buffer_size = sum(p.numel() for p in model.buffers()) * 4
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    print(f"Estimated model size: {model_size_mb:.2f} MB")


def get_cifar10_class_names() -> List[str]:
    """获取CIFAR-10类别名称"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck'] 