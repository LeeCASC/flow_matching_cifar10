import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Optional
import os
import requests
from PIL import Image
import json


class CustomImageDataset(Dataset):
    """自定义图片数据集"""
    def __init__(self, data_dir: str, image_files: List[str], transform=None):
        self.data_dir = data_dir
        self.image_files = image_files
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # 返回0作为标签（因为都是猫）


# ImageNet中猫咪相关的类别ID和名称
CAT_CLASSES = {
    281: 'tabby, tabby cat',
    282: 'tiger cat', 
    283: 'Persian cat',
    284: 'Siamese cat, Siamese',
    285: 'Egyptian cat',
    286: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
    287: 'lynx, catamount',
    288: 'leopard, Panthera pardus',
    289: 'snow leopard, ounce, Panthera uncia',
    290: 'jaguar, panther, Panthera onca, Felis onca',
    291: 'lion, king of beasts, Panthera leo',
    292: 'tiger, Panthera tigris',
    293: 'cheetah, chetah, Acinonyx jubatus'
}

# 主要的家猫类别（推荐用于训练）
DOMESTIC_CAT_CLASSES = {
    281: 'tabby, tabby cat',
    282: 'tiger cat', 
    283: 'Persian cat',
    284: 'Siamese cat, Siamese',
    285: 'Egyptian cat'
}


def get_imagenet_dataloader(data_dir: str = './data/imagenet', 
                           batch_size: int = 32,
                           image_size: int = 64,
                           use_domestic_only: bool = True,
                           train: bool = True,
                           num_workers: int = 4) -> DataLoader:
    """
    获取ImageNet猫咪类别的数据加载器
    
    Args:
        data_dir: ImageNet数据目录
        batch_size: 批次大小
        image_size: 图片尺寸 (推荐64或128)
        use_domestic_only: 是否只使用家猫类别 (推荐True)
        train: 是否使用训练集
        num_workers: 数据加载器工作进程数
    
    Returns:
        DataLoader对象
    """
    
    # 选择要使用的类别
    target_classes = DOMESTIC_CAT_CLASSES if use_domestic_only else CAT_CLASSES
    class_names = list(target_classes.values())
    class_ids = list(target_classes.keys())
    
    print(f"加载ImageNet猫咪数据...")
    print(f"目标类别数: {len(target_classes)}")
    print(f"使用{'家猫' if use_domestic_only else '所有猫科'}类别")
    
    # 数据预处理
    if train:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet标准化
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    try:
        # 检查是否有预处理的猫咪数据目录
        cats_dir = os.path.join(data_dir, 'cats')
        if os.path.exists(cats_dir):
            print("找到预处理的猫咪数据目录")
            # 使用ImageFolder加载预处理的猫咪图片
            dataset = torchvision.datasets.ImageFolder(
                root=cats_dir,  # 直接指向cats目录
                transform=transform
            )
            subset = dataset
            print(f"加载了 {len(dataset)} 张猫咪图片")
            
        elif os.path.exists(os.path.join(data_dir, 'train')):
            # 尝试加载完整的ImageNet数据集
            print("尝试从完整ImageNet数据集中筛选猫咪...")
            dataset = torchvision.datasets.ImageNet(
                root=data_dir,
                split='train' if train else 'val',
                transform=transform
            )
            
            # 筛选猫咪类别
            indices = []
            print("筛选猫咪类别数据...")
            for i, (_, label) in enumerate(dataset):
                if label in class_ids:
                    indices.append(i)
                if len(indices) % 1000 == 0 and len(indices) > 0:
                    print(f"已找到 {len(indices)} 张猫咪图片...")
                    
            subset = Subset(dataset, indices)
            print(f"找到 {len(subset)} 张猫咪图片")
        
        else:
            # 检查是否有单独的图片文件
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.JPEG']:
                files = [f for f in os.listdir(data_dir) 
                        if f.endswith(ext) and not f.startswith('._')]  # 过滤系统隐藏文件
                image_files.extend(files)
            
            if image_files:
                print(f"找到 {len(image_files)} 张图片文件")
                # 创建自定义数据集
                subset = CustomImageDataset(data_dir, image_files, transform)
            else:
                raise FileNotFoundError("未找到任何图片数据")
        
    except Exception as e:
        print(f"无法加载数据集: {e}")
        print("请使用以下方法之一准备数据:")
        print("1. python download_imagenet_cats.py --method oxford")  
        print("2. python download_imagenet_cats.py --method demo")
        print("3. 手动下载ImageNet数据到指定目录")
        raise e
    
    # 创建数据加载器
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """反归一化ImageNet图片用于显示"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    return tensor * std + mean


def save_images_imagenet(images: torch.Tensor, filename: str, nrow: int = 8):
    """保存ImageNet格式的图片"""
    # 反归一化
    images = denormalize_imagenet(images.cpu())
    images = torch.clamp(images, 0, 1)
    
    # 保存图片
    torchvision.utils.save_image(images, filename, nrow=nrow, normalize=False)
    print(f"保存图片到: {filename}")


def show_images_imagenet(images: torch.Tensor, title: str = "", nrow: int = 8, figsize: Tuple[int, int] = (12, 6)):
    """显示ImageNet格式的图片"""
    # 反归一化
    images = denormalize_imagenet(images.cpu())
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


def get_device() -> torch.device:
    """获取可用设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    
    return device


def estimate_imagenet_cat_size():
    """估算ImageNet猫咪数据集大小"""
    print("=" * 50)
    print("ImageNet猫咪数据集大小估算")
    print("=" * 50)
    
    domestic_cats = len(DOMESTIC_CAT_CLASSES)
    all_cats = len(CAT_CLASSES)
    
    print(f"家猫类别数: {domestic_cats}")
    print(f"所有猫科类别数: {all_cats}")
    
    # 每个类别约1000张图片，每张约200KB
    domestic_size = domestic_cats * 1000 * 200 / 1024 / 1024  # MB
    all_cats_size = all_cats * 1000 * 200 / 1024 / 1024  # MB
    
    print(f"\n估算数据大小:")
    print(f"家猫数据: 约 {domestic_cats * 1000:,} 张图片, {domestic_size:.1f} MB")
    print(f"所有猫科: 约 {all_cats * 1000:,} 张图片, {all_cats_size:.1f} MB")
    
    print(f"\n推荐:")
    print(f"- 初次试验: 使用家猫数据 ({domestic_size:.1f} MB)")
    print(f"- 高质量训练: 使用所有猫科数据 ({all_cats_size:.1f} MB)")
    print("=" * 50)


# 训练工具函数
def create_checkpoint_dir(checkpoint_dir: str = 'checkpoints_imagenet') -> str:
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
    print(f"保存检查点: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer = None) -> Tuple[int, float]:
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"加载检查点: {checkpoint_path}, epoch: {epoch}, loss: {loss:.4f}")
    
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


if __name__ == "__main__":
    # 显示数据集大小估算
    estimate_imagenet_cat_size() 