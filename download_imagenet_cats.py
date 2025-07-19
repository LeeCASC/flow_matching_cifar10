#!/usr/bin/env python3
"""
ImageNet猫咪数据下载脚本

由于ImageNet需要注册账号，这个脚本提供了几种获取猫咪数据的方法
"""

import os
import requests
import zipfile
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import json


def download_file(url, filename, desc="下载中"):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            pbar.update(size)


def download_cats_from_kaggle():
    """从Kaggle下载猫咪数据集"""
    print("🐱 从Kaggle下载猫咪数据集...")
    print("这是一个替代方案，包含高质量的猫咪图片")
    
    os.makedirs('./data/cats_kaggle', exist_ok=True)
    
    # 这里我们使用一个公开的猫咪数据集
    dataset_info = {
        'name': '猫咪图片数据集',
        'source': 'public domain',
        'size': '约5000张图片',
        'resolution': '各种尺寸'
    }
    
    print(f"数据集信息: {dataset_info}")
    print("由于版权限制，需要手动下载猫咪图片")
    print("\n推荐的免费猫咪数据源:")
    print("1. Unsplash猫咪图片: https://unsplash.com/s/photos/cat")
    print("2. Pexels猫咪图片: https://www.pexels.com/search/cat/")
    print("3. Oxford-IIIT Pet Dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/")
    
    return False


def setup_oxford_pets():
    """设置Oxford-IIIT Pet数据集（包含猫）"""
    print("🐱 下载Oxford-IIIT Pet数据集...")
    
    data_dir = './data/oxford_pets'
    os.makedirs(data_dir, exist_ok=True)
    
    # Oxford Pet数据集的URL
    images_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    annotations_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    
    images_file = os.path.join(data_dir, "images.tar.gz")
    annotations_file = os.path.join(data_dir, "annotations.tar.gz")
    
    try:
        # 下载图片数据
        if not os.path.exists(images_file):
            print("下载图片数据...")
            download_file(images_url, images_file, "下载图片")
        
        # 下载标注数据
        if not os.path.exists(annotations_file):
            print("下载标注数据...")
            download_file(annotations_url, annotations_file, "下载标注")
        
        # 解压文件
        import tarfile
        
        print("解压图片数据...")
        with tarfile.open(images_file, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        print("解压标注数据...")
        with tarfile.open(annotations_file, 'r:gz') as tar:
            tar.extractall(data_dir)
        
        # 筛选猫咪图片
        extract_cat_images(data_dir)
        
        print("✅ Oxford Pet数据集下载完成!")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False


def extract_cat_images(data_dir):
    """从Oxford Pet数据集中提取猫咪图片"""
    images_dir = os.path.join(data_dir, 'images')
    cats_dir = os.path.join(data_dir, 'cats')
    
    # 为ImageFolder创建正确的目录结构
    cats_class_dir = os.path.join(cats_dir, 'cats')  # ImageFolder需要子目录
    os.makedirs(cats_class_dir, exist_ok=True)
    
    # 猫咪品种名称（Oxford Pet数据集中的）
    cat_breeds = [
        'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
        'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Siamese', 'Sphynx'
    ]
    
    cat_count = 0
    
    print("筛选猫咪图片...")
    
    if os.path.exists(images_dir):
        for filename in tqdm(os.listdir(images_dir)):
            # 过滤系统隐藏文件和无效文件
            if (filename.endswith(('.jpg', '.jpeg', '.png', '.JPEG')) and 
                not filename.startswith('._') and 
                not filename.startswith('.')):
                
                # 检查是否是猫咪品种
                for breed in cat_breeds:
                    if filename.startswith(breed):
                        src_path = os.path.join(images_dir, filename)
                        dst_path = os.path.join(cats_class_dir, filename)
                        
                        # 只复制有效的图片文件
                        try:
                            import shutil
                            # 检查源文件是否存在且可读
                            if os.path.isfile(src_path) and os.path.getsize(src_path) > 0:
                                shutil.copy2(src_path, dst_path)
                                cat_count += 1
                        except Exception as e:
                            print(f"跳过文件 {filename}: {e}")
                        break
    
    print(f"提取了 {cat_count} 张猫咪图片")
    
    # 创建数据集配置
    config = {
        'name': 'Oxford Cats',
        'num_images': cat_count,
        'breeds': cat_breeds,
        'source': 'Oxford-IIIT Pet Dataset',
        'data_dir': cats_dir
    }
    
    with open(os.path.join(data_dir, 'dataset_info.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    return cats_dir


def create_simple_cat_dataset():
    """创建简单的猫咪演示数据集"""
    print("🎨 创建演示猫咪数据集...")
    
    # 创建一些简单的合成猫咪图片用于演示
    data_dir = './data/demo_cats'
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成一些简单的彩色方块作为演示数据
    import numpy as np
    from PIL import Image
    
    colors = [
        (255, 165, 0),   # 橙色 - 橘猫
        (105, 105, 105), # 灰色 - 灰猫
        (222, 184, 135), # 浅棕色 - 暹罗猫
        (255, 255, 255), # 白色 - 白猫
        (0, 0, 0),       # 黑色 - 黑猫
    ]
    
    for i, color in enumerate(colors):
        for j in range(20):  # 每种颜色生成20张
            # 创建64x64的彩色图片
            img_array = np.full((64, 64, 3), color, dtype=np.uint8)
            
            # 添加一些随机噪声使其更真实
            noise = np.random.randint(-30, 30, (64, 64, 3))
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_array)
            img.save(os.path.join(data_dir, f'cat_{i}_{j:02d}.png'))
    
    print(f"✅ 创建了 {len(colors) * 20} 张演示图片")
    print(f"数据目录: {data_dir}")
    
    return data_dir


def check_imagenet_official():
    """检查官方ImageNet数据集"""
    print("📋 官方ImageNet下载说明:")
    print("=" * 50)
    print("1. 注册ImageNet账号: https://image-net.org/")
    print("2. 下载ImageNet ILSVRC2012数据集")
    print("3. 解压到 ./data/imagenet/ 目录")
    print("4. 目录结构应该是:")
    print("   ./data/imagenet/")
    print("   ├── train/")
    print("   │   ├── n01440764/  # 类别文件夹")
    print("   │   └── ...")
    print("   └── val/")
    print("       ├── ILSVRC2012_val_00000001.JPEG")
    print("       └── ...")
    print("\n猫咪相关类别ID:")
    from utils_imagenet import DOMESTIC_CAT_CLASSES
    for class_id, class_name in DOMESTIC_CAT_CLASSES.items():
        print(f"   {class_id}: {class_name}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='ImageNet猫咪数据准备')
    parser.add_argument('--method', type=str, default='oxford',
                       choices=['oxford', 'demo', 'info'],
                       help='数据获取方法')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据存储目录')
    
    args = parser.parse_args()
    
    print("🐱 ImageNet猫咪数据准备工具")
    print("=" * 50)
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    if args.method == 'oxford':
        # 下载Oxford Pet数据集
        success = setup_oxford_pets()
        if success:
            print("\n✅ 数据准备完成!")
            print("现在可以运行:")
            print("python train_imagenet.py --data_dir ./data/oxford_pets/cats")
        
    elif args.method == 'demo':
        # 创建演示数据集
        demo_dir = create_simple_cat_dataset()
        print("\n✅ 演示数据创建完成!")
        print("现在可以运行:")
        print("python train_imagenet.py --data_dir ./data/demo_cats --epochs 10")
        print("注意: 这只是演示数据，效果有限")
        
    elif args.method == 'info':
        # 显示官方ImageNet信息
        check_imagenet_official()
    
    print("\n💡 推荐使用顺序:")
    print("1. 先用演示数据测试: python download_imagenet_cats.py --method demo")
    print("2. 然后用Oxford数据训练: python download_imagenet_cats.py --method oxford")
    print("3. 最后用官方ImageNet: python download_imagenet_cats.py --method info")


if __name__ == "__main__":
    main() 