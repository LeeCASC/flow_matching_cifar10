import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math

from diffusers import UNet2DModel


class FlowMatching(nn.Module):
    """使用Diffusers UNet的Flow Matching模型"""
    def __init__(self, 
                 sample_size: int = 32,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_types: Tuple[str, ...] = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
                 up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                 block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
                 layers_per_block: int = 2,
                 attention_head_dim: int = 8,
                 norm_num_groups: int = 32,
                 class_embed_type: Optional[str] = None,
                 sigma_min: float = 0.0):
        super().__init__()
        
        self.velocity_network = UNet2DModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            class_embed_type=class_embed_type,
            # 禁用class conditioning，我们只做无条件生成
            num_class_embeds=None,
        )
        
        self.sigma_min = sigma_min
        
    def forward(self, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """预测向量场"""
        # diffusers的UNet期望时间步在[0, 1000]范围内
        # 我们的t在[0, 1]范围内，需要缩放
        timesteps = (t * 1000).long()
        return self.velocity_network(x1, timesteps).sample
    
    def sample_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样路径上的点和对应的向量场"""
        # 线性插值路径: x_t = (1-t) * x0 + t * x1
        t = t.view(-1, 1, 1, 1)
        x_t = (1 - t) * x0 + t * x1
        
        # 目标向量场: v_t = x1 - x0
        target_velocity = x1 - x0
        
        return x_t, target_velocity
    
    def compute_loss(self, x1: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算Flow Matching损失"""
        batch_size = x1.shape[0]
        device = x1.device
        
        # 采样时间
        t = torch.rand(batch_size, device=device)
        
        # 如果没有提供x0，从标准高斯分布采样
        if x0 is None:
            x0 = torch.randn_like(x1)
            
        # 采样路径上的点和目标向量场
        x_t, target_velocity = self.sample_path(x0, x1, t)
        
        # 预测向量场
        predicted_velocity = self.forward(x_t, t)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return loss
    
    def sample(self, shape: Tuple[int, ...], num_steps: int = 100, 
               method: str = "euler") -> torch.Tensor:
        """使用ODE求解器生成样本"""
        device = next(self.parameters()).device
        
        # 从标准高斯噪声开始
        x = torch.randn(shape, device=device)
        
        dt = 1.0 / num_steps
        
        if method == "euler":
            # 欧拉方法
            for i in range(num_steps):
                t = torch.full((shape[0],), i * dt, device=device)
                with torch.no_grad():
                    v = self.forward(x, t)
                x = x + v * dt
        elif method == "rk4":
            # 四阶龙格-库塔方法
            for i in range(num_steps):
                t_current = i * dt
                t = torch.full((shape[0],), t_current, device=device)
                with torch.no_grad():
                    k1 = self.forward(x, t)
                    k2 = self.forward(x + 0.5 * dt * k1, 
                                    torch.full((shape[0],), t_current + 0.5 * dt, device=device))
                    k3 = self.forward(x + 0.5 * dt * k2, 
                                    torch.full((shape[0],), t_current + 0.5 * dt, device=device))
                    k4 = self.forward(x + dt * k3, 
                                    torch.full((shape[0],), t_current + dt, device=device))
                    x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return torch.clamp(x, -1, 1)


def create_unet_for_cifar10():
    """为CIFAR-10创建合适的UNet配置"""
    return FlowMatching(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(128, 256, 384, 512),
        layers_per_block=2,
        attention_head_dim=8,
        norm_num_groups=32,
    )


def create_small_unet_for_cifar10():
    """为快速实验创建小型UNet配置"""
    return FlowMatching(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(64, 128, 256),
        layers_per_block=2,
        attention_head_dim=8,
        norm_num_groups=16,  # 更小的组数适合更少的通道
    )


def create_large_unet_for_cifar10():
    """为高质量生成创建大型UNet配置"""
    return FlowMatching(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(256, 512, 768, 1024),
        layers_per_block=3,  # 更深的层
        attention_head_dim=16,  # 更大的注意力头
        norm_num_groups=32,
    ) 