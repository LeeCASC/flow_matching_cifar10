#!/usr/bin/env python3
"""
传统Diffusion模型实现
基于现有Flow Matching代码改造，使用相同的UNet架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math

from diffusers import UNet2DModel


class DiffusionModel(nn.Module):
    """使用Diffusers UNet的传统Diffusion模型"""
    
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
                 num_train_timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 beta_schedule: str = "linear"):
        super().__init__()
        
        # 使用与Flow Matching相同的UNet架构
        self.noise_predictor = UNet2DModel(
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
            num_class_embeds=None,  # 无条件生成
        )
        
        # Diffusion调度参数
        self.num_train_timesteps = num_train_timesteps
        self.register_buffer("timesteps", torch.arange(0, num_train_timesteps))
        
        # 噪声调度 - 支持不同的β调度
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # DDPM论文中使用的调度
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        elif beta_schedule == "cosine":
            # 改进的余弦调度
            s = 0.008
            x = torch.linspace(0, num_train_timesteps, num_train_timesteps + 1)
            alphas_cumprod = torch.cos((x / num_train_timesteps + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        self.register_buffer("betas", betas)
        
        # 计算alpha和alpha_cumprod
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer("alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # 计算用于采样的系数
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - self.alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod - 1))
        
        # 计算DDPM采样的系数
        posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", 
                           betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", 
                           (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """预测噪声"""
        # 确保时间步在正确范围内
        if t.max() <= 1.0:  # 如果t在[0,1]范围内，转换到[0, num_train_timesteps-1]
            timesteps = (t * (self.num_train_timesteps - 1)).long()
        else:
            timesteps = t.long()
        
        return self.noise_predictor(x_t, timesteps).sample

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向扩散过程：添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start: torch.Tensor) -> torch.Tensor:
        """计算Diffusion损失（噪声预测）"""
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # 随机采样时间步
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        
        # 采样噪声
        noise = torch.randn_like(x_start)
        
        # 前向扩散：添加噪声
        x_t = self.q_sample(x_start, t, noise)
        
        # 预测噪声
        predicted_noise = self.forward(x_t, t)
        
        # 计算MSE损失
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算反向过程的均值和方差"""
        # 预测噪声
        predicted_noise = self.forward(x_t, t)
        
        # 计算预测的x_0
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod.gather(-1, t).reshape(-1, 1, 1, 1)
        
        pred_x_start = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * predicted_noise
        pred_x_start = torch.clamp(pred_x_start, -1, 1)  # 裁剪到合理范围
        
        # 计算后验均值
        posterior_mean_coef1_t = self.posterior_mean_coef1.gather(-1, t).reshape(-1, 1, 1, 1)
        posterior_mean_coef2_t = self.posterior_mean_coef2.gather(-1, t).reshape(-1, 1, 1, 1)
        
        posterior_mean = posterior_mean_coef1_t * pred_x_start + posterior_mean_coef2_t * x_t
        
        # 后验方差
        posterior_variance_t = self.posterior_variance.gather(-1, t).reshape(-1, 1, 1, 1)
        
        return posterior_mean, posterior_variance_t

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """单步反向采样"""
        mean, variance = self.p_mean_variance(x_t, t)
        
        noise = torch.randn_like(x_t)
        # t=0时不添加噪声 
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)
        
        return mean + nonzero_mask * torch.sqrt(variance) * noise

    def sample(self, shape: Tuple[int, ...], num_steps: Optional[int] = None, 
               method: str = "ddpm") -> torch.Tensor:
        """生成样本"""
        device = next(self.parameters()).device
        
        if num_steps is None:
            num_steps = self.num_train_timesteps
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        if method == "ddpm":
            # 标准DDPM采样
            timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_steps).long()
            
            for i, t_val in enumerate(timesteps):
                t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
                x = self.p_sample(x, t)
                
        elif method == "ddim":
            # DDIM采样（确定性）
            eta = 0.0  # 0为完全确定性，1为DDPM
            
            timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_steps).long()
            
            for i, t_val in enumerate(timesteps):
                t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
                
                with torch.no_grad():
                    predicted_noise = self.forward(x, t)
                
                # DDIM更新公式
                alpha_t = self.alphas_cumprod[t_val]
                alpha_t_prev = self.alphas_cumprod[timesteps[i+1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
                
                sqrt_alpha_t = torch.sqrt(alpha_t)
                sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
                sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
                sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
                
                # 预测x_0
                pred_x_start = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
                pred_x_start = torch.clamp(pred_x_start, -1, 1)
                
                # DDIM方向
                direction = sqrt_one_minus_alpha_t_prev * predicted_noise
                
                # 添加随机性（eta控制）
                if eta > 0:
                    noise = torch.randn_like(x)
                    variance = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                    direction += variance * noise
                
                x = sqrt_alpha_t_prev * pred_x_start + direction
                
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return x


# 预定义配置函数（与Flow Matching保持一致）
def create_diffusion_for_imagenet(image_size: int = 64, model_size: str = "standard") -> DiffusionModel:
    """为ImageNet创建合适的Diffusion配置"""
    
    if model_size == "small":
        return DiffusionModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(128, 256, 384),
            layers_per_block=2,
            attention_head_dim=8,
            norm_num_groups=32,
            beta_schedule="cosine"  # 使用改进的余弦调度
        )
    elif model_size == "standard":
        return DiffusionModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(128, 256, 512, 768),
            layers_per_block=2,
            attention_head_dim=16,
            norm_num_groups=32,
            beta_schedule="cosine"
        )
    elif model_size == "large":
        return DiffusionModel(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(256, 512, 768, 1024),
            layers_per_block=3,
            attention_head_dim=32,
            norm_num_groups=32,
            beta_schedule="cosine"
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def create_diffusion_for_cifar10():
    """为CIFAR-10创建合适的Diffusion配置"""
    return DiffusionModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        block_out_channels=(128, 256, 384, 512),
        layers_per_block=2,
        attention_head_dim=8,
        norm_num_groups=32,
        beta_schedule="cosine"
    )


if __name__ == "__main__":
    # 测试模型创建
    model = create_diffusion_for_cifar10()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 32, 32)
    loss = model.compute_loss(x)
    print(f"损失值: {loss.item():.4f}")
    
    # 测试采样
    samples = model.sample((4, 3, 32, 32), num_steps=50, method="ddim")
    print(f"生成样本形状: {samples.shape}")