#!/usr/bin/env python3
"""
Conditional Flow Matching using UNet2DConditionModel (Diffusers) with CLIP text embeddings.

- Predicts velocity field v(x_t, t | text)
- Uses cross-attention with encoder_hidden_states from a text encoder (e.g., CLIPTextModel)
- Sampling supports Euler and RK4 ODE integration
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel


class FlowMatchingConditional(nn.Module):
    def __init__(self,
                 sample_size: int = 128,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_types: Tuple[str, ...] = (
                     "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                     "CrossAttnDownBlock2D", "DownBlock2D"
                 ),
                 up_block_types: Tuple[str, ...] = (
                     "UpBlock2D", "CrossAttnUpBlock2D",
                     "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"
                 ),
                 block_out_channels: Tuple[int, ...] = (128, 256, 512, 768),
                 layers_per_block: int = 2,
                 attention_head_dim: int = 8,
                 norm_num_groups: int = 32,
                 cross_attention_dim: int = 512):
        super().__init__()

        self.velocity_network = UNet2DConditionModel(
            sample_size=sample_size,
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            attention_head_dim=attention_head_dim,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # Diffusers UNet expects timesteps in [0, 1000]
        if t.max() <= 1.0:
            timesteps = (t * 1000).long()
        else:
            timesteps = t.long()
        out = self.velocity_network(x_t, timesteps, encoder_hidden_states=encoder_hidden_states)
        return out.sample

    @staticmethod
    def sample_path(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_ = t.view(-1, 1, 1, 1)
        x_t = (1.0 - t_) * x0 + t_ * x1
        target_velocity = x1 - x0
        return x_t, target_velocity

    def compute_loss(self, x1: torch.Tensor, encoder_hidden_states: torch.Tensor,
                     x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x1.shape[0]
        device = x1.device
        t = torch.rand(batch_size, device=device)
        if x0 is None:
            x0 = torch.randn_like(x1)
        x_t, target_velocity = self.sample_path(x0, x1, t)
        predicted_velocity = self.forward(x_t, t, encoder_hidden_states)
        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], encoder_hidden_states: torch.Tensor,
               num_steps: int = 50, method: str = "euler") -> torch.Tensor:
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        batch_size = shape[0]

        if method == "euler":
            for i in range(num_steps):
                t = torch.full((batch_size,), i * dt, device=device)
                v = self.forward(x, t, encoder_hidden_states)
                x = x + v * dt
        elif method == "rk4":
            for i in range(num_steps):
                t0 = i * dt
                t = torch.full((batch_size,), t0, device=device)
                k1 = self.forward(x, t, encoder_hidden_states)
                k2 = self.forward(x + 0.5 * dt * k1, torch.full((batch_size,), t0 + 0.5 * dt, device=device), encoder_hidden_states)
                k3 = self.forward(x + 0.5 * dt * k2, torch.full((batch_size,), t0 + 0.5 * dt, device=device), encoder_hidden_states)
                k4 = self.forward(x + dt * k3, torch.full((batch_size,), t0 + dt, device=device), encoder_hidden_states)
                x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown method: {method}")
        return x


def create_conditional_unet_for_coco(image_size: int = 128, model_size: str = "standard",
                                     cross_attention_dim: int = 512) -> FlowMatchingConditional:
    if model_size == "small":
        return FlowMatchingConditional(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(128, 256, 384),
            layers_per_block=2,
            attention_head_dim=8,
            norm_num_groups=32,
            cross_attention_dim=cross_attention_dim,
        )
    elif model_size == "standard":
        return FlowMatchingConditional(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(128, 256, 512, 768),
            layers_per_block=2,
            attention_head_dim=16,
            norm_num_groups=32,
            cross_attention_dim=cross_attention_dim,
        )
    elif model_size == "large":
        return FlowMatchingConditional(
            sample_size=image_size,
            in_channels=3,
            out_channels=3,
            down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels=(256, 512, 768, 1024),
            layers_per_block=3,
            attention_head_dim=32,
            norm_num_groups=32,
            cross_attention_dim=cross_attention_dim,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")
