"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN explanations.

原理简述
--------
对目标类别 c，取最后一层卷积输出的特征图 A（形状 [B,C,H,W]）。将标量得分
S = y^c（logits 第 c 维）对 A 反向传播，得到 ∂S/∂A（与 A 同形状）。对每个通道 k
在 H,W 上做全局平均得到权重 α_k^c（形状 [B,C,1,1]）。Grad-CAM 热点为：

    L^c = ReLU( Σ_k α_k^c · A_k )

再双线性上采样到输入分辨率，归一化后伪彩色显示。ReLU 只保留对类别 c 有正贡献的
空间位置。

关键 tensor 形状（单张图 B=1，输入 3×96×96）
------------------------------------------------
- x:              [1, 3, 96, 96]   归一化后的输入
- logits:         [1, num_classes]
- A (activations): [1, C, H, W]   最后一层 Conv2d 的输出
- ∂S/∂A:          [1, C, H, W]   与 A 同形
- α:              [1, C, 1, 1]   对 H,W 平均后的通道权重
- L (CAM):        [1, H, W]      通道加权求和 + ReLU
- 上采样后:       [1, 96, 96]    与输入空间尺寸对齐（便于叠加）
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def find_last_conv2d(model: nn.Module) -> nn.Conv2d:
    """Return the last nn.Conv2d in module traversal order (matches typical last-conv usage)."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("No nn.Conv2d found in model.")
    return convs[-1]


def denormalize_imagenet_style(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """Invert torchvision-style Normalize; tensor shape [B,3,H,W], values clamped to [0,1]."""
    mean_t = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor * std_t + mean_t).clamp(0.0, 1.0)


class GradCAM:
    """Grad-CAM on an arbitrary Conv2d layer (default: last Conv2d in the model)."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Conv2d] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else find_last_conv2d(model)
        self._activations: Optional[torch.Tensor] = None
        self._fwd_handle = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
            out.retain_grad()
            self._activations = out

        self._fwd_handle = self.target_layer.register_forward_hook(forward_hook)

    def remove_hooks(self) -> None:
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:
            pass

    def compute_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            cam_2d:        [H_in, W_in] 归一化到 [0,1] 的 CAM（便于上色）
            heatmap_rgb:   [H_in, W_in, 3] float32 [0,1] 伪彩色热力图
            overlay_rgb:   [H_in, W_in, 3] float32 [0,1] 与原图叠加
        """
        if input_tensor.dim() != 4:
            raise ValueError("input_tensor must be [B,3,H,W]")
        self.model.zero_grad(set_to_none=True)
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

        logits = self.model(input_tensor)
        if target_class < 0 or target_class >= logits.size(1):
            raise ValueError("target_class out of range")
        score = logits[0, target_class]
        score.backward(retain_graph=False)

        if self._activations is None or self._activations.grad is None:
            raise RuntimeError("Activations or gradients missing; check target_layer hooks.")

        activations = self._activations
        grads = self._activations.grad

        # α: [1,C,1,1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        # Weighted sum over channels -> [1,H,W]
        cam = (weights * activations).sum(dim=1, keepdim=False)
        cam = F.relu(cam)

        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        cam_up = F.interpolate(cam.unsqueeze(1), size=(h_in, w_in), mode="bilinear", align_corners=False)
        cam_up = cam_up.squeeze(1)[0]

        cam_np = cam_up.detach().cpu().float().numpy()
        c_min, c_max = cam_np.min(), cam_np.max()
        if c_max - c_min > 1e-8:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        cam_2d = torch.from_numpy(cam_np).to(input_tensor.device)

        try:
            cmap = matplotlib.colormaps["jet"]
        except (AttributeError, KeyError):
            cmap = plt.get_cmap("jet")
        heatmap_rgb = cmap(cam_np)[..., :3].astype(np.float32)

        img_denorm = denormalize_imagenet_style(input_tensor[:1])[0].detach().cpu().numpy()
        img_rgb = np.transpose(img_denorm, (1, 2, 0)).astype(np.float32)

        overlay_rgb = 0.5 * img_rgb + 0.5 * heatmap_rgb
        overlay_rgb = np.clip(overlay_rgb, 0.0, 1.0)

        heatmap_t = torch.from_numpy(heatmap_rgb).to(input_tensor.device)
        overlay_t = torch.from_numpy(overlay_rgb).to(input_tensor.device)
        return cam_2d, heatmap_t, overlay_t
