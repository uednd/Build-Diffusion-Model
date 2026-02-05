from __future__ import annotations

from typing import Optional, Tuple


def show_images(
    images,
    *,
    nrow: Optional[int] = None,
    padding: int = 2,
    normalize: bool = True,
    value_range: Tuple[float, float] = (-1.0, 1.0),
):
    """
    将一批图像张量拼接成网格并转换为 PIL Image。

    Args:
        images: 形状为 (B, C, H, W) 或 (C, H, W) 的 torch.Tensor。
        nrow: 每行显示的图像数量，默认使用批量大小。
        padding: 图像之间的间隔像素。
        normalize: 是否按 value_range 归一化到 [0, 1]。
        value_range: 归一化使用的值域，默认适配 [-1, 1]。

    Returns:
        PIL.Image.Image 对象。
    """
    import torch
    from torchvision.transforms.functional import to_pil_image
    from torchvision.utils import make_grid

    if not isinstance(images, torch.Tensor):
        raise TypeError("images 必须是 torch.Tensor")

    if images.ndim == 3:
        images = images.unsqueeze(0)

    if images.ndim != 4:
        raise ValueError("images 维度必须为 (B, C, H, W) 或 (C, H, W)")

    if nrow is None:
        nrow = images.shape[0]

    grid = make_grid(
        images.detach().float().cpu(),
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
    )

    return to_pil_image(grid)
