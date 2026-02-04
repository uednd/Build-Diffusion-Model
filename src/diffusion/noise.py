from __future__ import annotations

import torch


def corrupt(x: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
    """
    根据给定的 amount 值向输入张量添加噪声。

    Args:
        x: 输入张量。
        amount: 噪声比例张量。

    Returns:
        添加噪声后的张量。
    """
    amount = amount.view(-1, 1, 1, 1)
    noise = torch.rand_like(x)
    return x * (1 - amount) + noise * amount
