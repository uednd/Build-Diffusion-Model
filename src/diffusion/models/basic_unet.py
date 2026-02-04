from __future__ import annotations

from torch import nn


class BasicUnet(nn.Module):
    """
    一个简单的 UNet 网络实现。
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        """
        初始化 UNet 网络结构。

        Args:
            in_channels: 输入通道数。
            out_channels: 输出通道数。
        """
        super().__init__()  # 初始化

        # 下采样路径，包含三个卷积层
        self.down_layers = nn.ModuleList(
            [
                nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),  # 由输入通道数生成32个特征图，卷积核大小为5x5，填充为2以保持尺寸
                nn.Conv2d(32, 64, kernel_size=5, padding=2),           # 由32个特征图生成64个特征图
                nn.Conv2d(64, 64, kernel_size=5, padding=2),           # 不继续上升通道，防止过拟合
            ]
        )

        # 上采样路径，包含三个转置卷积层
        self.up_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(64, 64, kernel_size=5, padding=2),
                nn.ConvTranspose2d(64, 32, kernel_size=5, padding=2),
                nn.ConvTranspose2d(32, out_channels, kernel_size=5, padding=2),
            ]
        )
        self.act = nn.SiLU()  # 激活函数
        self.downscale = nn.MaxPool2d(2)  # 下采样使用最大池化法，窗口大小为2x2
        self.upscale = nn.Upsample(scale_factor=2)  # 上采样使用插值法

    def forward(self, x):
        """
        前向传播，输出预测噪声。

        Args:
            x: 输入张量。

        Returns:
            预测噪声张量。
        """
        h = []  # 创建一个空列表，保存下采样前的数据供上采样参考，以免上采样时丢失信息
        for i, layer in enumerate(self.down_layers):
            x = self.act(layer(x))  # 卷积 -> 激活
            if i < 2:
                h.append(x)  # 把当前特征图存入 h 列表
                x = self.downscale(x)  # 池化
        for i, layer in enumerate(self.up_layers):
            if i > 0:
                x = self.upscale(x)  # 插值
                x = x + h.pop()  # 与对应的下采样特征图相加（跳跃连接）
            x = self.act(layer(x))  # 转置卷积 -> 激活
        return x  # 返回预测的噪声
