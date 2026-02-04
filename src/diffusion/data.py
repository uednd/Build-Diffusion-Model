from __future__ import annotations


def create_mnist_dataset(
    *,
    root: str,
    train: bool = True,
    download: bool = True,
    transform=None,
):
    """
    创建 MNIST 数据集，默认使用 ToTensor 变换。

    Args:
        root: 数据集路径。
        train: 使用训练集，False为测试集。
        download: 若数据集不存在，下载数据集。
        transform: 数据变换。
    """
    import torchvision

    if transform is None:
        transform = torchvision.transforms.ToTensor()

    return torchvision.datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )


def create_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    persistent_workers: bool = False,
):
    """
    创建数据加载器。

    Args:
        dataset: 要加载的数据对象。
        batch_size: 每次迭代加载的样本数量。
        shuffle: 打乱数据顺序。
        num_workers: 并行处理数量。
        persistent_workers: 是否保持工作进程。
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
