from __future__ import annotations

import importlib.util
from typing import Iterable, Sequence, Tuple

DEFAULT_REQUIRED_PACKAGES: Sequence[Tuple[str, str]] = (
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("diffusers", "Diffusers"),
    ("matplotlib", "Matplotlib"),
)


def ensure_dependencies(
    required_packages: Iterable[Tuple[str, str]] = DEFAULT_REQUIRED_PACKAGES,
    *,
    verbose: bool = True,
) -> None:
    """
    依赖自检：缺失依赖时抛出异常。

    Args:
        required_packages: 依赖包列表。
        verbose: 是否输出检查日志。
    """
    if verbose:
        print(f"{'='*10} 执行依赖自检 {'='*10}")

    missing_packages = []
    for package_name, display_name in required_packages:
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package_name)
            continue
        module = __import__(package_name)
        version = getattr(module, "__version__", "未知版本")
        if verbose:
            print(f"{display_name}: {version}")

    if missing_packages:
        message = f"【ERROR】缺少以下依赖包: {', '.join(missing_packages)}"
        if verbose:
            print(f"\n{message}")
        raise ModuleNotFoundError(message)

    if verbose:
        print(f"{'='*10} 依赖自检完成 {'='*10}")


def select_device(torch_module=None, *, verbose: bool = True):
    """
    设备自检：优先使用 MPS / CUDA，否则回退到 CPU。

    Args:
        torch_module: torch 模块（用于注入以便测试）。
        verbose: 是否输出检查日志。

    Returns:
        可用的 torch.device。
    """
    if torch_module is None:
        import torch as torch_module

    if verbose:
        print(f"{'='*10} 执行硬件自检 {'='*10}")

    try:
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            device_name = "mps"
            if torch_module.backends.mps.is_built() and verbose:
                print("【INFO】Use Apple Silicon (MPS)")
        elif torch_module.cuda.is_available():
            device_name = "cuda"
            if verbose:
                print("【INFO】Use NVIDIA")
        else:
            device_name = "cpu"
            if verbose:
                print("【INFO】Use CPU")

        device = torch_module.device(device_name)
        _ = torch_module.ones(1, device=device)  # 触发一次张量创建以验证设备可用
        if verbose:
            print(f"{device} 可以使用")
    except Exception as exc:
        if verbose:
            print(f"【ERROR】{exc} 设备无法使用")
        device = torch_module.device("cpu")

    if verbose:
        print(f"{'='*10} 硬件自检完成 {'='*10}")

    return device
