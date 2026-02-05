from __future__ import annotations

from typing import Iterable, Optional


def login_hf(
    *,
    token_envs: Iterable[str] = ("HF_TOKEN", "HUGGINGFACE_TOKEN"),
    add_to_git_credential: bool = False,
    verbose: bool = True,
) -> Optional[str]:
    """
    使用环境变量中的 token 登录 Hugging Face。

    Args:
        token_envs: 依次尝试的环境变量名。
        add_to_git_credential: 是否写入 git credential。
        verbose: 是否输出提示信息。

    Returns:
        读取到的 token，若未找到则返回 None。
    """
    import os
    from huggingface_hub import login

    token = None
    for env_key in token_envs:
        value = os.getenv(env_key)
        if value:
            token = value
            break

    if not token:
        if verbose:
            print("未配置 HF_TOKEN，访问速率可能受限。")
        return None

    login(token=token, add_to_git_credential=add_to_git_credential)
    return token
