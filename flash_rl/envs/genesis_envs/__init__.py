from typing import Any


def get_genesis_env(
    env_name: str,
    num_envs: int,
    eval_mode: bool,
) -> Any:
    if env_name == "go2-walk":
        from .go2_walk import get_env

        env = get_env(num_envs, eval_mode)

    elif env_name == "go2-backflip":
        from .go2_backflip import get_env

        env = get_env(num_envs, eval_mode)

    elif env_name == "panda-grasp":
        from .panda_grasp import get_env

        env = get_env(num_envs)

    elif env_name == "go2-walk_easy":
        from .go2_walk_easy import get_env

        env = get_env(num_envs)

    else:
        raise ValueError

    return env
