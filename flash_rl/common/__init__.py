from typing import Any, Union

from flash_rl.common.logger import TensorboardTrainerLogger, WandbTrainerLogger  # noqa

TrainerLogger = Union[WandbTrainerLogger, TensorboardTrainerLogger]


def create_logger(cfg: Any) -> TrainerLogger:
    logger_type = getattr(cfg, "logger_type", "wandb")

    if logger_type == "wandb":
        return WandbTrainerLogger(cfg)
    elif logger_type == "tensorboard":
        return TensorboardTrainerLogger(cfg)
    else:
        raise ValueError


__all__ = [
    "WandbTrainerLogger",
    "TensorboardTrainerLogger",
    "create_logger",
]
