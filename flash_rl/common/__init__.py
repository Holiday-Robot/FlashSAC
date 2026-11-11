from typing import Any, Union

from flash_rl.common.logger import CsvTrainerLogger, TensorboardTrainerLogger, WandbTrainerLogger  # noqa

TrainerLogger = Union[WandbTrainerLogger, TensorboardTrainerLogger, CsvTrainerLogger]


def create_logger(cfg: Any) -> TrainerLogger:
    logger_type = getattr(cfg, "logger_type", "wandb")

    logger: TrainerLogger
    if logger_type == "wandb":
        logger = WandbTrainerLogger(cfg)
    elif logger_type == "tensorboard":
        logger = TensorboardTrainerLogger(cfg)
    else:
        raise ValueError

    if getattr(cfg, "log_csv", False):
        logger = CsvTrainerLogger(cfg, logger)
    return logger


__all__ = [
    "WandbTrainerLogger",
    "TensorboardTrainerLogger",
    "CsvTrainerLogger",
    "create_logger",
]
