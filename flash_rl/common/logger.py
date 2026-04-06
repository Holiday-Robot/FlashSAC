from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf


class WandbTrainerLogger:
    def __init__(self, cfg: Any):
        import wandb

        self._wandb = wandb
        self.cfg = cfg
        dict_cfg = OmegaConf.to_container(cfg, throw_on_missing=True)
        wandb.init(
            project=cfg.project_name,
            entity=cfg.entity_name,
            group=cfg.group_name,
            config=dict_cfg,  # type: ignore
        )
        self.media_dict: dict[str, Any] = {}
        self.reset()

    def update_metric(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if isinstance(v, (float, int)):
                self.average_meter_dict.update(k, v)
            elif isinstance(v, np.ndarray) and v.ndim == 5:
                self.media_dict[k] = self._wandb.Video(v, fps=30, format="gif")
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> None:
        log_data = {}
        log_data.update(self.average_meter_dict.averages())
        log_data.update(self.media_dict)
        self._wandb.log(log_data, step=step)

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict.clear()


class TensorboardTrainerLogger:
    def __init__(self, cfg: Any):
        from datetime import datetime

        from torch.utils.tensorboard import SummaryWriter

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        log_dir = f"runs/{cfg.group_name}/{cfg.exp_name}/{cfg.env.env_name}_seed{cfg.seed}_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)  # type: ignore[no-untyped-call]
        self.writer.add_text("config", OmegaConf.to_yaml(cfg))  # type: ignore[no-untyped-call]
        self.media_dict: dict[str, Any] = {}
        self.reset()

    def update_metric(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if isinstance(v, (float, int)):
                self.average_meter_dict.update(k, v)
            else:
                self.media_dict[k] = v

    def log_metric(self, step: int) -> None:
        for k, v in self.average_meter_dict.averages().items():
            self.writer.add_scalar(k, v, global_step=step)  # type: ignore[no-untyped-call]
        for k, v in self.media_dict.items():
            if isinstance(v, np.ndarray) and v.ndim == 5:
                # v: (B, T, C, H, W) uint8 — encode to animated gif, bypassing moviepy
                import io

                from PIL import Image
                from tensorboard.compat.proto.summary_pb2 import Summary

                frames = v[0].transpose(0, 2, 3, 1)  # (T, H, W, C)
                frames_pil = [Image.fromarray(f) for f in frames]
                buf = io.BytesIO()
                frames_pil[0].save(
                    buf,
                    format="GIF",
                    save_all=True,
                    append_images=frames_pil[1:],
                    duration=33,
                    loop=0,
                )
                _, h, w, _ = frames.shape
                image_summary = Summary.Image(encoded_image_string=buf.getvalue(), height=h, width=w)
                summary = Summary(value=[Summary.Value(tag=k, image=image_summary)])
                assert self.writer.file_writer is not None
                self.writer.file_writer.add_summary(summary, global_step=step)
        self.writer.flush()  # type: ignore[no-untyped-call]

    def reset(self) -> None:
        self.average_meter_dict = AverageMeterDict()
        self.media_dict.clear()


class AverageMeter:
    """
    Tracks and calculates the average and current values of a series of numbers.
    """

    def __init__(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        # TODO: description for using n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format: str) -> str:
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class AverageMeterDict:
    """
    Manages a collection of AverageMeter instances,
    allowing for grouped tracking and averaging of multiple metrics.
    """

    def __init__(self, meters: Optional[dict[str, AverageMeter]] = None):
        self.meters = meters if meters else {}

    def __getitem__(self, key: str) -> AverageMeter:
        if key not in self.meters:
            meter = AverageMeter()
            meter.update(0)
            return meter
        return self.meters[key]

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self) -> None:
        for meter in self.meters.values():
            meter.reset()

    def values(self, format_string: str = "{}") -> dict[str, float]:
        return {format_string.format(name): meter.val for name, meter in self.meters.items()}

    def averages(self, format_string: str = "{}") -> dict[str, float]:
        return {format_string.format(name): meter.avg for name, meter in self.meters.items()}
