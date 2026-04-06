import os
from typing import Any, Optional

import torch
import torch.nn as nn


class Network:
    """
    A bundle class for holding training related components (network, optimizer, lr-scheduler, etc.) in one entity.

    args:
        network: PyTorch nn.Module.
        optimizer: PyTorch optimizer (e.g., torch.optim.Adam).
        scheduler: PyTorch learning rate scheduler
        update_step: Number of update steps taken so far.
        compile_network: Whether to compile network.forward() function.
        compile_mode: Argument for torch.compile.
        use_weight_normalization: Whether to apply weight normalization.
                                  If `compile_network` is `True`, `normalize_fn` will be compiled using `compile_mode`.
        ema_source: Network to EMA. Must have the same nn.Module structure.
                    If `compile_network` is `True`, `ema_fn` will be compiled using `compile_mode`.
        ema_tau: EMA coefficient applied to source, i.e., `target = target*(1-tau) + source*tau`
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        update_step: int = 0,
        compile_network: bool = False,
        compile_mode: str = "default",
        use_weight_normalization: bool = False,
        ema_source: Optional["Network"] = None,
        ema_tau: Optional[float] = None,
    ):
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.update_step = update_step

        if compile_network:
            self.network = torch.compile(self.network, mode=compile_mode)  # type: ignore

        # Prepare weight normalization function
        self._use_weight_normalization = use_weight_normalization
        self._weight_normalize_fn = None
        if self._use_weight_normalization:
            _norm_modules = [m for m in network.modules() if hasattr(m, "normalize_parameters")]

            def _weight_normalize_fn() -> None:
                for m in _norm_modules:
                    m.normalize_parameters()  # type: ignore[operator]

            self._weight_normalize_fn = _weight_normalize_fn
            if compile_network:
                self._weight_normalize_fn = torch.compile(self._weight_normalize_fn, mode=compile_mode)

        # Prepare target EMA function
        self._ema_update_fn = None
        self._use_ema = ema_source is not None
        if self._use_ema:
            assert ema_source is not None
            assert ema_tau is not None
            _param_list: list[torch.Tensor] = list(self.network.parameters())
            _ema_param_list: list[torch.Tensor] = list(ema_source.network.parameters())

            def _ema_update_fn() -> None:
                torch._foreach_lerp_(_param_list, _ema_param_list, ema_tau)

            self._ema_update_fn = _ema_update_fn
            if compile_network:
                self._ema_update_fn = torch.compile(self._ema_update_fn, mode=compile_mode)

    @torch.no_grad()
    def normalize_parameters(self) -> None:
        assert self._weight_normalize_fn is not None
        self._weight_normalize_fn()

    @torch.no_grad()
    def ema_update_parameters(self) -> None:
        assert self._ema_update_fn is not None
        self._ema_update_fn()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.network(*args, **kwargs)

    def apply(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """
        Call a method on the wrapped nn.Module (self.network) by name.

        Example:
            net.apply("train")              -> self.network.train()
            net.apply("to", device)         -> self.network.to(device)
            net.apply("state_dict")         -> self.network.state_dict()
        """
        if not isinstance(method, str) or not method:
            raise ValueError(f"method must be a non-empty string, got: {method!r}")

        fn = getattr(self.network, method, None)
        if fn is None:
            raise AttributeError(f"{type(self.network).__name__} has no attribute '{method}'")
        if not callable(fn):
            raise TypeError(f"Attribute '{method}' of {type(self.network).__name__} is not callable")

        return fn(*args, **kwargs)

    def save(self, path: str) -> None:
        """
        Save parameters, optimizer state, scheduler state, and other metadata.
        args:
            path (str): The full file path to save the checkpoint (e.g. "checkpoints/actor.pt").
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "update_step": self.update_step,
        }
        torch.save(ckpt, path)

    def load(self, path: str, param_key: Optional[str] = None, load_optimizer: bool = True) -> None:
        """
        Load parameters, optimizer state, and other metadata from the given path.
        args:
            path (str): The full file path to the checkpoint (e.g. "checkpoints/actor.pt").
            param_key (str): If specified, only the subset of parameters is loaded.
            load_optimizer (bool): If False, only the parameters are loaded.
        """
        ckpt = torch.load(path, map_location=next(self.network.parameters()).device)

        if param_key:
            # Load only specific parameter key
            state_dict = self.network.state_dict()
            for key in state_dict.keys():
                if param_key in key:
                    if key in ckpt["network_state_dict"]:
                        state_dict[key] = ckpt["network_state_dict"][key]
            self.network.load_state_dict(state_dict)
        else:
            self.network.load_state_dict(ckpt["network_state_dict"])

        if load_optimizer:
            if self.optimizer is not None and ckpt["optimizer_state_dict"] is not None:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                self.update_step = ckpt["update_step"]
            else:
                print(
                    f"[Warning] load_optimizer=True but optimizer is None or checkpoint has no optimizer state."
                    f" Skipping optimizer load for {path}."
                )

            if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            else:
                print(
                    f"[Warning] load_optimizer=True but scheduler is None or checkpoint has no scheduler state."
                    f" Skipping scheduler load for {path}."
                )
