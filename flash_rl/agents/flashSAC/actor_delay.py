from typing import Any, Optional


class ActorDelay:
    """
    Decides, at every critic update, whether the actor is updated too.
    `delay` is the number of critic updates per actor update.

    fixed: delay = period (FlashSAC uses 2).
    ema:   every `interval` critic updates, with e = mean TD error over those updates,
               ema   = beta * ema + (1 - beta) * e
               delay = clip(scale * e / ema, min_delay, max_delay)
           The actor is updated less often while the critic error is above its recent average.
    """

    def __init__(
        self,
        mode: str,
        period: int,
        interval: int,
        beta: float,
        scale: float,
        min_delay: float,
        max_delay: float,
    ):
        assert mode in ("fixed", "ema"), f"Unknown actor delay: {mode}"
        assert interval >= 1 and 0.0 <= beta < 1.0 and 0.0 < min_delay <= max_delay
        self._mode = mode
        self._interval = interval
        self._beta = beta
        self._scale = scale
        self._min_delay = min_delay
        self._max_delay = max_delay

        # until the first interval ends, the error ratio is taken to be 1
        self.delay = float(period) if mode == "fixed" else self._clip(scale)
        self.ratio = 1.0
        self.ema: Optional[float] = None
        self._window_sum = 0.0
        self._window_count = 0
        # `delay` is real-valued: each critic update earns 1/delay credit, an actor update costs 1
        self._credit = 1.0

    def _clip(self, delay: float) -> float:
        return min(max(delay, self._min_delay), self._max_delay)

    def should_update_actor(self, update_step: int) -> bool:
        if self._mode == "fixed":
            return update_step % int(self.delay) == 0
        do_update = self._credit >= 1.0 - 1e-6
        if do_update:
            self._credit -= 1.0
        self._credit += 1.0 / self.delay
        return do_update

    def observe(self, td_error: float) -> None:
        """Record the TD error of one critic update."""
        if self._mode == "fixed":
            return
        self._window_sum += td_error
        self._window_count += 1
        if self._window_count < self._interval:
            return
        error = self._window_sum / self._window_count
        self._window_sum, self._window_count = 0.0, 0
        self.ema = error if self.ema is None else self._beta * self.ema + (1 - self._beta) * error
        self.ratio = error / (self.ema + 1e-8)
        self.delay = self._clip(self._scale * self.ratio)

    def state_dict(self) -> dict[str, Any]:
        keys = ("delay", "ratio", "ema", "_window_sum", "_window_count", "_credit")
        return {key: getattr(self, key) for key in keys}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key, value in state.items():
            setattr(self, key, value)
