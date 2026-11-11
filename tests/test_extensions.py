"""
CPU tests for the project extensions (SWD / LaBER sampling, adaptive actor delay, logging).

    python -m pytest tests/test_extensions.py
"""

import csv
import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")  # no compiler needed for the tests

import gymnasium as gym
import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.agents.flashSAC.actor_delay import ActorDelay
from flash_rl.agents.flashSAC.layer import EnsembleUnitBatchNorm
from flash_rl.buffers.torch_buffer import TorchUniformBuffer
from flash_rl.common.logger import CsvTrainerLogger

OBS_DIM, ACTION_DIM, NUM_ENVS = 5, 2, 8
OBS_SPACE = gym.spaces.Box(-np.inf, np.inf, (NUM_ENVS, OBS_DIM), np.float32)
ACTION_SPACE = gym.spaces.Box(-1.0, 1.0, (NUM_ENVS, ACTION_DIM), np.float32)

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))


def make_transition(step: int) -> dict[str, np.ndarray]:
    """A transition whose observation stores the interaction step it was added at."""
    rng = np.random.default_rng(step)
    return {
        "observation": np.full((NUM_ENVS, OBS_DIM), step, dtype=np.float32),
        "action": rng.uniform(-1, 1, (NUM_ENVS, ACTION_DIM)).astype(np.float32),
        "reward": rng.normal(size=NUM_ENVS).astype(np.float32),
        "terminated": rng.random(NUM_ENVS) < 0.05,
        "truncated": np.zeros(NUM_ENVS, dtype=bool),
        "next_observation": rng.normal(size=(NUM_ENVS, OBS_DIM)).astype(np.float32),
    }


def make_buffer(max_length: int, **kwargs) -> TorchUniformBuffer:
    return TorchUniformBuffer(
        observation_space=OBS_SPACE,
        action_space=ACTION_SPACE,
        n_step=1,
        gamma=0.99,
        max_length=max_length,
        min_length=1,
        sample_batch_size=64,
        device_type="cpu",
        **kwargs,
    )


def make_agent(*overrides: str):
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(
            "flashSAC_base",
            overrides=[
                "agent.device_type=cpu",
                "agent.buffer_device_type=cpu",
                "agent.use_compile=false",
                "agent.use_amp=false",
                "agent.buffer_max_length=4000",
                "agent.buffer_min_length=100",
                "agent.sample_batch_size=64",
                *overrides,
            ],
        )
    OmegaConf.resolve(cfg)
    agent = create_agent(OBS_SPACE, ACTION_SPACE, {}, cfg.agent)
    for step in range(100):
        agent.process_transition(make_transition(step))
    return agent


# max_length 100 is not a multiple of NUM_ENVS: the oldest step is partially overwritten after wrap-around
@pytest.mark.parametrize("max_length, num_steps", [(1000, 50), (96, 40), (100, 40)])
def test_buffer_age(max_length: int, num_steps: int) -> None:
    buffer = make_buffer(max_length)
    for step in range(num_steps):
        buffer.add(make_transition(step))
    batch = buffer.sample(batch_size=5000)
    added_at = batch["observation"][:, 0]
    assert torch.equal(batch["age"], (num_steps - 1) - added_at)


@pytest.mark.parametrize("max_length, num_steps", [(1000, 50), (96, 40), (100, 40)])
def test_swd_distribution(max_length: int, num_steps: int) -> None:
    torch.manual_seed(0)
    decay_steps, min_weight = 8, 0.1
    buffer = make_buffer(max_length, sampler="swd", swd_decay_steps=decay_steps, swd_min_weight=min_weight)
    for step in range(num_steps):
        buffer.add(make_transition(step))

    # expected: every stored transition weighted by max(min_weight, 1 - age / decay_steps), newest age = 1
    stored_age = (num_steps - 1) - buffer.get_observations()[:, 0] + 1
    weights = torch.clamp(1 - stored_age / decay_steps, min=min_weight)
    max_age = int(stored_age.max())
    expected = torch.zeros(max_age + 1).index_add_(0, stored_age.long(), weights)
    expected = expected / expected.sum()

    ages = buffer.sample(batch_size=400_000)["age"] + 1
    assert ages.max() <= max_age
    observed = torch.bincount(ages.long(), minlength=max_age + 1) / len(ages)
    assert torch.allclose(observed, expected, atol=3e-3)


def test_fixed_actor_delay_matches_modulo() -> None:
    for period in (1, 2, 4, 8):
        delay = ActorDelay("fixed", period, interval=10, beta=0.9, scale=2.0, min_delay=1.0, max_delay=8.0)
        for step in range(50):
            assert delay.should_update_actor(step) == (step % period == 0)
            delay.observe(1.0)
        assert delay.delay == period


def test_ema_actor_delay() -> None:
    delay = ActorDelay("ema", 2, interval=10, beta=0.9, scale=2.0, min_delay=1.0, max_delay=8.0)

    def run(td_error: float, num_steps: int) -> float:
        updates = 0
        for step in range(num_steps):
            updates += delay.should_update_actor(step)
            delay.observe(td_error)
        return updates / num_steps

    # constant error: ratio 1, so the delay stays at `scale` (FlashSAC's 2)
    assert run(1.0, 1000) == pytest.approx(0.5)
    assert delay.delay == pytest.approx(2.0)
    # error spike: the actor slows down, bounded by max_delay
    run(100.0, 10)
    assert delay.delay == 8.0
    assert run(100.0, 10) <= 0.2
    # error collapse: the actor speeds up, bounded by min_delay
    run(1e-3, 10)
    assert delay.delay == 1.0
    # the state survives a checkpoint
    restored = ActorDelay("ema", 2, interval=10, beta=0.9, scale=2.0, min_delay=1.0, max_delay=8.0)
    restored.load_state_dict(delay.state_dict())
    assert restored.state_dict() == delay.state_dict()


@pytest.mark.parametrize("sampler", ["uniform", "swd", "laber"])
@pytest.mark.parametrize("actor_delay", ["fixed", "ema"])
def test_agent_update(sampler: str, actor_delay: str) -> None:
    agent = make_agent(
        f"agent.sampler={sampler}",
        f"agent.actor_delay={actor_delay}",
        "agent.actor_delay_ema_interval=5",
        "agent.swd_decay_steps=20",
    )
    infos = [agent.update() for _ in range(30)]
    for info in infos:
        assert all(np.isfinite(v) for v in info.values()), info
    info = infos[-1]
    for key in ("critic/td_abs", "critic/surprise", "replay/age_mean", "replay/age_surprise_corr", "actor_delay/delay"):
        assert key in info
    assert ("laber/effective_sample_frac" in info) == (sampler == "laber")
    assert ("actor_delay/td_ema" in info) == (actor_delay == "ema")
    if actor_delay == "fixed":
        assert sum("actor/loss" in i for i in infos) == 15


def test_laber_selection() -> None:
    torch.manual_seed(0)
    agent = make_agent("agent.sampler=laber", "agent.laber_multiplier=8")
    candidates = agent._replay_buffer.sample(batch_size=512)
    candidates["actor_observation"] = candidates["observation"]
    candidates["actor_next_observation"] = candidates["next_observation"]
    batch, info = agent._select_laber_batch(dict(candidates))

    assert batch["observation"].shape == (64, OBS_DIM)
    assert batch["weight"].shape == (64,)
    assert 0.0 < float(info["laber/effective_sample_frac"]) <= 1.0
    assert 0.0 < info["laber/unique_frac"] <= 1.0

    # importance weights have mean 1 in expectation, i.e. the weighted critic loss is unbiased
    mean_weights = [float(agent._select_laber_batch(dict(candidates))[0]["weight"].mean()) for _ in range(300)]
    assert np.mean(mean_weights) == pytest.approx(1.0, abs=0.02)


def test_weighted_batchnorm_recovers_uniform_statistics() -> None:
    torch.manual_seed(0)
    norm = EnsembleUnitBatchNorm(num_ensemble=2, input_dim=3)
    population = torch.randn(2, 20_000, 3).exp()  # skewed, so that prioritizing large values shifts the statistics
    priorities = population[0].norm(dim=-1)
    idxs = torch.multinomial(priorities, 4096, replacement=True)
    batch, weights = population[:, idxs], priorities.mean() / priorities[idxs]

    def normalized_population_mean(sample_weights):
        norm.running_mean.zero_()
        norm.momentum = 1.0  # running_mean becomes the batch mean used for normalization
        norm(batch, training=True, sample_weights=sample_weights)
        return (norm.running_mean - population.mean(dim=1)).abs().max()

    assert normalized_population_mean(None) > 0.5  # selection bias
    assert normalized_population_mean(weights) < 0.1  # corrected
    # uniform weights reproduce the unweighted layer
    out = norm(batch, training=True)
    assert torch.allclose(norm(batch, training=True, sample_weights=torch.ones(4096)), out, atol=1e-4)


def test_buffer_save_load(tmp_path) -> None:
    buffer = make_buffer(1000, sampler="swd")
    for step in range(50):
        buffer.add(make_transition(step))
    path = str(tmp_path / "buffer" / "replay_buffer.pt")
    buffer.save(path)

    loaded = make_buffer(1000, sampler="swd")
    loaded.load(path)
    batch = loaded.sample(batch_size=2000)
    assert torch.equal(batch["age"], 49 - batch["observation"][:, 0])


def test_actor_lr_schedule() -> None:
    used = {}
    for schedule in ("actor_step", "global_step"):
        agent = make_agent(
            f"agent.actor_lr_schedule={schedule}",
            "agent.actor_update_period=4",
            "agent.learning_rate_decay_step=40",
            "agent.learning_rate_warmup_step=0",
        )
        # record the learning rate each optimizer actually steps with
        lrs: dict[str, list[float]] = {"actor": [], "critic": []}
        for name, network in (("actor", agent._actor), ("critic", agent._critic)):
            optimizer = network.optimizer

            def step(*args, _name=name, _optimizer=optimizer, _step=optimizer.step, **kwargs):
                lrs[_name].append(_optimizer.param_groups[0]["lr"])
                return _step(*args, **kwargs)

            optimizer.step = step
        for _ in range(37):
            agent.update()
        used[schedule] = lrs

    # global_step: every actor update uses exactly the critic's learning rate of that update
    assert used["global_step"]["actor"] == pytest.approx(used["global_step"]["critic"][::4], rel=1e-9)
    # actor_step (FlashSAC): the actor's schedule advances 4x slower than the critic's
    assert used["actor_step"]["actor"] == pytest.approx(used["actor_step"]["critic"][:10], rel=1e-9)


def test_csv_logger(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    class DummyLogger:
        def __init__(self) -> None:
            self.metrics: dict[str, float] = {}

        @property
        def average_meter_dict(self):
            return self

        def averages(self) -> dict[str, float]:
            return dict(self.metrics)

        def update_metric(self, **kwargs) -> None:
            self.metrics.update(kwargs)

        def log_metric(self, step: int) -> None:
            pass

        def reset(self) -> None:
            self.metrics = {}

    cfg = OmegaConf.create({"group_name": "g", "exp_name": "e", "seed": 0, "env": {"env_name": "Env"}})
    logger = CsvTrainerLogger(cfg, DummyLogger())
    for step, metrics in enumerate([{"a": 1.0}, {"a": 2.0}, {"a": 3.0, "b": 4.0}, {"b": 5.0}]):
        logger.update_metric(**metrics)
        logger.log_metric(step)
        logger.reset()

    with open(os.path.join(logger.log_dir, "metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    assert [r["env_step"] for r in rows] == ["0", "1", "2", "3"]
    assert [r["a"] for r in rows] == ["1.0", "2.0", "3.0", ""]
    assert [r["b"] for r in rows] == ["", "", "4.0", "5.0"]
    assert os.path.exists(os.path.join(logger.log_dir, "config.yaml"))
