from __future__ import annotations

from typing import Any, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import F32NDArray, NDArray


class MjlabVectorEnv(VectorEnv[F32NDArray, F32NDArray, F32NDArray]):
    """Gymnasium VectorEnv wrapping mjlab's ManagerBasedRlEnv for FlashSAC.

    Uses auto_reset=False so we can capture the true terminal observation before
    resetting. This populates infos["final_obs"] correctly for off-policy TD
    bootstrapping on truncated episodes — fixing the known limitation in the
    IsaacLab wrapper where terminal obs is unavailable.

    Observations are flattened from mjlab's dict format:
    - If both "actor" and "critic" groups exist: concatenated as [actor | critic],
      with env_info["actor_observation_size"] set so FlashSAC's agent can split them.
    - Otherwise: the single group is used as-is.

    Actions are passed through unchanged (mjlab action terms handle scaling internally).
    """

    def __init__(
        self,
        task_id: str,
        num_envs: int,
        seed: int,
        device: str = "cuda:0",
        to_numpy: bool = True,
    ) -> None:
        import mjlab.tasks  # noqa: F401  # populates the task registry via side effects
        from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        env_cfg = load_env_cfg(task_id)
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = seed
        env_cfg.auto_reset = False  # we handle resets to preserve the terminal obs

        self._env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        self._device = device
        self._to_numpy = to_numpy
        self.num_envs = num_envs

        # Determine obs layout
        obs_groups = list(self._env.single_observation_space.spaces.keys())
        self._has_asymmetric = "actor" in obs_groups and "critic" in obs_groups
        self._actor_obs_dim = int(self._env.single_observation_space.spaces["actor"].shape[0])
        if self._has_asymmetric:
            critic_dim = int(self._env.single_observation_space.spaces["critic"].shape[0])
            flat_dim = self._actor_obs_dim + critic_dim
        else:
            flat_dim = self._actor_obs_dim

        action_dim = int(self._env.single_action_space.shape[0])

        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, num_envs)

        # Expose for FlashSAC agent/env setup (mirrors IsaacLabVectorEnv)
        self.obs_size = (flat_dim,)
        self.action_size = (action_dim,)

        # Episode return/length tracking for training-time logging
        self._ep_returns = np.zeros(num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(num_envs, dtype=np.int32)

    def _flatten_obs(self, obs_dict: dict[str, Any]) -> F32NDArray:
        actor = obs_dict["actor"]
        flat = torch.cat([actor, obs_dict["critic"]], dim=-1) if self._has_asymmetric else actor
        return flat.cpu().numpy().astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[F32NDArray, dict[str, Any]]:
        obs_dict, _ = self._env.reset()
        self._ep_returns[:] = 0.0
        self._ep_lengths[:] = 0
        env_info: dict[str, Any] = {}
        if self._has_asymmetric:
            env_info["actor_observation_size"] = (self._actor_obs_dim,)
        return self._flatten_obs(obs_dict), env_info

    def step(
        self,
        actions: Union[F32NDArray, torch.Tensor],
    ) -> tuple[F32NDArray, F32NDArray, NDArray, NDArray, dict[str, Any]]:
        if isinstance(actions, np.ndarray):
            actions_t = torch.from_numpy(actions).float().to(self._device)
        else:
            actions_t = actions.to(self._device)

        obs_dict, rewards, terminateds, truncateds, extras = self._env.step(actions_t)

        rewards_np = rewards.cpu().numpy().astype(np.float32)
        self._ep_returns += rewards_np
        self._ep_lengths += 1

        # Capture terminal obs BEFORE resetting done envs
        terminal_obs = self._flatten_obs(obs_dict)

        # Reset done envs; mjlab raises RuntimeError on the next step() if we skip this.
        # reset() recomputes obs for ALL envs: done envs get fresh state, non-done envs
        # are unchanged — so the returned buf is already the correct next obs.
        dones = terminateds | truncateds
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            reset_obs_dict, _ = self._env.reset(env_ids=done_ids)
            next_obs = self._flatten_obs(reset_obs_dict)
        else:
            next_obs = terminal_obs

        infos: dict[str, Any] = {
            "final_obs": terminal_obs,  # true terminal obs; train.py uses this for done envs
        }

        # Emit episode return/length for done envs, merged with mjlab's per-reward-term extras
        done_ids_np = done_ids.cpu().numpy()
        raw_log = extras.get("log") or {}
        episode_info: dict[str, Any] = {
            k: float(v.mean().item()) if isinstance(v, torch.Tensor) else v
            for k, v in raw_log.items()
        }
        if len(done_ids_np) > 0:
            episode_info["episode_rewards"] = float(self._ep_returns[done_ids_np].mean())
            episode_info["episode_length"] = float(self._ep_lengths[done_ids_np].mean())
            self._ep_returns[done_ids_np] = 0.0
            self._ep_lengths[done_ids_np] = 0
        if episode_info:
            infos["episode_info"] = episode_info

        return (
            next_obs,
            rewards_np,
            terminateds.cpu().numpy(),
            truncateds.cpu().numpy(),
            infos,
        )

    def close(self, **kwargs: Any) -> None:
        if hasattr(self, "_env"):
            self._env.close()


    @classmethod
    def from_env(
        cls,
        env: Any,
        to_numpy: bool = True,
    ) -> "MjlabVectorEnv":
        """Wrap an already-created ManagerBasedRlEnv.

        Disables auto_reset on the env so terminal obs can be captured before
        the env resets done workers (required for correct off-policy bootstrapping).
        """
        env.cfg.auto_reset = False

        instance = cls.__new__(cls)
        instance._env = env
        instance._device = str(env.device)
        instance._to_numpy = to_numpy
        instance.num_envs = env.num_envs

        obs_groups = list(env.single_observation_space.spaces.keys())
        instance._has_asymmetric = "actor" in obs_groups and "critic" in obs_groups
        instance._actor_obs_dim = int(
            env.single_observation_space.spaces["actor"].shape[0]
        )
        if instance._has_asymmetric:
            critic_dim = int(env.single_observation_space.spaces["critic"].shape[0])
            flat_dim = instance._actor_obs_dim + critic_dim
        else:
            flat_dim = instance._actor_obs_dim

        action_dim = int(env.single_action_space.shape[0])

        instance.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        instance.observation_space = batch_space(
            instance.single_observation_space, env.num_envs
        )
        instance.single_action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32
        )
        instance.action_space = batch_space(
            instance.single_action_space, env.num_envs
        )

        instance.obs_size = (flat_dim,)
        instance.action_size = (action_dim,)
        instance._ep_returns = np.zeros(env.num_envs, dtype=np.float32)
        instance._ep_lengths = np.zeros(env.num_envs, dtype=np.int32)

        return instance


def make_mjlab_env(
    task_id: str,
    num_envs: int,
    seed: int,
    device: str = "cuda:0",
) -> MjlabVectorEnv:
    return MjlabVectorEnv(task_id=task_id, num_envs=num_envs, seed=seed, device=device)
