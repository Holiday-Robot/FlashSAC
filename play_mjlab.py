import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import random
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from omegaconf import OmegaConf

from flash_rl.agents import create_agent


class _MjlabViewerEnv:
    """Thin wrapper that adds get_observations() to ManagerBasedRlEnv (or VideoRecorder).

    NativeMujocoViewer / ViserPlayViewer require get_observations() on the env,
    but ManagerBasedRlEnv doesn't expose it directly. We delegate to
    observation_manager.compute() — the same call RslRlVecEnvWrapper makes.
    When VideoRecorder wraps the raw env, its __getattr__ delegates attribute
    lookups to the inner ManagerBasedRlEnv, so observation_manager still works.
    The viewer accesses .unwrapped for GPU sim state (.sim, .step_dt).
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        self.num_envs: int = env.num_envs

    @property
    def device(self) -> Any:
        return self._env.device

    @property
    def cfg(self) -> Any:
        return self._env.cfg

    @property
    def unwrapped(self) -> Any:
        # VideoRecorder.unwrapped returns the inner ManagerBasedRlEnv, so
        # the native viewer can reach .sim and .step_dt in both cases.
        return self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env

    def get_observations(self) -> dict[str, torch.Tensor]:
        return self._env.observation_manager.compute()  # type: ignore[no-any-return]

    def step(self, actions: torch.Tensor) -> Any:
        return self._env.step(actions)

    def reset(self, **kwargs: Any) -> Any:
        return self._env.reset(**kwargs)

    def close(self) -> None:
        self._env.close()


class _FlashSACPolicy:
    """Bridges FlashSAC's sample_actions to mjlab's PolicyProtocol.

    Receives the obs dict from env.get_observations() and mirrors exactly
    what MjlabVectorEnv._flatten_obs does: concatenates actor+critic when
    both groups are present, otherwise uses the single group as-is.
    sample_actions then handles the asymmetric_observation flag internally
    (slicing to actor-only dim if True, using the full flat obs if False).
    """

    def __init__(self, agent: Any, device: str, actor_key: str, has_asymmetric: bool) -> None:
        self._agent = agent
        self._device = device
        self._actor_key = actor_key
        self._has_asymmetric = has_asymmetric

    def __call__(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        actor_obs = obs_dict[self._actor_key]
        flat = torch.cat([actor_obs, obs_dict["critic"]], dim=-1) if self._has_asymmetric else actor_obs
        flat_obs_np = flat.cpu().numpy()
        actions_np = self._agent.sample_actions(
            interaction_step=0,
            prev_transition={"next_observation": flat_obs_np},
            training=False,
        )
        return torch.from_numpy(actions_np).to(self._device)


def play(args: argparse.Namespace) -> None:
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=args.config_path)
    cfg = hydra.compose(config_name=args.config_name, overrides=args.overrides)
    OmegaConf.resolve(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    import mjlab.tasks  # noqa: F401  # populates the task registry
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(cfg.env.env_name)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = cfg.seed
    env_cfg.auto_reset = True

    render_mode = "rgb_array" if args.video else None
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    # Optionally wrap with VideoRecorder (records the first episode).
    env: Any = raw_env
    if args.video:
        from mjlab.utils.wrappers import VideoRecorder

        env = VideoRecorder(
            raw_env,
            video_folder=args.video,
            episode_trigger=lambda ep: ep == 0,
            disable_logger=True,
        )
        print(f"[INFO] Video will be saved to: {args.video}")

    viewer_env = _MjlabViewerEnv(env)

    # Mirror the obs layout logic from MjlabVectorEnv so the agent matches training.
    obs_groups = list(raw_env.single_observation_space.spaces.keys())
    has_asymmetric = "actor" in obs_groups and "critic" in obs_groups
    actor_key = "actor" if "actor" in obs_groups else obs_groups[0]
    actor_dim = int(raw_env.single_observation_space.spaces[actor_key].shape[0])
    if has_asymmetric:
        critic_dim = int(raw_env.single_observation_space.spaces["critic"].shape[0])
        flat_dim = actor_dim + critic_dim
    else:
        flat_dim = actor_dim
    action_dim = int(raw_env.single_action_space.shape[0])

    single_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32)
    obs_space = batch_space(single_obs_space, args.num_envs)
    single_act_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)
    act_space = batch_space(single_act_space, args.num_envs)

    env_info: dict[str, Any] = {}
    if has_asymmetric:
        env_info["actor_observation_size"] = (actor_dim,)

    agent = create_agent(
        observation_space=obs_space,
        action_space=act_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(args.checkpoint_path)

    policy = _FlashSACPolicy(agent, device=device, actor_key=actor_key, has_asymmetric=has_asymmetric)

    # Pre-step once so sensor caches (e.g. raycast _cached_frame_pos) are populated
    # before the viewer calls sync_env_to_viewer on its first tick.
    env.reset()
    with torch.no_grad():
        _obs = env.observation_manager.compute()
        _actions = policy(_obs)
        env.step(_actions)

    # Resolve viewer backend.
    viewer_type = args.viewer
    if viewer_type == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer_type = "native" if has_display else "viser"
        print(f"[INFO] Viewer auto-selected: {viewer_type}")

    if viewer_type == "none":
        # Headless step loop — useful with --video to record without an interactive viewer.
        env.reset()
        for _ in range(args.num_steps):
            obs_dict = env.observation_manager.compute()
            actions = policy(obs_dict)
            env.step(actions)
    elif viewer_type == "native":
        from mjlab.viewer import NativeMujocoViewer

        NativeMujocoViewer(viewer_env, policy).run()
    elif viewer_type == "viser":
        from mjlab.viewer import ViserPlayViewer

        ViserPlayViewer(viewer_env, policy).run()
    else:
        raise ValueError(f"Unknown viewer: {viewer_type!r}. Choose from: none, native, viser, auto")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a trained FlashSAC agent in mjlab")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to agent checkpoint directory")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default=None, help="Torch device (default: cuda:0)")
    parser.add_argument(
        "--viewer",
        type=str,
        default="auto",
        choices=["auto", "native", "viser", "none"],
        help="Viewer backend: native (MuJoCo GUI), viser (browser), auto (detect from $DISPLAY), none (headless; use with --video)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1000,
        help="Steps to run in headless mode (--viewer none)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Save video of the first episode to OUTPUT_DIR (can be combined with --viewer)",
    )
    args = parser.parse_args()
    play(args)
