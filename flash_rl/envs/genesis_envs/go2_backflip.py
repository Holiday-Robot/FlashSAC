import genesis as gs
import torch

from .go2_walk import Go2WalkEnv

"""
https://github.com/ziyanx02/Genesis-backflip
- Added action_range in replacement to clip_actions
- Rearranged privileged obs to match the order of obs
"""


def gs_inv_quat(quat):
    qw, qx, qy, qz = quat.unbind(-1)
    inv_quat = torch.stack([1.0 * qw, -qx, -qy, -qz], dim=-1)
    return inv_quat


def gs_transform_by_quat(pos, quat):
    qw, qx, qy, qz = quat.unbind(-1)

    rot_matrix = torch.stack(
        [
            1.0 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    return rotated_pos


def gs_quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def gs_quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


def gs_quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, :1] * t + xyz.cross(t, dim=-1)).view(shape)


def gs_quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat(
        (
            a[:, :1],
            -a[:, 1:],
        ),
        dim=-1,
    ).view(shape)


class Go2BackflipEnv(Go2WalkEnv):
    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset root states - position
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx, 2] = 0.32
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        # update projected gravity
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(self.global_gravity, inv_base_quat)

        # reset root states - velocity
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        base_vel = torch.concat([self.base_lin_vel[envs_idx], self.base_ang_vel[envs_idx]], dim=1)
        self.robot.set_dofs_velocity(velocity=base_vel, dofs_idx_local=[0, 1, 2, 3, 4, 5], envs_idx=envs_idx)

        self._resample_commands(envs_idx)

        # reset buffers
        self.obs_history_buf[envs_idx] = 0.0
        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        self.last_last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.feet_air_time[envs_idx] = 0.0
        self.feet_max_height[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = 1

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.max_episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0
        # send timeout info to the algorithm
        if self.env_cfg["send_timeouts"]:
            self.extras["time_outs"] = self.time_out_buf

    def compute_observations(self):
        phase = torch.pi * self.episode_length_buf[:, None] * self.dt / 2
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                self.last_actions,  # 12
                torch.sin(phase),  # 1
                torch.cos(phase),  # 1
                torch.sin(phase / 2),  # 1
                torch.cos(phase / 2),  # 1
                torch.sin(phase / 4),  # 1
                torch.cos(phase / 4),  # 1
            ],
            axis=-1,
        )

        self.obs_history_buf = torch.cat([self.obs_history_buf[:, self.num_single_obs :], self.obs_buf.detach()], dim=1)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                [
                    self.obs_buf,  # 60
                    self.base_pos[:, 2:3],  # 1
                    self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                ],
                axis=-1,
            )

    def check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length

    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        current_time = self.episode_length_buf * self.dt
        phase = (current_time - 0.5).clamp(min=0, max=0.5)
        quat_pitch = gs_quat_from_angle_axis(
            4 * phase * torch.pi, torch.tensor([0, 1, 0], device=self.device, dtype=torch.float)
        )

        desired_base_quat = gs_quat_mul(quat_pitch, self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1))
        inv_desired_base_quat = gs_inv_quat(desired_base_quat)
        desired_projected_gravity = gs_transform_by_quat(self.global_gravity, inv_desired_base_quat)

        orientation_diff = torch.sum(torch.square(self.projected_gravity - desired_projected_gravity), dim=1)

        return orientation_diff

    def _reward_ang_vel_y(self):
        current_time = self.episode_length_buf * self.dt
        ang_vel = -self.base_ang_vel[:, 1].clamp(max=7.2, min=-7.2)
        return ang_vel * torch.logical_and(current_time > 0.5, current_time < 1.0)

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_lin_vel_z(self):
        current_time = self.episode_length_buf * self.dt
        lin_vel = self.robot.get_vel()[:, 2].clamp(max=3)
        return lin_vel * torch.logical_and(current_time > 0.5, current_time < 0.75)

    def _reward_height_control(self):
        # Penalize non flat base orientation
        current_time = self.episode_length_buf * self.dt
        target_height = 0.3
        height_diff = torch.square(target_height - self.base_pos[:, 2]) * torch.logical_or(
            current_time < 0.4, current_time > 1.4
        )
        return height_diff

    def _reward_actions_symmetry(self):
        actions_diff = torch.square(self.actions[:, 0] + self.actions[:, 3])
        actions_diff += torch.square(self.actions[:, 1:3] - self.actions[:, 4:6]).sum(dim=-1)
        actions_diff += torch.square(self.actions[:, 6] + self.actions[:, 9])
        actions_diff += torch.square(self.actions[:, 7:9] - self.actions[:, 10:12]).sum(dim=-1)
        return actions_diff

    def _reward_gravity_y(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_feet_distance(self):
        self.episode_length_buf * self.dt
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = gs_quat_apply(
                gs_quat_conjugate(self.base_quat), cur_footsteps_translated[:, i, :]
            )

        stance_width = 0.3 * torch.zeros(
            [
                self.num_envs,
                1,
            ],
            device=self.device,
        )
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)

        return stance_diff

    def _reward_feet_height_before_backflip(self):
        current_time = self.episode_length_buf * self.dt
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1) - 0.02
        return foot_height.clamp(min=0).sum(dim=1) * (current_time < 0.5)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        self.episode_length_buf * self.dt
        return (
            1.0 * (torch.norm(self.link_contact_forces[:, self.penalized_contact_link_indices, :], dim=-1) > 0.1)
        ).sum(dim=1)


def get_cfgs():
    env_cfg = {
        "urdf_path": "urdf/go2/urdf/go2.urdf",
        "links_to_keep": [
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ],
        "num_actions": 12,
        "num_dofs": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "termination_contact_link_names": ["base"],
        "penalized_contact_link_names": ["base", "thigh", "calf"],
        "feet_link_names": ["foot"],
        "base_link_name": ["base"],
        # PD
        "PD_stiffness": {"joint": 70.0},
        "PD_damping": {"joint": 3.0},
        "use_implicit_controller": False,
        # termination
        "termination_if_roll_greater_than": 0.4,
        "termination_if_pitch_greater_than": 0.4,
        "termination_if_height_lower_than": 0.2,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.36],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # random push
        "push_interval_s": -1,
        "max_push_vel_xy": 1.0,
        # time (second)
        "episode_length_s": 2.0,
        "resampling_time_s": 4.0,
        "command_type": "ang_vel_yaw",  # 'ang_vel_yaw' or 'heading'
        "action_scale": 0.5,
        "action_latency": 0.02,
        "action_range": 6.0,  # originally clip_actions
        "send_timeouts": True,
        "control_freq": 50,
        "decimation": 4,
        "feet_geom_offset": 1,
        "use_terrain": False,
        # domain randomization
        "randomize_friction": True,
        "friction_range": [0.2, 1.5],
        "randomize_base_mass": True,
        "added_mass_range": [-1.0, 3.0],
        "randomize_com_displacement": True,
        "com_displacement_range": [-0.01, 0.01],
        "randomize_motor_strength": False,
        "motor_strength_range": [0.9, 1.1],
        "randomize_motor_offset": True,
        "motor_offset_range": [-0.02, 0.02],
        "randomize_kp_scale": True,
        "kp_scale_range": [0.8, 1.2],
        "randomize_kd_scale": True,
        "kd_scale_range": [0.8, 1.2],
        # coupling
        "coupling": False,
    }
    obs_cfg = {
        "num_obs": 60,
        "num_history_obs": 1,
        "obs_noise": {
            "ang_vel": 0.1,
            "gravity": 0.02,
            "dof_pos": 0.01,
            "dof_vel": 0.5,
        },
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
        "num_priv_obs": 64,
    }
    reward_cfg = {
        "soft_dof_pos_limit": 0.9,
        "reward_scales": {
            "ang_vel_y": 5.0,
            "ang_vel_z": -1.0,
            "lin_vel_z": 20.0,
            "orientation_control": -1.0,
            "feet_height_before_backflip": -30.0,
            "height_control": -10.0,
            "actions_symmetry": -0.1,
            "gravity_y": -10.0,
            "feet_distance": -1.0,
            "action_rate": -0.001,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [-0.0, 0.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.0, 0.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_env(
    num_envs: int,
    eval_mode: bool,
) -> Go2BackflipEnv:
    try:
        gs.init(logging_level="warning")
    except Exception as e:
        print(e)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env = Go2BackflipEnv(
        num_envs=num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
        eval=eval_mode,
        debug=False,
    )
    return env
