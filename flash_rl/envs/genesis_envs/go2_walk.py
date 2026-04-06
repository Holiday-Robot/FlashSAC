import genesis as gs
import torch

from .go2_base import Go2BaseEnv

"""
https://github.com/ziyanx02/Genesis-backflip
- Added action_range in replacement to clip_actions
- Rearranged privileged obs to match the order of obs
- Privileged obs also gets its non-privileged part noised
"""


class Go2WalkEnv(Go2BaseEnv):
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
            dim=1,
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.base_pos[:, 2]
        base_height_target = self.reward_cfg["base_height_target"]
        return torch.square(base_height - base_height_target)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(
            1.0
            * (
                torch.norm(
                    self.link_contact_forces[:, self.penalized_contact_link_indices, :],
                    dim=-1,
                )
                > 0.1
            ),
            dim=1,
        )

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)  # upper limit
        return torch.sum(out_of_limits, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.link_contact_forces[:, self.feet_link_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum(
            (self.feet_air_time - 0.5) * first_contact, dim=1
        )  # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime


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
        "PD_stiffness": {"joint": 30.0},
        "PD_damping": {"joint": 1.5},
        "use_implicit_controller": False,
        # termination
        "termination_if_roll_greater_than": 0.4,
        "termination_if_pitch_greater_than": 0.4,
        "termination_if_height_lower_than": 0.0,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        # random push
        "push_interval_s": -1,
        "max_push_vel_xy": 1.0,
        # time (second)
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "command_type": "ang_vel_yaw",  # 'ang_vel_yaw' or 'heading'
        "action_scale": 0.25,
        "action_latency": 0.02,
        "action_range": 3.0,  # originally clip_actions
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
        "num_obs": 9 + 3 * env_cfg["num_dofs"],  # 45
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
        "num_priv_obs": 12 + 4 * env_cfg["num_dofs"],  # 60
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "soft_dof_pos_limit": 0.9,
        "base_height_target": 0.3,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -10.0,
            "base_height": -50.0,
            "torques": -0.0002,
            "collision": -1.0,
            "dof_vel": -0.0,
            "dof_acc": -2.5e-7,
            "feet_air_time": 1.0,
            "action_rate": -0.01,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [-1.0, 1.0],
        "lin_vel_y_range": [-1.0, 1.0],
        "ang_vel_range": [-1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_env(
    num_envs: int,
    eval_mode: bool,
) -> Go2WalkEnv:
    try:
        gs.init(logging_level="warning")
    except Exception as e:
        print(e)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env = Go2WalkEnv(
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
