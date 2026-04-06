import genesis as gs
import numpy as np
import torch
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

"""
https://github.com/ziyanx02/Genesis-backflip
"""


def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


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


def gs_quat2euler(quat):  # xyz
    # Extract quaternion components
    qw, qx, qy, qz = quat.unbind(-1)

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(torch.pi / 2),
        torch.asin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=-1)


def gs_euler2quat(xyz):  # xyz
    roll, pitch, yaw = xyz.unbind(-1)

    cosr = (roll * 0.5).cos()
    sinr = (roll * 0.5).sin()
    cosp = (pitch * 0.5).cos()
    sinp = (pitch * 0.5).sin()
    cosy = (yaw * 0.5).cos()
    siny = (yaw * 0.5).sin()

    qw = cosr * cosp * cosy + sinr * sinp * siny
    qx = sinr * cosp * cosy - cosr * sinp * siny
    qy = cosr * sinp * cosy + sinr * cosp * siny
    qz = cosr * cosp * siny - sinr * sinp * cosy

    return torch.stack([qw, qx, qy, qz], dim=-1)


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


def gs_quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return gs_quat_apply(quat_yaw, vec)


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


class Go2BaseEnv:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer,
        eval,
        debug,
        device="cuda",
    ) -> None:
        self.num_envs = 1 if num_envs == 0 else num_envs
        self.num_build_envs = num_envs
        self.num_single_obs = obs_cfg["num_obs"]
        assert obs_cfg["num_history_obs"] == 1, "num_history_obs is now fixed to 1."
        self.num_obs = self.num_single_obs * obs_cfg["num_history_obs"]
        self.num_privileged_obs = obs_cfg["num_priv_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.headless = not show_viewer
        self.eval = eval
        self.debug = debug

        self.dt = 1 / env_cfg["control_freq"]
        if env_cfg["use_implicit_controller"]:
            sim_dt = self.dt
            sim_substeps = env_cfg["decimation"]
        else:
            sim_dt = self.dt / env_cfg["decimation"]
            sim_substeps = 1
        self.max_episode_length_s = env_cfg["episode_length_s"]
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.obs_cfg = obs_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_cfg = reward_cfg
        self.reward_scales = reward_cfg["reward_scales"]
        self.env_cfg = env_cfg
        self.command_cfg = command_cfg

        self.command_type = env_cfg["command_type"]
        assert self.command_type in ["heading", "ang_vel_yaw"]

        self.action_latency = env_cfg["action_latency"]
        assert self.action_latency in [0, 0.02]

        self.num_dof = env_cfg["num_dofs"]
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            assert device in ["cpu", "cuda"]
            self.device = torch.device(device)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=sim_dt,
                substeps=sim_substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(1 / self.dt * self.env_cfg["decimation"]),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                n_rendered_envs=1,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=sim_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_self_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )

        for solver in self.scene.sim.solvers:
            if not isinstance(solver, RigidSolver):
                continue
            self.rigid_solver = solver
            break

        # add entities
        if self.env_cfg["use_terrain"]:
            self.terrain_cfg = self.env_cfg["terrain_cfg"]
            self.terrain = self.scene.add_entity(
                gs.morphs.Terrain(
                    n_subterrains=self.terrain_cfg["n_subterrains"],
                    horizontal_scale=self.terrain_cfg["horizontal_scale"],
                    vertical_scale=self.terrain_cfg["vertical_scale"],
                    subterrain_size=self.terrain_cfg["subterrain_size"],
                    subterrain_types=self.terrain_cfg["subterrain_types"],
                ),
            )
            terrain_margin_x = self.terrain_cfg["n_subterrains"][0] * self.terrain_cfg["subterrain_size"][0]
            terrain_margin_y = self.terrain_cfg["n_subterrains"][1] * self.terrain_cfg["subterrain_size"][1]
            self.terrain_margin = torch.tensor(
                [terrain_margin_x, terrain_margin_y], device=self.device, dtype=gs.tc_float
            )
            height_field = self.terrain.geoms[0].metadata["height_field"]
            self.height_field = (
                torch.tensor(height_field, device=self.device, dtype=gs.tc_float) * self.terrain_cfg["vertical_scale"]
            )
        else:
            self.scene.add_entity(
                gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
            )
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.env_cfg["urdf_path"],
                merge_fixed_links=True,
                links_to_keep=self.env_cfg["links_to_keep"],
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
            visualize_contact=self.debug,
        )

        if gs.platform != "macOS":
            self._set_camera()

        # build
        self.scene.build(n_envs=num_envs)

        self._init_buffers()
        self._prepare_reward_function()

        # domain randomization
        self._randomize_controls()
        self._randomize_rigids()

    def _prepare_reward_function(self):
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
            for name in self.reward_scales.keys()
        }

    def _init_buffers(self):
        self.base_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor(np.array([0.0, 0.0, -1.0]), device=self.device, dtype=gs.tc_float)
        self.forward_vec = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.forward_vec[:, 0] = 1.0

        self.obs_buf = torch.zeros((self.num_envs, self.num_single_obs), device=self.device, dtype=gs.tc_float)
        self.final_obs_history_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.obs_history_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.obs_noise = torch.zeros((self.num_envs, self.num_single_obs), device=self.device, dtype=gs.tc_float)
        self._prepare_obs_noise()
        self.privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros(
                (self.num_envs, self.num_privileged_obs),
                device=self.device,
                dtype=gs.tc_float,
            )
        )
        self.final_privileged_obs_buf = (
            None
            if self.num_privileged_obs is None
            else torch.zeros(
                (self.num_envs, self.num_privileged_obs),
                device=self.device,
                dtype=gs.tc_float,
            )
        )
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.rew_buf_pos = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.rew_buf_neg = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.time_out_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        # commands
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.stand_still = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        def find_link_indices(names):
            link_indices = list()
            for link in self.robot.links:
                flag = False
                for name in names:
                    if name in link.name:
                        flag = True
                if flag:
                    link_indices.append(link.idx - self.robot.link_start)
            return link_indices

        self.termination_contact_link_indices = find_link_indices(self.env_cfg["termination_contact_link_names"])
        self.penalized_contact_link_indices = find_link_indices(self.env_cfg["penalized_contact_link_names"])
        self.feet_link_indices = find_link_indices(self.env_cfg["feet_link_names"])
        assert len(self.termination_contact_link_indices) > 0
        assert len(self.penalized_contact_link_indices) > 0
        assert len(self.feet_link_indices) > 0
        self.feet_link_indices_world_frame = [i + 1 for i in self.feet_link_indices]

        # actions
        self.actions = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.last_last_actions = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.last_dof_vel = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=gs.tc_float)
        self.root_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_root_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.link_contact_forces = torch.zeros(
            (self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float
        )

        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.feet_max_height = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)),
            device=self.device,
            dtype=gs.tc_float,
        )

        self.last_contacts = torch.zeros(
            (self.num_envs, len(self.feet_link_indices)),
            device=self.device,
            dtype=gs.tc_int,
        )

        # extras
        self.continuous_push = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.env_identities = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=gs.tc_int,
        )
        self.common_step_counter = 0
        self.extras = {}

        self.terrain_heights = torch.zeros(
            (self.num_envs,),
            device=self.device,
            dtype=gs.tc_float,
        )

        # PD control
        stiffness = self.env_cfg["PD_stiffness"]
        damping = self.env_cfg["PD_damping"]

        self.p_gains, self.d_gains = [], []
        for dof_name in self.env_cfg["dof_names"]:
            for key in stiffness.keys():
                if key in dof_name:
                    self.p_gains.append(stiffness[key])
                    self.d_gains.append(damping[key])
        self.p_gains = torch.tensor(self.p_gains, device=self.device)
        self.d_gains = torch.tensor(self.d_gains, device=self.device)
        self.batched_p_gains = self.p_gains[None, :].repeat(self.num_envs, 1)
        self.batched_d_gains = self.d_gains[None, :].repeat(self.num_envs, 1)

        self.robot.set_dofs_kp(self.p_gains, self.motor_dofs)
        self.robot.set_dofs_kv(self.d_gains, self.motor_dofs)

        default_joint_angles = self.env_cfg["default_joint_angles"]
        self.default_dof_pos = torch.tensor(
            [default_joint_angles[name] for name in self.env_cfg["dof_names"]],
            device=self.device,
        )

        self.dof_pos_limits = torch.stack(self.robot.get_dofs_limit(self.motor_dofs), dim=1)
        self.torque_limits = self.robot.get_dofs_force_range(self.motor_dofs)[1]
        for i in range(self.dof_pos_limits.shape[0]):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.reward_cfg["soft_dof_pos_limit"]
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.reward_cfg["soft_dof_pos_limit"]

        self.motor_strengths = gs.ones((self.num_envs, self.num_dof), dtype=float)
        self.motor_offsets = gs.zeros((self.num_envs, self.num_dof), dtype=float)

        # gait control
        self.foot_positions = torch.ones(
            self.num_envs,
            len(self.feet_link_indices),
            3,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.foot_quaternions = torch.ones(
            self.num_envs,
            len(self.feet_link_indices),
            4,
            device=self.device,
            dtype=gs.tc_float,
        )
        self.foot_velocities = torch.ones(
            self.num_envs,
            len(self.feet_link_indices),
            3,
            device=self.device,
            dtype=gs.tc_float,
        )

        self.base_link_index = 1

        self.com = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
            dtype=gs.tc_float,
        )

    def _update_buffers(self):
        # update buffers
        # [:] is for non-parallelized scene
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        base_quat_rel = gs_quat_mul(
            self.base_quat, gs_inv_quat(self.base_init_quat.reshape(1, -1).repeat(self.num_envs, 1))
        )
        self.base_euler = gs_quat2euler(base_quat_rel)

        inv_quat_yaw = gs_quat_from_angle_axis(
            -self.base_euler[:, 2], torch.tensor([0, 0, 1], device=self.device, dtype=torch.float)
        )

        inv_base_quat = gs_inv_quat(self.base_quat)
        self.base_lin_vel[:] = gs_transform_by_quat(self.robot.get_vel(), inv_quat_yaw)
        self.base_ang_vel[:] = gs_transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = gs_transform_by_quat(self.global_gravity, inv_base_quat)

        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.link_contact_forces[:] = torch.tensor(
            self.robot.get_links_net_contact_force(),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.com[:] = self.rigid_solver.get_links_root_COM(
            [
                self.base_link_index,
            ]
        ).squeeze(dim=1)

        self.foot_positions[:] = self.rigid_solver.get_links_pos(self.feet_link_indices_world_frame)
        self.foot_quaternions[:] = self.rigid_solver.get_links_quat(self.feet_link_indices_world_frame)
        self.foot_velocities[:] = self.rigid_solver.get_links_vel(self.feet_link_indices_world_frame)

        if self.env_cfg["use_terrain"]:
            clipped_base_pos = self.base_pos[:, :2].clamp(
                min=torch.zeros(2, device=self.device), max=self.terrain_margin
            )
            height_field_ids = (clipped_base_pos / self.terrain_cfg["horizontal_scale"] - 0.5).floor().int()
            height_field_ids.clamp(min=0)
            # print(self.height_field[height_field_ids[:, 0], height_field_ids[:, 1]])
            self.terrain_heights = self.height_field[height_field_ids[:, 0], height_field_ids[:, 1]]

    def _compute_torques(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.env_cfg["action_scale"]
        torques = (
            self.batched_p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets)
            - self.batched_d_gains * self.dof_vel
        )
        return torques * self.motor_strengths

    def _compute_target_dof_pos(self, actions):
        # control_type = 'P'
        actions_scaled = actions * self.env_cfg["action_scale"]
        target_dof_pos = actions_scaled + self.default_dof_pos

        return target_dof_pos

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(
                self.link_contact_forces[:, self.termination_contact_link_indices, :],
                dim=-1,
            )
            > 1.0,
            dim=1,
        )
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= torch.logical_or(
            torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"],
            torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"],
        )
        if self.env_cfg["use_terrain"]:
            self.reset_buf |= torch.logical_or(
                self.base_pos[:, 0] > self.terrain_margin[0],
                self.base_pos[:, 1] > self.terrain_margin[1],
            )
            self.reset_buf |= torch.logical_or(
                self.base_pos[:, 0] < 1,
                self.base_pos[:, 1] < 1,
            )
        self.reset_buf |= self.base_pos[:, 2] < self.env_cfg["termination_if_height_lower_than"]
        self.reset_buf |= self.time_out_buf

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def get_observations(self):
        extras = {}
        return self.obs_history_buf, extras

    def get_privileged_observations(self):
        extras = {}
        return self.privileged_obs_buf, extras

    def post_physics_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1

        self._update_buffers()

        resampling_time_s = self.env_cfg["resampling_time_s"]
        envs_idx = (self.episode_length_buf % int(resampling_time_s / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(envs_idx)
        self._randomize_rigids(envs_idx)
        self._randomize_controls(envs_idx)
        if self.command_type == "heading":
            forward = gs_transform_by_quat(self.forward_vec, self.base_quat)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        # random push
        push_interval_s = self.env_cfg["push_interval_s"]
        if push_interval_s > 0 and not (self.debug or self.eval):
            max_push_vel_xy = self.env_cfg["max_push_vel_xy"]
            dofs_vel = self.robot.get_dofs_velocity()  # (num_envs, num_dof) [0:3] ~ base_link_vel
            push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
            push_vel[((self.common_step_counter + self.env_identities) % int(push_interval_s / self.dt) != 0)] = 0
            dofs_vel[:, :2] += push_vel
            self.robot.set_dofs_velocity(dofs_vel)

        self.check_termination()
        self.compute_reward()

        if torch.any(self.reset_buf):
            self.extras["episode_length"] = (
                (self.episode_length_buf * self.reset_buf).sum() / self.reset_buf.sum()
            ).item()
        envs_idx = self.reset_buf.nonzero(as_tuple=False).flatten()
        if self.num_build_envs > 0:
            self.compute_observations()
            self.final_obs_history_buf = self.obs_history_buf.detach().clone()
            self.final_privileged_obs_buf = self.privileged_obs_buf.detach().clone()
            self.reset_idx(envs_idx)
        # self.rigid_solver.forward_kinematics() # no need currently
        self.compute_observations()

        if not self.headless and self.debug:
            self._draw_debug_vis()

        self.last_actions[:] = self.actions[:]
        self.last_last_actions[:] = self.last_actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.robot.get_vel()

    def compute_observations(self):
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands[:, :3] * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        # add noise
        if not self.eval:
            self.obs_buf += gs_rand_float(-1.0, 1.0, (self.num_single_obs,), self.device) * self.obs_noise

        clip_obs = 100.0
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        self.obs_history_buf = torch.cat([self.obs_history_buf[:, self.num_single_obs :], self.obs_buf.detach()], dim=1)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.cat(
                [
                    self.obs_buf,  # 45 (noised)
                    self.base_lin_vel * self.obs_scales["lin_vel"],  # 3
                    self.last_actions,  # 12
                ],
                axis=-1,
            )
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

    def _prepare_obs_noise(self):
        self.obs_noise[:, :3] = self.obs_cfg["obs_noise"]["ang_vel"]
        self.obs_noise[:, 3:6] = self.obs_cfg["obs_noise"]["gravity"]
        self.obs_noise[:, 21:33] = self.obs_cfg["obs_noise"]["dof_pos"]
        self.obs_noise[:, 33:45] = self.obs_cfg["obs_noise"]["dof_vel"]

    def _resample_commands(self, envs_idx):
        # resample commands

        # lin_vel
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, :2] *= (torch.norm(self.commands[envs_idx, :2], dim=1) > 0.2).unsqueeze(1)

        # ang_vel
        if self.command_type == "heading":
            self.commands[envs_idx, 3] = gs_rand_float(-3.14, 3.14, (len(envs_idx),), self.device)
        elif self.command_type == "ang_vel_yaw":
            self.commands[envs_idx, 2] = gs_rand_float(
                *self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device
            )
            self.commands[envs_idx, 2] *= torch.abs(self.commands[envs_idx, 2]) > 0.2

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = (self.default_dof_pos) + gs_rand_float(
            -0.3, 0.3, (len(envs_idx), self.num_dof), self.device
        )
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset root states - position
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_pos[envs_idx, :2] += gs_rand_float(-1.0, 1.0, (len(envs_idx), 2), self.device)
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        base_euler = gs_rand_float(-0.1, 0.1, (len(envs_idx), 3), self.device)
        base_euler[:, 2] = gs_rand_float(0.0, 3.14, (len(envs_idx),), self.device)
        self.base_quat[envs_idx] = gs_quat_mul(
            gs_euler2quat(base_euler),
            self.base_quat[envs_idx],
        )
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.zero_all_dofs_velocity(envs_idx)

        # update projected gravity
        inv_base_quat = gs_inv_quat(self.base_quat)
        self.projected_gravity = gs_transform_by_quat(self.global_gravity, inv_base_quat)

        # reset root states - velocity
        self.base_lin_vel[envs_idx] = 0  # gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device)
        self.base_ang_vel[envs_idx] = 0.0  # gs_rand_float(-0.5, 0.5, (len(envs_idx), 3), self.device)
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
        self.time_out_buf[envs_idx] = 0

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        return None, None

    def step(self, actions):
        action_range = self.env_cfg["action_range"]
        self.actions = torch.clip(actions, -action_range, action_range)
        exec_actions = self.last_actions if self.action_latency > 0 else self.actions
        dof_pos_list = []
        dof_vel_list = []
        if self.env_cfg["use_implicit_controller"]:
            target_dof_pos = self._compute_target_dof_pos(exec_actions)
            self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
            self.scene.step()
        else:
            for i in range(self.env_cfg["decimation"]):
                self.torques = self._compute_torques(exec_actions)
                if self.num_build_envs == 0:
                    torques = self.torques.squeeze()
                    self.robot.control_dofs_force(torques, self.motor_dofs)
                else:
                    self.robot.control_dofs_force(self.torques, self.motor_dofs)
                self.scene.step()
                self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
                self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

                if i == 0 or i == 2:
                    dof_pos_list.append(self.robot.get_dofs_position().detach().cpu())
                    dof_vel_list.append(self.robot.get_dofs_velocity().detach().cpu())

        self.dof_pos_list = dof_pos_list
        self.dof_vel_list = dof_vel_list

        self.post_physics_step()

        self.extras["privileged_observations"] = self.privileged_obs_buf
        self.extras["final_observations"] = self.final_obs_history_buf
        self.extras["final_privileged_observations"] = self.final_privileged_obs_buf

        return (
            self.obs_history_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    # ------------ domain randomization----------------

    def _randomize_rigids(self, env_ids=None):
        if self.eval:
            return

        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg["randomize_friction"]:
            self._randomize_link_friction(env_ids)
        if self.env_cfg["randomize_base_mass"]:
            self._randomize_base_mass(env_ids)
        if self.env_cfg["randomize_com_displacement"]:
            self._randomize_com_displacement(env_ids)

    def _randomize_controls(self, env_ids=None):
        if self.eval:
            return

        if env_ids is None:
            env_ids = torch.arange(0, self.num_envs)
        elif len(env_ids) == 0:
            return

        if self.env_cfg["randomize_motor_strength"]:
            self._randomize_motor_strength(env_ids)
        if self.env_cfg["randomize_motor_offset"]:
            self._randomize_motor_offset(env_ids)
        if self.env_cfg["randomize_kp_scale"]:
            self._randomize_kp(env_ids)
        if self.env_cfg["randomize_kd_scale"]:
            self._randomize_kd(env_ids)

    def _randomize_link_friction(self, env_ids):
        min_friction, max_friction = self.env_cfg["friction_range"]

        solver = self.rigid_solver

        ratios = (
            gs.rand((len(env_ids), 1), dtype=float).repeat(1, solver.n_geoms) * (max_friction - min_friction)
            + min_friction
        )
        solver.set_geoms_friction_ratio(ratios, torch.arange(0, solver.n_geoms), env_ids)

    def _randomize_base_mass(self, env_ids):
        min_mass, max_mass = self.env_cfg["added_mass_range"]
        base_link_id = 1

        added_mass = gs.rand((len(env_ids), 1), dtype=float) * (max_mass - min_mass) + min_mass

        self.rigid_solver.set_links_mass_shift(
            added_mass,
            [
                base_link_id,
            ],
            env_ids,
        )

    def _randomize_com_displacement(self, env_ids):
        min_displacement, max_displacement = self.env_cfg["com_displacement_range"]
        base_link_id = 1

        com_displacement = (
            gs.rand((len(env_ids), 1, 3), dtype=float) * (max_displacement - min_displacement) + min_displacement
        )
        # com_displacement[:, :, 0] -= 0.02

        self.rigid_solver.set_links_COM_shift(
            com_displacement,
            [
                base_link_id,
            ],
            env_ids,
        )

    def _randomize_motor_strength(self, env_ids):
        min_strength, max_strength = self.env_cfg["motor_strength_range"]
        self.motor_strengths[env_ids, :] = (
            gs.rand((len(env_ids), 1), dtype=float) * (max_strength - min_strength) + min_strength
        )

    def _randomize_motor_offset(self, env_ids):
        min_offset, max_offset = self.env_cfg["motor_offset_range"]
        self.motor_offsets[env_ids, :] = (
            gs.rand((len(env_ids), self.num_dof), dtype=float) * (max_offset - min_offset) + min_offset
        )

    def _randomize_kp(self, env_ids):
        min_scale, max_scale = self.env_cfg["kp_scale_range"]
        kp_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) * (max_scale - min_scale) + min_scale
        self.batched_p_gains[env_ids, :] = kp_scales * self.p_gains[None, :]

    def _randomize_kd(self, env_ids):
        min_scale, max_scale = self.env_cfg["kd_scale_range"]
        kd_scales = gs.rand((len(env_ids), self.num_dof), dtype=float) * (max_scale - min_scale) + min_scale
        self.batched_d_gains[env_ids, :] = kd_scales * self.d_gains[None, :]

    def _draw_debug_vis(self):
        """Draws visualizations for dubugging (slows down simulation a lot).
        Default behaviour: draws height measurement points
        """
        self.scene.clear_debug_objects()

        foot_poss = self.foot_positions[0].reshape(-1, 3)
        # self.scene.draw_debug_spheres(poss=foot_poss, radius=0.03, color=(1, 0, 0, 0.7))

        foot_poss = foot_poss.cpu()
        self.scene.draw_debug_line(foot_poss[0], foot_poss[3], radius=0.002, color=(1, 0, 0, 0.7))
        self.scene.draw_debug_line(foot_poss[1], foot_poss[2], radius=0.002, color=(1, 0, 0, 0.7))

        com = self.com[0]
        # self.scene.draw_debug_sphere(pos=com, radius=0.1, color=(0, 0, 1, 0.7))

        com[2] = 0.02 + self.terrain_heights[0]
        self.scene.draw_debug_sphere(pos=com, radius=0.02, color=(0, 0, 1, 0.7))

    def _set_camera(self):
        """Set camera position and direction"""
        self._floating_camera = self.scene.add_camera(
            pos=np.array([0, -1, 1]),
            lookat=np.array([0, 0, 0]),
            # res=(720, 480),
            fov=40,
            GUI=False,
        )

        self._recording = False
        self._recorded_frames = []

    def render(self):
        robot_pos = np.array(self.base_pos[0].cpu())
        self._floating_camera.set_pose(
            pos=robot_pos + np.array([-1, -1, 0.5]), lookat=robot_pos + np.array([0, 0, -0.1])
        )
        frame, _, _, _ = self._floating_camera.render()
        return frame
