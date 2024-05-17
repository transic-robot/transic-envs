from __future__ import annotations

from isaacgym import gymtorch
from isaacgym import gymapi
import torch

import transic_envs.utils.torch_jit_utils as torch_jit_utils
from transic_envs.envs.core import TRANSICEnvPCD, TRANSICEnvOSC


class ReachAndGraspSingleEnv(TRANSICEnvOSC):
    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        self._target_lift_height = cfg["env"]["targetLiftHeight"]
        self._distance_reward = cfg["env"]["distanceReward"]
        self._lift_reward = cfg["env"]["liftReward"]
        self._success_reward = cfg["env"]["successReward"]
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)
        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.sim_device)
        pos = torch_jit_utils.tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0),
            self.franka_dof_upper_limits,
        )

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # Deploy updates for Franka
        multi_env_ids_int32 = self._global_franka_indices[env_ids].flatten()
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._effort_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0

        # Update leg pose
        num_resets = len(env_ids)
        sampled_leg_state = torch.zeros(num_resets, 13, device=self.sim_device)
        # Sampling is "centered" around middle of table
        base_tag_x = self._front_obstacle_pose.p.x - 0.38
        workspace_x_center = 0.5 * (
            float(self._front_obstacle_pose.p.x) + base_tag_x
        )  # 0.28 is the x coordinate of base tag
        workspace_y_center = 0.0
        x_noise = 0.5 * 0.38 * 0.8
        y_noise = 0.175 * 0.9
        centered_leg_xy_state = torch.tensor(
            [workspace_x_center, workspace_y_center],
            device=self.sim_device,
            dtype=torch.float32,
        )
        # Set z value, which is fixed height
        sampled_leg_state[:, 2] = self._table_surface_z + 0.015
        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_leg_state[:, 6] = 1.0
        sampled_leg_state[:, :2] = centered_leg_xy_state.unsqueeze(
            0
        ) + 2.0 * torch.tensor(
            [x_noise, y_noise], dtype=torch.float32, device=self.sim_device
        ) * (
            torch.rand(num_resets, 2, device=self.sim_device) - 0.5
        )
        # Sample rotation value
        aa_rot = torch.zeros(num_resets, 3, device=self.sim_device)
        aa_rot[:, 2] = (
            2.0 * 3.14159 * (torch.rand(num_resets, device=self.sim_device) - 0.5)
        )
        sampled_leg_state[:, 3:7] = torch_jit_utils.quat_mul(
            torch_jit_utils.axisangle2quat(aa_rot), sampled_leg_state[:, 3:7]
        )
        # Set leg state
        self._init_fparts_states["leg"][env_ids, :] = sampled_leg_state

        # Write these new init states to the sim states
        self._fparts_states["leg"][env_ids] = self._init_fparts_states["leg"][env_ids]

        # Deploy leg state update
        multi_env_ids_leg_int32 = self._global_furniture_part_indices["leg"][
            env_ids
        ].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_leg_int32),
            len(multi_env_ids_leg_int32),
        )

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
        ) = compute_reach_and_grasp_single_reward(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            states=self.states,
            max_episode_length=self.max_episode_length,
            target_lift_height=self._target_lift_height,
            distance_reward=self._distance_reward,
            lift_reward=self._lift_reward,
            success_reward=self._success_reward,
        )


class ReachAndGraspSinglePCDEnv(TRANSICEnvPCD):
    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
    ):
        self._target_lift_height = cfg["env"]["targetLiftHeight"]
        self._distance_reward = cfg["env"]["distanceReward"]
        self._lift_reward = cfg["env"]["liftReward"]
        self._success_reward = cfg["env"]["successReward"]
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
        )

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)
        # Reset agent
        reset_noise = torch.rand((len(env_ids), 9), device=self.sim_device)
        pos = torch_jit_utils.tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + self.franka_dof_noise * 2.0 * (reset_noise - 0.5),
            self.franka_dof_lower_limits.unsqueeze(0),
            self.franka_dof_upper_limits,
        )

        # Overwrite gripper init pos (no noise since these are always position controlled)
        pos[:, -2:] = self.franka_default_dof_pos[-2:]

        # Reset the internal obs accordingly
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # Set any position control to the current position, and any vel / effort control to be 0
        # NOTE: Task takes care of actually propagating these controls in sim using the SimActions API
        self._pos_control[env_ids, :] = pos

        # Deploy updates for Franka
        multi_env_ids_int32 = self._global_franka_indices[env_ids].flatten()
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0

        # Update leg pose
        num_resets = len(env_ids)
        sampled_leg_state = torch.zeros(num_resets, 13, device=self.sim_device)
        # Sampling is "centered" around middle of table
        base_tag_x = self._front_obstacle_pose.p.x - 0.38
        workspace_x_center = 0.5 * (
            float(self._front_obstacle_pose.p.x) + base_tag_x
        )  # 0.28 is the x coordinate of base tag
        workspace_y_center = 0.0
        x_noise = 0.5 * 0.38 * 0.8
        y_noise = 0.175 * 0.9
        centered_leg_xy_state = torch.tensor(
            [workspace_x_center, workspace_y_center],
            device=self.sim_device,
            dtype=torch.float32,
        )
        # Set z value, which is fixed height
        sampled_leg_state[:, 2] = self._table_surface_z + 0.015
        # Initialize rotation, which is no rotation (quat w = 1)
        sampled_leg_state[:, 6] = 1.0
        sampled_leg_state[:, :2] = centered_leg_xy_state.unsqueeze(
            0
        ) + 2.0 * torch.tensor(
            [x_noise, y_noise], dtype=torch.float32, device=self.sim_device
        ) * (
            torch.rand(num_resets, 2, device=self.sim_device) - 0.5
        )
        # Sample rotation value
        aa_rot = torch.zeros(num_resets, 3, device=self.sim_device)
        aa_rot[:, 2] = (
            2.0 * 3.14159 * (torch.rand(num_resets, device=self.sim_device) - 0.5)
        )
        sampled_leg_state[:, 3:7] = torch_jit_utils.quat_mul(
            torch_jit_utils.axisangle2quat(aa_rot), sampled_leg_state[:, 3:7]
        )
        # Set leg state
        self._init_fparts_states["leg"][env_ids, :] = sampled_leg_state

        # Write these new init states to the sim states
        self._fparts_states["leg"][env_ids] = self._init_fparts_states["leg"][env_ids]

        # Deploy leg state update
        multi_env_ids_leg_int32 = self._global_furniture_part_indices["leg"][
            env_ids
        ].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_leg_int32),
            len(multi_env_ids_leg_int32),
        )

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
        ) = compute_reach_and_grasp_single_reward(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            states=self.states,
            max_episode_length=self.max_episode_length,
            target_lift_height=self._target_lift_height,
            distance_reward=self._distance_reward,
            lift_reward=self._lift_reward,
            success_reward=self._success_reward,
        )


@torch.jit.script
def compute_reach_and_grasp_single_reward(
    reset_buf,
    progress_buf,
    states: dict[str, torch.Tensor],
    max_episode_length: int,
    target_lift_height: float,
    distance_reward: float,
    lift_reward: float,
    success_reward: float,
):
    leg_pos = states["leg_pos"]
    leg_rot = states["leg_rot"]
    leg_vel = states["leg_vel"]
    fintergip_center_pos = states["ftip_center_pos"]
    eef_rot = states["eef_quat"]
    eef_lf_pos = states["eef_lf_pos"]
    eef_rf_pos = states["eef_rf_pos"]

    front_wall_cf = states["front_wall_cf"]
    left_wall_cf = states["left_wall_cf"]
    right_wall_cf = states["right_wall_cf"]
    wall_contact_mask = (
        torch.all(torch.isclose(front_wall_cf, torch.zeros_like(front_wall_cf)), dim=-1)
        & torch.all(torch.isclose(left_wall_cf, torch.zeros_like(left_wall_cf)), dim=-1)
        & torch.all(
            torch.isclose(right_wall_cf, torch.zeros_like(right_wall_cf)), dim=-1
        )
    )

    leg_hor_vel_norm = torch.linalg.vector_norm(leg_vel[:, :2], dim=-1)
    leg_stable_mask = leg_hor_vel_norm < 5e-2

    # distance from hand to the leg
    leg_pos_relative = leg_pos - fintergip_center_pos
    d = torch.norm(leg_pos_relative, dim=-1)
    d_lf = torch.norm(leg_pos - eef_lf_pos, dim=-1)
    d_rf = torch.norm(leg_pos - eef_rf_pos, dim=-1)
    leg_rot_euler_z = torch_jit_utils.get_euler_xyz(leg_rot)[-1]
    eef_rot_euler_z = torch_jit_utils.get_euler_xyz(eef_rot)[-1]
    leg_rot_euler_z = torch_jit_utils.normalize_angle(leg_rot_euler_z)
    eef_rot_euler_z = torch_jit_utils.normalize_angle(eef_rot_euler_z)
    # gripper and table leg should be orthogonal in Z rotation
    d_orthogonal = torch.abs(torch.cos(leg_rot_euler_z - eef_rot_euler_z))
    distance = (d + d_lf + d_rf + d_orthogonal) / 4
    dist_reward = 1 - torch.tanh(10.0 * distance)

    # reward for lifting leg
    leg_bottom_z = leg_pos[:, 2] - 0.015
    leg_lifted = leg_bottom_z >= 0.01

    # reward for reaching target height
    succeeded = leg_bottom_z > target_lift_height

    succeeded = succeeded * leg_stable_mask * wall_contact_mask
    rewards = (
        (
            distance_reward * dist_reward
            + lift_reward * leg_lifted
            + success_reward * succeeded
        )
        * leg_stable_mask
        * wall_contact_mask
    )
    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1) | succeeded,
        torch.ones_like(reset_buf),
        reset_buf,
    )

    return rewards, reset_buf, succeeded
