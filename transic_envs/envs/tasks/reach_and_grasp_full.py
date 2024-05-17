from typing import Dict

from isaacgym import gymapi, gymtorch
import torch
import numpy as np


from transic_envs.asset_root import ASSET_ROOT
import transic_envs.utils.torch_jit_utils as torch_jit_utils
from transic_envs.envs.core import TRANSICEnvPCD, TRANSICEnvOSC
from transic_envs.envs.core.sim_config import sim_config
from transic_envs.utils.pose_utils import rot_mat, get_mat
import transic_envs.utils.fb_control_utils as C


class ReachAndGraspFullEnv(TRANSICEnvOSC):
    initial_height = 0.012851864099502563

    all_legs_reset_pos = [
        np.array([-0.20, 0.07, -0.015]),
        np.array([-0.12, 0.07, -0.015]),
        np.array([0.12, 0.07, -0.015]),
        np.array([0.20, 0.07, -0.015]),
    ]
    all_legs_reset_ori = [
        rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
        rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
        rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
        rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
    ]
    all_possible_assemble_poses = [
        get_mat(np.array([0.05625, 0.046875, 0.05625]), [0, 0, 0]),
        get_mat(np.array([-0.05625, 0.046875, 0.05625]), [0, 0, 0]),
        get_mat(np.array([0.05625, 0.046875, -0.05625]), [0, 0, 0]),
        get_mat(np.array([-0.05625, 0.046875, -0.05625]), [0, 0, 0]),
    ]

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
        self._success_weight = cfg["env"]["successWeight"]
        self._failure_weight = cfg["env"]["failureWeight"]
        assert self._failure_weight >= 0, "failure weight should be non-negative"
        self._target_lift_height = cfg["env"]["targetLiftHeight"]
        self._distance_reward = cfg["env"]["distanceReward"]
        self._progress_reward = cfg["env"]["progressReward"]
        self._dq_penalty = cfg["env"]["dqPenalty"]
        assert self._dq_penalty >= 0, "dq penalty should be non-negative"

        self._selected_leg_idx = cfg["env"]["selectedLegIdx"]

        self._task_progress_buf = None

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )

    def allocate_buffers(self):
        super().allocate_buffers()
        self._task_progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0

        self._task_progress_buf[env_ids] = 0

        # Randomize initial furniture part poses
        self.furniture.reset()

        # Update furniture parts poses
        pos, ori = [], []
        for part in self.furniture.parts:
            pos.append(part.part_config["reset_pos"][0])  # (3,)
            ori.append(part.part_config["reset_ori"][0])  # (4,)
        pos = np.stack(pos)[:, np.newaxis, :]  # (num_parts, 1, 3)
        ori = np.stack(ori)[:, np.newaxis, ...]  # (num_parts, 1, 4, 4)
        pos = pos.repeat(len(env_ids), 1)  # (num_parts, num_resets, 3)
        ori = ori.repeat(len(env_ids), 1)  # (num_parts, num_resets, 4, 4)
        if self._selected_leg_idx is not None:
            assert self._selected_leg_idx in [0, 1, 2, 3]
            other_leg_idxs = np.array(
                [x for x in range(4) if x != self._selected_leg_idx]
            )  # (3,)
            pose_idxs = np.tile(other_leg_idxs, (len(env_ids), 1))  # (num_resets, 3)
            pose_idxs = np.apply_along_axis(
                np.random.permutation, 1, pose_idxs
            )  # (num_resets, 3)
            pose_idxs = np.concatenate(
                [
                    pose_idxs,
                    np.ones((len(env_ids), 1), dtype=int) * self._selected_leg_idx,
                ],
                axis=1,
            )  # (num_resets, 4)
        else:
            # randomly position other three table legs
            pose_idxs = np.tile(np.arange(4), (len(env_ids), 1))  # (num_resets, 4)
            pose_idxs = np.apply_along_axis(
                np.random.permutation, 1, pose_idxs
            )  # (num_resets, 4)
        legs_pos = np.array(self.all_legs_reset_pos)[pose_idxs]  # (num_resets, 4, 3)
        legs_ori = np.array(self.all_legs_reset_ori)[pose_idxs]  # (num_resets, 4, 4, 4)
        legs_pos = legs_pos.transpose(1, 0, 2)  # (4, num_resets, 3)
        legs_ori = legs_ori.transpose(1, 0, 2, 3)  # (4, num_resets, 4, 4)
        pos[1:5, :, :] = legs_pos
        ori[1:5, :, :, :] = legs_ori

        # randomize pos and ori
        pos[:, :, :2] += np.random.uniform(
            -0.015, 0.015, size=(len(self.furniture.parts), len(env_ids), 2)
        )
        pos = torch.tensor(pos, device=self.sim_device)
        # convert pos to homogenous matrix
        pos_mat = (
            torch.eye(4, device=self.sim_device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(len(self.furniture.parts), len(env_ids), 1, 1)
        )  # (num_parts, num_resets, 4, 4)
        pos_mat[:, :, :3, 3] = pos
        pos_mat = pos_mat.reshape(-1, 4, 4)
        pos_mat = (
            self.april_to_sim_mat @ pos_mat
        )  # (4, 4) @ (num_parts * num_resets, 4, 4) -> (num_parts * num_resets, 4, 4)
        pos_mat = pos_mat.reshape(len(self.furniture.parts), len(env_ids), 4, 4)
        new_pos = pos_mat[:, :, :3, 3]  # (num_parts, num_resets, 3)
        # randomly spawn the tabletop in the valid region
        new_pos[0, :, 0] = torch.tensor(
            np.random.uniform(
                low=0.2315 * 0.9, high=(0.38 - 0.08125) * 0.9, size=(len(env_ids))
            ),
            device=self.sim_device,
            dtype=new_pos.dtype,
        )
        new_pos[0, :, 1] = torch.tensor(
            np.random.uniform(
                low=0,
                high=(0.17 - 0.08125) * 0.9,
                size=(len(env_ids)),
            ),
            device=self.sim_device,
            dtype=new_pos.dtype,
        )

        ori = torch.tensor(ori, device=self.sim_device)  # (num_parts, num_resets, 4, 4)
        ori_noise = np.zeros((len(self.furniture.parts), len(env_ids), 3))
        ori_noise[:, :, 2] = np.random.uniform(
            np.radians(-15),
            np.radians(15),
            size=(len(self.furniture.parts), len(env_ids)),
        )
        ori_noise = torch.tensor(ori_noise, device=self.sim_device, dtype=ori.dtype)
        ori_noise = torch_jit_utils.axisangle2quat(
            ori_noise
        )  # (num_parts, num_resets, 4) in xyzw order
        # change to wxyz order
        ori_noise = torch.cat([ori_noise[:, :, 3:], ori_noise[:, :, :3]], dim=-1)
        ori_noise = torch_jit_utils.quaternion_to_matrix(
            ori_noise
        )  # (num_parts, num_resets, 3, 3)
        # convert to homogeneous matrix
        ori_noise_homo = (
            torch.eye(4, dtype=ori.dtype, device=self.sim_device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(len(self.furniture.parts), len(env_ids), 1, 1)
        )  # (num_parts, num_resets, 4, 4)
        ori_noise_homo[:, :, :3, :3] = ori_noise
        ori_noise_homo[:, :, 3, 3] = 1
        ori = ori.reshape(-1, 4, 4)
        ori_noise_homo = ori_noise_homo.reshape(-1, 4, 4)
        ori = ori_noise_homo @ ori  # (N, 4, 4) @ (N, 4, 4) -> (N, 4, 4)
        ori = (
            self.april_to_sim_mat @ ori
        )  # (4, 4) @ (num_parts * num_resets, 4, 4) -> (num_parts * num_resets, 4, 4)
        ori_quat = torch_jit_utils.matrix_to_quaternion(
            ori[:, :3, :3]
        )  # (num_parts * num_resets, 4) in wxyz order
        # convert to xyzw order
        ori_quat = torch.cat([ori_quat[:, 1:], ori_quat[:, :1]], dim=-1)
        ori_quat = ori_quat.reshape(len(self.furniture.parts), len(env_ids), 4)

        # determine randomly assembled legs
        tabletop_pose_mat = (
            torch.eye(4, device=self.sim_device).unsqueeze(0).repeat(len(env_ids), 1, 1)
        )  # (num_resets, 4, 4)
        table_ori = torch.cat(
            [ori_quat[0, :, 3:], ori_quat[0, :, :3]], dim=-1
        )  # (num_resets, 4)
        tabletop_rot_mat = torch_jit_utils.quaternion_to_matrix(table_ori)
        tabletop_pose_mat[:, :3, :3] = tabletop_rot_mat
        tabletop_pose_mat[:, :3, 3] = new_pos[0, :, :3]
        tabletop_pose_mat = (
            tabletop_pose_mat.unsqueeze(1).repeat(1, 3, 1, 1).reshape(-1, 4, 4)
        )

        all_possible_assemble_poses = np.stack(
            self.all_possible_assemble_poses
        )  # (4, 4, 4)
        all_assemble_poses = all_possible_assemble_poses[np.newaxis, ...].repeat(
            len(env_ids), 0
        )  # (num_resets, 4, 4, 4)
        other_legs_pose_idxs = pose_idxs[:, :3]
        all_assemble_poses = all_assemble_poses[
            np.arange(len(env_ids))[:, None], other_legs_pose_idxs, :, :
        ]  # (num_resets, 3, 4, 4)
        all_assemble_poses = torch.tensor(
            all_assemble_poses, device=self.sim_device, dtype=torch.float32
        )
        all_assemble_poses = all_assemble_poses.reshape(-1, 4, 4)
        leg_assemble_poses = (
            tabletop_pose_mat @ all_assemble_poses
        )  # (num_resets * 3, 4, 4)
        leg_assemble_poses = leg_assemble_poses.reshape(
            len(env_ids), 3, 4, 4
        )  # (num_resets, 3, 4, 4)
        leg_assemble_pos = leg_assemble_poses[:, :, :3, 3]  # (num_resets, 3, 3)
        leg_assemble_ori = leg_assemble_poses[:, :, :3, :3]  # (num_resets, 3, 3, 3)
        leg_assemble_ori = torch_jit_utils.matrix_to_quaternion(
            leg_assemble_ori
        )  # (num_resets, 3, 4)
        leg_assemble_ori = torch.cat(
            [leg_assemble_ori[..., 3:], leg_assemble_ori[..., :3]], dim=-1
        )  # (num_resets, 3, 4)
        leg_assemble_pos = leg_assemble_pos.reshape(-1, 3)  # (num_resets * 3, 3)
        leg_assemble_ori = leg_assemble_ori.reshape(-1, 4)  # (num_resets * 3, 4)
        if_assembled_mask = (
            torch.rand(
                size=(len(env_ids) * 3, 1),
                device=self.sim_device,
                dtype=torch.float32,
            )
            > 0.5
        )
        old_leg_pos = new_pos[1:4].reshape(-1, 3)  # (num_resets * 3, 3)
        old_leg_ori = ori_quat[1:4].reshape(-1, 4)  # (num_resets * 3, 4)
        new_leg_pos = (
            if_assembled_mask * leg_assemble_pos + ~if_assembled_mask * old_leg_pos
        )
        new_leg_ori = (
            if_assembled_mask * leg_assemble_ori + ~if_assembled_mask * old_leg_ori
        )
        new_leg_pos = new_leg_pos.reshape(len(env_ids), 3, 3)  # (num_resets, 3, 3)
        new_leg_ori = new_leg_ori.reshape(len(env_ids), 3, 4)  # (num_resets, 3, 4)
        new_leg_pos = new_leg_pos.transpose(0, 1)  # (3, num_resets, 3)
        new_leg_ori = new_leg_ori.transpose(0, 1)  # (3, num_resets, 4)
        new_pos[1:4, :, :] = new_leg_pos
        ori_quat[1:4, :, :] = new_leg_ori

        reset_pos = torch.cat([new_pos, ori_quat], dim=-1)  # (num_parts, num_resets, 7)
        vel = torch.zeros(
            (len(self.furniture.parts), len(env_ids), 6),
            device=self.sim_device,
            dtype=reset_pos.dtype,
        )
        reset_state = torch.cat([reset_pos, vel], dim=-1)  # (num_parts, num_resets, 13)

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

        for part, part_state in zip(self.furniture.parts, reset_state):
            # Set furniture part state
            self._init_fparts_states[part.name][env_ids, :] = part_state
            # Write these new init states to the sim states
            self._fparts_states[part.name][env_ids] = self._init_fparts_states[
                part.name
            ][env_ids]
        # Collect all part ids and deploy state update
        multi_env_ids_int32 = [
            self._global_furniture_part_indices[part_name][env_ids].flatten()
            for part_name in self._fparts_names
        ]
        multi_env_ids_int32 = torch.cat(multi_env_ids_int32, dim=0)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
            self.failure_buf[:],
            self._task_progress_buf[:],
        ) = compute_reward(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            success_buf=self.success_buf,
            failure_buf=self.failure_buf,
            task_progress_buf=self._task_progress_buf,
            progress_reward_weight=self._progress_reward,
            states=self.states,
            initial_height=self.initial_height,
            target_lift_height=self._target_lift_height,
            distance_reward_weight=self._distance_reward,
            max_episode_length=self.max_episode_length,
            success_reward_weight=self._success_weight,
            failure_weight=self._failure_weight,
            dq_penalty=self._dq_penalty,
        )


class ReachAndGraspFullPCDEnv(TRANSICEnvPCD):
    initial_height = 0.012851864099502563

    all_legs_reset_pos = [
        np.array([-0.20, 0.07, -0.015]),
        np.array([-0.12, 0.07, -0.015]),
        np.array([0.12, 0.07, -0.015]),
        np.array([0.20, 0.07, -0.015]),
    ]
    all_legs_reset_ori = [
        rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
        rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
        rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
        rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
    ]
    all_possible_assemble_poses = [
        get_mat(np.array([0.05625, 0.046875, 0.05625]), [0, 0, 0]),
        get_mat(np.array([-0.05625, 0.046875, 0.05625]), [0, 0, 0]),
        get_mat(np.array([0.05625, 0.046875, -0.05625]), [0, 0, 0]),
        get_mat(np.array([-0.05625, 0.046875, -0.05625]), [0, 0, 0]),
    ]

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
        self._success_weight = cfg["env"]["successWeight"]
        self._target_lift_height = cfg["env"]["targetLiftHeight"]

        self._task_progress_buf = None

        self._selected_leg_idx = cfg["env"]["selectedLegIdx"]
        assert self._selected_leg_idx is not None, "selected leg index should be given"

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
        )

    def allocate_buffers(self):
        super().allocate_buffers()
        self._task_progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )

    def _import_furniture_assets(self):
        self._fparts_assets = {}
        for part in self.furniture.parts:
            if part.name in self._fparts_assets:
                continue
            asset_option = sim_config["asset"][
                "square_table_leg4" if part.name == "leg" else part.name
            ]
            self._fparts_assets[part.name] = self.gym.load_asset(
                self.sim,
                ASSET_ROOT,
                part.asset_file,
                asset_option,
            )
        # import raw pcds
        self._fparts_pcds = {}
        for part in self.furniture.parts:
            if part.name in self._fparts_pcds:
                continue
            self._fparts_pcds[part.name] = C.xyz_to_homogeneous(
                torch.tensor(
                    part.pointcloud, device=self.sim_device, dtype=torch.float32
                ),
                device=self.sim_device,
            )

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0

        self._task_progress_buf[env_ids] = 0

        # Randomize initial furniture part poses
        self.furniture.reset()

        # Update furniture parts poses
        pos, ori = [], []
        for part in self.furniture.parts:
            pos.append(part.part_config["reset_pos"][0])  # (3,)
            ori.append(part.part_config["reset_ori"][0])  # (4,)
        pos = np.stack(pos)[:, np.newaxis, :]  # (num_parts, 1, 3)
        ori = np.stack(ori)[:, np.newaxis, ...]  # (num_parts, 1, 4, 4)
        pos = pos.repeat(len(env_ids), 1)  # (num_parts, num_resets, 3)
        ori = ori.repeat(len(env_ids), 1)  # (num_parts, num_resets, 4, 4)
        # randomly position other three table legs
        assert self._selected_leg_idx in [0, 1, 2, 3]
        other_leg_idxs = np.array(
            [x for x in range(4) if x != self._selected_leg_idx]
        )  # (3,)
        pose_idxs = np.tile(other_leg_idxs, (len(env_ids), 1))  # (num_resets, 3)
        pose_idxs = np.apply_along_axis(
            np.random.permutation, 1, pose_idxs
        )  # (num_resets, 3)
        pose_idxs = np.concatenate(
            [
                pose_idxs,
                np.ones((len(env_ids), 1), dtype=int) * self._selected_leg_idx,
            ],
            axis=1,
        )  # (num_resets, 4)
        legs_pos = np.array(self.all_legs_reset_pos)[pose_idxs]  # (num_resets, 4, 3)
        legs_ori = np.array(self.all_legs_reset_ori)[pose_idxs]  # (num_resets, 4, 4, 4)
        legs_pos = legs_pos.transpose(1, 0, 2)  # (4, num_resets, 3)
        legs_ori = legs_ori.transpose(1, 0, 2, 3)  # (4, num_resets, 4, 4)
        pos[1:5, :, :] = legs_pos
        ori[1:5, :, :, :] = legs_ori

        # randomize pos and ori
        pos[:, :, :2] += np.random.uniform(
            -0.015, 0.015, size=(len(self.furniture.parts), len(env_ids), 2)
        )
        pos = torch.tensor(pos, device=self.sim_device)
        # convert pos to homogenous matrix
        pos_mat = (
            torch.eye(4, device=self.sim_device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(len(self.furniture.parts), len(env_ids), 1, 1)
        )  # (num_parts, num_resets, 4, 4)
        pos_mat[:, :, :3, 3] = pos
        pos_mat = pos_mat.reshape(-1, 4, 4)
        pos_mat = (
            self.april_to_sim_mat @ pos_mat
        )  # (4, 4) @ (num_parts * num_resets, 4, 4) -> (num_parts * num_resets, 4, 4)
        pos_mat = pos_mat.reshape(len(self.furniture.parts), len(env_ids), 4, 4)
        new_pos = pos_mat[:, :, :3, 3]  # (num_parts, num_resets, 3)
        # randomly spawn the tabletop in the valid region
        new_pos[0, :, 0] = torch.tensor(
            np.random.uniform(
                low=0.2315 * 0.9, high=(0.38 - 0.08125) * 0.9, size=(len(env_ids))
            ),
            device=self.sim_device,
            dtype=new_pos.dtype,
        )
        new_pos[0, :, 1] = torch.tensor(
            np.random.uniform(
                low=0,
                high=(0.17 - 0.08125) * 0.9,
                size=(len(env_ids)),
            ),
            device=self.sim_device,
            dtype=new_pos.dtype,
        )

        ori = torch.tensor(ori, device=self.sim_device)  # (num_parts, num_resets, 4, 4)
        ori_noise = np.zeros((len(self.furniture.parts), len(env_ids), 3))
        ori_noise[:, :, 2] = np.random.uniform(
            np.radians(-15),
            np.radians(15),
            size=(len(self.furniture.parts), len(env_ids)),
        )
        ori_noise = torch.tensor(ori_noise, device=self.sim_device, dtype=ori.dtype)
        ori_noise = torch_jit_utils.axisangle2quat(
            ori_noise
        )  # (num_parts, num_resets, 4) in xyzw order
        # change to wxyz order
        ori_noise = torch.cat([ori_noise[:, :, 3:], ori_noise[:, :, :3]], dim=-1)
        ori_noise = torch_jit_utils.quaternion_to_matrix(
            ori_noise
        )  # (num_parts, num_resets, 3, 3)
        # convert to homogeneous matrix
        ori_noise_homo = (
            torch.eye(4, dtype=ori.dtype, device=self.sim_device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(len(self.furniture.parts), len(env_ids), 1, 1)
        )  # (num_parts, num_resets, 4, 4)
        ori_noise_homo[:, :, :3, :3] = ori_noise
        ori_noise_homo[:, :, 3, 3] = 1
        ori = ori.reshape(-1, 4, 4)
        ori_noise_homo = ori_noise_homo.reshape(-1, 4, 4)
        ori = ori_noise_homo @ ori  # (N, 4, 4) @ (N, 4, 4) -> (N, 4, 4)
        ori = (
            self.april_to_sim_mat @ ori
        )  # (4, 4) @ (num_parts * num_resets, 4, 4) -> (num_parts * num_resets, 4, 4)
        ori_quat = torch_jit_utils.matrix_to_quaternion(
            ori[:, :3, :3]
        )  # (num_parts * num_resets, 4) in wxyz order
        # convert to xyzw order
        ori_quat = torch.cat([ori_quat[:, 1:], ori_quat[:, :1]], dim=-1)
        ori_quat = ori_quat.reshape(len(self.furniture.parts), len(env_ids), 4)

        # determine randomly assembled legs
        tabletop_pose_mat = (
            torch.eye(4, device=self.sim_device).unsqueeze(0).repeat(len(env_ids), 1, 1)
        )  # (num_resets, 4, 4)
        table_ori = torch.cat(
            [ori_quat[0, :, 3:], ori_quat[0, :, :3]], dim=-1
        )  # (num_resets, 4)
        tabletop_rot_mat = torch_jit_utils.quaternion_to_matrix(table_ori)
        tabletop_pose_mat[:, :3, :3] = tabletop_rot_mat
        tabletop_pose_mat[:, :3, 3] = new_pos[0, :, :3]
        tabletop_pose_mat = (
            tabletop_pose_mat.unsqueeze(1).repeat(1, 3, 1, 1).reshape(-1, 4, 4)
        )

        all_possible_assemble_poses = np.stack(
            self.all_possible_assemble_poses
        )  # (4, 4, 4)
        all_assemble_poses = all_possible_assemble_poses[np.newaxis, ...].repeat(
            len(env_ids), 0
        )  # (num_resets, 4, 4, 4)
        other_legs_pose_idxs = pose_idxs[:, :3]
        all_assemble_poses = all_assemble_poses[
            np.arange(len(env_ids))[:, None], other_legs_pose_idxs, :, :
        ]  # (num_resets, 3, 4, 4)
        all_assemble_poses = torch.tensor(
            all_assemble_poses, device=self.sim_device, dtype=torch.float32
        )
        all_assemble_poses = all_assemble_poses.reshape(-1, 4, 4)
        leg_assemble_poses = (
            tabletop_pose_mat @ all_assemble_poses
        )  # (num_resets * 3, 4, 4)
        leg_assemble_poses = leg_assemble_poses.reshape(
            len(env_ids), 3, 4, 4
        )  # (num_resets, 3, 4, 4)
        leg_assemble_pos = leg_assemble_poses[:, :, :3, 3]  # (num_resets, 3, 3)
        leg_assemble_ori = leg_assemble_poses[:, :, :3, :3]  # (num_resets, 3, 3, 3)
        leg_assemble_ori = torch_jit_utils.matrix_to_quaternion(
            leg_assemble_ori
        )  # (num_resets, 3, 4)
        leg_assemble_ori = torch.cat(
            [leg_assemble_ori[..., 3:], leg_assemble_ori[..., :3]], dim=-1
        )  # (num_resets, 3, 4)
        leg_assemble_pos = leg_assemble_pos.reshape(-1, 3)  # (num_resets * 3, 3)
        leg_assemble_ori = leg_assemble_ori.reshape(-1, 4)  # (num_resets * 3, 4)
        if_assembled_mask = (
            torch.rand(
                size=(len(env_ids) * 3, 1),
                device=self.sim_device,
                dtype=torch.float32,
            )
            > 0.5
        )
        old_leg_pos = new_pos[1:4].reshape(-1, 3)  # (num_resets * 3, 3)
        old_leg_ori = ori_quat[1:4].reshape(-1, 4)  # (num_resets * 3, 4)
        new_leg_pos = (
            if_assembled_mask * leg_assemble_pos + ~if_assembled_mask * old_leg_pos
        )
        new_leg_ori = (
            if_assembled_mask * leg_assemble_ori + ~if_assembled_mask * old_leg_ori
        )
        new_leg_pos = new_leg_pos.reshape(len(env_ids), 3, 3)  # (num_resets, 3, 3)
        new_leg_ori = new_leg_ori.reshape(len(env_ids), 3, 4)  # (num_resets, 3, 4)
        new_leg_pos = new_leg_pos.transpose(0, 1)  # (3, num_resets, 3)
        new_leg_ori = new_leg_ori.transpose(0, 1)  # (3, num_resets, 4)
        new_pos[1:4, :, :] = new_leg_pos
        ori_quat[1:4, :, :] = new_leg_ori

        reset_pos = torch.cat([new_pos, ori_quat], dim=-1)  # (num_parts, num_resets, 7)
        vel = torch.zeros(
            (len(self.furniture.parts), len(env_ids), 6),
            device=self.sim_device,
            dtype=reset_pos.dtype,
        )
        reset_state = torch.cat([reset_pos, vel], dim=-1)  # (num_parts, num_resets, 13)

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

        for part, part_state in zip(self.furniture.parts, reset_state):
            # Set furniture part state
            self._init_fparts_states[part.name][env_ids, :] = part_state
            # Write these new init states to the sim states
            self._fparts_states[part.name][env_ids] = self._init_fparts_states[
                part.name
            ][env_ids]
        # Collect all part ids and deploy state update
        multi_env_ids_int32 = [
            self._global_furniture_part_indices[part_name][env_ids].flatten()
            for part_name in self._fparts_names
        ]
        multi_env_ids_int32 = torch.cat(multi_env_ids_int32, dim=0)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32),
        )

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
            self.failure_buf[:],
            self._task_progress_buf[:],
        ) = compute_reward(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            success_buf=self.success_buf,
            failure_buf=self.failure_buf,
            task_progress_buf=self._task_progress_buf,
            progress_reward_weight=0,
            states=self.states,
            initial_height=self.initial_height,
            target_lift_height=self._target_lift_height,
            distance_reward_weight=0,
            max_episode_length=self.max_episode_length,
            success_reward_weight=self._success_weight,
            failure_weight=0,
            dq_penalty=0,
        )


@torch.jit.script
def compute_reward(
    *,
    reset_buf,
    progress_buf,
    success_buf,
    failure_buf,
    task_progress_buf,
    progress_reward_weight: float,
    states: Dict[str, torch.Tensor],
    initial_height: float,
    target_lift_height: float,
    distance_reward_weight: float,
    max_episode_length: int,
    success_reward_weight: float,
    failure_weight: float,
    dq_penalty: float,
):
    target_leg_pos = states["leg_pos"]
    leg_rot = states["leg_rot"]
    leg_vel = states["leg_vel"]
    fintergip_center_pos = states["ftip_center_pos"]
    eef_lf_pos = states["eef_lf_pos"]
    eef_rf_pos = states["eef_rf_pos"]
    eef_rot = states["eef_quat"]

    # distance from hand to the leg
    d = torch.norm(target_leg_pos - fintergip_center_pos, dim=-1)
    d_lf = torch.norm(target_leg_pos - eef_lf_pos, dim=-1)
    d_rf = torch.norm(target_leg_pos - eef_rf_pos, dim=-1)
    leg_rot_euler_z = torch_jit_utils.get_euler_xyz(leg_rot)[-1]
    eef_rot_euler_z = torch_jit_utils.get_euler_xyz(eef_rot)[-1]
    leg_rot_euler_z = torch_jit_utils.normalize_angle(leg_rot_euler_z)
    eef_rot_euler_z = torch_jit_utils.normalize_angle(eef_rot_euler_z)
    # gripper and table leg should be orthogonal in Z rotation
    d_orthogonal = torch.abs(torch.cos(leg_rot_euler_z - eef_rot_euler_z))
    distance = (d + d_lf + d_rf + d_orthogonal) / 4
    dist_reward = 1 - torch.tanh(10.0 * distance)

    lifted_height = target_leg_pos[:, 2] - initial_height

    # compute normalized task progress
    normalized_task_progress = torch.clamp(
        lifted_height / target_lift_height, min=0, max=1
    )
    delta_progress = torch.clamp(
        normalized_task_progress - task_progress_buf, min=0, max=1
    )
    # update task progress buffer
    new_normalized_task_progress = torch.where(
        delta_progress > 0,
        normalized_task_progress,
        task_progress_buf,
    )

    dq_norm = torch.norm(states["dq"], dim=-1)

    leg_stable_mask = torch.linalg.vector_norm(leg_vel[:, :2], dim=-1) < 5e-2

    succeeded = (lifted_height > target_lift_height) & leg_stable_mask

    failure = torch.zeros_like(succeeded)

    distance_reward = distance_reward_weight * dist_reward
    success_reward = success_reward_weight * succeeded
    failure_reward = failure_weight * failure
    progress_reward = progress_reward_weight * delta_progress
    dq_penalty = dq_penalty * dq_norm
    reward = (
        distance_reward + success_reward - failure_reward + progress_reward - dq_penalty
    )

    success = succeeded | success_buf
    failure = failure | failure_buf

    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1) | success | failure,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    return reward, reset_buf, success, failure, new_normalized_task_progress
