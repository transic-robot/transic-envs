from __future__ import annotations
import os

from isaacgym import gymtorch
import torch
import numpy as np

from transic_envs.asset_root import ASSET_ROOT
from transic_envs.utils.pose_utils import rot_mat, get_mat
import transic_envs.utils.torch_jit_utils as torch_jit_utils
from transic_envs.envs.core import TRANSICEnvPCD, TRANSICEnvOSC


class InsertFullEnv(TRANSICEnvOSC):
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
        self._distance_reward = cfg["env"]["distanceReward"]
        self._success_reward = cfg["env"]["successReward"]

        self._rotation_noise = cfg["env"]["rotationNoise"]

        self._leg_target_pos = None
        self._target_leg_idx = None

        self._pos_error_threshold = 0.007
        self._ori_error_threshold = 15 / 180 * np.pi

        try:
            import casadi
            import urdf_parser_py
        except ImportError:
            raise ImportError(
                "Packages `casadi` and `urdf-parser-py` are required for the env `InsertFull`. Install them with `pip install casadi urdf-parser-py`."
            )
        from transic_envs.utils.urdf2casadi import URDFparser

        franka_parser = URDFparser()
        franka_parser.from_file(
            os.path.join(
                ASSET_ROOT,
                "franka_description/robots/franka_panda_finray.urdf",
            )
        )
        self._ftip_center_fk_fn = franka_parser.get_forward_kinematics(
            root="panda_link0", tip="tri_finger_center"
        )["T_fk"]

        self._selected_leg_idx = cfg["env"]["selectedLegIdx"]

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )

        self.franka_dof_lower_limits[-2] = 0.0145
        self.franka_dof_lower_limits[-1] = 0.0145

    def init_data(self):
        super().init_data()
        self._leg_target_pos = torch.zeros(
            (self.num_envs, 3), device=self.sim_device, dtype=torch.float32
        )
        self._target_leg_idx = torch.zeros(
            (self.num_envs,),
            device=self.sim_device,
            dtype=torch.long,
        )

    def _update_states(self):
        super()._update_states()

        # compute target assemble pose for leg
        tabletop_rot_quat = self.states["square_table_top_rot"]  # (N, 4) in xyzw order
        # change to wxyz order
        tabletop_rot_quat = torch.cat(
            [tabletop_rot_quat[:, -1:], tabletop_rot_quat[:, :-1]], dim=-1
        )
        tabletop_rot_mat = torch_jit_utils.quaternion_to_matrix(
            tabletop_rot_quat
        )  # (N, 3, 3)
        tabletop_pose_mat = (
            torch.eye(4, device=self.sim_device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1, 1)
        )
        tabletop_pose_mat[:, :3, :3] = tabletop_rot_mat
        tabletop_pose_mat[:, :3, 3] = self.states["square_table_top_pos"]  # (N, 3)

        all_possible_assemble_poses = torch.tensor(
            np.stack(self.all_possible_assemble_poses), device=self.sim_device
        )  # (4, 4, 4)
        selected_assemble_poses = all_possible_assemble_poses[
            self._target_leg_idx
        ]  # (N, 4, 4)

        furniture_assembled_pose = (
            tabletop_pose_mat @ selected_assemble_poses
        )  # (N, 4, 4)
        furniture_assembled_pos = furniture_assembled_pose[:, :3, 3]
        furniture_assembled_pos[:, 2] += 0.02
        self._leg_target_pos = furniture_assembled_pos
        self.states["target_xyz"] = furniture_assembled_pos[:, :3]

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

        # Randomize initial furniture part poses
        self.furniture.reset()

        # compute pose of finger tip center
        qs = self._q[env_ids, :7]
        ftip_center_poses = []
        for q in qs:
            q = list(q.cpu().numpy())
            ftip_center_poses.append(
                np.array(self._ftip_center_fk_fn(q)).astype(np.float32).reshape(4, 4)
            )
        ftip_center_poses = torch.tensor(
            np.stack(ftip_center_poses),
            device=self.sim_device,
            dtype=torch.float32,
        )  # (N, 4, 4)

        # Update leg pose
        num_resets = len(env_ids)

        leg2ftip_center_transform = torch.eye(
            4, device=self.sim_device, dtype=torch.float32
        )  # (4, 4)
        leg2ftip_center_transform[2, 3] = -0.02 if self.franka_dof_noise > 0 else -0.04

        leg2ftip_center_transform = leg2ftip_center_transform.unsqueeze(0).repeat(
            num_resets, 1, 1
        )
        leg_pose = leg2ftip_center_transform @ ftip_center_poses  # (N, 4, 4)
        leg_pos = leg_pose[:, :3, 3] + self._base_state[env_ids, :3]

        leg2ftip_center_rotation = torch_jit_utils.quat_from_euler_xyz(
            roll=torch.ones((1,), device=self.sim_device, dtype=torch.float32)
            * -np.pi
            / 2,
            pitch=torch.ones((1,), device=self.sim_device, dtype=torch.float32) * 0,
            yaw=torch.ones((1,), device=self.sim_device, dtype=torch.float32) * 0,
        )  # (1, 4) in xyzw order
        # # change to wxyz order
        leg2ftip_center_rotation = torch.cat(
            [leg2ftip_center_rotation[:, -1:], leg2ftip_center_rotation[:, :-1]], dim=-1
        )
        leg2ftip_center_rotation = torch_jit_utils.quaternion_to_matrix(
            leg2ftip_center_rotation
        )  # (1, 3, 3)
        leg_rot = leg2ftip_center_rotation @ leg_pose[:, :3, :3]  # (N, 3, 3)
        leg_rot = torch_jit_utils.matrix_to_quaternion(leg_rot)  # (N, 4) in wxyz order
        # change to xyzw order
        leg_rot = torch.cat([leg_rot[:, -1:], leg_rot[:, :-1]], dim=-1)

        sampled_leg_state = torch.zeros(num_resets, 13, device=self.sim_device)
        sampled_leg_state[:, :3] = leg_pos
        sampled_leg_state[:, 3:7] = leg_rot
        # Update table pose
        sampled_table_state = torch.zeros(num_resets, 13, device=self.sim_device)
        sampled_table_state[:, :7] = torch.tensor(
            [
                [
                    0.23,
                    0.05,
                    0.4307,
                    5.0000e-01,
                    -5.0000e-01,
                    -5.0000e-01,
                    5.0000e-01,
                ]
            ],
            dtype=sampled_table_state.dtype,
            device=sampled_table_state.device,
        ).repeat(num_resets, 1)
        # add noise to table xy
        # +- 2 cm
        table_xy_noise = (
            torch.rand((num_resets, 2), device=self.sim_device, dtype=torch.float32) * 2
            - 1
        ) * 0.02
        sampled_table_state[:, :2] += table_xy_noise

        table_ori_noise = torch.zeros((num_resets, 3), device=self.sim_device)
        table_ori_noise[:, 2] = (
            (torch.rand((num_resets,), device=self.sim_device) * 2 - 1)
            * self._rotation_noise
            / 180
            * np.pi
        )
        table_ori_noise = torch_jit_utils.axisangle2quat(table_ori_noise)
        table_ori = torch_jit_utils.quat_mul(
            table_ori_noise, sampled_table_state[:, 3:7]
        )
        sampled_table_state[:, 3:7] = table_ori

        # handle other furniture parts
        pos, ori = [], []
        for part in self.furniture.parts:
            pos.append(part.part_config["reset_pos"][0])  # (3,)
            ori.append(part.part_config["reset_ori"][0])  # (4,)
        pos = np.stack(pos)[:, np.newaxis, :]  # (num_parts, 1, 3)
        ori = np.stack(ori)[:, np.newaxis, ...]  # (num_parts, 1, 4, 4)
        pos = pos.repeat(num_resets, 1)  # (num_parts, num_resets, 3)
        ori = ori.repeat(num_resets, 1)  # (num_parts, num_resets, 4, 4)
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
            pose_idxs = np.tile(np.arange(4), (num_resets, 1))  # (num_resets, 4)
            pose_idxs = np.apply_along_axis(
                np.random.permutation, 1, pose_idxs
            )  # (num_resets, 4)
        self._target_leg_idx[env_ids] = torch.tensor(
            pose_idxs[:, 3], device=self.sim_device, dtype=torch.long
        )

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
            torch.eye(4, device=self.sim_device).unsqueeze(0).repeat(num_resets, 1, 1)
        )  # (num_resets, 4, 4)
        table_ori = torch.cat(
            [table_ori[:, 3:], table_ori[:, :3]], dim=-1
        )  # (num_resets, 4)
        tabletop_rot_mat = torch_jit_utils.quaternion_to_matrix(table_ori)
        tabletop_pose_mat[:, :3, :3] = tabletop_rot_mat
        tabletop_pose_mat[:, :3, 3] = sampled_table_state[:, :3]

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
        new_pos[0, :, :] = sampled_table_state[:, :3]
        ori_quat[0, :, :] = sampled_table_state[:, 3:7]
        new_pos[4, :, :] = sampled_leg_state[:, :3]
        ori_quat[4, :, :] = sampled_leg_state[:, 3:7]

        reset_pose = torch.cat(
            [new_pos, ori_quat], dim=-1
        )  # (num_parts, num_resets, 7)
        vel = torch.zeros(
            (len(self.furniture.parts), len(env_ids), 6),
            device=self.sim_device,
            dtype=reset_pose.dtype,
        )
        reset_state = torch.cat(
            [reset_pose, vel], dim=-1
        )  # (num_parts, num_resets, 13)

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
        ) = compute_insert_reward(
            self.reset_buf,
            self.progress_buf,
            self.states,
            self.max_episode_length,
            self._distance_reward,
            self._success_reward,
            self._pos_error_threshold,
            self._ori_error_threshold,
            self._leg_target_pos,
        )


class InsertFullPCDEnv(TRANSICEnvPCD):
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
        self._distance_reward = cfg["env"]["distanceReward"]
        self._success_reward = cfg["env"]["successReward"]

        self._leg_target_pos = None
        self._target_leg_idx = None

        self._pos_error_threshold = 0.007
        self._ori_error_threshold = 15 / 180 * np.pi

        try:
            import casadi
            import urdf_parser_py
        except ImportError:
            raise ImportError(
                "Packages `casadi` and `urdf-parser-py` are required for the env `InsertFullPCD`. Install them with `pip install casadi urdf-parser-py`."
            )
        from transic_envs.utils.urdf2casadi import URDFparser

        franka_parser = URDFparser()
        franka_parser.from_file(
            os.path.join(
                ASSET_ROOT,
                "franka_description/robots/franka_panda_finray.urdf",
            )
        )
        self._ftip_center_fk_fn = franka_parser.get_forward_kinematics(
            root="panda_link0", tip="tri_finger_center"
        )["T_fk"]

        self._selected_leg_idx = cfg["env"]["selectedLegIdx"]
        assert self._selected_leg_idx is not None and self._selected_leg_idx in [
            0,
            1,
            2,
            3,
        ]

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
        )

        self.franka_dof_lower_limits[-2] = 0.0145
        self.franka_dof_lower_limits[-1] = 0.0145

    def init_data(self):
        super().init_data()
        self._leg_target_pos = torch.zeros(
            (self.num_envs, 3), device=self.sim_device, dtype=torch.float32
        )
        self._target_leg_idx = torch.zeros(
            (self.num_envs,),
            device=self.sim_device,
            dtype=torch.long,
        )

    def _update_states(self):
        super()._update_states()

        # compute target assemble pose for leg
        tabletop_rot_quat = self.states["square_table_top_rot"]  # (N, 4) in xyzw order
        # change to wxyz order
        tabletop_rot_quat = torch.cat(
            [tabletop_rot_quat[:, -1:], tabletop_rot_quat[:, :-1]], dim=-1
        )
        tabletop_rot_mat = torch_jit_utils.quaternion_to_matrix(
            tabletop_rot_quat
        )  # (N, 3, 3)
        tabletop_pose_mat = (
            torch.eye(4, device=self.sim_device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1, 1)
        )
        tabletop_pose_mat[:, :3, :3] = tabletop_rot_mat
        tabletop_pose_mat[:, :3, 3] = self.states["square_table_top_pos"]  # (N, 3)

        all_possible_assemble_poses = torch.tensor(
            np.stack(self.all_possible_assemble_poses), device=self.sim_device
        )  # (4, 4, 4)
        selected_assemble_poses = all_possible_assemble_poses[
            self._target_leg_idx
        ]  # (N, 4, 4)

        furniture_assembled_pose = (
            tabletop_pose_mat @ selected_assemble_poses
        )  # (N, 4, 4)
        furniture_assembled_pos = furniture_assembled_pose[:, :3, 3]
        furniture_assembled_pos[:, 2] += 0.02
        self._leg_target_pos = furniture_assembled_pos
        self.states["target_xyz"] = furniture_assembled_pos[:, :3]

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

        # Randomize initial furniture part poses
        self.furniture.reset()

        # compute pose of finger tip center
        qs = self._q[env_ids, :7]
        ftip_center_poses = []
        for q in qs:
            q = list(q.cpu().numpy())
            ftip_center_poses.append(
                np.array(self._ftip_center_fk_fn(q)).astype(np.float32).reshape(4, 4)
            )
        ftip_center_poses = torch.tensor(
            np.stack(ftip_center_poses),
            device=self.sim_device,
            dtype=torch.float32,
        )  # (N, 4, 4)

        # Update leg pose
        num_resets = len(env_ids)

        leg2ftip_center_transform = torch.eye(
            4, device=self.sim_device, dtype=torch.float32
        )  # (4, 4)
        leg2ftip_center_transform[2, 3] = -0.02 if self.franka_dof_noise > 0 else -0.04

        leg2ftip_center_transform = leg2ftip_center_transform.unsqueeze(0).repeat(
            num_resets, 1, 1
        )
        leg_pose = leg2ftip_center_transform @ ftip_center_poses  # (N, 4, 4)
        leg_pos = leg_pose[:, :3, 3] + self._base_state[env_ids, :3]

        leg2ftip_center_rotation = torch_jit_utils.quat_from_euler_xyz(
            roll=torch.ones((1,), device=self.sim_device, dtype=torch.float32)
            * -np.pi
            / 2,
            pitch=torch.ones((1,), device=self.sim_device, dtype=torch.float32) * 0,
            yaw=torch.ones((1,), device=self.sim_device, dtype=torch.float32) * 0,
        )  # (1, 4) in xyzw order
        # # change to wxyz order
        leg2ftip_center_rotation = torch.cat(
            [leg2ftip_center_rotation[:, -1:], leg2ftip_center_rotation[:, :-1]], dim=-1
        )
        leg2ftip_center_rotation = torch_jit_utils.quaternion_to_matrix(
            leg2ftip_center_rotation
        )  # (1, 3, 3)
        leg_rot = leg2ftip_center_rotation @ leg_pose[:, :3, :3]  # (N, 3, 3)
        leg_rot = torch_jit_utils.matrix_to_quaternion(leg_rot)  # (N, 4) in wxyz order
        # change to xyzw order
        leg_rot = torch.cat([leg_rot[:, -1:], leg_rot[:, :-1]], dim=-1)

        sampled_leg_state = torch.zeros(num_resets, 13, device=self.sim_device)
        sampled_leg_state[:, :3] = leg_pos
        sampled_leg_state[:, 3:7] = leg_rot
        # Update table pose
        sampled_table_state = torch.zeros(num_resets, 13, device=self.sim_device)
        sampled_table_state[:, :7] = torch.tensor(
            [
                [
                    0.23,
                    0.05,
                    0.4307,
                    5.0000e-01,
                    -5.0000e-01,
                    -5.0000e-01,
                    5.0000e-01,
                ]
            ],
            dtype=sampled_table_state.dtype,
            device=sampled_table_state.device,
        ).repeat(num_resets, 1)
        # add noise to table xy
        # +- 2 cm
        table_xy_noise = (
            torch.rand((num_resets, 2), device=self.sim_device, dtype=torch.float32) * 2
            - 1
        ) * 0.02
        sampled_table_state[:, :2] += table_xy_noise

        table_ori_noise = torch.zeros((num_resets, 3), device=self.sim_device)
        # use rotation_noise < 45 to avoid ambiguity
        rotation_noise = 30
        table_ori_noise[:, 2] = (
            (torch.rand((num_resets,), device=self.sim_device) * 2 - 1)
            * rotation_noise
            / 180
            * np.pi
        )
        table_ori_noise = torch_jit_utils.axisangle2quat(table_ori_noise)
        table_ori = torch_jit_utils.quat_mul(
            table_ori_noise, sampled_table_state[:, 3:7]
        )
        sampled_table_state[:, 3:7] = table_ori

        # handle other furniture parts
        pos, ori = [], []
        for part in self.furniture.parts:
            pos.append(part.part_config["reset_pos"][0])  # (3,)
            ori.append(part.part_config["reset_ori"][0])  # (4,)
        pos = np.stack(pos)[:, np.newaxis, :]  # (num_parts, 1, 3)
        ori = np.stack(ori)[:, np.newaxis, ...]  # (num_parts, 1, 4, 4)
        pos = pos.repeat(num_resets, 1)  # (num_parts, num_resets, 3)
        ori = ori.repeat(num_resets, 1)  # (num_parts, num_resets, 4, 4)
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
        self._target_leg_idx[env_ids] = torch.tensor(
            pose_idxs[:, 3], device=self.sim_device, dtype=torch.long
        )

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
            torch.eye(4, device=self.sim_device).unsqueeze(0).repeat(num_resets, 1, 1)
        )  # (num_resets, 4, 4)
        table_ori = torch.cat(
            [table_ori[:, 3:], table_ori[:, :3]], dim=-1
        )  # (num_resets, 4)
        tabletop_rot_mat = torch_jit_utils.quaternion_to_matrix(table_ori)
        tabletop_pose_mat[:, :3, :3] = tabletop_rot_mat
        tabletop_pose_mat[:, :3, 3] = sampled_table_state[:, :3]

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
        new_pos[0, :, :] = sampled_table_state[:, :3]
        ori_quat[0, :, :] = sampled_table_state[:, 3:7]
        new_pos[4, :, :] = sampled_leg_state[:, :3]
        ori_quat[4, :, :] = sampled_leg_state[:, 3:7]

        reset_pose = torch.cat(
            [new_pos, ori_quat], dim=-1
        )  # (num_parts, num_resets, 7)
        vel = torch.zeros(
            (len(self.furniture.parts), len(env_ids), 6),
            device=self.sim_device,
            dtype=reset_pose.dtype,
        )
        reset_state = torch.cat(
            [reset_pose, vel], dim=-1
        )  # (num_parts, num_resets, 13)

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
        ) = compute_insert_reward(
            self.reset_buf,
            self.progress_buf,
            self.states,
            self.max_episode_length,
            self._distance_reward,
            self._success_reward,
            self._pos_error_threshold,
            self._ori_error_threshold,
            self._leg_target_pos,
        )


@torch.jit.script
def compute_insert_reward(
    reset_buf,
    progress_buf,
    states: dict[str, torch.Tensor],
    max_episode_length: int,
    distance_reward: float,
    success_reward: float,
    pos_error_threshold: float,
    ori_error_threshold: float,
    target_leg_pos,
):
    leg_pos = states["square_table_leg4_pos"]
    leg_rot = states["square_table_leg4_rot"]
    leg_rot_roll, leg_rot_pitch, _ = torch_jit_utils.get_euler_xyz(leg_rot)
    leg_rot_roll = torch_jit_utils.normalize_angle(leg_rot_roll)
    leg_rot_pitch = torch_jit_utils.normalize_angle(leg_rot_pitch)

    roll_diff = (90 / 180 * np.pi - leg_rot_roll).abs()
    pitch_diff = (0 - leg_rot_pitch).abs()
    roll_distance = torch.sin(roll_diff)
    pitch_distance = torch.sin(pitch_diff)

    pos_diff = (leg_pos - target_leg_pos).abs().sum(dim=-1)
    rot_diff = 0.5 * (roll_distance + pitch_distance)

    distance = (pos_diff + rot_diff) / 2
    dist_reward = 1 - torch.tanh(10.0 * distance)

    succeeded = (
        (pos_diff < pos_error_threshold)
        & (roll_diff < ori_error_threshold)
        & (pitch_diff < ori_error_threshold)
    )
    rewards = distance_reward * dist_reward + success_reward * succeeded

    failed = leg_pos[:, 2] < 0.44
    failed = torch.zeros_like(failed)

    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1) | succeeded | failed,
        torch.ones_like(reset_buf),
        reset_buf,
    )

    return rewards, reset_buf, succeeded, failed
