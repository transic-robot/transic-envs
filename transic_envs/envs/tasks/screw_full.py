from __future__ import annotations

from isaacgym import gymtorch
import torch
import numpy as np

from transic_envs.utils.pose_utils import rot_mat, get_mat
import transic_envs.utils.torch_jit_utils as torch_jit_utils
from transic_envs.envs.core import TRANSICEnvPCD, TRANSICEnvOSC


class ScrewFullEnv(TRANSICEnvOSC):
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
        try:
            import roboticstoolbox as rtb
            from spatialmath import SE3
        except ImportError:
            print(
                "roboticstoolbox is not installed. Please install it to use this env."
            )
        self._rtb_franka = rtb.models.Panda()

        # no DoF noise since we will set it on top of table leg through inverse kinematics
        cfg["env"]["frankaDofNoise"] = 0
        self._screw_reward = cfg["env"]["screwReward"]
        self._success_reward = cfg["env"]["successReward"]
        failure_penalty = cfg["env"]["failurePenalty"]
        assert failure_penalty >= 0
        self._failure_penalty = failure_penalty
        eef_deviate_penalty = cfg["env"]["eefDeviatePenalty"]
        assert eef_deviate_penalty >= 0
        self._eef_deviate_penalty = eef_deviate_penalty
        self._initial_q7_noise_level = cfg["env"]["initialQ7NoiseLevel"]
        self._initial_q1_to_q6_noise_level = cfg["env"]["initialQ1toQ6NoiseLevel"]

        self._prev_leg_pos, self._prev_leg_rot = None, None

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

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        num_resets = len(env_ids)

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
        rotation_noise = 45
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

        # Update leg pose
        pose_idxs = np.tile(np.arange(4), (num_resets, 1))  # (num_resets, 4)
        pose_idxs = np.apply_along_axis(
            np.random.permutation, 1, pose_idxs
        )  # (num_resets, 4)
        other_leg_pose_idxs = pose_idxs[:, :3]

        all_possible_assemble_poses = np.stack(
            self.all_possible_assemble_poses
        )  # (4, 4, 4)
        all_assemble_poses = all_possible_assemble_poses[np.newaxis, ...].repeat(
            len(env_ids), 0
        )  # (num_resets, 4, 4, 4)
        all_assemble_poses = all_assemble_poses[
            np.arange(len(env_ids))[:, None], pose_idxs, ...
        ]  # (num_resets, 4, 4, 4)
        all_assemble_poses = torch.tensor(
            all_assemble_poses, device=self.sim_device, dtype=torch.float32
        )
        all_assemble_poses = all_assemble_poses.reshape(
            -1, 4, 4
        )  # (num_resets * 4, 4, 4)
        # compute target leg pose
        tabletop_pose_mat = (
            torch.eye(4, device=self.sim_device).unsqueeze(0).repeat(num_resets, 1, 1)
        )  # (num_resets, 4, 4)
        # change to wxyz order
        table_ori = torch.cat(
            [table_ori[..., 3:], table_ori[..., :3]], dim=-1
        )  # (num_resets, 4)
        tabletop_rot_mat = torch_jit_utils.quaternion_to_matrix(table_ori)
        tabletop_pose_mat[:, :3, :3] = tabletop_rot_mat
        tabletop_pose_mat[:, :3, 3] = sampled_table_state[:, :3]
        tabletop_pose_mat = (
            tabletop_pose_mat.unsqueeze(1).repeat(1, 4, 1, 1).reshape(-1, 4, 4)
        )
        leg_assembled_pose = (
            tabletop_pose_mat @ all_assemble_poses
        )  # (num_resets * 4, 4, 4)
        leg_assemble_pose = leg_assembled_pose.reshape(
            num_resets, 4, 4, 4
        )  # (num_resets, 4, 4, 4)
        leg_assembled_pos = leg_assemble_pose[:, 3, :3, 3]  # (num_resets, 3)
        leg_assembled_pos[:, 2] += 0.02
        leg_assembled_rot = leg_assemble_pose[:, 3, :3, :3]  # (num_resets, 3, 3)
        leg_assembled_rot = torch_jit_utils.matrix_to_quaternion(leg_assembled_rot)
        leg_assembled_rot = torch.cat(
            [leg_assembled_rot[..., 3:], leg_assembled_rot[..., :3]], dim=-1
        )  # (num_resets, 4)
        sampled_leg_state = torch.zeros(num_resets, 13, device=self.sim_device)
        sampled_leg_state[:, :3] = leg_assembled_pos
        sampled_leg_state[:, 3:7] = leg_assembled_rot

        # Now compute franka pose through forward kinematics
        eef_pos = leg_assembled_pos - self._base_state[env_ids, :3]
        eef_pos = eef_pos.cpu().numpy()
        eef_pos[:, 2] += 0.1
        franka_qs_ik = []
        from spatialmath import SE3

        for goal_pos in eef_pos:
            Tep = SE3.Trans(goal_pos[0], goal_pos[1], goal_pos[2]) * SE3.RPY(
                180, 0, 0, unit="deg"
            )
            sol = self._rtb_franka.ik_LM(
                Tep, q0=self.franka_default_dof_pos[:7].cpu().numpy()
            )
            new_q = torch.tensor(sol[0], device=self.sim_device)
            franka_qs_ik.append(new_q)
        franka_qs_ik = torch.stack(franka_qs_ik, dim=0)
        pos = torch.concatenate(
            [
                franka_qs_ik,
                torch.ones((num_resets, 2), device=self.sim_device, dtype=torch.float32)
                * 0.035,
            ],
            dim=-1,
        ).to(dtype=torch.float32, device=self.sim_device)
        # add noise to joint 7
        joint_7_noise = (
            (torch.rand((num_resets,), device=self.sim_device) * 2 - 1)
            * self._initial_q7_noise_level
            * 2.8973
            * 2
        )
        q7 = pos[:, 6] + joint_7_noise
        q7 = torch.clamp(q7, -2.8973, 2.8973)
        pos[:, 6] = q7
        # add noise to joint 1 to 6
        q1_to_q6_noise = torch.rand((num_resets, 6), device=self.sim_device) * 2 - 1
        q1_to_q6_noise = (
            q1_to_q6_noise
            * (self.franka_dof_upper_limits[:6] - self.franka_dof_lower_limits[:6])
            * self._initial_q1_to_q6_noise_level
        )
        q1_to_q6 = pos[:, :6] + q1_to_q6_noise
        q1_to_q6 = torch.clamp(
            q1_to_q6,
            self.franka_dof_lower_limits[:6],
            self.franka_dof_upper_limits[:6],
        )
        pos[:, :6] = q1_to_q6

        # handle other legs
        other_legs_pos = np.array(self.all_legs_reset_pos)[
            other_leg_pose_idxs
        ]  # (num_resets, 3, 3)
        other_legs_ori = np.array(self.all_legs_reset_ori)[
            other_leg_pose_idxs
        ]  # (num_resets, 3, 4, 4)
        other_legs_pos = other_legs_pos.transpose(1, 0, 2)  # (3, num_resets, 3)
        other_legs_ori = other_legs_ori.transpose(1, 0, 2, 3)  # (3, num_resets, 4, 4)

        # randomize pos and ori
        other_legs_pos[:, :, :2] += np.random.uniform(
            -0.015, 0.015, size=(3, num_resets, 2)
        )
        other_legs_pos = torch.tensor(other_legs_pos, device=self.sim_device)
        # convert pos to homogenous matrix
        other_legs_pos_mat = (
            torch.eye(4, device=self.sim_device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(3, num_resets, 1, 1)
        )  # (3, num_resets, 4, 4)
        other_legs_pos_mat[:, :, :3, 3] = other_legs_pos
        other_legs_pos_mat = other_legs_pos_mat.reshape(-1, 4, 4)
        other_legs_pos_mat = (
            self.april_to_sim_mat @ other_legs_pos_mat
        )  # (4, 4) @ (3 * num_resets, 4, 4) -> (3 * num_resets, 4, 4)
        other_legs_pos_mat = other_legs_pos_mat.reshape(3, num_resets, 4, 4)
        other_legs_new_pos = other_legs_pos_mat[:, :, :3, 3]  # (3, num_resets, 3)
        other_legs_ori = torch.tensor(
            other_legs_ori, device=self.sim_device
        )  # (3, num_resets, 4, 4)
        other_legs_ori_noise = np.zeros((3, num_resets, 3))
        other_legs_ori_noise[:, :, 2] = np.random.uniform(
            np.radians(-15),
            np.radians(15),
            size=(3, num_resets),
        )
        other_legs_ori_noise = torch.tensor(
            other_legs_ori_noise,
            device=self.sim_device,
            dtype=torch.float,
        )
        other_legs_ori_noise = torch_jit_utils.axisangle2quat(
            other_legs_ori_noise
        )  # (3, num_resets, 4) in xyzw order
        # change to wxyz order
        other_legs_ori_noise = torch.cat(
            [other_legs_ori_noise[:, :, 3:], other_legs_ori_noise[:, :, :3]], dim=-1
        )
        other_legs_ori_noise = torch_jit_utils.quaternion_to_matrix(
            other_legs_ori_noise
        )  # (3, num_resets, 3, 3)
        # convert to homogeneous matrix
        other_legs_ori_noise_homo = (
            torch.eye(4, dtype=other_legs_ori_noise.dtype, device=self.sim_device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(3, num_resets, 1, 1)
        )  # (3, num_resets, 4, 4)
        other_legs_ori_noise_homo[:, :, :3, :3] = other_legs_ori_noise
        other_legs_ori_noise_homo[:, :, 3, 3] = 1
        other_legs_ori = other_legs_ori.reshape(-1, 4, 4)
        other_legs_ori_noise_homo = other_legs_ori_noise_homo.reshape(-1, 4, 4)
        other_legs_ori = (
            other_legs_ori_noise_homo @ other_legs_ori
        )  # (N, 4, 4) @ (N, 4, 4) -> (N, 4, 4)
        other_legs_ori = (
            self.april_to_sim_mat @ other_legs_ori
        )  # (4, 4) @ (3 * num_resets, 4, 4) -> (3 * num_resets, 4, 4)
        other_legs_ori_quat = torch_jit_utils.matrix_to_quaternion(
            other_legs_ori[:, :3, :3]
        )  # (3 * num_resets, 4) in wxyz order
        # convert to xyzw order
        other_legs_ori_quat = torch.cat(
            [other_legs_ori_quat[:, 1:], other_legs_ori_quat[:, :1]], dim=-1
        )
        other_legs_ori_quat = other_legs_ori_quat.reshape(3, num_resets, 4)

        # determine randomly assembled legs
        if_assembled_mask = (
            torch.rand(
                size=(num_resets * 3, 1),
                device=self.sim_device,
                dtype=torch.float32,
            )
            > 0.5
        )
        other_legs_old_pos = other_legs_new_pos.reshape(-1, 3)  # (num_resets * 3, 3)
        other_legs_old_ori = other_legs_ori_quat.reshape(-1, 4)  # (num_resets * 3, 4)
        leg_assemble_pos = leg_assemble_pose[:, :3, :3, 3].reshape(
            -1, 3
        )  # (num_resets * 3, 3)
        other_legs_new_pos = (
            if_assembled_mask * leg_assemble_pos
            + ~if_assembled_mask * other_legs_old_pos
        )
        leg_assemble_ori = leg_assemble_pose[:, :3, :3, :3].reshape(
            -1, 3, 3
        )  # (num_resets * 3, 3, 3)
        leg_assemble_ori = torch_jit_utils.matrix_to_quaternion(
            leg_assemble_ori
        )  # (num_resets * 3, 4)
        leg_assemble_ori = torch.cat(
            [leg_assemble_ori[..., 3:], leg_assemble_ori[..., :3]], dim=-1
        )  # (num_resets * 3, 4)
        other_legs_new_ori = (
            if_assembled_mask * leg_assemble_ori
            + ~if_assembled_mask * other_legs_old_ori
        )
        other_legs_new_pos = other_legs_new_pos.reshape(
            len(env_ids), 3, 3
        )  # (num_resets, 3, 3)
        other_legs_new_ori = other_legs_new_ori.reshape(
            len(env_ids), 3, 4
        )  # (num_resets, 3, 4)
        other_legs_new_pos = other_legs_new_pos.transpose(0, 1)  # (3, num_resets, 3)
        other_legs_new_ori = other_legs_new_ori.transpose(0, 1)  # (3, num_resets, 4)

        all_reset_pose = torch.zeros(
            len(self.furniture.parts), len(env_ids), 7, device=self.sim_device
        )
        all_reset_pose[0] = sampled_table_state[:, :7]
        all_reset_pose[1:4, :, :3] = other_legs_new_pos
        all_reset_pose[1:4, :, 3:7] = other_legs_new_ori
        all_reset_pose[4] = sampled_leg_state[:, :7]
        vel = torch.zeros(
            (len(self.furniture.parts), len(env_ids), 6),
            device=self.sim_device,
            dtype=all_reset_pose.dtype,
        )
        reset_state = torch.cat(
            [all_reset_pose, vel], dim=-1
        )  # (num_parts, num_resets, 13)

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

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0

    def _update_states(self):
        super()._update_states()
        leg_state = self._fparts_states["square_table_leg4"]
        leg_pos, leg_rot, leg_vel = (
            leg_state[:, :3],
            leg_state[:, 3:7],
            leg_state[:, 7:10],
        )
        self.states.update(
            {
                "leg_pos": leg_pos,
                "leg_rot": leg_rot,
                "leg_vel": leg_vel,
                "prev_leg_pos": self._prev_leg_pos
                if self._prev_leg_pos is not None
                else leg_pos,
                "prev_leg_rot": self._prev_leg_rot
                if self._prev_leg_rot is not None
                else leg_rot,
            }
        )
        self._prev_leg_pos, self._prev_leg_rot = leg_pos.clone(), leg_rot.clone()

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
            self.failure_buf[:],
        ) = compute_screw_reward(
            self.reset_buf,
            self.progress_buf,
            self.states,
            self.max_episode_length,
            self._eef_deviate_penalty,
            self._screw_reward,
            self._success_reward,
            self._failure_penalty,
        )


class ScrewFullPCDEnv(TRANSICEnvPCD):
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
        try:
            import roboticstoolbox as rtb
            from spatialmath import SE3
        except ImportError:
            print(
                "roboticstoolbox is not installed. Please install it to use this env."
            )
        self._rtb_franka = rtb.models.Panda()

        # no DoF noise since we will set it on top of table leg through inverse kinematics
        cfg["env"]["frankaDofNoise"] = 0

        self._screw_reward = cfg["env"]["screwReward"]
        self._success_reward = cfg["env"]["successReward"]
        failure_penalty = cfg["env"]["failurePenalty"]
        assert failure_penalty >= 0
        self._failure_penalty = failure_penalty
        eef_deviate_penalty = cfg["env"]["eefDeviatePenalty"]
        assert eef_deviate_penalty >= 0
        self._eef_deviate_penalty = eef_deviate_penalty
        self._initial_q7_noise_level = cfg["env"]["initialQ7NoiseLevel"]
        self._initial_q1_to_q6_noise_level = cfg["env"]["initialQ1toQ6NoiseLevel"]

        self._prev_leg_pos, self._prev_leg_rot = None, None

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

    def reset_idx(self, env_ids):
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        num_resets = len(env_ids)

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
        rotation_noise = 45
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

        # Update leg pose
        pose_idxs = np.tile(np.arange(4), (num_resets, 1))  # (num_resets, 4)
        pose_idxs = np.apply_along_axis(
            np.random.permutation, 1, pose_idxs
        )  # (num_resets, 4)
        other_leg_pose_idxs = pose_idxs[:, :3]

        all_possible_assemble_poses = np.stack(
            self.all_possible_assemble_poses
        )  # (4, 4, 4)
        all_assemble_poses = all_possible_assemble_poses[np.newaxis, ...].repeat(
            len(env_ids), 0
        )  # (num_resets, 4, 4, 4)
        all_assemble_poses = all_assemble_poses[
            np.arange(len(env_ids))[:, None], pose_idxs, ...
        ]  # (num_resets, 4, 4, 4)
        all_assemble_poses = torch.tensor(
            all_assemble_poses, device=self.sim_device, dtype=torch.float32
        )
        all_assemble_poses = all_assemble_poses.reshape(
            -1, 4, 4
        )  # (num_resets * 4, 4, 4)
        # compute target leg pose
        tabletop_pose_mat = (
            torch.eye(4, device=self.sim_device).unsqueeze(0).repeat(num_resets, 1, 1)
        )  # (num_resets, 4, 4)
        # change to wxyz order
        table_ori = torch.cat(
            [table_ori[..., 3:], table_ori[..., :3]], dim=-1
        )  # (num_resets, 4)
        tabletop_rot_mat = torch_jit_utils.quaternion_to_matrix(table_ori)
        tabletop_pose_mat[:, :3, :3] = tabletop_rot_mat
        tabletop_pose_mat[:, :3, 3] = sampled_table_state[:, :3]
        tabletop_pose_mat = (
            tabletop_pose_mat.unsqueeze(1).repeat(1, 4, 1, 1).reshape(-1, 4, 4)
        )
        leg_assembled_pose = (
            tabletop_pose_mat @ all_assemble_poses
        )  # (num_resets * 4, 4, 4)
        leg_assemble_pose = leg_assembled_pose.reshape(
            num_resets, 4, 4, 4
        )  # (num_resets, 4, 4, 4)
        leg_assembled_pos = leg_assemble_pose[:, 3, :3, 3]  # (num_resets, 3)
        leg_assembled_pos[:, 2] += 0.02
        leg_assembled_rot = leg_assemble_pose[:, 3, :3, :3]  # (num_resets, 3, 3)
        leg_assembled_rot = torch_jit_utils.matrix_to_quaternion(leg_assembled_rot)
        leg_assembled_rot = torch.cat(
            [leg_assembled_rot[..., 3:], leg_assembled_rot[..., :3]], dim=-1
        )  # (num_resets, 4)
        sampled_leg_state = torch.zeros(num_resets, 13, device=self.sim_device)
        sampled_leg_state[:, :3] = leg_assembled_pos
        sampled_leg_state[:, 3:7] = leg_assembled_rot

        # Now compute franka pose through forward kinematics
        eef_pos = leg_assembled_pos - self._base_state[env_ids, :3]
        eef_pos = eef_pos.cpu().numpy()
        eef_pos[:, 2] += 0.1
        franka_qs_ik = []
        from spatialmath import SE3

        for goal_pos in eef_pos:
            Tep = SE3.Trans(goal_pos[0], goal_pos[1], goal_pos[2]) * SE3.RPY(
                180, 0, 0, unit="deg"
            )
            sol = self._rtb_franka.ik_LM(
                Tep, q0=self.franka_default_dof_pos[:7].cpu().numpy()
            )
            new_q = torch.tensor(sol[0], device=self.sim_device)
            franka_qs_ik.append(new_q)
        franka_qs_ik = torch.stack(franka_qs_ik, dim=0)
        q_gripper = 0.035
        pos = torch.concatenate(
            [
                franka_qs_ik,
                torch.ones((num_resets, 2), device=self.sim_device, dtype=torch.float32)
                * q_gripper,
            ],
            dim=-1,
        ).to(dtype=torch.float32, device=self.sim_device)
        # add noise to joint 7
        joint_7_noise = (
            (torch.rand((num_resets,), device=self.sim_device) * 2 - 1)
            * self._initial_q7_noise_level
            * 2.8973
            * 2
        )
        q7 = pos[:, 6] + joint_7_noise
        q7 = torch.clamp(q7, -2.8973, 2.8973)
        pos[:, 6] = q7
        # add noise to joint 1 to 6
        q1_to_q6_noise = torch.rand((num_resets, 6), device=self.sim_device) * 2 - 1
        q1_to_q6_noise = (
            q1_to_q6_noise
            * (self.franka_dof_upper_limits[:6] - self.franka_dof_lower_limits[:6])
            * self._initial_q1_to_q6_noise_level
        )
        q1_to_q6 = pos[:, :6] + q1_to_q6_noise
        q1_to_q6 = torch.clamp(
            q1_to_q6,
            self.franka_dof_lower_limits[:6],
            self.franka_dof_upper_limits[:6],
        )
        pos[:, :6] = q1_to_q6

        # handle other legs
        other_legs_pos = np.array(self.all_legs_reset_pos)[
            other_leg_pose_idxs
        ]  # (num_resets, 3, 3)
        other_legs_ori = np.array(self.all_legs_reset_ori)[
            other_leg_pose_idxs
        ]  # (num_resets, 3, 4, 4)
        other_legs_pos = other_legs_pos.transpose(1, 0, 2)  # (3, num_resets, 3)
        other_legs_ori = other_legs_ori.transpose(1, 0, 2, 3)  # (3, num_resets, 4, 4)

        # randomize pos and ori
        other_legs_pos[:, :, :2] += np.random.uniform(
            -0.015, 0.015, size=(3, num_resets, 2)
        )
        other_legs_pos = torch.tensor(other_legs_pos, device=self.sim_device)
        # convert pos to homogenous matrix
        other_legs_pos_mat = (
            torch.eye(4, device=self.sim_device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(3, num_resets, 1, 1)
        )  # (3, num_resets, 4, 4)
        other_legs_pos_mat[:, :, :3, 3] = other_legs_pos
        other_legs_pos_mat = other_legs_pos_mat.reshape(-1, 4, 4)
        other_legs_pos_mat = (
            self.april_to_sim_mat @ other_legs_pos_mat
        )  # (4, 4) @ (3 * num_resets, 4, 4) -> (3 * num_resets, 4, 4)
        other_legs_pos_mat = other_legs_pos_mat.reshape(3, num_resets, 4, 4)
        other_legs_new_pos = other_legs_pos_mat[:, :, :3, 3]  # (3, num_resets, 3)
        other_legs_ori = torch.tensor(
            other_legs_ori, device=self.sim_device
        )  # (3, num_resets, 4, 4)
        other_legs_ori_noise = np.zeros((3, num_resets, 3))
        other_legs_ori_noise[:, :, 2] = np.random.uniform(
            np.radians(-15),
            np.radians(15),
            size=(3, num_resets),
        )
        other_legs_ori_noise = torch.tensor(
            other_legs_ori_noise,
            device=self.sim_device,
            dtype=torch.float,
        )
        other_legs_ori_noise = torch_jit_utils.axisangle2quat(
            other_legs_ori_noise
        )  # (3, num_resets, 4) in xyzw order
        # change to wxyz order
        other_legs_ori_noise = torch.cat(
            [other_legs_ori_noise[:, :, 3:], other_legs_ori_noise[:, :, :3]], dim=-1
        )
        other_legs_ori_noise = torch_jit_utils.quaternion_to_matrix(
            other_legs_ori_noise
        )  # (3, num_resets, 3, 3)
        # convert to homogeneous matrix
        other_legs_ori_noise_homo = (
            torch.eye(4, dtype=other_legs_ori_noise.dtype, device=self.sim_device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(3, num_resets, 1, 1)
        )  # (3, num_resets, 4, 4)
        other_legs_ori_noise_homo[:, :, :3, :3] = other_legs_ori_noise
        other_legs_ori_noise_homo[:, :, 3, 3] = 1
        other_legs_ori = other_legs_ori.reshape(-1, 4, 4)
        other_legs_ori_noise_homo = other_legs_ori_noise_homo.reshape(-1, 4, 4)
        other_legs_ori = (
            other_legs_ori_noise_homo @ other_legs_ori
        )  # (N, 4, 4) @ (N, 4, 4) -> (N, 4, 4)
        other_legs_ori = (
            self.april_to_sim_mat @ other_legs_ori
        )  # (4, 4) @ (3 * num_resets, 4, 4) -> (3 * num_resets, 4, 4)
        other_legs_ori_quat = torch_jit_utils.matrix_to_quaternion(
            other_legs_ori[:, :3, :3]
        )  # (3 * num_resets, 4) in wxyz order
        # convert to xyzw order
        other_legs_ori_quat = torch.cat(
            [other_legs_ori_quat[:, 1:], other_legs_ori_quat[:, :1]], dim=-1
        )
        other_legs_ori_quat = other_legs_ori_quat.reshape(3, num_resets, 4)

        # determine randomly assembled legs
        if_assembled_mask = (
            torch.rand(
                size=(num_resets * 3, 1),
                device=self.sim_device,
                dtype=torch.float32,
            )
            > 0.5
        )
        other_legs_old_pos = other_legs_new_pos.reshape(-1, 3)  # (num_resets * 3, 3)
        other_legs_old_ori = other_legs_ori_quat.reshape(-1, 4)  # (num_resets * 3, 4)
        leg_assemble_pos = leg_assemble_pose[:, :3, :3, 3].reshape(
            -1, 3
        )  # (num_resets * 3, 3)
        other_legs_new_pos = (
            if_assembled_mask * leg_assemble_pos
            + ~if_assembled_mask * other_legs_old_pos
        )
        leg_assemble_ori = leg_assemble_pose[:, :3, :3, :3].reshape(
            -1, 3, 3
        )  # (num_resets * 3, 3, 3)
        leg_assemble_ori = torch_jit_utils.matrix_to_quaternion(
            leg_assemble_ori
        )  # (num_resets * 3, 4)
        leg_assemble_ori = torch.cat(
            [leg_assemble_ori[..., 3:], leg_assemble_ori[..., :3]], dim=-1
        )  # (num_resets * 3, 4)
        other_legs_new_ori = (
            if_assembled_mask * leg_assemble_ori
            + ~if_assembled_mask * other_legs_old_ori
        )
        other_legs_new_pos = other_legs_new_pos.reshape(
            len(env_ids), 3, 3
        )  # (num_resets, 3, 3)
        other_legs_new_ori = other_legs_new_ori.reshape(
            len(env_ids), 3, 4
        )  # (num_resets, 3, 4)
        other_legs_new_pos = other_legs_new_pos.transpose(0, 1)  # (3, num_resets, 3)
        other_legs_new_ori = other_legs_new_ori.transpose(0, 1)  # (3, num_resets, 4)

        all_reset_pose = torch.zeros(
            len(self.furniture.parts), len(env_ids), 7, device=self.sim_device
        )
        all_reset_pose[0] = sampled_table_state[:, :7]
        all_reset_pose[1:4, :, :3] = other_legs_new_pos
        all_reset_pose[1:4, :, 3:7] = other_legs_new_ori
        all_reset_pose[4] = sampled_leg_state[:, :7]
        vel = torch.zeros(
            (len(self.furniture.parts), len(env_ids), 6),
            device=self.sim_device,
            dtype=all_reset_pose.dtype,
        )
        reset_state = torch.cat(
            [all_reset_pose, vel], dim=-1
        )  # (num_parts, num_resets, 13)

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

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0

    def _update_states(self):
        super()._update_states()
        leg_state = self._fparts_states["square_table_leg4"]
        leg_pos, leg_rot, leg_vel = (
            leg_state[:, :3],
            leg_state[:, 3:7],
            leg_state[:, 7:10],
        )
        self.states.update(
            {
                "leg_pos": leg_pos,
                "leg_rot": leg_rot,
                "leg_vel": leg_vel,
                "prev_leg_pos": self._prev_leg_pos
                if self._prev_leg_pos is not None
                else leg_pos,
                "prev_leg_rot": self._prev_leg_rot
                if self._prev_leg_rot is not None
                else leg_rot,
            }
        )
        self._prev_leg_pos, self._prev_leg_rot = leg_pos.clone(), leg_rot.clone()

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
            self.failure_buf[:],
        ) = compute_screw_reward(
            self.reset_buf,
            self.progress_buf,
            self.states,
            self.max_episode_length,
            self._eef_deviate_penalty,
            self._screw_reward,
            self._success_reward,
            self._failure_penalty,
        )


@torch.jit.script
def compute_screw_reward(
    reset_buf,
    progress_buf,
    states: dict[str, torch.Tensor],
    max_episode_length: int,
    eef_deviate_penalty: float,
    skew_reward: float,
    success_reward: float,
    failure_penalty: float,
):
    eef_rot = states["eef_quat"]
    eef_x_angle, eef_y_angle, _ = torch_jit_utils.get_euler_xyz(eef_rot)
    eef_x_angle = torch_jit_utils.normalize_angle(eef_x_angle)
    eef_y_angle = torch_jit_utils.normalize_angle(eef_y_angle)

    eef_x_angle_distance = (
        torch.ones_like(eef_x_angle) * np.pi - eef_x_angle.abs()
    ).abs()
    eef_y_angle_distance = eef_y_angle.abs()
    eef_angle_distance = (eef_x_angle_distance + eef_y_angle_distance) / 2
    eef_dist_reward = torch.tanh(eef_angle_distance)

    leg_pos = states["leg_pos"]
    leg_rot = states["leg_rot"]  # (N, 4) in xyzw order
    x_angle, y_angle, z_angle = torch_jit_utils.get_euler_xyz(leg_rot)
    x_angle = torch_jit_utils.normalize_angle(x_angle)
    y_angle = torch_jit_utils.normalize_angle(y_angle)
    z_angle = torch_jit_utils.normalize_angle(z_angle)
    x_angle = x_angle * 180 / 3.1415926
    y_angle = y_angle * 180 / 3.1415926
    vertical_mask = ((x_angle - 90).abs() < 10) & ((y_angle - 0).abs() < 10)

    prev_leg_pos, prev_leg_rot = states["prev_leg_pos"], states["prev_leg_rot"]
    prev_z_angle = torch_jit_utils.get_euler_xyz(prev_leg_rot)[2]
    prev_z_angle = torch_jit_utils.normalize_angle(prev_z_angle)

    failed = (leg_pos[:, 2] < 0.44) | ~vertical_mask | (leg_pos[:, 2] > 0.499)

    d_rot = (-np.pi - z_angle).abs()
    correct_rot_direction = (prev_z_angle < 0) & (z_angle < 0)
    succeeded = (d_rot < (5 * np.pi / 180)) & correct_rot_direction

    dist_reward = 1 - torch.tanh(d_rot)
    rewards = (
        (skew_reward * dist_reward + success_reward * succeeded) * (~failed)
        - eef_deviate_penalty * eef_dist_reward
        - failure_penalty * failed
    )
    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1) | failed | succeeded,
        torch.ones_like(reset_buf),
        reset_buf,
    )

    return rewards, reset_buf, succeeded, failed
