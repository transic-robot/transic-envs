from typing import Dict

from isaacgym import gymapi, gymtorch
import torch
import numpy as np

from transic_envs.asset_root import ASSET_ROOT
import transic_envs.utils.torch_jit_utils as torch_jit_utils
from transic_envs.envs.core.furniture import SquareTable
from transic_envs.envs.core import TRANSICEnvPCD, TRANSICEnvOSCBase
from transic_envs.envs.core.sim_config import sim_config
from transic_envs.utils.pose_utils import rot_mat, get_mat


class LiftLeanedLegEnv(TRANSICEnvOSCBase):
    initial_height = 0.03393867611885071

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
            raise ImportError(
                "roboticstoolbox is not installed. Please install it to use this env by `pip3 install roboticstoolbox-python`."
            )
        self._rtb_franka = rtb.models.Panda()

        self._success_weight = cfg["env"]["successWeight"]
        self._failure_weight = cfg["env"]["failureWeight"]
        assert self._failure_weight >= 0, "failure weight should be non-negative"
        self._target_lift_height = cfg["env"]["targetLiftHeight"]
        self._distance_reward = cfg["env"]["distanceReward"]
        self._progress_reward = cfg["env"]["progressReward"]
        self._success_eef_tilt_threshold = cfg["env"]["successEEFTiltThreshold"]
        self._dq_penalty = cfg["env"]["dqPenalty"]
        assert self._dq_penalty >= 0, "dq penalty should be non-negative"
        self._eef_tilt_penalty = cfg["env"]["eefTiltPenalty"]
        assert self._eef_tilt_penalty >= 0, "eef tilt penalty should be non-negative"
        self._target_obj_v_penalty = cfg["env"]["targetObjVPenalty"]
        assert (
            self._target_obj_v_penalty >= 0
        ), "target obj v penalty should be non-negative"

        self._task_progress_buf = None

        self.furniture = SquareTable(cfg["seed"])

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
            num_furniture_parts=len(self.furniture.parts),
        )

    def pre_physics_step(self, actions):
        # wait few steps until the leg is stable
        actions[self.progress_buf <= 15] = 0
        super().pre_physics_step(actions)

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
            asset_option = sim_config["asset"][part.name]
            self._fparts_assets[part.name] = self.gym.load_asset(
                self.sim,
                ASSET_ROOT,
                part.asset_file,
                asset_option,
            )

    def _create_furniture_parts(self, env_prt, i):
        for part in self.furniture.parts:
            actor = self.gym.create_actor(
                env_prt,
                self._fparts_assets[part.name],
                gymapi.Transform(),
                part.name,
                i,
                0,
            )
            # Set properties of part
            part_props = self.gym.get_actor_rigid_shape_properties(env_prt, actor)
            part_props[0].friction = sim_config["parts"]["friction"]
            self.gym.set_actor_rigid_shape_properties(env_prt, actor, part_props)
            if part.name not in self._fparts_names:
                self._fparts_names.append(part.name)

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
        pose_idxs = np.tile(np.arange(4), (len(env_ids), 1))  # (num_resets, 4)
        pose_idxs = np.apply_along_axis(
            np.random.permutation, 1, pose_idxs
        )  # (num_resets, 4)
        pose_idxs = pose_idxs[:, :3]  # (num_resets, 3)
        three_legs_pos = np.array(self.all_legs_reset_pos)[
            pose_idxs
        ]  # (num_resets, 3, 3)
        three_legs_ori = np.array(self.all_legs_reset_ori)[
            pose_idxs
        ]  # (num_resets, 3, 4, 4)
        three_legs_pos = three_legs_pos.transpose(1, 0, 2)  # (3, num_resets, 3)
        three_legs_ori = three_legs_ori.transpose(1, 0, 2, 3)  # (3, num_resets, 4, 4)
        pos[1:4, :, :] = three_legs_pos
        ori[1:4, :, :, :] = three_legs_ori

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
        # for the target leg, make it orthogonal to the left wall
        clockwise_rotate_idxs = np.random.randint(
            2,
            size=(len(env_ids)),
        ).astype(bool)
        ori_noise[4, clockwise_rotate_idxs, 2] = np.random.uniform(
            np.radians(60),
            np.radians(120),
            size=(len(env_ids)),
        )[clockwise_rotate_idxs]
        ori_noise[4, ~clockwise_rotate_idxs, 2] = np.random.uniform(
            np.radians(-120),
            np.radians(-60),
            size=(len(env_ids)),
        )[~clockwise_rotate_idxs]
        init_eef_rz = ori_noise[4, :, 2].copy()

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

        # lean the target leg on the left wall
        # uniformly sample x from [0.2315, 0.3065]
        new_pos[4, :, 0] = (
            torch.rand((len(env_ids),), dtype=new_pos.dtype, device=new_pos.device)
            * (0.3065 - 0.2315)
            + 0.2315
        )
        new_pos[4, :, 1] = -0.175
        new_pos[4, :, 2] += 0.015

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
        all_assemble_poses = all_assemble_poses[
            np.arange(len(env_ids))[:, None], pose_idxs, :, :
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
        target_leg_pos = reset_pos[4, :, :3] - self._base_state[env_ids, :3]
        target_leg_pos[:, 2] += (
            torch.rand((len(env_ids),), device=self.sim_device) * 0.05 + 0.05
        )
        target_leg_pos = target_leg_pos.cpu().numpy()

        from spatialmath import SE3

        franka_qs_ik = []
        for each_pos, each_init_eef_rz in zip(target_leg_pos, init_eef_rz):
            Tep = (
                SE3.Trans(each_pos[0], each_pos[1], each_pos[2])
                * SE3.Rx(np.pi)
                * SE3.Ry(0)
                * SE3.Rz(each_init_eef_rz)
                * SE3.RPY(0, -np.pi / 4, 0, unit="rad")
            )
            sol = self._rtb_franka.ik_LM(
                Tep,
                q0=self.franka_default_dof_pos[:7].cpu().numpy(),
            )
            new_q = torch.tensor(sol[0], device=self.sim_device)
            franka_qs_ik.append(new_q)
        franka_qs_ik = torch.stack(franka_qs_ik, dim=0)
        # add initial franka dof noise
        franka_qs_ik = torch_jit_utils.tensor_clamp(
            franka_qs_ik
            + self.franka_dof_noise
            * 2.0
            * (torch.rand((len(env_ids), 7), device=self.sim_device) - 0.5),
            self.franka_dof_lower_limits[:-2].unsqueeze(0),
            self.franka_dof_upper_limits[:-2],
        )

        pos = torch.zeros((len(env_ids), 9), device=self.sim_device)
        pos[:, :7] = franka_qs_ik
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
            states=self.states,
            init_fparts_states=self._init_fparts_states,
            base_states=self._base_state,
            initial_height=self.initial_height,
            target_lift_height=self._target_lift_height,
            distance_reward_weight=self._distance_reward,
            max_episode_length=self.max_episode_length,
            success_reward_weight=self._success_weight,
            eef_tilt_penalty=self._eef_tilt_penalty,
            dq_penalty=self._dq_penalty,
            target_obj_v_penalty=self._target_obj_v_penalty,
            eef_tilt_threshold=float(
                np.cos(np.deg2rad(90 - self._success_eef_tilt_threshold))
            ),
        )


class LiftLeanedLegPCDEnv(TRANSICEnvPCD):
    initial_height = 0.03393867611885071

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
            raise ImportError(
                "roboticstoolbox is not installed. Please install it to use this env by `pip3 install roboticstoolbox-python`."
            )
        self._rtb_franka = rtb.models.Panda()

        self._success_weight = cfg["env"]["successWeight"]
        self._target_lift_height = cfg["env"]["targetLiftHeight"]
        self._success_eef_tilt_threshold = cfg["env"]["successEEFTiltThreshold"]
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

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0

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
        pose_idxs = np.tile(np.arange(4), (len(env_ids), 1))  # (num_resets, 4)
        pose_idxs = np.apply_along_axis(
            np.random.permutation, 1, pose_idxs
        )  # (num_resets, 4)
        pose_idxs = pose_idxs[:, :3]  # (num_resets, 3)
        three_legs_pos = np.array(self.all_legs_reset_pos)[
            pose_idxs
        ]  # (num_resets, 3, 3)
        three_legs_ori = np.array(self.all_legs_reset_ori)[
            pose_idxs
        ]  # (num_resets, 3, 4, 4)
        three_legs_pos = three_legs_pos.transpose(1, 0, 2)  # (3, num_resets, 3)
        three_legs_ori = three_legs_ori.transpose(1, 0, 2, 3)  # (3, num_resets, 4, 4)
        pos[1:4, :, :] = three_legs_pos
        ori[1:4, :, :, :] = three_legs_ori

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
        # for the target leg, make it orthogonal to the left wall
        clockwise_rotate_idxs = np.random.randint(
            2,
            size=(len(env_ids)),
        ).astype(bool)
        ori_noise[4, clockwise_rotate_idxs, 2] = np.random.uniform(
            np.radians(60),
            np.radians(120),
            size=(len(env_ids)),
        )[clockwise_rotate_idxs]
        ori_noise[4, ~clockwise_rotate_idxs, 2] = np.random.uniform(
            np.radians(-120),
            np.radians(-60),
            size=(len(env_ids)),
        )[~clockwise_rotate_idxs]
        init_eef_rz = ori_noise[4, :, 2].copy()

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

        # lean the target leg on the left wall
        # uniformly sample x from [0.2315, 0.3065]
        new_pos[4, :, 0] = (
            torch.rand((len(env_ids),), dtype=new_pos.dtype, device=new_pos.device)
            * (0.3065 - 0.2315)
            + 0.2315
        )
        new_pos[4, :, 1] = -0.175
        new_pos[4, :, 2] += 0.015

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
        all_assemble_poses = all_assemble_poses[
            np.arange(len(env_ids))[:, None], pose_idxs, :, :
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
        target_leg_pos = reset_pos[4, :, :3] - self._base_state[env_ids, :3]
        target_leg_pos[:, 2] += (
            torch.rand((len(env_ids),), device=self.sim_device) * 0.05
            + 0.05  # EEF is 5~10cm above the target leg
        )
        target_leg_pos = target_leg_pos.cpu().numpy()

        from spatialmath import SE3

        franka_qs_ik = []
        for each_pos, each_init_eef_rz in zip(target_leg_pos, init_eef_rz):
            Tep = (
                SE3.Trans(each_pos[0], each_pos[1], each_pos[2])
                * SE3.Rx(np.pi)
                * SE3.Ry(0)
                * SE3.Rz(each_init_eef_rz)
                * SE3.RPY(0, -np.pi / 4, 0, unit="rad")
            )
            sol = self._rtb_franka.ik_LM(
                Tep,
                q0=self.franka_default_dof_pos[:7].cpu().numpy(),
            )
            new_q = torch.tensor(sol[0], device=self.sim_device)
            franka_qs_ik.append(new_q)
        franka_qs_ik = torch.stack(franka_qs_ik, dim=0)
        # add initial franka dof noise
        franka_qs_ik = torch_jit_utils.tensor_clamp(
            franka_qs_ik
            + self.franka_dof_noise
            * 2.0
            * (torch.rand((len(env_ids), 7), device=self.sim_device) - 0.5),
            self.franka_dof_lower_limits[:-2].unsqueeze(0),
            self.franka_dof_upper_limits[:-2],
        )

        pos = torch.zeros((len(env_ids), 9), device=self.sim_device)
        pos[:, :7] = franka_qs_ik
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
            _,
        ) = compute_reward(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            success_buf=self.success_buf,
            failure_buf=self.failure_buf,
            task_progress_buf=torch.zeros_like(self.rew_buf),
            states=self.states,
            init_fparts_states=self._init_fparts_states,
            base_states=self._base_state,
            initial_height=self.initial_height,
            target_lift_height=self._target_lift_height,
            distance_reward_weight=0,
            max_episode_length=self.max_episode_length,
            success_reward_weight=self._success_weight,
            eef_tilt_penalty=0,
            dq_penalty=0,
            target_obj_v_penalty=0,
            eef_tilt_threshold=float(
                np.cos(np.deg2rad(90 - self._success_eef_tilt_threshold))
            ),
        )


@torch.jit.script
def compute_reward(
    *,
    reset_buf,
    progress_buf,
    success_buf,
    failure_buf,
    task_progress_buf,
    states: Dict[str, torch.Tensor],
    init_fparts_states: Dict[str, torch.Tensor],
    base_states,
    initial_height: float,
    target_lift_height: float,
    distance_reward_weight: float,
    max_episode_length: int,
    success_reward_weight: float,
    eef_tilt_penalty: float,
    dq_penalty: float,
    target_obj_v_penalty: float,
    eef_tilt_threshold: float,
):
    target_leg_pos = states["square_table_leg4_pos"]
    fintergip_center_pos = states["ftip_center_pos"]
    eef_lf_pos = states["eef_lf_pos"]
    eef_rf_pos = states["eef_rf_pos"]

    # distance from hand to the leg
    d = torch.norm(target_leg_pos - fintergip_center_pos, dim=-1)
    d_lf = torch.norm(target_leg_pos - eef_lf_pos, dim=-1)
    d_rf = torch.norm(target_leg_pos - eef_rf_pos, dim=-1)
    distance = (d + d_lf + d_rf) / 3
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

    target_obj_v_norm = torch.norm(states["square_table_leg4_vel"], dim=-1)

    target_leg_rot = states["square_table_leg4_rot"]
    eef_rot = states["eef_quat"]
    rot_diff_rad = torch_jit_utils.quat_diff_rad(target_leg_rot, eef_rot)
    eef_tilt = torch.cos(rot_diff_rad).abs()

    succeeded = (lifted_height > target_lift_height) & (eef_tilt <= eef_tilt_threshold)

    # a task is failed if the tabletop is moved by 2cm
    init_tabletop_xy = init_fparts_states["square_table_top"][..., :2]
    curr_tabletop_xy = states["square_table_top_pos"][..., :2] + base_states[..., :2]
    displacement = torch.norm(init_tabletop_xy - curr_tabletop_xy, dim=-1)
    failure = displacement > 0.02
    failure = torch.zeros_like(failure)

    distance_reward = distance_reward_weight * dist_reward
    success_reward = success_reward_weight * succeeded
    dq_penalty = dq_penalty * dq_norm
    target_obj_v_penalty = target_obj_v_penalty * target_obj_v_norm
    eef_tilt_penalty = eef_tilt_penalty * eef_tilt
    reward = (
        distance_reward
        + success_reward
        - dq_penalty
        - target_obj_v_penalty
        - eef_tilt_penalty
    )

    success = succeeded | success_buf
    failure = failure | failure_buf

    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1)
        | ((states["square_table_leg4_pos"][:, 2] - 0.012).abs() < 0.005)
        | success
        | failure,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    return reward, reset_buf, success, failure, new_normalized_task_progress
