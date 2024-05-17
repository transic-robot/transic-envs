from typing import Dict

import os

import isaacgym
from isaacgym import gymapi
import torch
import numpy as np
from gym import spaces

from transic_envs.asset_root import ASSET_ROOT
import transic_envs.utils.fb_control_utils as C
from transic_envs.envs.core.base import TRANSICEnvJPC
from transic_envs.utils.pointcloud_visualizer import PointCloudVisualizer


class TRANSICEnvPCD(TRANSICEnvJPC):
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
        self._pcd_mask_ratio = cfg["env"]["pcdMaskRatio"]
        self._pcd_N = cfg["env"]["pcdN"]
        assert (self._pcd_mask_ratio is not None or self._pcd_N is not None) and not (
            self._pcd_mask_ratio is not None and self._pcd_N is not None
        ), "only one of pointcloud_mask_ratio and pointcloud_N should be provided"
        self._sampled_points = None

        # point cloud augmentation
        self._pc_augmentation_enabled = cfg["env"]["pcAugmentation"]["enabled"]
        self._pc_aug_apply_p = cfg["env"]["pcAugmentation"]["applyP"]
        self._pc_aug_random_trans_low = torch.tensor(
            cfg["env"]["pcAugmentation"]["randomTransLow"], device=sim_device
        ).view(1, 1, 3)
        self._pc_aug_random_trans_high = torch.tensor(
            cfg["env"]["pcAugmentation"]["randomTransHigh"], device=sim_device
        ).view(1, 1, 3)
        self._pc_aug_jitter_ratio = cfg["env"]["pcAugmentation"]["jitterRatio"]
        self._pc_aug_jitter_sigma = cfg["env"]["pcAugmentation"]["jitterSigma"]
        self._pc_aug_jitter_dist = None
        self._pc_aug_jitter_high = cfg["env"]["pcAugmentation"]["jitterHigh"]
        self._pc_aug_jitter_low = cfg["env"]["pcAugmentation"]["jitterLow"]

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
        )
        obs_space_dict = self.obs_space.spaces
        obs_space_dict.update(
            {
                "pcd/coordinate": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.sampled_points, 3),
                ),
                "pcd/ee_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.sampled_points,),
                    dtype=bool,
                ),
            }
        )
        self.obs_space = spaces.Dict(obs_space_dict)

    def enable_pointcloud_augmentation(self):
        self._pc_augmentation_enabled = True

    def disable_pointcloud_augmentation(self):
        self._pc_augmentation_enabled = False

    @torch.no_grad()
    def recover_pcd_from_offline_data(
        self,
        *,
        state_dict: Dict[str, torch.Tensor],
    ):
        pcds_full, ee_mask_full = [], []

        furniture_pcds = []
        for part_name, part_pcd in self._fparts_pcds.items():
            part_poses = state_dict[part_name]  # (B, 7)
            part_pos, part_quat = part_poses[:, :3], part_poses[:, 3:]
            # quaternion_to_matrix assumes real part first
            part_quat = part_quat[..., [3, 0, 1, 2]]
            part_tf = C.batched_pose2mat(
                part_pos, part_quat, device=self.sim_device
            )  # (B, 4, 4)
            part_pcd_transformed = part_tf @ part_pcd.T  # (B, 4, n_points)
            part_pcd_transformed = part_pcd_transformed.transpose(1, 2)[
                :, :, :3
            ]  # (B, n_points, 3)
            furniture_pcds.append(part_pcd_transformed)
        furniture_pcds = torch.cat(furniture_pcds, dim=1)  # (B, n_points, 3)
        furniture_ee_masks = torch.zeros(
            (furniture_pcds.shape[0], furniture_pcds.shape[1]),
            device=self.sim_device,
            dtype=torch.bool,
        )
        pcds_full.append(furniture_pcds)
        ee_mask_full.append(furniture_ee_masks)

        B = furniture_pcds.shape[0]
        wall_pcd = self.static_wall_pcd.unsqueeze(0).repeat(B, 1, 1)
        wall_ee_masks = torch.zeros(
            (wall_pcd.shape[0], wall_pcd.shape[1]),
            device=self.sim_device,
            dtype=torch.bool,
        )
        pcds_full.append(wall_pcd)
        ee_mask_full.append(wall_ee_masks)

        # for EE fingers
        rot_bias = (
            C.axisangle2quat(
                torch.tensor([0, 0, np.pi], dtype=torch.float32, device=self.sim_device)
            )
            .unsqueeze(0)
            .repeat(B, 1)
        )  # (B, 4)
        left_finger_poses = state_dict["leftfinger"]
        left_finger_pos, left_finger_quat = (
            left_finger_poses[:, :3],
            left_finger_poses[:, 3:],
        )
        left_finger_quat = C.quat_mul(left_finger_quat, rot_bias)
        left_finger_quat = left_finger_quat[..., [3, 0, 1, 2]]
        left_finger_tf = C.batched_pose2mat(
            left_finger_pos, left_finger_quat, device=self.sim_device
        )
        left_finger_pcd_transformed = (
            left_finger_tf @ self._franka_finger_pcd.T
        )  # (B, 4, n_points)
        left_finger_pcd_transformed = left_finger_pcd_transformed.transpose(1, 2)[
            :, :, :3
        ]  # (B, n_points, 3)
        right_finger_poses = state_dict["rightfinger"]
        right_finger_pos, right_finger_quat = (
            right_finger_poses[:, :3],
            right_finger_poses[:, 3:],
        )
        # right finger needs to be flipped
        flip_rot = (
            C.axisangle2quat(
                torch.tensor([0, 0, np.pi], dtype=torch.float32, device=self.sim_device)
            )
            .unsqueeze(0)
            .repeat(B, 1)
        )  # (B, 4)
        right_finger_quat = C.quat_mul(right_finger_quat, rot_bias)
        right_finger_quat = C.quat_mul(right_finger_quat, flip_rot)
        right_finger_quat = right_finger_quat[..., [3, 0, 1, 2]]
        right_finger_tf = C.batched_pose2mat(
            right_finger_pos, right_finger_quat, device=self.sim_device
        )
        right_finger_pcd_transformed = (
            right_finger_tf @ self._franka_finger_pcd.T
        )  # (B, 4, n_points)
        right_finger_pcd_transformed = right_finger_pcd_transformed.transpose(1, 2)[
            :, :, :3
        ]  # (B, n_points, 3)

        ee_finger_pcds = torch.cat(
            [left_finger_pcd_transformed, right_finger_pcd_transformed], dim=1
        )  # (B, n_points, 3)
        ee_finger_ee_masks = torch.ones(
            (ee_finger_pcds.shape[0], ee_finger_pcds.shape[1]),
            device=self.sim_device,
            dtype=torch.bool,
        )
        pcds_full.append(ee_finger_pcds)
        ee_mask_full.append(ee_finger_ee_masks)
        pcds_full = torch.cat(pcds_full, dim=1)  # (B, n_points, 3)
        ee_mask_full = torch.cat(ee_mask_full, dim=1)  # (B, n_points)

        # transform pcd coordinate into Franka base frame
        base_state = state_dict["franka_base"][:, :3].unsqueeze(1)  # (B, 1, 3)
        pcds_full = pcds_full - base_state
        return pcds_full, ee_mask_full

    def compute_observations(self):
        super().compute_observations()

        # compute pcd observation
        # for static walls
        wall_pcd = self.static_wall_pcd.unsqueeze(0).repeat(
            self.num_envs, 1, 1
        )  # (n_envs, n_points, 3)
        wall_ee_masks = torch.zeros(
            (self.num_envs, wall_pcd.shape[1]),
            device=self.sim_device,
            dtype=torch.bool,
        )

        # for furniture parts
        furniture_pcds = []
        for part_name, part_pcd in self._fparts_pcds.items():
            part_poses = self._fparts_states[part_name][:, :7]  # (n_envs, 7)
            part_pos, part_quat = part_poses[:, :3], part_poses[:, 3:]
            # quaternion_to_matrix assumes real part first
            part_quat = part_quat[..., [3, 0, 1, 2]]
            part_tf = C.batched_pose2mat(
                part_pos, part_quat, device=self.sim_device
            )  # (n_envs, 4, 4)
            part_pcd_transformed = part_tf @ part_pcd.T  # (n_envs, 4, n_points)
            part_pcd_transformed = part_pcd_transformed.transpose(1, 2)[
                :, :, :3
            ]  # (n_envs, n_points, 3)
            furniture_pcds.append(part_pcd_transformed)
        furniture_pcds = torch.cat(furniture_pcds, dim=1)  # (n_envs, n_points, 3)
        furniture_ee_masks = torch.zeros(
            (self.num_envs, furniture_pcds.shape[1]),
            device=self.sim_device,
            dtype=torch.bool,
        )

        # for EE fingers
        rot_bias = (
            C.axisangle2quat(
                torch.tensor([0, 0, np.pi], dtype=torch.float32, device=self.sim_device)
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )  # (num_envs, 4)
        left_finger_poses = self._rigid_body_state[
            :, self.franka_handles["leftfinger"], :
        ][:, :7]
        left_finger_pos, left_finger_quat = (
            left_finger_poses[:, :3],
            left_finger_poses[:, 3:],
        )
        left_finger_quat = C.quat_mul(left_finger_quat, rot_bias)
        left_finger_quat = left_finger_quat[..., [3, 0, 1, 2]]
        left_finger_tf = C.batched_pose2mat(
            left_finger_pos, left_finger_quat, device=self.sim_device
        )
        left_finger_pcd_transformed = (
            left_finger_tf @ self._franka_finger_pcd.T
        )  # (n_envs, 4, n_points)
        left_finger_pcd_transformed = left_finger_pcd_transformed.transpose(1, 2)[
            :, :, :3
        ]  # (n_envs, n_points, 3)

        right_finger_poses = self._rigid_body_state[
            :, self.franka_handles["rightfinger"], :
        ][:, :7]
        right_finger_pos, right_finger_quat = (
            right_finger_poses[:, :3],
            right_finger_poses[:, 3:],
        )
        # right finger needs to be flipped
        flip_rot = (
            C.axisangle2quat(
                torch.tensor([0, 0, np.pi], dtype=torch.float32, device=self.sim_device)
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )  # (num_envs, 4)
        right_finger_quat = C.quat_mul(right_finger_quat, rot_bias)
        right_finger_quat = C.quat_mul(right_finger_quat, flip_rot)
        right_finger_quat = right_finger_quat[..., [3, 0, 1, 2]]
        right_finger_tf = C.batched_pose2mat(
            right_finger_pos, right_finger_quat, device=self.sim_device
        )
        right_finger_pcd_transformed = (
            right_finger_tf @ self._franka_finger_pcd.T
        )  # (n_envs, 4, n_points)
        right_finger_pcd_transformed = right_finger_pcd_transformed.transpose(1, 2)[
            :, :, :3
        ]  # (n_envs, n_points, 3)

        ee_finger_pcds = torch.cat(
            [left_finger_pcd_transformed, right_finger_pcd_transformed], dim=1
        )  # (n_envs, n_points, 3)
        ee_finger_ee_masks = torch.ones(
            (self.num_envs, ee_finger_pcds.shape[1]),
            device=self.sim_device,
            dtype=torch.bool,
        )

        pcds_full = [furniture_pcds, ee_finger_pcds]
        ee_mask_full = [furniture_ee_masks, ee_finger_ee_masks]
        pcds_full.append(wall_pcd)
        ee_mask_full.append(wall_ee_masks)

        pcds_full = torch.cat(
            pcds_full,
            dim=1,
        )
        ee_mask_full = torch.cat(
            ee_mask_full,
            dim=1,
        )
        pcds_sampled, ee_mask_sampled = sample_points(
            pcds_full, ee_mask_full, self.sampled_points
        )

        # transform pcd coordinate into Franka base frame
        pcds_sampled = pcds_sampled - self._base_state[:, :3].unsqueeze(
            1
        )  # (n_envs, n_points, 3) - (n_envs, 1, 3)
        if self._pc_augmentation_enabled:
            pcds_sampled, ee_mask_sampled = apply_pc_aug_random_trans(
                pcds_sampled,
                ee_mask_sampled,
                self.sim_device,
                self._pc_aug_apply_p,
                self._pc_aug_random_trans_high,
                self._pc_aug_random_trans_low,
            )
            if np.random.rand() < self._pc_aug_apply_p:
                jitter_points = int(self.sampled_points * self._pc_aug_jitter_ratio)
                if self._pc_aug_jitter_dist is None:
                    jitter_std = torch.tensor(
                        [self._pc_aug_jitter_sigma] * 3,
                        dtype=torch.float32,
                        device=self.sim_device,
                    ).view(1, 3)
                    # repeat along n_points
                    jitter_std = jitter_std.repeat(jitter_points, 1)
                    jitter_mean = torch.zeros_like(jitter_std)
                    self._pc_aug_jitter_dist = torch.distributions.normal.Normal(
                        jitter_mean, jitter_std
                    )
                jitter_value = self._pc_aug_jitter_dist.sample()
                pcds_sampled, ee_mask_sampled = apply_pc_aug_jitter(
                    pcds_sampled,
                    ee_mask_sampled,
                    jitter_value,
                    jitter_points,
                    self._pc_aug_jitter_low,
                    self._pc_aug_jitter_high,
                )

        self.obs_dict["pcd/coordinate"][:] = pcds_sampled
        self.obs_dict["pcd/ee_mask"][:] = ee_mask_sampled

    def allocate_buffers(self):
        super().allocate_buffers()
        self.obs_dict.update(
            {
                "pcd/coordinate": torch.zeros(
                    (self.num_envs, self.sampled_points, 3), device=self.sim_device
                ),
                "pcd/ee_mask": torch.zeros(
                    (self.num_envs, self.sampled_points),
                    device=self.sim_device,
                    dtype=torch.bool,
                ),
            }
        )

    @property
    def static_wall_pcd(self):
        if self._wall_pcd_transformed is None:
            self._wall_pcd_transformed = self._prepare_wall_pcd()
            del self._front_wall_pcd
            del self._side_wall_pcd
        return self._wall_pcd_transformed

    @property
    def total_points(self):
        rtn = 2 * self._franka_finger_pcd.shape[0] + sum(
            x.shape[0] for x in self._fparts_pcds.values()
        )
        rtn += self.static_wall_pcd.shape[0]
        return rtn

    @property
    def sampled_points(self):
        if self._sampled_points is None:
            if self._pcd_mask_ratio is not None:
                self._sampled_points = int(
                    (1 - self._pcd_mask_ratio) * self.total_points
                )
                print(
                    f"[INFO] Pointcloud mask ratio = {self._pcd_mask_ratio}, inferred sampled points = {self._sampled_points}"
                )
            else:
                self._sampled_points = self._pcd_N
                inferred_mask_ratio = 1 - self._sampled_points / self.total_points
                print(
                    f"[INFO] Sampled points = {self._sampled_points}, inferred pointcloud mask ratio = {inferred_mask_ratio}"
                )
        return self._sampled_points

    def _import_franka_pcd(self):
        finger_pointcloud_file = "franka_description/meshes/collision/finray_finger.npy"
        self._franka_finger_pcd = C.xyz_to_homogeneous(
            torch.tensor(
                np.load(os.path.join(ASSET_ROOT, finger_pointcloud_file)),
                device=self.sim_device,
                dtype=torch.float32,
            ),
            device=self.sim_device,
        )

    def _import_furniture_assets(self):
        super()._import_furniture_assets()
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

    def _import_obstacle_pcds(self):
        self._wall_pcd_transformed = None

        front_wall_pcd_file = "furniture_bench/mesh/obstacle_front.npy"
        side_wall_pcd_file = "furniture_bench/mesh/obstacle_side.npy"
        front_wall_pcd = torch.tensor(
            np.load(os.path.join(ASSET_ROOT, front_wall_pcd_file)),
            device=self.sim_device,
            dtype=torch.float32,
        )
        side_wall_pcd = torch.tensor(
            np.load(os.path.join(ASSET_ROOT, side_wall_pcd_file)),
            device=self.sim_device,
            dtype=torch.float32,
        )
        self._front_wall_pcd = C.xyz_to_homogeneous(
            front_wall_pcd, device=self.sim_device
        )
        self._side_wall_pcd = C.xyz_to_homogeneous(
            side_wall_pcd, device=self.sim_device
        )

    def _prepare_wall_pcd(self):
        """
        Pointcloud for the wall
        """
        front_wall_tf = gymapi.Transform()
        front_wall_tf.p = self._front_obstacle_pose.p
        front_wall_tf.r = self._front_obstacle_pose.r
        wall_pos = torch.tensor(
            [
                [
                    self._front_obstacle_pose.p.x,
                    self._front_obstacle_pose.p.y,
                    self._front_obstacle_pose.p.z,
                ],
                [
                    self._left_obstacle_pose.p.x,
                    self._left_obstacle_pose.p.y,
                    self._left_obstacle_pose.p.z,
                ],
                [
                    self._right_obstacle_pose.p.x,
                    self._right_obstacle_pose.p.y,
                    self._right_obstacle_pose.p.z,
                ],
            ],
            device=self.sim_device,
        )
        wall_rot = torch.tensor(
            [
                [
                    self._front_obstacle_pose.r.w,
                    self._front_obstacle_pose.r.x,
                    self._front_obstacle_pose.r.y,
                    self._front_obstacle_pose.r.z,
                ],
                [
                    self._left_obstacle_pose.r.w,
                    self._left_obstacle_pose.r.x,
                    self._left_obstacle_pose.r.y,
                    self._left_obstacle_pose.r.z,
                ],
                [
                    self._right_obstacle_pose.r.w,
                    self._right_obstacle_pose.r.x,
                    self._right_obstacle_pose.r.y,
                    self._right_obstacle_pose.r.z,
                ],
            ],
            device=self.sim_device,
        )
        wall_tf = C.batched_pose2mat(
            wall_pos, wall_rot, device=self.sim_device
        )  # (3, 4, 4)

        front_wall_pcd_transformed = (wall_tf[0] @ self._front_wall_pcd.T).T
        front_wall_pcd_transformed = front_wall_pcd_transformed[:, :3]  # (n_points, 3)
        left_wall_pcd_transformed = (wall_tf[1] @ self._side_wall_pcd.T).T
        left_wall_pcd_transformed = left_wall_pcd_transformed[:, :3]
        right_wall_pcd_transformed = (wall_tf[2] @ self._side_wall_pcd.T).T
        right_wall_pcd_transformed = right_wall_pcd_transformed[:, :3]
        wall_pcd_transformed = torch.cat(
            [
                front_wall_pcd_transformed,
                left_wall_pcd_transformed,
                right_wall_pcd_transformed,
            ],
            dim=0,
        )  # (n_points, 3)
        return wall_pcd_transformed

    def _set_renderers(self, display):
        super()._set_renderers(display)
        self._pcd_viewer = PointCloudVisualizer() if display else None

    def render(self, mode="rgb_array"):
        super().render()
        if self._pcd_viewer is not None:
            self._pcd_viewer(self.obs_dict["pcd/coordinate"][-1].cpu().numpy())


@torch.jit.script
def sample_points(points, ee_segm, sample_num: int):
    """
    points: (n_envs, n_points, 3)
    """
    sampling_idx = torch.randperm(points.shape[1])[:sample_num]
    sampled_points = points[:, sampling_idx, :]
    ee_segm = ee_segm[:, sampling_idx]
    return sampled_points, ee_segm


@torch.jit.script
def apply_pc_aug_random_trans(
    pcd,
    ee_mask,
    sim_device: torch.device,
    pc_aug_apply_p: float,
    pc_aug_random_trans_high,
    pc_aug_random_trans_low,
):
    """
    Randomly translate whole point cloud
    """
    apply = torch.rand(pcd.shape[0], device=sim_device) < pc_aug_apply_p  # (n_envs)
    random_translation = torch.rand(
        (pcd.shape[0], 1, 3), device=sim_device
    )  # (n_envs, 1, 3)
    random_translation = (
        random_translation * (pc_aug_random_trans_high - pc_aug_random_trans_low)
        + pc_aug_random_trans_low
    )
    random_translation = random_translation * apply.view(-1, 1, 1)
    pcd = pcd + random_translation
    return pcd, ee_mask


@torch.jit.script
def apply_pc_aug_jitter(
    pcd,
    ee_mask,
    jitter_value,
    jitter_points: int,
    pc_aug_jitter_low: float,
    pc_aug_jitter_high: float,
):
    jitter_point_idx = torch.randperm(pcd.shape[1])[:jitter_points]
    jitter_value = jitter_value.unsqueeze(0)  # (1, jitter_points, 3)
    # cap to high and low
    jitter_value = torch.clamp(jitter_value, pc_aug_jitter_low, pc_aug_jitter_high)
    pcd[:, jitter_point_idx] = pcd[:, jitter_point_idx] + jitter_value
    return pcd, ee_mask
