import numpy as np

from transic_envs.utils.pose_utils import rot_mat, is_similar_rot
from transic_envs.envs.core.furniture.furniture_parts.base import Part


class TableTop(Part):
    def __init__(self, part_config: dict, part_idx: int, seed: int):
        super().__init__(part_config, part_idx, seed)

        self.gripper_action = -1
        self.body_grip_width = 0.01

        self.skill_complete_next_states = [
            "push",
            "go_up",
        ]  # Specificy next state after skill is complete.
        self.reset()

    def reset(self):
        self.pre_assemble_done = False
        self._state = "reach_body_grasp_xy"
        self.gripper_action = -1

    def is_in_reset_ori(self, pose, from_skill, ori_bound):
        reset_ori = (
            self.reset_ori[from_skill] if len(self.reset_ori) > 1 else self.reset_ori[0]
        )
        for _ in range(4):
            if is_similar_rot(pose[:3, :3], reset_ori[:3, :3], ori_bound=ori_bound):
                return True
            pose = pose @ rot_mat(np.array([0, np.pi / 2, 0]), hom=True)
        return False
