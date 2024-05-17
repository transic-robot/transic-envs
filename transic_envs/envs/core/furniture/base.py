"""Define base class for all furniture. It contains the core functions and properties for the furniture (e.g., furniture parts, computing reward function, getting observation,etc.)"""
from abc import ABC
from typing import Optional, List

import numpy as np
from gym import logger

from transic_envs.envs.core.furniture.config import config
from transic_envs.envs.core.furniture.furniture_parts.base import Part
from transic_envs.envs.core.furniture.obstacles import (
    ObstacleFront,
    ObstacleRight,
    ObstacleLeft,
)


class Furniture(ABC):
    def __init__(self, seed: Optional[int] = None):
        self.parts: List[Part] = []
        self.num_parts = len(self.parts)
        self.random = np.random.RandomState(seed)

        self.ori_bound = 0.94
        self.parts_pos_lim = config["furniture"]["position_limits"]

        # Defined in the child class.
        self.reset_temporal_xys = None
        self.reset_temporal_idxs = {}
        self.should_assembled_first = {}
        self.should_be_assembled = []
        self.assembled_rel_poses = {}

        # Reset assembled set.
        self.assembled_set = set()
        self.position_only = set()
        self.max_env_steps = 3000

        self._init_obstacle()

        self.reset_pos_diff_threshold = [0.015, 0.015, 0.015]  # 1.5cm.
        self.reset_ori_bound = 0.96  # 15 degrees.
        self.max_env_steps_skills = [0, 250, 250, 250, 250, 350]
        self.max_env_steps_from_skills = [
            sum(self.max_env_steps_skills[i:])
            for i in range(len(self.max_env_steps_skills) - 1)
        ]

    @property
    def new_random_seed(self):
        return self.random.randint(0, 2**31 - 1)

    def randomize_init_pose(
        self, from_skill, pos_range=[-0.05, 0.05], rot_range=45
    ) -> bool:
        """Randomize the furniture initial pose."""
        trial = 0
        max_trial = 300000
        while True:
            trial += 1
            for part in self.parts:
                part.randomize_init_pose(from_skill, pos_range, rot_range)
            if trial > max_trial:
                logger.error("Failed to randomize init pose")
                return False
            if self._in_boundary(from_skill) and not self._check_collision():
                logger.info("Found collision-free init pose")
                return True

    def randomize_skill_init_pose(self, from_skill) -> bool:
        """Randomize the furniture initial pose."""
        trial = 0
        max_trial = 300000
        while True:
            trial += 1
            for i, part in enumerate(self.parts):
                if part.part_moved_skill_idx <= from_skill:
                    # Reduce randomized range the part that has been moved from the skill.
                    part.randomize_init_pose(
                        from_skill=from_skill,
                        pos_range=[-0.0, 0.0],
                        rot_range=0,
                    )
                elif (
                    part.part_attached_skill_idx <= from_skill
                    and self.skill_attach_part_idx == i
                ):
                    attached_part, attach_to = self.attach(part)
                    if attached_part:
                        self.set_attached_pose(part, attach_to, from_skill)
                else:
                    part.randomize_init_pose(from_skill=from_skill)
            if trial > max_trial:
                logger.error("Failed to randomize init pose")
                return False
            if not self._check_collision(from_skill) and self._in_boundary(from_skill):
                logger.info("Found initialization pose")
                return True

    def _check_collision(self):
        """Simple rectangle collision check between two parts."""
        for i in range(self.num_parts):
            for j in range(i + 1, self.num_parts):
                if self.parts[i].is_collision(self.parts[j]):
                    return True

        for i in range(self.num_parts):
            for obstacle in self.obstacles:
                if self.parts[i].is_collision(obstacle):
                    return True

        return False

    def _in_boundary(self, from_skill):
        """Check whether the furniture is in the boundary."""
        for part in self.parts:
            if not part.in_boundary(self.parts_pos_lim, from_skill):
                return False
        return True

    def reset(self):
        """Reset filter and assembled parts."""
        self.assembled_set = set()
        for part in self.parts:
            part.reset()

    def all_assembled(self) -> bool:
        if len(self.assembled_set) == len(self.should_be_assembled):
            return True
        return False

    def _init_obstacle(self):
        """Initialize the obstacle."""
        self.obstacles = [ObstacleFront(), ObstacleLeft(), ObstacleRight()]
