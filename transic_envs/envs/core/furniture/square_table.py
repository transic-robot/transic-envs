from transic_envs.utils.pose_utils import get_mat
from transic_envs.envs.core.furniture.config import config
from transic_envs.envs.core.furniture.base import Furniture
from transic_envs.envs.core.furniture.furniture_parts import (
    SquareTableLeg,
    SquareTableTop,
)


class SquareTable(Furniture):
    def __init__(self, seed):
        super().__init__(seed=seed)
        furniture_conf = config["furniture"]["square_table"]
        self.furniture_conf = furniture_conf

        self.parts = [
            SquareTableTop(furniture_conf["square_table_top"], 0, self.new_random_seed),
            SquareTableLeg(
                furniture_conf["square_table_leg1"], 1, self.new_random_seed
            ),
            SquareTableLeg(
                furniture_conf["square_table_leg2"], 2, self.new_random_seed
            ),
            SquareTableLeg(
                furniture_conf["square_table_leg3"], 3, self.new_random_seed
            ),
            SquareTableLeg(
                furniture_conf["square_table_leg4"], 4, self.new_random_seed
            ),
        ]
        self.num_parts = len(self.parts)

        self.assembled_rel_poses[(0, 1)] = [
            get_mat([-0.05625, 0.046875, -0.05625], [0, 0, 0]),
            get_mat([0.05625, 0.046875, -0.05625], [0, 0, 0]),
            get_mat([-0.05625, 0.046875, 0.05625], [0, 0, 0]),
            get_mat([0.05625, 0.046875, 0.05625], [0, 0, 0]),
        ]

        self.assembled_rel_poses[(0, 2)] = self.assembled_rel_poses[(0, 1)]
        self.assembled_rel_poses[(0, 3)] = self.assembled_rel_poses[(0, 1)]
        self.assembled_rel_poses[(0, 4)] = self.assembled_rel_poses[(0, 1)]

        self.should_be_assembled = [(0, 4), (0, 3), (0, 1), (0, 2)]
        self.skill_attach_part_idx = 4

    def get_grasp_part_idx(self, from_skill):
        if from_skill == 1:
            return 0
        elif from_skill == 3:
            return 4
        else:
            assert False

    def z_noise(self, from_skill):
        # Zero noise for collision.
        return 0


class OneLeg(SquareTable):
    def __init__(self, seed):
        super().__init__(seed)
        self.should_be_assembled = [(0, 4)]


class JustOneLeg(OneLeg):
    def __init__(self, seed):
        super().__init__(seed)
        self.parts = [self.parts[2]]
        self.parts[0].name = "leg"
        self.num_parts = len(self.parts)


class TableWithOneLeg(SquareTable):
    def __init__(self, seed):
        super().__init__(seed)
        self.parts = [self.parts[0], self.parts[4]]
        self.num_parts = len(self.parts)


class SquareTablePatchFix(SquareTable):
    def __init__(self, seed):
        super().__init__(seed=seed)
        furniture_conf = config["furniture"]["square_table_patchfix"]
        self.furniture_conf = furniture_conf

        self.parts = [
            SquareTableTop(furniture_conf["square_table_top"], 0, self.new_random_seed),
            SquareTableLeg(
                furniture_conf["square_table_leg1"], 1, self.new_random_seed
            ),
            SquareTableLeg(
                furniture_conf["square_table_leg2"], 2, self.new_random_seed
            ),
            SquareTableLeg(
                furniture_conf["square_table_leg3"], 3, self.new_random_seed
            ),
            SquareTableLeg(furniture_conf["leg"], 4, self.new_random_seed),
        ]
        self.num_parts = len(self.parts)

        self.assembled_rel_poses[(0, 1)] = [
            get_mat([-0.05625, 0.046875, -0.05625], [0, 0, 0]),
            get_mat([0.05625, 0.046875, -0.05625], [0, 0, 0]),
            get_mat([-0.05625, 0.046875, 0.05625], [0, 0, 0]),
            get_mat([0.05625, 0.046875, 0.05625], [0, 0, 0]),
        ]

        self.assembled_rel_poses[(0, 2)] = self.assembled_rel_poses[(0, 1)]
        self.assembled_rel_poses[(0, 3)] = self.assembled_rel_poses[(0, 1)]
        self.assembled_rel_poses[(0, 4)] = self.assembled_rel_poses[(0, 1)]

        self.should_be_assembled = [(0, 4), (0, 3), (0, 1), (0, 2)]
        self.skill_attach_part_idx = 4
