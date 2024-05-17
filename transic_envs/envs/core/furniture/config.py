from typing import Any, Dict

import numpy as np

from transic_envs.utils.pose_utils import get_mat, rot_mat


ROBOT_HEIGHT = 0.00214874


config: Dict[str, Any] = {
    "robot": {
        "tag_base_from_robot_base": get_mat(
            (0.23 + 0.0715, 0, -ROBOT_HEIGHT), (np.pi, 0, np.pi / 2)
        ),
    },
    "furniture": {
        "position_limits": [
            [-0.21, 0.21],
            [0.07, 0.37],
        ],
        "square_table": {
            "square_table_top": {
                "name": "square_table_top",
                "asset_file": "furniture_bench/urdf/square_table/square_table_top.urdf",
                "ids": [4, 5, 6, 7],
                "reset_pos": [
                    np.array([0.0, 0.24, -0.015625]),
                    np.array([0.0, 0.24, -0.015625]),
                    np.array([0.08, 0.27, -0.015], dtype=np.float32),
                    np.array([0.08, 0.27, -0.015625], dtype=np.float32),
                    np.array([0.08, 0.26, -0.015625], dtype=np.float32),
                ],
                "reset_ori": [
                    rot_mat(np.array([-np.pi / 2, 0, np.pi]), hom=True),
                    rot_mat(np.array([-np.pi / 2, 0, np.pi]), hom=True),
                    np.array(
                        [
                            [-9.9997163e-01, -9.2363916e-07, -7.5414525e-03, 0],
                            [7.5414525e-03, -3.8682265e-07, -9.9997157e-01, 0],
                            [9.2084520e-07, -1.0000000e00, 3.2843309e-07, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                    np.array(
                        [
                            [-9.9997163e-01, -9.2363916e-07, -7.5414525e-03, 0],
                            [7.5414525e-03, -3.8682265e-07, -9.9997157e-01, 0],
                            [9.2084520e-07, -1.0000000e00, 3.2843309e-07, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                    np.array(
                        [
                            [-9.9997151e-01, -6.1001629e-07, -7.5414428e-03, 0],
                            [7.5414428e-03, 2.0808420e-07, -9.9997151e-01, 0],
                            [6.1141327e-07, -1.0000000e00, -1.4954367e-07, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                ],
                "part_moved_skill_idx": 2,
            },
            "square_table_leg1": {
                "name": "square_table_leg1",
                "asset_file": "furniture_bench/urdf/square_table/square_table_leg1.urdf",
                "ids": [8, 9, 10, 11],
                "reset_pos": [np.array([-0.20, 0.07, -0.015])] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    np.array([0.05625, 0.046875, 0.05625]), [0, 0, 0]
                ),
            },
            "square_table_leg2": {
                "name": "square_table_leg2",
                "asset_file": "furniture_bench/urdf/square_table/square_table_leg2.urdf",
                "ids": [12, 13, 14, 15],
                "reset_pos": [np.array([-0.12, 0.07, -0.015])] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    np.array([-0.05625, 0.046875, 0.05625]), [0, 0, 0]
                ),
            },
            "square_table_leg3": {
                "name": "square_table_leg3",
                "asset_file": "furniture_bench/urdf/square_table/square_table_leg3.urdf",
                "ids": [16, 17, 18, 19],
                "reset_pos": [np.array([0.12, 0.07, -0.015])] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    np.array([0.05625, 0.046875, -0.05625]), [0, 0, 0]
                ),
            },
            "square_table_leg4": {
                "name": "square_table_leg4",
                "asset_file": "furniture_bench/urdf/square_table/square_table_leg4.urdf",
                "ids": [20, 21, 22, 23],
                "reset_pos": [
                    np.array([0.20, 0.07, -0.015]),
                    np.array([0.20, 0.071, -0.015], dtype=np.float32),
                    np.array([0.20, 0.071, -0.015], dtype=np.float32),
                    np.array([0.20, 0.071, -0.015], dtype=np.float32),
                    np.array([0.136, 0.336, -0.07763672], dtype=np.float32),
                ],
                "reset_ori": [
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    np.array(
                        [
                            [0.09753844, 0.03445375, -0.9946352, 0],
                            [0.9915855, 0.08210686, 0.10008356, 0],
                            [0.08511469, -0.9960278, -0.02615529, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                ],
                "default_assembled_pose": get_mat(
                    np.array([-0.05625, 0.046875, -0.05625]), [0, 0, 0]
                ),
                "part_attached_skill_idx": 4,
            },
        },
        "square_table_patchfix": {
            "square_table_top": {
                "name": "square_table_top",
                "asset_file": "furniture_bench/urdf/square_table/square_table_top.urdf",
                "ids": [4, 5, 6, 7],
                "reset_pos": [
                    np.array([0.0, 0.24, -0.015625]),
                    np.array([0.0, 0.24, -0.015625]),
                    np.array([0.08, 0.27, -0.015], dtype=np.float32),
                    np.array([0.08, 0.27, -0.015625], dtype=np.float32),
                    np.array([0.08, 0.26, -0.015625], dtype=np.float32),
                ],
                "reset_ori": [
                    rot_mat(np.array([-np.pi / 2, 0, np.pi]), hom=True),
                    rot_mat(np.array([-np.pi / 2, 0, np.pi]), hom=True),
                    np.array(
                        [
                            [-9.9997163e-01, -9.2363916e-07, -7.5414525e-03, 0],
                            [7.5414525e-03, -3.8682265e-07, -9.9997157e-01, 0],
                            [9.2084520e-07, -1.0000000e00, 3.2843309e-07, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                    np.array(
                        [
                            [-9.9997163e-01, -9.2363916e-07, -7.5414525e-03, 0],
                            [7.5414525e-03, -3.8682265e-07, -9.9997157e-01, 0],
                            [9.2084520e-07, -1.0000000e00, 3.2843309e-07, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                    np.array(
                        [
                            [-9.9997151e-01, -6.1001629e-07, -7.5414428e-03, 0],
                            [7.5414428e-03, 2.0808420e-07, -9.9997151e-01, 0],
                            [6.1141327e-07, -1.0000000e00, -1.4954367e-07, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                ],
                "part_moved_skill_idx": 2,
            },
            "square_table_leg1": {
                "name": "square_table_leg1",
                "asset_file": "furniture_bench/urdf/square_table/square_table_leg1.urdf",
                "ids": [8, 9, 10, 11],
                "reset_pos": [np.array([-0.20, 0.07, -0.015])] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    np.array([0.05625, 0.046875, 0.05625]), [0, 0, 0]
                ),
            },
            "square_table_leg2": {
                "name": "square_table_leg2",
                "asset_file": "furniture_bench/urdf/square_table/square_table_leg2.urdf",
                "ids": [12, 13, 14, 15],
                "reset_pos": [np.array([-0.12, 0.07, -0.015])] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    np.array([-0.05625, 0.046875, 0.05625]), [0, 0, 0]
                ),
            },
            "square_table_leg3": {
                "name": "square_table_leg3",
                "asset_file": "furniture_bench/urdf/square_table/square_table_leg3.urdf",
                "ids": [16, 17, 18, 19],
                "reset_pos": [np.array([0.12, 0.07, -0.015])] * 5,
                "reset_ori": [rot_mat(np.array([0, -np.pi / 2, 0]), hom=True)] * 5,
                "default_assembled_pose": get_mat(
                    np.array([0.05625, 0.046875, -0.05625]), [0, 0, 0]
                ),
            },
            "leg": {
                "name": "leg",
                "asset_file": "furniture_bench/urdf/square_table/square_table_leg4.urdf",
                "ids": [20, 21, 22, 23],
                "reset_pos": [
                    np.array([0.20, 0.07, -0.015]),
                    np.array([0.20, 0.071, -0.015], dtype=np.float32),
                    np.array([0.20, 0.071, -0.015], dtype=np.float32),
                    np.array([0.20, 0.071, -0.015], dtype=np.float32),
                    np.array([0.136, 0.336, -0.07763672], dtype=np.float32),
                ],
                "reset_ori": [
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    rot_mat(np.array([0, -np.pi / 2, 0]), hom=True),
                    np.array(
                        [
                            [0.09753844, 0.03445375, -0.9946352, 0],
                            [0.9915855, 0.08210686, 0.10008356, 0],
                            [0.08511469, -0.9960278, -0.02615529, 0],
                            [0, 0, 0, 1],
                        ],
                        dtype=np.float32,
                    ),
                ],
                "default_assembled_pose": get_mat(
                    np.array([-0.05625, 0.046875, -0.05625]), [0, 0, 0]
                ),
                "part_attached_skill_idx": 4,
            },
        },
    },
}
