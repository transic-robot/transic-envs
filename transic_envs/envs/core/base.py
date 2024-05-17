from __future__ import annotations

from isaacgym import gymtorch
from isaacgym import gymapi
import torch
import numpy as np

from transic_envs.asset_root import ASSET_ROOT
from transic_envs.utils.pose_utils import get_mat
from transic_envs.envs.core.vec_task import VecTask
from transic_envs.envs.core.sim_config import sim_config
import transic_envs.utils.torch_jit_utils as torch_jit_utils
from transic_envs.envs.core.furniture import furniture_factory
from transic_envs.envs.core.furniture.config import config, ROBOT_HEIGHT


class TRANSICEnvOSCBase(VecTask):
    franka_asset_file = "franka_description/robots/franka_panda_finray.urdf"

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
        num_furniture_parts: int = 0,
    ):
        self._record = record
        self.cfg = cfg

        use_quat_rot = self.use_quat_rot = self.cfg["env"]["useQuatRot"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self._num_fparts = num_furniture_parts

        self.cfg["env"]["numActions"] = 8 if use_quat_rot else 7

        # a dict containing prop obs name to dump and their dimensions
        # used for distillation
        self._prop_dump_info = self.cfg["env"]["propDumpInfo"]

        # Values to be filled in at runtime
        self.states = {}
        self.franka_handles = {}  # will be dict mapping names to relevant sim handles
        self.fparts_handles = {}  # for furniture part handlers
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed
        self._fparts_names = []  # Names of all furniture parts
        self._init_fparts_states = None  # Initial state of all furniture parts
        self._fparts_states = None  # State of all furniture parts

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = (
            None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        )
        self.net_cf = None  # contact force
        self._eef_state = None  # end effector state (at grasping point)
        self._ftip_center_state = None  # center of fingertips
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._franka_effort_limits = None  # Actuator effort limits for franka
        self._global_franka_indices = (
            None  # Unique indices corresponding to all envs in flattened array
        )
        self._global_furniture_part_indices = {}

        self._front_wall_idxs = None
        self._left_wall_idxs = None
        self._right_wall_idxs = None
        self._fparts_idxs = None

        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )

        # Franka defaults
        default_pose = self.cfg["env"].get("frankaDefaultDofPos", None)
        default_pose = default_pose or [
            0.12162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
            0.035,
            0.035,
        ]

        self.franka_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)

        # OSC Gains
        self.kp = torch.tensor([150.0] * 6, device=self.sim_device)
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = torch.tensor([10.0] * 7, device=self.sim_device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # Set control limits
        self.cmd_limit = torch.tensor(
            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.sim_device
        ).unsqueeze(0)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _import_furniture_assets(self):
        pass

    def _import_franka_pcd(self):
        pass

    def _import_obstacle_pcds(self):
        pass

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 1.3, 0.8, 0.03, table_asset_options)

        # import front obstacle asset
        front_obstacle_asset_options = gymapi.AssetOptions()
        front_obstacle_asset_options.fix_base_link = True
        front_obstacle_asset_file = "furniture_bench/urdf/obstacle_front.urdf"
        front_obstacle_asset = self.gym.load_asset(
            self.sim,
            ASSET_ROOT,
            front_obstacle_asset_file,
            front_obstacle_asset_options,
        )

        # import side obstacle asset
        side_obstacle_asset_options = gymapi.AssetOptions()
        side_obstacle_asset_options.fix_base_link = True
        side_obstacle_asset_file = "furniture_bench/urdf/obstacle_side.urdf"
        side_obstacle_asset = self.gym.load_asset(
            self.sim, ASSET_ROOT, side_obstacle_asset_file, side_obstacle_asset_options
        )

        # import background if recording
        if self._record:
            bg_asset_options = gymapi.AssetOptions()
            bg_asset_options.fix_base_link = True
            background_asset_file = "furniture_bench/urdf/background.urdf"
            background_asset = self.gym.load_asset(
                self.sim, ASSET_ROOT, background_asset_file, bg_asset_options
            )

        # import obstacle pcds
        self._import_obstacle_pcds()

        # import furniture assets
        self._import_furniture_assets()

        # import franka pcds
        self._import_franka_pcd()

        # load franka asset
        franka_asset_file = self.franka_asset_file
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(
            self.sim, ASSET_ROOT, franka_asset_file, asset_options
        )
        franka_dof_stiffness = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 5000.0, 5000.0],
            dtype=torch.float,
            device=self.sim_device,
        )
        franka_dof_damping = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 1.0e2, 1.0e2],
            dtype=torch.float,
            device=self.sim_device,
        )

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        self.franka_grip_site_idx = franka_link_dict["panda_grip_site"]

        print(f"Num Franka Bodies: {self.num_franka_bodies}")
        print(f"Num Franka DOFs: {self.num_franka_dofs}")

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        self._franka_effort_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props["driveMode"][i] = (
                gymapi.DOF_MODE_POS if i > 6 else gymapi.DOF_MODE_EFFORT
            )
            franka_dof_props["stiffness"][i] = franka_dof_stiffness[i]
            franka_dof_props["damping"][i] = franka_dof_damping[i]

            self.franka_dof_lower_limits.append(franka_dof_props["lower"][i])
            self.franka_dof_upper_limits.append(franka_dof_props["upper"][i])
            self._franka_effort_limits.append(franka_dof_props["effort"][i])

        self.franka_dof_lower_limits = torch.tensor(
            self.franka_dof_lower_limits, device=self.sim_device
        )
        self.franka_dof_upper_limits = torch.tensor(
            self.franka_dof_upper_limits, device=self.sim_device
        )
        self._franka_effort_limits = torch.tensor(
            self._franka_effort_limits, device=self.sim_device
        )
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props["effort"][7] = 200
        franka_dof_props["effort"][8] = 200

        # Define start pose for franka and table
        table_pos = gymapi.Vec3(0.8, 0.8, 0.4)
        self.franka_pose = gymapi.Transform()
        table_half_width = 0.015
        self._table_surface_z = table_surface_z = table_pos.z + table_half_width
        self.franka_pose.p = gymapi.Vec3(
            0.5 * -table_pos.x + 0.1, 0, table_surface_z + ROBOT_HEIGHT
        )

        # Define start pose for obstacles
        base_tag_pose = gymapi.Transform()
        base_tag_pos = get_mat(
            (0.23 + 0.0715, 0, -ROBOT_HEIGHT), (np.pi, 0, np.pi / 2)
        )[:3, 3]
        base_tag_pose.p = self.franka_pose.p + gymapi.Vec3(
            base_tag_pos[0], base_tag_pos[1], -ROBOT_HEIGHT
        )
        base_tag_pose.p.z = table_surface_z
        self._front_obstacle_pose = gymapi.Transform()
        self._front_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01, 0.0, table_surface_z + 0.015
        )
        self._front_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )
        self._right_obstacle_pose = gymapi.Transform()
        self._right_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
            -0.175,
            table_surface_z + 0.015,
        )
        self._right_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )
        self._left_obstacle_pose = gymapi.Transform()
        self._left_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
            0.175,
            table_surface_z + 0.015,
        )
        self._left_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )

        if self._record:
            bg_pos = gymapi.Vec3(-0.8, 0, 0.75)
            bg_pose = gymapi.Transform()
            bg_pose.p = gymapi.Vec3(bg_pos.x, bg_pos.y, bg_pos.z)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = (
            num_franka_bodies
            + 4
            + self._num_fparts
            + (1 if self._record else 0)  # for background
        )  # 1 for table, front obstacle, left obstacle, right obstacle
        max_agg_shapes = (
            num_franka_shapes + 4 + self._num_fparts + (1 if self._record else 0)
        )

        self.frankas = []
        self.envs = []

        # Create environments
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # camera handler for view rendering
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env_ptr,
                        isaac_gym=self.gym,
                    )
                )

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            franka_actor = self.gym.create_actor(
                env_ptr,
                franka_asset,
                self.franka_pose,
                "franka",
                i,
                0,
            )
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, table_pos.z)
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 0
            )
            table_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, table_actor
            )
            table_props[0].friction = 0.10
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(
                env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1)
            )
            front_obstacle_actor = self.gym.create_actor(
                env_ptr,
                front_obstacle_asset,
                self._front_obstacle_pose,
                "obstacle_front",
                i,
                0,
            )
            left_obstacle_actor = self.gym.create_actor(
                env_ptr,
                side_obstacle_asset,
                self._left_obstacle_pose,
                "obstacle_left",
                i,
                0,
            )
            right_obstacle_actor = self.gym.create_actor(
                env_ptr,
                side_obstacle_asset,
                self._right_obstacle_pose,
                "obstacle_right",
                i,
                0,
            )

            if self._record:
                bg_actor = self.gym.create_actor(
                    env_ptr,
                    background_asset,
                    bg_pose,
                    "background",
                    i,
                    0,
                )

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create furniture parts
            self._create_furniture_parts(env_ptr, i)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        self.franka_from_origin_mat = get_mat(
            [self.franka_pose.p.x, self.franka_pose.p.y, self.franka_pose.p.z],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]
        april_to_sim_mat = self.franka_from_origin_mat @ self.base_tag_from_robot_mat
        self.april_to_sim_mat = torch.from_numpy(april_to_sim_mat).to(
            device=self.sim_device
        )

        # Setup init state buffer for all furniture parts
        self._init_fparts_states = {
            part_name: torch.zeros(self.num_envs, 13, device=self.sim_device)
            for part_name in self._fparts_names
        }

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.franka_handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_hand"
            ),
            "base": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_link0"
            ),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_leftfinger_tip"
            ),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_rightfinger_tip"
            ),
            "leftfinger": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_leftfinger"
            ),
            "rightfinger": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_rightfinger"
            ),
            "grip_site": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_grip_site"
            ),
            "fingertip_center": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "tri_finger_center"
            ),
        }
        self.fparts_handles = {
            part_name: self.gym.find_actor_index(env_ptr, part_name, gymapi.DOMAIN_ENV)
            for part_name in self._fparts_names
        }
        self.walls_handles = {
            part_name: self.gym.find_actor_index(env_ptr, part_name, gymapi.DOMAIN_ENV)
            for part_name in ["obstacle_front", "obstacle_left", "obstacle_right"]
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.franka_handles["grip_site"], :]
        self._ftip_center_state = self._rigid_body_state[
            :, self.franka_handles["fingertip_center"], :
        ]
        self._base_state = self._rigid_body_state[:, self.franka_handles["base"], :]
        self._eef_lf_state = self._rigid_body_state[
            :, self.franka_handles["leftfinger_tip"], :
        ]
        self._eef_rf_state = self._rigid_body_state[
            :, self.franka_handles["rightfinger_tip"], :
        ]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self._j_eef = jacobian[
            :, self.franka_grip_site_idx - 1, :, :7
        ]  # -1 due to fixed base link.
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :7, :7]
        self._fparts_states = {
            part_name: self._root_state[:, part_idx, :]
            for part_name, part_idx in self.fparts_handles.items()
        }
        self._walls_states = {
            part_name: self._root_state[:, part_idx, :]
            for part_name, part_idx in self.walls_handles.items()
        }
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(_net_cf)

        # Initialize actions
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize control
        self._arm_control = self._effort_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_franka_indices = torch.tensor(
            [
                self.gym.find_actor_index(env, "franka", gymapi.DOMAIN_SIM)
                for env in self.envs
            ],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_furniture_part_indices = {
            part_name: torch.tensor(
                [
                    self.gym.find_actor_index(env, part_name, gymapi.DOMAIN_SIM)
                    for env in self.envs
                ],
                dtype=torch.int32,
                device=self.sim_device,
            ).view(self.num_envs, -1)
            for part_name in self._fparts_names
        }

    def allocate_buffers(self):
        # will also allocate extra buffers for data dumping, used for distillation
        super().allocate_buffers()

        # basic prop fields
        self.dump_fileds = {
            k: torch.zeros(
                (self.num_envs, v),
                device=self.device,
                dtype=torch.float,
            )
            for k, v in self._prop_dump_info.items()
        }

        # for constructing PCD, only save dynamic parts (i.e., furniture parts & robot fingers)
        self.dump_fileds.update(
            {
                k: torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                )
                for k in self._fparts_states.keys()
            }
        )
        self.dump_fileds.update(
            {
                "leftfinger": torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                ),
                "rightfinger": torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                ),
                "franka_base": torch.zeros(
                    (self.num_envs, 7),
                    device=self.device,
                    dtype=torch.float,
                ),
            }
        )

    def _create_furniture_parts(self, env_prt, i):
        pass

    def _update_states(self):
        self.states.update(
            {
                # Franka
                "q": self._q[:, :],
                "cos_q": torch.cos(self._q[:, :]),
                "sin_q": torch.sin(self._q[:, :]),
                "dq": self._qd[:, :],
                "q_gripper": self._q[:, -2:],
                "eef_pos": self._eef_state[:, :3] - self._base_state[:, :3],
                "eef_quat": self._eef_state[:, 3:7],
                "ftip_center_pos": self._ftip_center_state[:, :3]
                - self._base_state[:, :3],
                "ftip_center_quat": self._ftip_center_state[:, 3:7],
                "gripper_width": torch.sum(self._q[:, -2:], dim=-1, keepdim=True),
                "eef_vel": self._eef_state[:, 7:],  # still required for OSC
                "eef_lf_pos": self._eef_lf_state[:, :3] - self._base_state[:, :3],
                "eef_rf_pos": self._eef_rf_state[:, :3] - self._base_state[:, :3],
            }
        )

        fparts_states = {}
        for name, state in self._fparts_states.items():
            pos, rot, vel = (
                state[:, :3] - self._base_state[:, :3],
                state[:, 3:7],
                state[:, 7:13],
            )
            fparts_states[f"{name}_pos"] = pos
            fparts_states[f"{name}_rot"] = rot
            fparts_states[f"{name}_vel"] = vel
        self.states.update(fparts_states)
        walls_states = {}
        for name, state in self._walls_states.items():
            pos, rot = state[:, :3] - self._base_state[:, :3], state[:, 3:7]
            walls_states[f"{name}_pos"] = pos
        self.states.update(walls_states)

        if self._front_wall_idxs is None:
            front_wall_idxs, left_wall_idxs, right_wall_idxs = [], [], []
            for env_handle in self.envs:
                front_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_front"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
                left_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_left"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
                right_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_right"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
            self._front_wall_idxs = torch.tensor(
                front_wall_idxs, device=self.sim_device
            )
            self._left_wall_idxs = torch.tensor(left_wall_idxs, device=self.sim_device)
            self._right_wall_idxs = torch.tensor(
                right_wall_idxs, device=self.sim_device
            )

        if self._fparts_idxs is None:
            fparts_idxs = {name: [] for name in self._fparts_states.keys()}
            for env_handle in self.envs:
                for name, handle in self.fparts_handles.items():
                    fparts_idxs[name].append(
                        self.gym.get_actor_rigid_body_index(
                            env_handle,
                            handle,
                            0,
                            gymapi.DOMAIN_SIM,
                        )
                    )
            self._fparts_idxs = {
                k: torch.tensor(v, device=self.sim_device)
                for k, v in fparts_idxs.items()
            }

        self.states.update(
            {
                "front_wall_cf": self.net_cf[self._front_wall_idxs],
                "left_wall_cf": self.net_cf[self._left_wall_idxs],
                "right_wall_cf": self.net_cf[self._right_wall_idxs],
            }
        )
        self.states.update(
            {
                f"{name}_cf": self.net_cf[idxs]
                for name, idxs in self._fparts_idxs.items()
            }
        )

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_dummy_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.states,
            self.max_episode_length,
        )

    def compute_observations(self):
        self._refresh()
        self.obs_dict["proprioception"][:] = torch.cat(
            [
                self.states[ob][:, :-2]
                if ob in ["q", "cos_q", "sin_q", "dq"]
                else self.states[ob]
                for ob in self._obs_keys
            ],
            dim=-1,
        )
        if len(self._privileged_obs_keys) > 0:
            self.obs_dict["privileged"][:] = torch.cat(
                [self.states[ob] for ob in self._privileged_obs_keys], dim=-1
            )

        # update fields to dump
        # prop fields
        for prop_name in self._prop_dump_info.keys():
            if prop_name in ["q", "cos_q", "sin_q", "dq"]:
                self.dump_fileds[prop_name][:] = self.states[prop_name][:, :-2]
            else:
                self.dump_fileds[prop_name][:] = self.states[prop_name][:]
        # furniture parts
        for fpart_name, fpart_state in self._fparts_states.items():
            self.dump_fileds[fpart_name][:] = fpart_state[:, :7]
        # eef fingers
        self.dump_fileds["leftfinger"][:] = self._rigid_body_state[
            :, self.franka_handles["leftfinger"], :
        ][:, :7]
        self.dump_fileds["rightfinger"][:] = self._rigid_body_state[
            :, self.franka_handles["rightfinger"], :
        ][:, :7]
        self.dump_fileds["franka_base"][:] = self._base_state[:, :7]
        return self.obs_dict

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

    def _compute_osc_torques(self, dpose, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[env_ids, :7], self._qd[env_ids, :7]
        mm = self._mm[env_ids]
        j_eef = self._j_eef[env_ids]
        mm_inv = torch.inverse(mm)
        m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = (
            torch.transpose(j_eef, 1, 2)
            @ m_eef
            @ (self.kp * dpose - self.kd * self.states["eef_vel"][env_ids]).unsqueeze(
                -1
            )
        )

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
            (self.franka_default_dof_pos[:7] - q + np.pi) % (2 * np.pi) - np.pi
        )
        u_null[:, 7:] *= 0
        u_null = mm @ u_null.unsqueeze(-1)
        u += (
            torch.eye(7, device=self.device).unsqueeze(0)
            - torch.transpose(j_eef, 1, 2) @ j_eef_inv
        ) @ u_null

        # Clip the values to be within valid effort range
        u = torch_jit_utils.tensor_clamp(
            u.squeeze(-1),
            -self._franka_effort_limits[:7].unsqueeze(0),
            self._franka_effort_limits[:7].unsqueeze(0),
        )

        return u

    def pre_physics_step(self, actions):
        if self.use_quat_rot:
            assert (
                actions.shape[-1] == 8
            ), "Must provide 8D action for FrankaCubeStackQuatRot"
            pos, quat_rot, gripper = actions[:, :3], actions[:, 3:7], actions[:, 7:]
            # rot_angle: (...,)
            # rot_axis: (..., 3)
            rot_angle, rot_axis = torch_jit_utils.quat_to_angle_axis(quat_rot)
            # get rotation along each axis
            rot = torch.stack([rot_angle * rot_axis[..., i] for i in range(3)], dim=-1)
            actions = torch.cat([pos, rot, gripper], dim=-1)
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        u_arm, u_gripper = self.actions[:, :-1], self.actions[:, -1]

        # Control arm (scale value first)
        u_arm = u_arm * self.cmd_limit / self.action_scale
        u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # Control gripper
        u_fingers = torch.zeros_like(self._gripper_control)
        u_fingers[:, 0] = torch.where(
            u_gripper >= 0.0,
            self.franka_dof_upper_limits[-2].item(),
            self.franka_dof_lower_limits[-2].item(),
        )
        u_fingers[:, 1] = torch.where(
            u_gripper >= 0.0,
            self.franka_dof_upper_limits[-1].item(),
            self.franka_dof_lower_limits[-1].item(),
        )
        # Write gripper command to appropriate tensor buffer
        self._gripper_control[:, :] = u_fingers

        # Deploy actions
        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self._pos_control)
        )
        self.gym.set_dof_actuation_force_tensor(
            self.sim, gymtorch.unwrap_tensor(self._effort_control)
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera


class TRANSICEnvOSC(TRANSICEnvOSCBase):
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
        furniture = cfg["env"]["furniture"]
        self.furniture = furniture_factory(furniture, cfg["seed"])
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

    def _import_furniture_assets(self):
        self._fparts_assets = {}
        for part in self.furniture.parts:
            if part.name in self._fparts_assets:
                continue
            assert_option = sim_config["asset"][part.name]
            self._fparts_assets[part.name] = self.gym.load_asset(
                self.sim,
                ASSET_ROOT,
                part.asset_file,
                assert_option,
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
        super().reset_idx(env_ids)

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

        reset_pos = torch.cat([new_pos, ori_quat], dim=-1)  # (num_parts, num_resets, 7)
        vel = torch.zeros(
            (len(self.furniture.parts), len(env_ids), 6),
            device=self.sim_device,
            dtype=reset_pos.dtype,
        )
        reset_state = torch.cat([reset_pos, vel], dim=-1)  # (num_parts, num_resets, 13)

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


class TRANSICEnvJPCBase(VecTask):
    franka_asset_file = "franka_description/robots/franka_panda_finray.urdf"

    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        num_furniture_parts: int = 0,
    ):
        self._record = record
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.franka_dof_noise = self.cfg["env"]["frankaDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self._num_fparts = num_furniture_parts

        self.cfg["env"]["numActions"] = 8

        # Values to be filled in at runtime
        self.states = {}
        self.franka_handles = {}  # will be dict mapping names to relevant sim handles
        self.fparts_handles = {}  # for furniture part handlers
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed
        self._fparts_names = []  # Names of all furniture parts
        self._init_fparts_states = None  # Initial state of all furniture parts
        self._fparts_states = None  # State of all furniture parts

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = (
            None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        )
        self.net_cf = None  # contact force
        self._eef_state = None  # end effector state (at grasping point)
        self._ftip_center_state = None  # center of fingertips
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._arm_control = None  # Tensor buffer for controlling arm
        self._gripper_control = None  # Tensor buffer for controlling gripper
        self._pos_control = None  # Position actions
        self._global_franka_indices = (
            None  # Unique indices corresponding to all envs in flattened array
        )
        self._global_furniture_part_indices = {}

        self._front_wall_idxs = None
        self._left_wall_idxs = None
        self._right_wall_idxs = None
        self._fparts_idxs = None

        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
        )

        # Franka defaults
        default_pose = self.cfg["env"].get("frankaDefaultDofPos", None)
        default_pose = default_pose or [
            0.12162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
            0.035,
            0.035,
        ]

        self.franka_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)

        # Set control limits
        self.cmd_limit_high = torch.tensor(
            [0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.sim_device
        ).unsqueeze(0)
        self.cmd_limit_low = -self.cmd_limit_high

        damping = 0.05
        self._ik_lambda = torch.eye(6, device=self.sim_device) * (damping**2)

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _import_furniture_assets(self):
        pass

    def _import_franka_pcd(self):
        pass

    def _import_obstacle_pcds(self):
        pass

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 1.3, 0.8, 0.03, table_asset_options)

        # import front obstacle asset
        front_obstacle_asset_options = gymapi.AssetOptions()
        front_obstacle_asset_options.fix_base_link = True
        front_obstacle_asset_file = "furniture_bench/urdf/obstacle_front.urdf"
        front_obstacle_asset = self.gym.load_asset(
            self.sim,
            ASSET_ROOT,
            front_obstacle_asset_file,
            front_obstacle_asset_options,
        )

        # import side obstacle asset
        side_obstacle_asset_options = gymapi.AssetOptions()
        side_obstacle_asset_options.fix_base_link = True
        side_obstacle_asset_file = "furniture_bench/urdf/obstacle_side.urdf"
        side_obstacle_asset = self.gym.load_asset(
            self.sim, ASSET_ROOT, side_obstacle_asset_file, side_obstacle_asset_options
        )

        # import background if recording
        if self._record:
            bg_asset_options = gymapi.AssetOptions()
            bg_asset_options.fix_base_link = True
            background_asset_file = "furniture_bench/urdf/background.urdf"
            background_asset = self.gym.load_asset(
                self.sim, ASSET_ROOT, background_asset_file, bg_asset_options
            )

        # import obstacle pcds
        self._import_obstacle_pcds()

        # import furniture assets
        self._import_furniture_assets()

        # import franka pcds
        self._import_franka_pcd()

        # load franka asset
        franka_asset_file = self.franka_asset_file
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(
            self.sim, ASSET_ROOT, franka_asset_file, asset_options
        )
        franka_dof_stiffness = torch.tensor(
            [400, 400, 400, 400, 400, 400, 400, 5000.0, 5000.0],
            dtype=torch.float,
            device=self.sim_device,
        )
        franka_dof_damping = torch.tensor(
            [80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2],
            dtype=torch.float,
            device=self.sim_device,
        )
        franka_dof_effort = torch.tensor(
            [200, 200, 200, 200, 200, 200, 200, 200, 200],
            dtype=torch.float,
            device=self.sim_device,
        )

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        franka_link_dict = self.gym.get_asset_rigid_body_dict(franka_asset)
        self.franka_grip_site_idx = franka_link_dict["panda_grip_site"]

        print(f"Num Franka Bodies: {self.num_franka_bodies}")
        print(f"Num Franka DOFs: {self.num_franka_dofs}")

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            franka_dof_props["stiffness"][i] = franka_dof_stiffness[i]
            franka_dof_props["damping"][i] = franka_dof_damping[i]
            franka_dof_props["effort"][i] = franka_dof_effort[i]

            self.franka_dof_lower_limits.append(franka_dof_props["lower"][i])
            self.franka_dof_upper_limits.append(franka_dof_props["upper"][i])

        self.franka_dof_lower_limits = torch.tensor(
            self.franka_dof_lower_limits, device=self.sim_device
        )
        self.franka_dof_upper_limits = torch.tensor(
            self.franka_dof_upper_limits, device=self.sim_device
        )
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1

        # Define start pose for franka and table
        table_pos = gymapi.Vec3(0.8, 0.8, 0.4)
        self.franka_pose = gymapi.Transform()
        table_half_width = 0.015
        self._table_surface_z = table_surface_z = table_pos.z + table_half_width
        self.franka_pose.p = gymapi.Vec3(
            0.5 * -table_pos.x + 0.1, 0, table_surface_z + ROBOT_HEIGHT
        )

        # Define start pose for obstacles
        base_tag_pose = gymapi.Transform()
        base_tag_pos = get_mat(
            (0.23 + 0.0715, 0, -ROBOT_HEIGHT), (np.pi, 0, np.pi / 2)
        )[:3, 3]
        base_tag_pose.p = self.franka_pose.p + gymapi.Vec3(
            base_tag_pos[0], base_tag_pos[1], -ROBOT_HEIGHT
        )
        base_tag_pose.p.z = table_surface_z
        self._front_obstacle_pose = gymapi.Transform()
        self._front_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01, 0.0, table_surface_z + 0.015
        )
        self._front_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )
        self._right_obstacle_pose = gymapi.Transform()
        self._right_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
            -0.175,
            table_surface_z + 0.015,
        )
        self._right_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )
        self._left_obstacle_pose = gymapi.Transform()
        self._left_obstacle_pose.p = gymapi.Vec3(
            base_tag_pose.p.x + 0.37 + 0.01 - 0.075,
            0.175,
            table_surface_z + 0.015,
        )
        self._left_obstacle_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0.5 * np.pi
        )

        if self._record:
            bg_pos = gymapi.Vec3(-0.8, 0, 0.75)
            bg_pose = gymapi.Transform()
            bg_pose.p = gymapi.Vec3(bg_pos.x, bg_pos.y, bg_pos.z)

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        max_agg_bodies = (
            num_franka_bodies
            + 4
            + self._num_fparts
            + (1 if self._record else 0)  # for background
        )  # 1 for table, front obstacle, left obstacle, right obstacle
        max_agg_shapes = (
            num_franka_shapes + 4 + self._num_fparts + (1 if self._record else 0)
        )

        self.frankas = []
        self.envs = []

        # Create environments
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # camera handler for view rendering
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env_ptr,
                        isaac_gym=self.gym,
                    )
                )

            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: franka should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create franka
            franka_actor = self.gym.create_actor(
                env_ptr,
                franka_asset,
                self.franka_pose,
                "franka",
                i,
                0,
            )
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, table_pos.z)
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i, 0
            )
            table_props = self.gym.get_actor_rigid_shape_properties(
                env_ptr, table_actor
            )
            table_props[0].friction = 0.10
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(
                env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1)
            )
            front_obstacle_actor = self.gym.create_actor(
                env_ptr,
                front_obstacle_asset,
                self._front_obstacle_pose,
                "obstacle_front",
                i,
                0,
            )
            left_obstacle_actor = self.gym.create_actor(
                env_ptr,
                side_obstacle_asset,
                self._left_obstacle_pose,
                "obstacle_left",
                i,
                0,
            )
            right_obstacle_actor = self.gym.create_actor(
                env_ptr,
                side_obstacle_asset,
                self._right_obstacle_pose,
                "obstacle_right",
                i,
                0,
            )

            if self._record:
                bg_actor = self.gym.create_actor(
                    env_ptr,
                    background_asset,
                    bg_pose,
                    "background",
                    i,
                    0,
                )

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # Create furniture parts
            self._create_furniture_parts(env_ptr, i)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)

        self.franka_from_origin_mat = get_mat(
            [self.franka_pose.p.x, self.franka_pose.p.y, self.franka_pose.p.z],
            [0, 0, 0],
        )
        self.base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]
        april_to_sim_mat = self.franka_from_origin_mat @ self.base_tag_from_robot_mat
        self.april_to_sim_mat = torch.tensor(april_to_sim_mat, device=self.sim_device)

        # Setup init state buffer for all furniture parts
        self._init_fparts_states = {
            part_name: torch.zeros(self.num_envs, 13, device=self.sim_device)
            for part_name in self._fparts_names
        }

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        franka_handle = 0
        self.franka_handles = {
            # Franka
            "hand": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_hand"
            ),
            "base": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_link0"
            ),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_leftfinger_tip"
            ),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_rightfinger_tip"
            ),
            "leftfinger": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_leftfinger"
            ),
            "rightfinger": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_rightfinger"
            ),
            "grip_site": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "panda_grip_site"
            ),
            "fingertip_center": self.gym.find_actor_rigid_body_handle(
                env_ptr, franka_handle, "tri_finger_center"
            ),
        }
        self.fparts_handles = {
            part_name: self.gym.find_actor_index(env_ptr, part_name, gymapi.DOMAIN_ENV)
            for part_name in self._fparts_names
        }
        self.walls_handles = {
            part_name: self.gym.find_actor_index(env_ptr, part_name, gymapi.DOMAIN_ENV)
            for part_name in ["obstacle_front", "obstacle_left", "obstacle_right"]
        }

        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(
            self.num_envs, -1, 2
        )
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(
            self.num_envs, -1, 13
        )
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.franka_handles["grip_site"], :]
        self._ftip_center_state = self._rigid_body_state[
            :, self.franka_handles["fingertip_center"], :
        ]
        self._base_state = self._rigid_body_state[:, self.franka_handles["base"], :]
        self._eef_lf_state = self._rigid_body_state[
            :, self.franka_handles["leftfinger_tip"], :
        ]
        self._eef_rf_state = self._rigid_body_state[
            :, self.franka_handles["rightfinger_tip"], :
        ]
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        self._j_eef = jacobian[
            :, self.franka_grip_site_idx - 1, :, :7
        ]  # -1 due to fixed base link.
        self._fparts_states = {
            part_name: self._root_state[:, part_idx, :]
            for part_name, part_idx in self.fparts_handles.items()
        }
        self._walls_states = {
            part_name: self._root_state[:, part_idx, :]
            for part_name, part_idx in self.walls_handles.items()
        }
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.net_cf = gymtorch.wrap_tensor(_net_cf)

        # Initialize actions
        self._pos_control = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )

        # Initialize control
        self._arm_control = self._pos_control[:, :7]
        self._gripper_control = self._pos_control[:, 7:9]

        # Initialize indices
        self._global_franka_indices = torch.tensor(
            [
                self.gym.find_actor_index(env, "franka", gymapi.DOMAIN_SIM)
                for env in self.envs
            ],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)
        self._global_furniture_part_indices = {
            part_name: torch.tensor(
                [
                    self.gym.find_actor_index(env, part_name, gymapi.DOMAIN_SIM)
                    for env in self.envs
                ],
                dtype=torch.int32,
                device=self.sim_device,
            ).view(self.num_envs, -1)
            for part_name in self._fparts_names
        }

    def _create_furniture_parts(self, env_prt, i):
        pass

    def _update_states(self):
        self.states.update(
            {
                # Franka
                "q": self._q[:, :],
                "cos_q": torch.cos(self._q[:, :]),
                "sin_q": torch.sin(self._q[:, :]),
                "dq": self._qd[:, :],
                "q_gripper": self._q[:, -2:],
                "eef_pos": self._eef_state[:, :3] - self._base_state[:, :3],
                "eef_quat": self._eef_state[:, 3:7],
                "ftip_center_pos": self._ftip_center_state[:, :3]
                - self._base_state[:, :3],
                "ftip_center_quat": self._ftip_center_state[:, 3:7],
                "gripper_width": torch.sum(self._q[:, -2:], dim=-1, keepdim=True),
                "eef_vel": self._eef_state[:, 7:],  # still required for OSC
                "eef_lf_pos": self._eef_lf_state[:, :3] - self._base_state[:, :3],
                "eef_rf_pos": self._eef_rf_state[:, :3] - self._base_state[:, :3],
            }
        )

        fparts_states = {}
        for name, state in self._fparts_states.items():
            pos, rot, vel = (
                state[:, :3] - self._base_state[:, :3],
                state[:, 3:7],
                state[:, 7:13],
            )
            fparts_states[f"{name}_pos"] = pos
            fparts_states[f"{name}_rot"] = rot
            fparts_states[f"{name}_vel"] = vel
        self.states.update(fparts_states)
        walls_states = {}
        for name, state in self._walls_states.items():
            pos, rot = state[:, :3] - self._base_state[:, :3], state[:, 3:7]
            walls_states[f"{name}_pos"] = pos
        self.states.update(walls_states)

        if self._front_wall_idxs is None:
            front_wall_idxs, left_wall_idxs, right_wall_idxs = [], [], []
            for env_handle in self.envs:
                front_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_front"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
                left_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_left"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )
                right_wall_idxs.append(
                    self.gym.get_actor_rigid_body_index(
                        env_handle,
                        self.walls_handles["obstacle_right"],
                        0,
                        gymapi.DOMAIN_SIM,
                    )
                )

            self._front_wall_idxs = torch.tensor(
                front_wall_idxs, device=self.sim_device
            )
            self._left_wall_idxs = torch.tensor(left_wall_idxs, device=self.sim_device)
            self._right_wall_idxs = torch.tensor(
                right_wall_idxs, device=self.sim_device
            )

        if self._fparts_idxs is None:
            fparts_idxs = {name: [] for name in self._fparts_states.keys()}
            for env_handle in self.envs:
                for name, handle in self.fparts_handles.items():
                    fparts_idxs[name].append(
                        self.gym.get_actor_rigid_body_index(
                            env_handle,
                            handle,
                            0,
                            gymapi.DOMAIN_SIM,
                        )
                    )
            self._fparts_idxs = {
                k: torch.tensor(v, device=self.sim_device)
                for k, v in fparts_idxs.items()
            }

        self.states.update(
            {
                "front_wall_cf": self.net_cf[self._front_wall_idxs],
                "left_wall_cf": self.net_cf[self._left_wall_idxs],
                "right_wall_cf": self.net_cf[self._right_wall_idxs],
            }
        )
        self.states.update(
            {
                f"{name}_cf": self.net_cf[idxs]
                for name, idxs in self._fparts_idxs.items()
            }
        )

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_dummy_reward(
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.states,
            self.max_episode_length,
        )

    def compute_observations(self):
        self._refresh()
        self.obs_dict["proprioception"][:] = torch.cat(
            [
                self.states[ob][:, :-2]
                if ob in ["q", "cos_q", "sin_q", "dq"]
                else self.states[ob]
                for ob in self._obs_keys
            ],
            dim=-1,
        )
        if len(self._privileged_obs_keys) > 0:
            self.obs_dict["privileged"][:] = torch.cat(
                [self.states[ob] for ob in self._privileged_obs_keys], dim=-1
            )
        return self.obs_dict

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

    def _compute_ik(self, dpose):
        j_eef_T = torch.transpose(self._j_eef, 1, 2)
        u = (
            j_eef_T
            @ torch.inverse(self._j_eef @ j_eef_T + self._ik_lambda)
            @ dpose.unsqueeze(-1)
        ).view(self.num_envs, -1)
        return u

    def step(self, actions):
        """
        Will internally invoke JPC controller N times until achieve the desired joint position
        Here `actions` specify desired joint position
        """
        # randomize actions
        if self.dr_randomizations.get("actions", None):
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)
        self.actions = actions.clone().to(self.device)

        # Split arm and gripper command
        goal_q, gripper_actions = self.actions[:, :-1], self.actions[:, -1]

        # invoke JPC controller N times until achieve the desired joint position
        for _ in range(10):
            arm_control_target = goal_q[:]
            arm_control_target = torch.clamp(
                arm_control_target,
                min=self.franka_dof_lower_limits[:7],
                max=self.franka_dof_upper_limits[:7],
            )
            self._arm_control[:, :] = arm_control_target[:, :]

            # gripper control
            u_fingers = torch.zeros_like(self._gripper_control)
            u_fingers[:, 0] = torch.where(
                gripper_actions >= 0.0,
                self.franka_dof_upper_limits[-2].item(),
                self.franka_dof_lower_limits[-2].item(),
            )
            u_fingers[:, 1] = torch.where(
                gripper_actions >= 0.0,
                self.franka_dof_upper_limits[-1].item(),
                self.franka_dof_lower_limits[-1].item(),
            )
            # Write gripper command to appropriate tensor buffer
            self._gripper_control[:, :] = u_fingers

            # Deploy actions
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self._pos_control)
            )

            # step physics and render each frame
            for i in range(self.control_freq_inv):
                self.gym.simulate(self.sim)

            if self.camera_obs is not None:
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)

            if self.camera_obs is not None:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

            # post physics
            self.compute_observations()

            if self.camera_obs is not None:
                self.gym.end_access_image_tensors(self.sim)

            if self._rgb_viewr_renderer is not None:
                self.render()

            # break when all joints are within 1e-3 of desired joint position
            if torch.all(
                torch.max(
                    torch.abs(self.states["q"][:, :7] - goal_q),
                    dim=-1,
                )[0]
                < 1e-3
            ):
                break

        # now update buffer
        self.progress_buf += 1
        self.randomize_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_reward(self.actions)

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (
            self.reset_buf != 0
        )

        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](
                self.obs_buf
            )

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(
                self.obs_buf, -self.clip_obs, self.clip_obs
            ).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def pre_physics_step(self, actions):
        pass

    def post_physics_step(self):
        pass

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera


class TRANSICEnvJPC(TRANSICEnvJPCBase):
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
        furniture = cfg["env"]["furniture"]
        self.furniture = furniture_factory(furniture, cfg["seed"])
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            num_furniture_parts=len(self.furniture.parts),
        )

    def _import_furniture_assets(self):
        self._fparts_assets = {}
        for part in self.furniture.parts:
            if part.name in self._fparts_assets:
                continue
            assert_option = sim_config["asset"][part.name]
            self._fparts_assets[part.name] = self.gym.load_asset(
                self.sim,
                ASSET_ROOT,
                part.asset_file,
                assert_option,
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
        super().reset_idx(env_ids)

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

        reset_pos = torch.cat([new_pos, ori_quat], dim=-1)  # (num_parts, num_resets, 7)
        vel = torch.zeros(
            (len(self.furniture.parts), len(env_ids), 6),
            device=self.sim_device,
            dtype=reset_pos.dtype,
        )
        reset_state = torch.cat([reset_pos, vel], dim=-1)  # (num_parts, num_resets, 13)

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


@torch.jit.script
def compute_dummy_reward(reset_buf, progress_buf, actions, states, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], float) -> Tuple[Tensor, Tensor]

    # dummy rewards
    rewards = torch.zeros_like(reset_buf)
    reset_buf = torch.where(
        progress_buf >= max_episode_length - 1,
        torch.ones_like(reset_buf),
        reset_buf,
    )

    return rewards, reset_buf
