"""Define additional parameters based on real-world config for simulator."""

from isaacgym import gymapi
from transic_envs.envs.core.furniture.config import config


sim_config = config.copy()

# Simulator options.
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = True
sim_params.physx.solver_type = 1
sim_params.physx.bounce_threshold_velocity = 0.02
sim_params.physx.num_position_iterations = 20
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.rest_offset = 0.0
sim_params.physx.contact_offset = 0.002
sim_params.physx.friction_offset_threshold = 0.01
sim_params.physx.friction_correlation_distance = 0.0005
sim_params.physx.use_gpu = True

sim_config["sim_params"] = sim_params
sim_config["parts"] = {"friction": 0.15}
sim_config["table"] = {"friction": 0.10}
sim_config["asset"] = {}

# Parameters for the robot.
sim_config["robot"].update(
    {
        "kp": [90, 90, 90, 70.0, 60.0, 80.0],  # Default positional gains.
        "kv": None,  # Default velocity gains.
        "arm_frictions": [
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
            0.05,
        ],  # Default arm friction.
        "gripper_frictions": [15.0, 15.0],  # Default gripper friction.
        "gripper_torque": 13,  # Default torque for gripper.
    }
)

# Parameters for the light.
sim_config["lights"] = [
    {
        "color": [0.8, 0.8, 0.8],
        "ambient": [0.35, 0.35, 0.35],
        "direction": [0.1, -0.03, 0.2],
    }
]

"""
Set density for each furniture part.
  - The volume is estimated using Belnder.
  - The mass is estimated using 3D printer slicer.
"""


def default_asset_options():
    asset_options = gymapi.AssetOptions()
    asset_options.flip_visual_attachments = False
    asset_options.fix_base_link = False
    asset_options.thickness = 0.0
    asset_options.density = 600.0
    # asset_options.armature = 0.01
    asset_options.linear_damping = 0.0
    asset_options.max_linear_velocity = 1000.0
    asset_options.angular_damping = 0.0
    asset_options.max_angular_velocity = 1000.0
    asset_options.disable_gravity = False
    asset_options.enable_gyroscopic_forces = True

    return asset_options


# Volume: 302802 mm^3
# Mass: 151g
square_table_top_asset_options = default_asset_options()
square_table_top_asset_options.density = 498.68
sim_config["asset"]["square_table_top"] = square_table_top_asset_options

# Volume: 62435.mm^3
# Mass: 23.1g
leg_asset_options = default_asset_options()
leg_asset_options.density = 369.98
sim_config["asset"]["square_table_leg1"] = leg_asset_options
leg_asset_options = default_asset_options()
leg_asset_options.density = 369.98
sim_config["asset"]["square_table_leg2"] = leg_asset_options
leg_asset_options = default_asset_options()
leg_asset_options.density = 369.98
sim_config["asset"]["square_table_leg3"] = leg_asset_options
leg_asset_options = default_asset_options()
leg_asset_options.density = 369.98
sim_config["asset"]["square_table_leg4"] = leg_asset_options
leg_asset_options = default_asset_options()
leg_asset_options.density = 369.98
sim_config["asset"]["leg"] = leg_asset_options  # for square_table_patch_fix
