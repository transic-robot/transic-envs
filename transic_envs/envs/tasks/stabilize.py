from typing import Tuple, Dict

from isaacgym import gymapi, gymtorch
import torch

from transic_envs.envs.core import TRANSICEnvPCD, TRANSICEnvOSC


class StabilizeEnv(TRANSICEnvOSC):
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
        self._success_weight = cfg["env"]["successWeight"]
        self._failure_weight = cfg["env"]["failureWeight"]
        qd_penalty = cfg["env"]["qdPenalty"]
        assert qd_penalty >= 0
        self._qd_penalty = qd_penalty
        action_penalty = cfg["env"]["actionPenalty"]
        assert action_penalty >= 0
        self._action_penalty = action_penalty

        self._valid_x_range = (0.2315 - -0.3, 0.3815 - -0.3)
        self._valid_y_range = (-0.175, 0.175)

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
            self.failure_buf[:],
        ) = compute_stabilize_reward(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            success_buf=self.success_buf,
            failure_buf=self.failure_buf,
            states=self.states,
            action=actions,
            goal_x=0.28 - -0.3,
            goal_y=0.08,
            max_episode_length=self.max_episode_length,
            success_weight=self._success_weight,
            failure_weight=self._failure_weight,
            qd_penalty=self._qd_penalty,
            action_penalty=self._action_penalty,
            valid_x_range=self._valid_x_range,
            valid_y_range=self._valid_y_range,
        )


class StabilizePCDEnv(TRANSICEnvPCD):
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
        self._success_weight = cfg["env"]["successWeight"]
        self._failure_weight = cfg["env"]["failureWeight"]
        qd_penalty = cfg["env"]["qdPenalty"]
        assert qd_penalty >= 0
        self._qd_penalty = qd_penalty
        action_penalty = cfg["env"]["actionPenalty"]
        assert action_penalty >= 0
        self._action_penalty = action_penalty

        self._valid_x_range = (0.2315 - -0.3, 0.3815 - -0.3)
        self._valid_y_range = (-0.175, 0.175)

        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
        )

    def compute_reward(self, actions):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
            self.failure_buf[:],
        ) = compute_stabilize_reward(
            reset_buf=self.reset_buf,
            progress_buf=self.progress_buf,
            success_buf=self.success_buf,
            failure_buf=self.failure_buf,
            states=self.states,
            action=actions,
            goal_x=0.28 - -0.3,
            goal_y=0.08,
            max_episode_length=self.max_episode_length,
            success_weight=self._success_weight,
            failure_weight=self._failure_weight,
            qd_penalty=self._qd_penalty,
            action_penalty=self._action_penalty,
            valid_x_range=self._valid_x_range,
            valid_y_range=self._valid_y_range,
        )


@torch.jit.script
def compute_stabilize_reward(
    reset_buf,
    progress_buf,
    success_buf,
    failure_buf,
    states: Dict[str, torch.Tensor],
    action,
    goal_x: float,
    goal_y: float,
    max_episode_length: int,
    success_weight: float,
    failure_weight: float,
    qd_penalty: float,
    action_penalty: float,
    valid_x_range: Tuple[float, float],
    valid_y_range: Tuple[float, float],
):
    # get pos for all legs
    legs_pos = torch.stack(
        [states[f"square_table_leg{i + 1}_pos"] for i in range(4)], dim=1
    )  # (num_envs, 4, 3)
    legs_x, legs_y = legs_pos[..., 0], legs_pos[..., 1]  # (num_envs, 4)
    legs_x_in_range = (legs_x >= valid_x_range[0]) & (
        legs_x <= valid_x_range[1]
    )  # (num_envs, n_legs)
    legs_y_in_range = (legs_y >= valid_y_range[0]) & (legs_y <= valid_y_range[1])
    legs_in_range = legs_x_in_range & legs_y_in_range  # (num_envs, n_legs)
    any_leg_in_range = torch.any(legs_in_range, dim=-1)  # (num_envs,)
    # legs shouldn't be in region
    failure = any_leg_in_range | failure_buf

    # get pos for table
    table_pos = states["square_table_top_pos"]  # (num_envs, 3)
    table_x, table_y = table_pos[..., 0], table_pos[..., 1]  # (num_envs,)
    table_x_in_range = (table_x >= valid_x_range[0]) & (
        table_x <= valid_x_range[1]
    )  # (num_envs,)
    table_y_in_range = (table_y >= valid_y_range[0]) & (table_y <= valid_y_range[1])
    table_in_range = table_x_in_range & table_y_in_range  # (num_envs,)

    close_to_goal = ((table_x - goal_x).abs() < 0.01) & (
        (table_y - goal_y).abs() < 0.01
    )
    succeeded = close_to_goal & table_in_range & ~failure

    # qd penalty
    qd_norm = torch.linalg.norm(states["dq"], dim=-1)  # (num_envs,)

    # action penalty
    action_norm = torch.linalg.norm(action, dim=-1)

    reward = (
        succeeded * success_weight
        + failure * failure_weight
        - qd_penalty * qd_norm
        - action_penalty * action_norm
    )
    reset_buf = torch.where(
        (progress_buf >= max_episode_length - 1) | succeeded | failure,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    success_buf = succeeded | success_buf
    failure_buf = failure | failure_buf
    return reward, reset_buf, success_buf, failure_buf
