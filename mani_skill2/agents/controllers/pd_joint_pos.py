from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from gymnasium import spaces

from ..base_controller import BaseController, ControllerConfig


class PDJointPosController(BaseController):
    config: "PDJointPosControllerConfig"

    def _get_joint_limits(self):
        qlimits = self.articulation.get_qlimits()[self.joint_indices]
        # Override if specified
        if self.config.lower is not None:
            qlimits[:, 0] = self.config.lower
        if self.config.upper is not None:
            qlimits[:, 1] = self.config.upper
        return qlimits

    def _initialize_action_space(self):
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        self.action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        n = len(self.joints)
        stiffness = np.broadcast_to(self.config.stiffness, n)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            joint.set_drive_property(
                stiffness[i], damping[i], force_limit=force_limit[i], mode=self.config.drive_mode
            )
            joint.set_friction(friction[i])

    def reset(self):
        super().reset()
        self._step = 0  # counter of simulation steps after action is set
        self._start_qpos = self.qpos
        self._target_qpos = self.qpos
        self._last_drive_qpos_targets = self.qpos
        self._interpolation_path = None
        self._last_interpolation_success = False

    def set_drive_targets(self, targets):
        self._last_drive_qpos_targets = targets
        for i, joint in enumerate(self.joints):
            joint.set_drive_target(targets[i])

    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)

        self._step = 0
        self._start_qpos = self.qpos
        _last_target_qpos = self._target_qpos

        if self.config.use_delta:
            if self.config.use_target:
                if self.config.delta_target_from_last_drive_target:
                    self._target_qpos = self._last_drive_qpos_targets
                    self._start_qpos = self._last_drive_qpos_targets
                self._target_qpos = self._target_qpos + action
                if self.config.clip_target:
                    joint_limits = self.articulation.get_qlimits()[self.joint_indices]
                    self._target_qpos = np.clip(
                        self._target_qpos,
                        joint_limits[:, 0] - self.config.clip_target_thres,
                        joint_limits[:, 1] + self.config.clip_target_thres,
                    )
            else:
                self._target_qpos = self._start_qpos + action
        else:
            # Compatible with mimic
            self._target_qpos = np.broadcast_to(action, self._start_qpos.shape)

        if self.config.small_action_repeat_last_target:
            small_action_idx = np.where(np.abs(action) < 1e-3)[0]
            self._target_qpos[small_action_idx] = _last_target_qpos[small_action_idx]

        if self.config.interpolate:
            self._setup_qpos_interpolation()
        else:
            self.set_drive_targets(self._target_qpos)
            
    def _setup_qpos_interpolation(self):
        if self.config.interpolate_by_planner:
            # self._interpolation_path, self._last_interpolation_success = self.plan_joint_path(
            #     self._start_qpos, self._target_qpos, 
            #     self.config.interpolate_planner_vlim, self.config.interpolate_planner_alim,
            # )
            if (self._interpolation_path is None) or (not self._last_interpolation_success) or (self.config.interpolate_planner_init_no_vel):
                self._interpolation_path, self._last_interpolation_success = self.plan_joint_path(
                    self._start_qpos, self._target_qpos, 
                    self.config.interpolate_planner_vlim, self.config.interpolate_planner_alim,
                )
                # print(self.qpos, self._start_qpos, self._target_qpos, self._interpolation_path[0], self._interpolation_path[min(self._sim_steps, len(self._interpolation_path) - 1)])
            else:
                last_start_qpos = self._interpolation_path[0]
                # last_end_qpos = self.qpos
                if self.config.use_target:
                    last_end_qpos = self.qpos
                    # length_last_path = min(self._sim_steps, len(self._interpolation_path) - 1) + 1 # including start and end
                    # last_end_qpos = self._interpolation_path[length_last_path - 1]
                else:
                    length_last_path = min(self._sim_steps, len(self._interpolation_path) - 1) + 1 # including start and end
                    last_end_qpos = self._interpolation_path[length_last_path - 1]
                    # last_end_qpos = self._start_qpos
                # include last interpolation result during current interpolation to keep velocity continuous
                self._interpolation_path, success = self.plan_joint_path(
                    last_start_qpos, self._target_qpos, 
                    self.config.interpolate_planner_vlim, self.config.interpolate_planner_alim,
                    additional_waypoints=[last_end_qpos]
                )
                if success:
                    closest_idx_to_last_end_qpos = np.argmin(np.linalg.norm(self._interpolation_path - last_end_qpos[None, :], axis=-1))
                    self._interpolation_path = self._interpolation_path[closest_idx_to_last_end_qpos:]
                self._last_interpolation_success = success
        else:
            step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
            # linear interpolation
            self._interpolation_path = np.array(
                [self._start_qpos + step_size * i for i in range(self._sim_steps + 1)]
            )
        # print("interpolation path length", len(self._interpolation_path))
        # print("interpolation at next ctrl", self._interpolation_path[min(self._sim_steps, len(self._interpolation_path) - 1)])
        # print()

    def before_simulation_step(self):
        self._step += 1
        
        # Compute the next target
        if self.config.interpolate:
            targets = self._interpolation_path[min(self._step, len(self._interpolation_path) - 1)]
            self.set_drive_targets(targets)

    def get_state(self) -> dict:
        if self.config.use_target:
            return {"target_qpos": self._target_qpos}
        return {}

    def set_state(self, state: dict):
        if self.config.use_target:
            self._target_qpos = state["target_qpos"]


@dataclass
class PDJointPosControllerConfig(ControllerConfig):
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    drive_mode: str = "force"
    use_delta: bool = False
    use_target: bool = False
    delta_target_from_last_drive_target: bool = False
    clip_target: bool = False
    clip_target_thres: float = 0.01
    interpolate: bool = False
    interpolate_by_planner: bool = False
    interpolate_planner_init_no_vel: bool = False
    interpolate_planner_vlim: float = 1.5
    interpolate_planner_alim: float = 2.0
    small_action_repeat_last_target: bool = False
    normalize_action: bool = True
    controller_cls = PDJointPosController


class PDJointPosMimicController(PDJointPosController):
    def _get_joint_limits(self):
        joint_limits = super()._get_joint_limits()
        diff = joint_limits[0:-1] - joint_limits[1:]
        assert np.allclose(diff, 0), "Mimic joints should have the same limit"
        return joint_limits[0:1]


class PDJointPosMimicControllerConfig(PDJointPosControllerConfig):
    controller_cls = PDJointPosMimicController
