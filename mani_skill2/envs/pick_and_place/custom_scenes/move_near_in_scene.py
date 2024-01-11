from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional
from copy import deepcopy

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2 import ASSET_DIR, format_path
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.utils.geometry import (
    get_axis_aligned_bbox_for_actor,
    angle_between_vec
)

from .base_env import CustomSceneEnv


class MoveNearInSceneEnv(CustomSceneEnv):
    DEFAULT_ASSET_ROOT: str
    DEFAULT_SCENE_ROOT: str
    DEFAULT_MODEL_JSON: str

    obj: sapien.Actor  # target object
    
    def __init__(
        self,
        **kwargs,
    ):
        self.episode_objs = [None] * 3
        self.episode_model_ids = [None] * 3
        self.episode_model_scales = [None] * 3
        self.episode_model_bbox_sizes = [None] * 3
        self.episode_model_init_xyzs = [None] * 3
        
        self.obj_init_options = {}
        
        super().__init__(**kwargs)

    # def _setup_lighting(self):
    #     super()._setup_lighting()
    #     self._scene.add_directional_light([-1, 1, -1], [1, 1, 1])
        
    def _load_actors(self):
        self._load_arena_helper()        
        self._load_model()
        for obj in self.episode_objs:
            obj.set_damping(0.1, 0.1)
            
    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError
    
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        
        self.obj_init_options = options.get("obj_init_options", {})
        
        self.set_episode_rng(seed)
        model_scales = options.get("model_scales", None)
        model_ids = options.get("model_ids", None)
        reconfigure = options.get("reconfigure", False)
        _reconfigure = self._set_model(model_ids, model_scales)
        reconfigure = _reconfigure or reconfigure
                
        options["reconfigure"] = reconfigure
        
        return super().reset(seed=self._episode_seed, options=options)

    # def _setup_lighting(self):
    #     super()._setup_lighting()
    #     # self._scene.add_directional_light([0, 0, -1], [1, 1, 1])
    #     self._scene.add_point_light([-0.2, 0.0, 1.4], [1, 1, 1])
        
    def _set_model(self, model_ids, model_scales):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        if model_ids is None:
            model_ids = []
            for _ in range(3):
                model_ids.append(random_choice(self.model_ids, self._episode_rng))
        if set(model_ids) != set(self.episode_model_ids):
            self.model_ids = model_ids
            reconfigure = True

        if model_scales is None:
            model_scales = []
            for model_id in self.model_ids:
                model_scales.append(self.model_db[model_id].get("scales"))
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure
    
    def _settle(self, t):
        sim_steps = int(self.sim_freq * t)
        for _ in range(sim_steps):
            self._scene.step()

    def _initialize_actors(self):
        # The object will fall from a certain initial height
        obj_init_xy = self.obj_init_options.get("init_xy", None)
        if obj_init_xy is None:
            obj_init_xy = np.array([-0.2, 0.2]) + self._episode_rng.uniform(-0.05, 0.05, [2])
        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z + 0.5 # let object fall onto the table
        obj_init_rot_quat = self.obj_init_options.get("init_rot_quat", None)
        p = np.hstack([obj_init_xy, obj_init_z])
        q = [1, 0, 0, 0] if obj_init_rot_quat is None else obj_init_rot_quat

        # Rotate along z-axis
        if self.obj_init_options.get("init_rand_rot_z", False):
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = qmult(euler2quat(0, 0, ori), q)

        # Rotate along a random axis by a small angle
        if (init_rand_axis_rot_range := self.obj_init_options.get("init_rand_axis_rot_range", 0.0)) > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, init_rand_axis_rot_range)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.obj.set_pose(Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later
        self.agent.robot.set_pose(Pose([-10, 0, 0]))

        # Lock rotation around x and y
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)   
        
        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)
        
        self.obj_height_after_settle = self.obj.pose.p[2]
        
    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.obj.pose.transform(self.obj.cmass_local_pose)
    
    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj_pose),
                tcp_to_obj_pos=self.obj_pose.p - self.tcp.pose.p,
            )
        return obs

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def evaluate(self, **kwargs):
        is_grasped = self.agent.check_grasp(self.obj, max_angle=80)
        if is_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
            self.lifted_obj_during_consecutive_grasp = False
            
        contacts = self._scene.get_contacts()
        flag = True
        robot_link_names = [x.name for x in self.agent.robot.get_links()]
        for contact in contacts:
            actor_0, actor_1 = contact.actor0, contact.actor1
            other_obj_contact_actor_name = None
            if actor_0.name == self.obj.name:
                other_obj_contact_actor_name = actor_1.name
            elif actor_1.name == self.obj.name:
                other_obj_contact_actor_name = actor_0.name
            if other_obj_contact_actor_name is not None:
                # the object is in contact with an actor
                contact_impulse = np.sum([point.impulse for point in contact.points], axis=0)
                if (other_obj_contact_actor_name not in robot_link_names) and (np.linalg.norm(contact_impulse) > 1e-6):
                    # the object has contact with an actor other than the robot link, so the object is not yet lifted up
                    # print(other_obj_contact_actor_name, np.linalg.norm(contact_impulse))
                    flag = False
                    break

        consecutive_grasp = (self.consecutive_grasp >= 5)
        diff_obj_height = self.obj.pose.p[2] - self.obj_height_after_settle
        self.lifted_obj_during_consecutive_grasp = self.lifted_obj_during_consecutive_grasp or flag
        
        if self.require_lifting_obj_for_success:
            success = self.lifted_obj_during_consecutive_grasp
        else:
            success = consecutive_grasp
        return dict(
            is_grasped=is_grasped,
            consecutive_grasp=consecutive_grasp,
            lifted_object=self.lifted_obj_during_consecutive_grasp,
            lifted_object_significantly=self.lifted_obj_during_consecutive_grasp and (diff_obj_height > 0.02),
            success=success,
        )

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0
        if info["success"]:
            reward = 1.0
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 1.0