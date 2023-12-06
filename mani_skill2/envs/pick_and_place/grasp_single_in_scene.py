from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

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

from .base_env import StationaryManipulationEnv


class GraspSingleInSceneEnv(StationaryManipulationEnv):
    DEFAULT_ASSET_ROOT: str
    DEFAULT_SCENE_ROOT: str
    DEFAULT_MODEL_JSON: str

    obj: sapien.Actor  # target object

    def __init__(
        self,
        asset_root: str = None,
        scene_root: str = None,
        scene_name: str = None,
        model_json: str = None,
        model_ids: List[str] = (),
        obj_init_rand_rot_z_enabled=True,
        obj_init_rand_rot_range=0,
        obj_init_fixed_xy_pos=None,
        obj_init_fixed_z_rot=None,
        obj_init_rot_quat=None,
        robot_init_fixed_xy_pos=None,
        robot_init_fixed_rot_quat=None,
        **kwargs,
    ):
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))
        
        if scene_root is None:
            scene_root = self.DEFAULT_SCENE_ROOT
        self.scene_root = Path(format_path(scene_root))
        self.scene_name = scene_name

        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        # NOTE(jigu): absolute path will overwrite asset_root
        model_json = self.asset_root / format_path(model_json)

        if not model_json.exists():
            raise FileNotFoundError(
                f"{model_json} is not found."
                "Please download the corresponding assets:"
                "`python -m mani_skill2.utils.download_asset ${ENV_ID}`."
            )
        self.model_db: Dict[str, Dict] = load_json(model_json)

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json
        self.model_ids = model_ids

        self.model_id = model_ids[0]
        self.model_scale = None
        self.model_bbox_size = None

        self.obj_init_rot_quat = obj_init_rot_quat # the rotation quaternion to initialize the target object, before random perturbation below
        self.obj_init_rand_rot_z_enabled = obj_init_rand_rot_z_enabled # whether to randomize the z rotation of target object upon environment reset
        self.obj_init_rand_rot_range = obj_init_rand_rot_range # the range to rotate the target object along a random axis by a small angle upon environment reset
        self.obj_init_fixed_xy_pos = obj_init_fixed_xy_pos # the xy position to fix the target object upon environment reset
        self.obj_init_fixed_z_rot = obj_init_fixed_z_rot # the z rotation to fix the target object upon environment reset
        self.robot_init_fixed_xy_pos = robot_init_fixed_xy_pos # the xy position to fix the robot upon environment reset
        self.robot_init_fixed_rot_quat = robot_init_fixed_rot_quat # the rotation quaternion to fix the robot upon environment reset
        
        self.arena = None
        self.obj_init_actual_z = None # actual target object z position at env reset
        self.obj_init_actual_xy_center = None # actual target object xy position at env reset
        
        self.obj = None
        
        self._check_assets()
        super().__init__(**kwargs)

    # def _setup_lighting(self):
    #     super()._setup_lighting()
    #     self._scene.add_directional_light([-1, 1, -1], [1, 1, 1])
        
    def _check_assets(self):
        """Check whether the assets exist."""
        pass

    def _load_actors(self):
        builder = self._scene.create_actor_builder()
        if self.scene_name is None:
            # scene_path = str(self.scene_root / "stages/Baked_sc1_staging_00.glb")
            scene_path = str(self.scene_root / "stages/Baked_sc1_staging_table83_82cm.glb") # hardcoded for now
            # scene_path = str(self.scene_root / "stages/Baked_sc1_staging_table_small_83_82cm.glb") # hardcoded for now
        else:
            scene_path = str(self.scene_root / "stages" / f"{self.scene_name}.glb")
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
        # NOTE: use nonconvex collision for static scene
        builder.add_nonconvex_collision_from_file(scene_path, scene_pose)
        builder.add_visual_from_file(scene_path, scene_pose)
        self.arena = builder.build_static()
        # Add offset so that the workspace is next to the table
        scene_offset = np.array([-1.6616, -3.0337, 0.0])
        self.arena.set_pose(sapien.Pose(-scene_offset))
        # self.obj_init_actual_z = 0.66467 + 0.5 # table height + 0.5
        self.obj_init_actual_z = 0.8382 + 0.5 # table height + 0.5
        if self.obj_init_fixed_xy_pos is not None:
            self.obj_init_actual_xy_center = np.array(self.obj_init_fixed_xy_pos)
        else:
            self.obj_init_actual_xy_center = np.array([-0.2, 0.2]) + self._episode_rng.uniform(-0.05, 0.05, [2]) # hardcoded for now
        
        self._load_model()
        self.obj.set_damping(0.1, 0.1)

    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError
    
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        self.set_episode_rng(seed)
        model_scale = options.pop("model_scale", None)
        model_id = options.pop("model_id", None)
        reconfigure = options.pop("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure
        return super().reset(seed=self._episode_seed, options=options)

    # def _setup_lighting(self):
    #     super()._setup_lighting()
    #     # self._scene.add_directional_light([0, 0, -1], [1, 1, 1])
    #     self._scene.add_point_light([-0.2, 0.0, 1.4], [1, 1, 1])
        
    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        if model_scale is None:
            model_scales = self.model_db[self.model_id].get("scales")
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
        p = np.hstack([self.obj_init_actual_xy_center, self.obj_init_actual_z])
        q = [1, 0, 0, 0] if self.obj_init_rot_quat is None else self.obj_init_rot_quat

        # Rotate along z-axis
        if self.obj_init_rand_rot_z_enabled:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = qmult(euler2quat(0, 0, ori), q)
        if self.obj_init_fixed_z_rot is not None:
            ori = self.obj_init_fixed_z_rot
            q = qmult(euler2quat(0, 0, ori), q)

        # Rotate along a random axis by a small angle
        if self.obj_init_rand_rot_range > 0 and self.obj_init_fixed_z_rot is None:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, self.obj_init_rand_rot_range)
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
            self._settle(0.5)

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
        is_grasped = self.agent.check_grasp(self.obj, max_angle=85)
        return dict(
            is_grasped=is_grasped,
            success=is_grasped
        )

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0
        if info["success"]:
            reward = 1.0
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 1.0


# ---------------------------------------------------------------------------- #
# YCB
# ---------------------------------------------------------------------------- #
def build_actor_ycb(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=1000,
    root_dir=ASSET_DIR / "mani_skill2_ycb",
):
    builder = scene.create_actor_builder()
    model_dir = Path(root_dir) / "models" / model_id

    collision_file = str(model_dir / "collision.obj")
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = str(model_dir / "textured.obj")
    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    actor = builder.build()
    return actor


@register_env("GraspSingleYCBInScene-v0", max_episode_steps=200)
class GraspSingleYCBInSceneEnv(GraspSingleInSceneEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"

    def _check_assets(self):
        models_dir = self.asset_root / "models"
        for model_id in self.model_ids:
            model_dir = models_dir / model_id
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"{model_dir} is not found."
                    "Please download (ManiSkill2) YCB models:"
                    "`python -m mani_skill2.utils.download_asset ycb`."
                )

            collision_file = model_dir / "collision.obj"
            if not collision_file.exists():
                raise FileNotFoundError(
                    "convex.obj has been renamed to collision.obj. "
                    "Please re-download YCB models."
                )

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)
        self.obj = build_actor_ycb(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            physical_material=self._scene.create_physical_material(
                static_friction=2.0, dynamic_friction=2.0, restitution=0.0
            ),
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id

    def _get_init_z(self):
        bbox_min = self.model_db[self.model_id]["bbox"]["min"]
        return -bbox_min[2] * self.model_scale + 0.05

    def _initialize_agent(self):
        if self.robot_uid == "google_robot_static":
            qpos = np.array(
                [-0.2639457174606611, # -0.1639457174606611
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774,
                0, 0,
                -0.00285961, 0.7851361]
            )
            # qpos = np.array([-0.168, -0.001, 0.596, 1.211, 0.011, 1.591, -0.98, 0, 0, 0.003, 0.785])
            # qpos = np.array([0.225, 0.381, 0.396, 0.895, -0.141, 1.611, -0.921, 0., 0., -0.003, 0.785])
            self.agent.reset(qpos)
            if self.robot_init_fixed_xy_pos is not None:
                robot_init_xyz = [self.robot_init_fixed_xy_pos[0], self.robot_init_fixed_xy_pos[1], 0.06205]
            else:
                init_x = self._episode_rng.uniform(0.30, 0.40)
                init_y = self._episode_rng.uniform(0.0, 0.2)
                robot_init_xyz = [init_x, init_y, 0.06205]
            if self.robot_init_fixed_rot_quat is not None:
                robot_init_rot_quat = self.robot_init_fixed_rot_quat
            else:
                robot_init_rot_quat = [0, 0, 0, 1]
            self.agent.robot.set_pose(Pose(robot_init_xyz, robot_init_rot_quat))
        else:
            raise NotImplementedError(self.robot_uid)
        
        
# ---------------------------------------------------------------------------- #
# Custom Assets
# ---------------------------------------------------------------------------- #
def build_actor_custom(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=1000,
    root_dir=ASSET_DIR / "custom",
):
    builder = scene.create_actor_builder()
    model_dir = Path(root_dir) / "models" / model_id

    collision_file = str(model_dir / "collision.obj")
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = str(model_dir / "textured.dae")
    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    actor = builder.build()
    return actor


@register_env("GraspSingleCustomInScene-v0", max_episode_steps=200)
class GraspSingleCustomInSceneEnv(GraspSingleYCBInSceneEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/custom"
    DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
    DEFAULT_MODEL_JSON = "info_pick_custom_v0.json"

    def _check_assets(self):
        models_dir = self.asset_root / "models"
        for model_id in self.model_ids:
            model_dir = models_dir / model_id
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"{model_dir} is not found."
                )

            collision_file = model_dir / "collision.obj"
            if not collision_file.exists():
                raise FileNotFoundError(
                    "convex.obj has been renamed to collision.obj. "
                )        
                
    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)
        self.obj = build_actor_custom(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            physical_material=self._scene.create_physical_material(
                static_friction=0.5, dynamic_friction=0.5, restitution=0.0
            ),
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id
        
        
        
        
@register_env("GraspSingleYCBCanInScene-v0", max_episode_steps=200)
class GraspSingleYCBCanInSceneEnv(GraspSingleYCBInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_can_v0.json"
    
@register_env("GraspSingleYCBTomatoCanInScene-v0", max_episode_steps=200)
class GraspSingleYCBTomatoCanInSceneEnv(GraspSingleYCBInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_can_v0.json"
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["005_tomato_soup_can"]
        super().__init__(**kwargs)
    
@register_env("GraspSingleYCBBoxInScene-v0", max_episode_steps=200)
class GraspSingleYCBBoxInSceneEnv(GraspSingleYCBInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_box_v0.json"
    
    def _load_model(self):
        density = 500 # override by hand
        self.obj = build_actor_ycb(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            physical_material=self._scene.create_physical_material(
                static_friction=2.0, dynamic_friction=2.0, restitution=0.0
            ),
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id
    
@register_env("GraspSingleYCBFruitInScene-v0", max_episode_steps=200)
class GraspSingleYCBFruitInSceneEnv(GraspSingleYCBInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_fruit_v0.json"
    
@register_env("GraspSingleYCBSomeInScene-v0", max_episode_steps=200)
class GraspSingleYCBSomeInSceneEnv(GraspSingleYCBInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pickintobowl_v0.json"
    

@register_env("KnockSingleYCBBoxOverInScene-v0", max_episode_steps=200)
class KnockSingleYCBBoxOverInSceneEnv(GraspSingleYCBInSceneEnv):
    DEFAULT_MODEL_JSON = "info_knock_box_v0.json"
    # TODO: override success condition
    
    
    
"""
Custom Assets
"""
class GraspSingleWithDistractorInSceneEnv(GraspSingleCustomInSceneEnv):
    distractor_model_ids = []
    distractor_obj = None
    
    def _initialize_actors(self):
        super()._initialize_actors()
        
        assert len(self.distractor_model_ids) > 0
        # The object will fall from a certain initial height
        distractor_model_id = random_choice(self.distractor_model_ids, self._episode_rng)
        distractor_model_scales = self.model_db[distractor_model_id].get("scales")
        if distractor_model_scales is None:
            distractor_model_scale = 1.0
        else:
            distractor_model_scale = random_choice(distractor_model_scales, self._episode_rng)
        self.distractor_obj = build_actor_custom(
            distractor_model_id,
            self._scene,
            scale=distractor_model_scale,
            density=self.model_db[distractor_model_id].get("density", 1000),
            physical_material=self._scene.create_physical_material(
                static_friction=0.5, dynamic_friction=0.5, restitution=0.0
            ),
            root_dir=self.asset_root,
        )
        self.distractor_obj.name = distractor_model_id
        
        while True:
            distractor_rand_xy_center = self.obj_init_actual_xy_center + self._episode_rng.uniform(-0.3, 0.3, [2]) # hardcoded for now
            distractor_rand_xy_center = np.clip(distractor_rand_xy_center, [-0.35, 0.0], [-0.1, 0.4])
            if np.linalg.norm(distractor_rand_xy_center - self.obj_init_actual_xy_center) > 0.2:
                break
        # distractor_rand_xy_center = [-0.1, 0.2]
            
        p = np.hstack([distractor_rand_xy_center, self.obj_init_actual_z])
        q = [1, 0, 0, 0] if self.obj_init_rot_quat is None else np.array(self.obj_init_rot_quat)
        self.distractor_obj.set_pose(Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later
        self.agent.robot.set_pose(Pose([-10, 0, 0]))

        # Lock rotation around x and y
        self.distractor_obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)
        # Unlock motion
        self.distractor_obj.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.distractor_obj.set_pose(self.distractor_obj.pose)
        self.distractor_obj.set_velocity(np.zeros(3))
        self.distractor_obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.distractor_obj.velocity)
        ang_vel = np.linalg.norm(self.distractor_obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(0.5)
            
    def reset(self, *args, **kwargs):
        if self.distractor_obj is not None:
            self._scene.remove_actor(self.distractor_obj)
        return super().reset(*args, **kwargs)
    
class GraspSingleCanInSceneEnv(GraspSingleCustomInSceneEnv):
    def __init__(self, upright=False, **kwargs):
        if upright:
            kwargs['obj_init_rot_quat'] = euler2quat(np.pi/2, 0, 0)
            kwargs['obj_init_rand_rot_z_enabled'] = False
            kwargs['obj_init_rand_rot_range'] = 0
        super().__init__(**kwargs)
    
@register_env("GraspSingleCokeCanInScene-v0", max_episode_steps=200)
class GraspSingleCokeCanInSceneEnv(GraspSingleCanInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["coke_can"]
        super().__init__(**kwargs)
        
@register_env("GraspSingleCokeCanWithDistractorInScene-v0", max_episode_steps=200)
class GraspSingleCokeCanWithDistractorInSceneEnv(GraspSingleCokeCanInSceneEnv, GraspSingleWithDistractorInSceneEnv):
    distractor_model_ids = ['7up_can']
        
@register_env("GraspSingleUpRightCokeCanInScene-v0", max_episode_steps=200)
class GraspSingleUpRightCokeCanInSceneEnv(GraspSingleCokeCanInSceneEnv):
    def __init__(self, **kwargs):
        super().__init__(upright=True, **kwargs)
        
@register_env("GraspSingleUpRightCokeCanWithDistractorInScene-v0", max_episode_steps=200)
class GraspSingleUpRightCokeCanWithDistractorInSceneEnv(GraspSingleUpRightCokeCanInSceneEnv, GraspSingleWithDistractorInSceneEnv):
    distractor_model_ids = ['7up_can']
        
@register_env("GraspSingleLightCokeCanInScene-v0", max_episode_steps=200)
class GraspSingleLightCokeCanInSceneEnv(GraspSingleCanInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["coke_can"]
        super().__init__(**kwargs)
        
        for model_id in self.model_ids:
            self.model_db[model_id]["density"] = 100
            
@register_env("GraspSingleUpRightLightCokeCanInScene-v0", max_episode_steps=200)
class GraspSingleUpRightLightCokeCanInSceneEnv(GraspSingleLightCokeCanInSceneEnv):
    def __init__(self, **kwargs):
        super().__init__(upright=True, **kwargs)
        
@register_env("GraspSingleOpenedCokeCanInScene-v0", max_episode_steps=200)
class GraspSingleOpenedCokeCanInSceneEnv(GraspSingleCanInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["opened_coke_can"]
        super().__init__(**kwargs)
        
@register_env("GraspSingleUpRightOpenedCokeCanInScene-v0", max_episode_steps=200)
class GraspSingleUpRightOpenedCokeCanInSceneEnv(GraspSingleOpenedCokeCanInSceneEnv):
    def __init__(self, **kwargs):
        super().__init__(upright=True, **kwargs)
        
@register_env("GraspSingleUpRightOpenedCokeCanWithDistractorInScene-v0", max_episode_steps=200)
class GraspSingleUpRightOpenedCokeCanWithDistractorInSceneEnv(GraspSingleUpRightOpenedCokeCanInSceneEnv, GraspSingleWithDistractorInSceneEnv):
    distractor_model_ids = ['7up_can']
    
@register_env("GraspSinglePepsiCanInScene-v0", max_episode_steps=200)
class GraspSinglePepsiCanInSceneEnv(GraspSingleCanInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["pepsi_can"]
        super().__init__(**kwargs)
        
@register_env("GraspSingleUpRightPepsiCanInScene-v0", max_episode_steps=200)
class GraspSingleUpRightPepsiCanInSceneEnv(GraspSinglePepsiCanInSceneEnv):
    def __init__(self, **kwargs):
        super().__init__(upright=True, **kwargs)
        
@register_env("GraspSingle7upCanInScene-v0", max_episode_steps=200)
class GraspSingle7upCanInSceneEnv(GraspSingleCanInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["7up_can"]
        super().__init__(**kwargs)
    
@register_env("GraspSingleSpriteCanInScene-v0", max_episode_steps=200)
class GraspSingleSpriteCanInSceneEnv(GraspSingleCanInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["sprite_can"]
        super().__init__(**kwargs)
        
@register_env("GraspSingleFantaCanInScene-v0", max_episode_steps=200)
class GraspSingleFantaCanInSceneEnv(GraspSingleCanInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["fanta_can"]
        super().__init__(**kwargs)