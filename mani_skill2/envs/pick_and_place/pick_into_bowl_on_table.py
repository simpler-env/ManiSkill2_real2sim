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


class PickSingleIntoTargetEnv(StationaryManipulationEnv):
    DEFAULT_ASSET_ROOT: str
    DEFAULT_SCENE_ROOT: str
    DEFAULT_MODEL_JSON: str

    obj: sapien.Actor  # target object

    def __init__(
        self,
        asset_root: str = None,
        scene_root: str = None,
        model_json: str = None,
        model_ids: List[str] = (),
        obj_init_rot_z=True,
        obj_init_rot=0,
        goal_thresh=0.025,
        **kwargs,
    ):
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))
        
        if scene_root is None:
            scene_root = self.DEFAULT_SCENE_ROOT
        self.scene_root = Path(format_path(scene_root))

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

        self.obj_init_rot_z = obj_init_rot_z
        self.obj_init_rot = obj_init_rot
        self.goal_thresh = goal_thresh
        
        self.arena = None
        self.obj_init_z = None
        self.obj_init_xy_center = None
        
        self.obj = None
        self.bowl = None

        self._check_assets()
        super().__init__(**kwargs)

    def _check_assets(self):
        """Check whether the assets exist."""
        pass

    def _load_actors(self):
        builder = self._scene.create_actor_builder()
        scene_path = str(self.scene_root / "stages/Baked_sc1_staging_00.glb")
        # scene_path = str(self.scene_root / "stages/Baked_sc1_staging_table85cm.glb") # hardcoded for now
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
        # NOTE: use nonconvex collision for static scene
        builder.add_nonconvex_collision_from_file(scene_path, scene_pose)
        builder.add_visual_from_file(scene_path, scene_pose)
        self.arena = builder.build_static()
        # Add offset so that the workspace is next to the table
        scene_offset = np.array([-1.6616, -3.0337, 0.0])
        self.arena.set_pose(sapien.Pose(-scene_offset))
        self.obj_init_z = 0.66467 + 0.5 # table height + 0.5
        # self.obj_init_z = 0.85 + 0.5 # table height + 0.5
        self.obj_init_xy_center = np.array([-0.2, 0.0])
        
        self._load_model()
        self.obj.set_damping(0.1, 0.1)
        self.bowl.set_damping(0.1, 0.1)

    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError
    
    def _load_bowl(self):
        """Load the bowl to put the target object in."""
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
        while True:
            # The object will fall from a certain height
            xy = self.obj_init_xy_center + self._episode_rng.uniform(-0.05, 0.05, [2])
            z = self.obj_init_z
            p = np.hstack([xy, z])
            q = [1, 0, 0, 0]

            # Rotate along z-axis
            if self.obj_init_rot_z:
                ori = self._episode_rng.uniform(0, 2 * np.pi)
                q = euler2quat(0, 0, ori)

            # Rotate along a random axis by a small angle
            if self.obj_init_rot > 0:
                axis = self._episode_rng.uniform(-1, 1, 3)
                axis = axis / max(np.linalg.norm(axis), 1e-6)
                ori = self._episode_rng.uniform(0, self.obj_init_rot)
                q = qmult(q, axangle2quat(axis, ori, True))
            self.obj.set_pose(Pose(p, q))
            
            bowl_r_from_obj = self._episode_rng.uniform(0.15, 0.18)
            bowl_xy_rot_from_obj = self._episode_rng.uniform(0, 2 * np.pi)
            bowl_xy = self.obj_init_xy_center + np.array([np.cos(bowl_xy_rot_from_obj), np.sin(bowl_xy_rot_from_obj)]) * bowl_r_from_obj
            bowl_xy[0] = np.clip(bowl_xy[0], None, -0.1)
            bowl_p = np.hstack([bowl_xy, z])
            self.bowl.set_pose(Pose(bowl_p, q))
            
            # Check if there is any collision between the bowl and the object
            bowl_mins, bowl_maxs = get_axis_aligned_bbox_for_actor(self.bowl)
            obj_mins, obj_maxs = get_axis_aligned_bbox_for_actor(self.obj)
            
            if not all(
                ((bowl_mins >= obj_mins) & (bowl_mins <= obj_maxs))
                | ((bowl_maxs >= obj_mins) & (bowl_maxs <= obj_maxs))
                | ((bowl_mins >= obj_mins) & (bowl_maxs <= obj_maxs))
                | ((bowl_mins <= obj_mins) & (bowl_maxs >= obj_maxs))
            ):
                break

        # Move the robot far away to avoid collision
        # The robot should be initialized later
        self.agent.robot.set_pose(Pose([-10, 0, 0]))

        # Lock rotation around x and y
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self.bowl.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        self.bowl.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self.bowl.set_pose(self.bowl.pose)
        self.bowl.set_velocity(np.zeros(3))
        self.bowl.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity) + np.linalg.norm(self.bowl.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity) + np.linalg.norm(self.bowl.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(0.5)

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.obj.pose.transform(self.obj.cmass_local_pose)
    
    @property
    def bowl_pose(self):
        return self.bowl.pose.transform(self.bowl.cmass_local_pose)

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                goal_pos=self.bowl_pose.p,
                tcp_to_goal_pos=self.bowl_pose.p - self.tcp.pose.p,
                obj_pose=vectorize_pose(self.obj_pose),
                tcp_to_obj_pos=self.obj_pose.p - self.tcp.pose.p,
                obj_to_goal_pos=self.bowl_pose.p - self.obj_pose.p,
            )
        return obs

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def evaluate(self, **kwargs):
        obj_to_goal_pos = self.bowl_pose.p - self.obj_pose.p
        
        bowl_mins, bowl_maxs = get_axis_aligned_bbox_for_actor(self.bowl)
        bowl_maxs[2] += 0.02 # clearance
        obj_center = self.obj_pose.p
        print(bowl_mins, bowl_maxs, obj_center)
        is_obj_placed = all((obj_center >= bowl_mins) & (obj_center <= bowl_maxs))
        
        z_axis_world = np.array([0, 0, 1])
        bowl_up_axis = self.bowl.get_pose().to_transformation_matrix()[:3, :3] @ z_axis_world
        is_bowl_upwards = abs(angle_between_vec(bowl_up_axis,
                                                z_axis_world)) < 0.1 * np.pi
        
        # is_robot_static = self.check_robot_static()
        
        return dict(
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_bowl_upwards=is_bowl_upwards,
            # is_robot_static=is_robot_static,
            success=is_obj_placed and is_bowl_upwards, # and is_robot_static,
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


@register_env("PickSingleYCBIntoBowl-v0", max_episode_steps=200)
class PickSingleYCBIntoBowlEnv(PickSingleIntoTargetEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
    DEFAULT_MODEL_JSON = "info_pickintobowl_v0.json"

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

        self.bowl = build_actor_ycb(
            "024_bowl",
            self._scene,
            scale=1.0,
            density=1000,
            root_dir=self.asset_root,
        )
        self.bowl.name = "024_bowl"

    def _get_init_z(self):
        bbox_min = self.model_db[self.model_id]["bbox"]["min"]
        return -bbox_min[2] * self.model_scale + 0.05

    def _initialize_agent(self):
        if self.robot_uid == "google_robot_static":
            qpos = np.array(
                [-0.2639457174606611,
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774,
                0, 0,
                -0.00285961, 0.7851361]
            )
            self.agent.reset(qpos)
            init_x = self._episode_rng.uniform(0.30, 0.40)
            init_y = self._episode_rng.uniform(0.0, 0.1)
            self.agent.robot.set_pose(Pose([init_x, init_y, 0.06205], [0, 0, 0, 1]))
        else:
            raise NotImplementedError(self.robot_uid)