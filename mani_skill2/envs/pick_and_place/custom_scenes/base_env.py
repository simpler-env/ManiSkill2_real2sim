from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import os
import sapien.core as sapien

from mani_skill2 import ASSET_DIR, format_path
from mani_skill2.utils.io_utils import load_json

from ..base_env import StationaryManipulationEnv

class CustomSceneEnv(StationaryManipulationEnv):
    DEFAULT_ASSET_ROOT: str
    DEFAULT_SCENE_ROOT: str
    DEFAULT_MODEL_JSON: str
    
    def __init__(
            self, 
            asset_root: str = None,
            scene_root: str = None,
            scene_name: str = None,
            scene_offset: Optional[List[float]] = None,
            scene_pose: Optional[List[float]] = None,
            scene_table_height: float = 0.85,
            model_json: str = None,
            model_ids: List[str] = (),
            **kwargs
        ):
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))
        
        if scene_root is None:
            scene_root = self.DEFAULT_SCENE_ROOT
        self.scene_root = Path(format_path(scene_root))
        self.scene_name = scene_name
        self.scene_offset = scene_offset
        self.scene_pose = scene_pose
        self.scene_table_height = scene_table_height

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
        self._check_assets()
        
        self.arena = None
        self.robot_init_options = {}
        
        super().__init__(**kwargs)
    
    def _check_assets(self):
        """Check whether the assets exist."""
        pass
    
    def _load_arena_helper(self, add_collision=True):
        builder = self._scene.create_actor_builder()
        if self.scene_name is None:
            if "google_robot_static" in self.robot_uid:
                scene_path = str(self.scene_root / "stages/google_pick_coke_can_1_v4.glb") # hardcoded for now
            elif self.robot_uid == "widowx":
                scene_path = str(self.scene_root / "stages/bridge_table_1_v1.glb") # hardcoded for now
        elif self.scene_name == "dummy":
            scene_path = None  # no scene
        else:
            scene_path = str(self.scene_root / "stages" / f"{self.scene_name}.glb")
        if self.scene_offset is None:
            if "google_robot_static" in self.robot_uid:
                scene_offset = np.array([-1.6616, -3.0337, 0.0])
            elif self.robot_uid == "widowx":
                scene_offset = np.array([-2.0634, -2.8313, 0.0])
        else:
            scene_offset = np.array(self.scene_offset)
        if self.scene_pose is None:
            scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])  # y-axis up for Habitat scenes
        else:
            scene_pose = sapien.Pose(q=self.scene_pose)
        
        if self.scene_name != "dummy":
            # NOTE: use nonconvex collision for static scene
            if add_collision:
                builder.add_nonconvex_collision_from_file(scene_path, scene_pose)
            builder.add_visual_from_file(scene_path, scene_pose)
        else:
            builder.add_box_visual(half_size=np.array([10.0, 10.0, 0.017/2]))
        self.arena = builder.build_static()
        # Add offset so that the workspace is next to the table
        
        self.arena.set_pose(sapien.Pose(-scene_offset))
        
    def _settle(self, t):
        sim_steps = int(self.sim_freq * t)
        for _ in range(sim_steps):
            self._scene.step()
    
    def reset(self, seed=None, options=None):
        self.robot_init_options = options.get("robot_init_options", {})
        return super().reset(seed=seed, options=options)
    
    def _initialize_agent(self):
        if "google_robot_static" in self.robot_uid:
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
            robot_init_height = 0.06205 + 0.017 # base height + ground offset
            robot_init_rot_quat = [0, 0, 0, 1]
        elif self.robot_uid == 'widowx':
            # qpos = np.array([0, 0, 0, -np.pi, np.pi / 2, np.pi, 0.037, 0.037])
            # qpos = np.array([-0.00153398, 0.04448544, -0.21629129, -3.14466056, 1.36524296, np.pi, 0.037, 0.037]) # 1.31324296 for fully perpendicular-to-table init
            qpos = np.array([-0.00153398,  0.04448544,  0.21629129, -0.00306796,  1.36524296, 0., 0.037, 0.037])
            # qpos = np.array([-0.002, 0.025, -0.21, -3.139, 1.308, 3.031, 0.037, 0.037])
            robot_init_height = 0.87 # 0.888
            robot_init_rot_quat = [0, 0, 0, 1]
        else:
            raise NotImplementedError(self.robot_uid)
        
        if self.robot_init_options.get("qpos", None) is not None:
            qpos = self.robot_init_options["qpos"]
        self.agent.reset(qpos)
        
        if self.robot_init_options.get("init_height", None) is not None:
            robot_init_height = self.robot_init_options["init_height"]
        if self.robot_init_options.get("init_rot_quat", None) is not None:
            robot_init_rot_quat = self.robot_init_options["init_rot_quat"]
        
        if (robot_init_xy := self.robot_init_options.get("init_xy", None)) is not None:
            robot_init_xyz = [robot_init_xy[0], robot_init_xy[1], robot_init_height]
        else:
            if 'google_robot' in self.robot_uid:
                init_x = self._episode_rng.uniform(0.30, 0.40)
                init_y = self._episode_rng.uniform(0.0, 0.2)
            elif 'widowx' in self.robot_uid:
                init_x = 0.147
                init_y = 0.028
            else:
                init_x, init_y = 0.0, 0.0
            robot_init_xyz = [init_x, init_y, robot_init_height]
        
        self.agent.robot.set_pose(sapien.Pose(robot_init_xyz, robot_init_rot_quat))
        
    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh
    
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0
        if info["success"]:
            reward = 1.0
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 1.0
    
    @staticmethod
    def _get_instruction_obj_name(s):
        s = s.split('_')
        rm_list = ['opened', 'light', 'generated', 'modified', 'objaverse', 'bridge', 'baked']
        cleaned = []
        for w in s:
            if w not in rm_list:
                cleaned.append(w)
        return ' '.join(cleaned)
    
    
# ---------------------------------------------------------------------------- #
# YCB
# ---------------------------------------------------------------------------- #

class CustomYCBInSceneEnv(CustomSceneEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"
    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

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

    @staticmethod
    def _build_actor_helper(
        model_id: str,
        scene: sapien.Scene,
        scale: float = 1.0,
        physical_material: sapien.PhysicalMaterial = None,
        density: float = 1000.0,
        root_dir: str = ASSET_DIR / "mani_skill2_ycb",
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
        if not os.path.exists(visual_file):
            visual_file = str(model_dir / "textured.dae")
            if not os.path.exists(visual_file):
                visual_file = str(model_dir / "textured.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

        actor = builder.build()
        return actor
    
    
# ---------------------------------------------------------------------------- #
# Custom Assets
# ---------------------------------------------------------------------------- #

class CustomOtherObjectsInSceneEnv(CustomYCBInSceneEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/custom"
    DEFAULT_SCENE_ROOT = "{ASSET_DIR}/hab2_bench_assets"
    DEFAULT_MODEL_JSON = "info_pick_custom_v0.json"
    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

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
                
                

class CustomBridgeObjectsInSceneEnv(CustomOtherObjectsInSceneEnv):
    DEFAULT_MODEL_JSON = "info_bridge_custom_v0.json"