"""
python -m mani_skill2.examples.demo_manual_control \
    -e OpenDrawerCustomInScene-v0 \
    -c arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner \
    -o rgbd robot google_robot_static \
    sim_freq @500 control_freq @15 scene_name frl_apartment_stage_simple scene_offset @[-1.8,-2.5,0.0]
"""
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import sapien.core as sapien
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_entity_by_name

from .base_env import CustomOtherObjectsInSceneEnv, CustomSceneEnv


class OpenDrawerInSceneEnv(CustomSceneEnv):
    drawer_id: str

    def __init__(self, light_mode=None, camera_mode=None, **kwargs):
        self.light_mode = light_mode
        self.camera_mode = camera_mode
        super().__init__(**kwargs)

    # def _get_default_scene_config(self):
    #     scene_config = super()._get_default_scene_config()
    #     scene_config.enable_pcm = True
    #     return scene_config

    def _initialize_agent(self):
        init_qpos = np.array(
            [
                -0.2639457174606611,
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774,
                0,
                0,
                -0.00285961,
                0.7851361,
            ]
        )
        if self.camera_mode == "variant":
            init_qpos[-2] += -0.025
            init_qpos[-1] += 0.008
        self.robot_init_options.setdefault("qpos", init_qpos)
        super()._initialize_agent()

    def _setup_lighting(self):
        # self.enable_shadow = True
        # super()._setup_lighting()

        direction = [-0.2, 0, -1]
        if self.light_mode == "vertical":
            direction = [-0.1, 0, -1]

        color = [1, 1, 1]
        if self.light_mode == "darker":
            color = [0.5, 0.5, 0.5]
        elif self.light_mode == "brighter":
            color = [2, 2, 2]
        elif self.light_mode == "yellow":
            color = np.array([186 / 255, 152 / 255,124 / 255]) ** (1 / 2.2)

        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            direction, color, shadow=True, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([-1, 1, -0.05], [0.5] * 3)
        self._scene.add_directional_light([-1, -1, -0.05], [0.5] * 3)

    def _load_actors(self):
        self._load_arena_helper(add_collision=False)

    def _load_articulations(self):
        filename = str(self.asset_root / "mk_station.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.art_obj = loader.load(filename)
        # TODO: This pose can be tuned for different rendering approachs.
        self.art_obj.set_pose(sapien.Pose([-0.295, 0, 0.017], [1, 0, 0, 0]))
        for joint in self.art_obj.get_active_joints():
            # friction seems more important
            # joint.set_friction(0.1)
            joint.set_friction(0.05)
            joint.set_drive_property(stiffness=0, damping=1)

        self.obj = get_entity_by_name(
            self.art_obj.get_links(), f"{self.drawer_id}_drawer"
        )
        joint_names = [j.name for j in self.art_obj.get_active_joints()]
        self.joint_idx = joint_names.index(f"{self.drawer_id}_drawer_joint")

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        return dict(success=qpos >= 0.15, qpos=qpos)

    def get_language_instruction(self):
        return f"open {self.drawer_id} drawer"


class OpenDrawerCustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    pass


@register_env("OpenTopDrawerCustomInScene-v0", max_episode_steps=200)
class OpenTopDrawerCustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_id = "top"


@register_env("OpenMiddleDrawerCustomInScene-v0", max_episode_steps=200)
class OpenMiddleDrawerCustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_id = "middle"


@register_env("OpenBottomDrawerCustomInScene-v0", max_episode_steps=200)
class OpenBottomDrawerCustomInSceneEnv(OpenDrawerCustomInSceneEnv):
    drawer_id = "bottom"


class CloseDrawerInSceneEnv(OpenDrawerInSceneEnv):
    def _initialize_articulations(self):
        super()._initialize_articulations()
        qpos = self.art_obj.get_qpos()
        qpos[self.joint_idx] = 0.2
        self.art_obj.set_qpos(qpos)

    def evaluate(self, **kwargs):
        qpos = self.art_obj.get_qpos()[self.joint_idx]
        return dict(success=qpos <= 0.05, qpos=qpos)

    def get_language_instruction(self):
        return f"close {self.drawer_id} drawer"


class CloseDrawerCustomInSceneEnv(CloseDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    pass


@register_env("CloseTopDrawerCustomInScene-v0", max_episode_steps=200)
class CloseTopDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_id = "top"


@register_env("CloseMiddleDrawerCustomInScene-v0", max_episode_steps=200)
class CloseMiddleDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_id = "middle"


@register_env("CloseBottomDrawerCustomInScene-v0", max_episode_steps=200)
class CloseBottomDrawerCustomInSceneEnv(CloseDrawerCustomInSceneEnv):
    drawer_id = "bottom"
