from collections import OrderedDict
from typing import List, Optional

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2.utils.common import random_choice
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose

from .base_env import CustomSceneEnv, CustomYCBInSceneEnv, CustomOtherObjectsInSceneEnv


class OpenDrawerInSceneEnv(CustomSceneEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _setup_lighting(self):
        # self.enable_shadow = True
        # super()._setup_lighting()
        
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            [-0.05, 0, -1], [1, 1, 1], shadow=True, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([-1, 0, 0], [0.5] * 3)

    def _load_actors(self):
        self._load_actor_helper(add_collision=False)

    def _load_articulations(self):
        filename = str(self.asset_root / "mk_station.urdf")
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._art_obj = loader.load(filename)
        # TODO: This pose can be tuned for different rendering approachs.
        self._art_obj.set_pose(sapien.Pose([-0.5, 0, 0.0 + 0.017], [1, 0, 0, 0]))
        for joint in self._art_obj.get_active_joints():
            # joint.set_friction(0.05)
            # joint.set_drive_property(stiffness=0, damping=1)
            # friction seems more important
            joint.set_friction(0.1)
            joint.set_drive_property(stiffness=0, damping=1)
        # TODO: select drawer
        self.obj = self._art_obj.get_links()[1]

    def evaluate(self, **kwargs):
        qpos = self._art_obj.get_qpos()[0]
        return dict(success=qpos >= 0.18, qpos=qpos)


@register_env("OpenDrawerCustomInScene-v0", max_episode_steps=200)
class OpenDrawerCustomInSceneEnv(OpenDrawerInSceneEnv, CustomOtherObjectsInSceneEnv):
    pass
