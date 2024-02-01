from typing import Type, Union

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

import cv2

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.agents.robots.xmate3 import Xmate3Robotiq
from mani_skill2.agents.robots.googlerobot import (
    GoogleRobotStaticBase, GoogleRobotStaticBaseColorAdjust,
    GoogleRobotStaticBaseWorseControl1, GoogleRobotStaticBaseWorseControl2, GoogleRobotStaticBaseWorseControl3,
    GoogleRobotStaticBaseWorseControl4, GoogleRobotStaticBaseWorseControl5
)
from mani_skill2.agents.robots.widowx import WidowX
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    look_at,
    set_articulation_render_material,
    vectorize_pose,
)


class StationaryManipulationEnv(BaseEnv):
    SUPPORTED_ROBOTS = {"panda": Panda, "xmate3_robotiq": Xmate3Robotiq, 
                        "google_robot_static": GoogleRobotStaticBase, 
                        "google_robot_static_color_adjust": GoogleRobotStaticBaseColorAdjust,
                        "google_robot_static_worse_control1": GoogleRobotStaticBaseWorseControl1,
                        "google_robot_static_worse_control2": GoogleRobotStaticBaseWorseControl2,
                        "google_robot_static_worse_control3": GoogleRobotStaticBaseWorseControl3,
                        "google_robot_static_worse_control4": GoogleRobotStaticBaseWorseControl4,
                        "google_robot_static_worse_control5": GoogleRobotStaticBaseWorseControl5,
                        "widowx": WidowX}
    agent: Union[Panda, Xmate3Robotiq, GoogleRobotStaticBase, WidowX]

    def __init__(self, *args, robot="panda", robot_init_qpos_noise=0.02, 
                 rgb_overlay_path=None, rgb_overlay_cameras=[], rgb_overlay_mode='background', **kwargs):
        self.robot_uid = robot
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        if rgb_overlay_path is not None:
            self.rgb_overlay_img = cv2.cvtColor(cv2.imread(rgb_overlay_path), cv2.COLOR_BGR2RGB) / 255 # (H, W, 3); float32
        else:
            self.rgb_overlay_img = None
        if not isinstance(rgb_overlay_cameras, list):
            rgb_overlay_cameras = [rgb_overlay_cameras]
        self.rgb_overlay_cameras = rgb_overlay_cameras
        self.rgb_overlay_mode = rgb_overlay_mode
        
        super().__init__(*args, **kwargs)

    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=False,
        render_material: sapien.RenderMaterial = None,
    ):
        if render_material is None:
            render_material = self._renderer.create_material()
            render_material.set_base_color(np.hstack([color, 1.0]))

        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=render_material)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)

    def _build_sphere_site(self, radius, color=(0, 1, 0), name="goal_site"):
        """Build a sphere site (visual only). Used to indicate goal position."""
        builder = self._scene.create_actor_builder()
        builder.add_sphere_visual(radius=radius, color=color)
        sphere = builder.build_static(name)
        # NOTE(jigu): Must hide after creation to avoid pollute observations!
        sphere.hide_visual()
        return sphere

    def _configure_agent(self):
        agent_cls: Type[BaseAgent] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self._agent_cfg = agent_cls.get_default_config()

    def _load_agent(self):
        agent_cls: Type[Panda] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self.agent = agent_cls(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), self.agent.config.ee_link_name
        )
        if not getattr(self, "disable_bad_material", False):
            set_articulation_render_material(self.agent.robot, specular=0.9, roughness=0.3)

    def _initialize_agent(self):
        if self.robot_uid == "panda":
            # fmt: off
            # EE at [0.615, 0, 0.17]
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        elif self.robot_uid == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.562, 0, 0]))
        elif "google_robot_static" in self.robot_uid:
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
            self.agent.robot.set_pose(Pose([0, 0, 0.06205]))
        elif self.robot_uid == 'widowx':
            qpos = np.array([-0.00153398,  0.04448544,  0.21629129, -0.00306796,  1.36524296, 0., 0.037, 0.037])
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)

    def _initialize_agent_v1(self):
        """Higher EE pos."""
        if self.robot_uid == "panda":
            # fmt: off
            qpos = np.array(
                [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        elif self.robot_uid == "xmate3_robotiq":
            qpos = np.array([0, 0.6, 0, 1.3, 0, 1.3, -1.57, 0, 0])
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.562, 0, 0]))
        elif "google_robot_static" in self.robot_uid:
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
            self.agent.robot.set_pose(Pose([0, 0, 0.06205]))
        elif self.robot_uid == 'widowx':
            qpos = np.array([-0.00153398,  0.04448544,  0.21629129, -0.00306796,  1.36524296, 0., 0.037, 0.037])
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)

    def _register_cameras(self):
        pose = look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return CameraConfig(
            "base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )

    def _register_render_cameras(self):
        if self.robot_uid == "panda":
            pose = look_at([0.4, 0.4, 0.8], [0.0, 0.0, 0.4])
        else:
            pose = look_at([0.5, 0.5, 1.0], [0.0, 0.0, 0.5])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        return obs
    
    def get_obs(self):
        obs = super().get_obs()
        
        if self._obs_mode == "image" and self.rgb_overlay_img is not None:
            # get the actor ids of objects to manipulate; note that objects here are not articulated
            target_object_actor_ids = [x.id for x in self.get_actors() if x.name not in ['ground', 'goal_site', '', 'arena']]
            target_object_actor_ids = np.array(target_object_actor_ids, dtype=np.int32)

            # get the robot link ids (links are subclass of actors)
            robot_links = self.agent.robot.get_links() # e.g., [Actor(name="root", id="1"), Actor(name="root_arm_1_link_1", id="2"), Actor(name="root_arm_1_link_2", id="3"), ...]
            robot_link_ids = np.array([x.id for x in robot_links], dtype=np.int32)

            other_link_ids = []
            for art_obj in self._scene.get_all_articulations():
                if art_obj is self.agent.robot:
                    continue
                for link in art_obj.get_links():
                    other_link_ids.append(link.id)
            other_link_ids = np.array(other_link_ids, dtype=np.int32)

            # obtain segmentations of the target object(s) and the robot
            for camera_name in self.rgb_overlay_cameras:
                assert 'Segmentation' in obs['image'][camera_name].keys(), 'Image overlay requires segment info in the observation!'
                seg = obs['image'][camera_name]['Segmentation'] # (H, W, 4); [..., 0] is mesh-level; [..., 1] is actor-level; [..., 2:] is zero (unused)
                actor_seg = seg[..., 1]
                mask = np.ones_like(actor_seg, dtype=np.float32)
                if ('background' in self.rgb_overlay_mode and 'object' not in self.rgb_overlay_mode) or ('debug' in self.rgb_overlay_mode):
                    mask[np.isin(actor_seg, np.concatenate([robot_link_ids, target_object_actor_ids, other_link_ids]))] = 0.0
                elif 'background' in self.rgb_overlay_mode:
                    mask[np.isin(actor_seg, robot_link_ids)] = 0.0
                mask = mask[..., np.newaxis]
                
                rgb_overlay_img = cv2.resize(self.rgb_overlay_img, (obs['image'][camera_name]['Color'].shape[1], obs['image'][camera_name]['Color'].shape[0]))
                if 'debug' not in self.rgb_overlay_mode:
                    obs['image'][camera_name]['Color'][..., :3] = obs['image'][camera_name]['Color'][..., :3] * (1 - mask) + rgb_overlay_img * mask
                else:
                    # debug
                    obs['image'][camera_name]['Color'][..., :3] = obs['image'][camera_name]['Color'][..., :3] * (1 - mask) + rgb_overlay_img * mask
                    # obs['image'][camera_name]['Color'][..., :3] = obs['image'][camera_name]['Color'][..., :3] * 0.5 + rgb_overlay_img * 0.5
                
        return obs
