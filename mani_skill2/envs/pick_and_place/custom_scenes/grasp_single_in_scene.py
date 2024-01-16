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


class GraspSingleInSceneEnv(CustomSceneEnv):
    obj: sapien.Actor  # target object

    def __init__(
        self,
        require_lifting_obj_for_success: bool = True,
        distractor_model_ids: Optional[List[str]] = None,
        original_lighting: bool = False,
        darker_lighting: bool = False,
        **kwargs,
    ):
        if isinstance(distractor_model_ids, str):
            distractor_model_ids = [distractor_model_ids]
        self.distractor_model_ids = distractor_model_ids

        self.model_id = None
        self.model_scale = None
        self.model_bbox_size = None
        
        self.selected_distractor_model_ids = None
        self.selected_distractor_model_scales = None
        
        self.obj = None
        self.distractor_objs = []
        
        self.obj_init_options = {}
        self.distractor_obj_init_options = {}
        
        self.require_lifting_obj_for_success = require_lifting_obj_for_success
        self.consecutive_grasp = 0
        self.lifted_obj_during_consecutive_grasp = False
        self.obj_height_after_settle = None
        
        self.original_lighting = original_lighting
        self.darker_lighting = darker_lighting
        
        super().__init__(**kwargs)

    # def _setup_lighting(self):
    #     super()._setup_lighting()
    #     self._scene.add_directional_light([-1, 1, -1], [1, 1, 1])
        
    def _load_actors(self):
        self._load_arena_helper()        
        self._load_model()
        self.obj.set_damping(0.1, 0.1)
            
    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError
    
    def reset(self, seed=None, options=None):
        for distractor_obj in self.distractor_objs:
            self._scene.remove_actor(distractor_obj)
        
        if options is None:
            options = dict()
            
        self.obj_init_options = options.get("obj_init_options", {})
        self.distractor_obj_init_options = options.get("distractor_obj_init_options", {})
        
        self.set_episode_rng(seed)
        model_scale = options.get("model_scale", None)
        model_id = options.get("model_id", None)
        reconfigure = options.get("reconfigure", False)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        if self.distractor_model_ids is not None:
            distractor_model_scales = options.get("distractor_model_scales", None)
            distractor_model_ids = options.get("distractor_model_ids", None)
            if distractor_model_ids is not None:
                reconfigure = True
                self._set_distractor_models(distractor_model_ids, distractor_model_scales)
                
        options["reconfigure"] = reconfigure
        
        self.consecutive_grasp = 0
        self.lifted_obj_during_consecutive_grasp = False
        self.obj_height_after_settle = None
        
        return super().reset(seed=self._episode_seed, options=options)

    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        if self.original_lighting:
            self._scene.add_directional_light(
                [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
            )
            self._scene.add_directional_light([0, 0, -1], [1, 1, 1])
            return
        elif self.darker_lighting:
            self._scene.add_directional_light(
                [1, 1, -1], [0.3, 0.3, 0.3], shadow=shadow, scale=5, shadow_map_size=2048
            )
            self._scene.add_directional_light([0, 0, -1], [0.3, 0.3, 0.3])
            return
        
        # add_directional_light: direction vec, color vec
        # Only the first of directional lights can have shadow
        # self._scene.add_directional_light(
        #     [0.05, 0, -1], [100, 100, 100], shadow=shadow, scale=5, shadow_map_size=2048
        # )
        
        # 84% for rt-1-new-best standing coke can
        # self._scene.add_directional_light(
        #     [0.05, 0, -1], [3, 3, 3], shadow=shadow, scale=5, shadow_map_size=2048
        # )
        
        # 88% for rt-1-new-best standing coke can
        # self._scene.add_directional_light(
        #     [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
        # )
        # self._scene.add_directional_light(
        #     [0.05, 0, -1], [2, 2, 2]
        # )
        
        # 84% for rt-1-new-best standing coke can
        # self._scene.add_directional_light(
        #     [0.05, 0, -1], [2, 2, 2]
        # )
        
        # 60% for rt-1-new-best standing coke can
        # self._scene.add_directional_light(
        #     [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
        # )
        # self._scene.add_directional_light(
        #     [0.05, 0, -1], [1, 1, 1]
        # )
        
        # 72% for rt-1-new-best standing coke can
        # self._scene.add_directional_light(
        #     [-1, -0.5, -1], [0.7, 0.7, 0.7], shadow=shadow, scale=5, shadow_map_size=2048
        # )
        # self._scene.add_directional_light(
        #     [1, 1, -1], [0.7, 0.7, 0.7]
        # )
        # self._scene.add_directional_light(
        #     [0, 0, -1], [1.6, 1.6, 1.6]
        # )
        
        # 92% for rt-1-new-best standing coke can
        # self._scene.add_directional_light(
        #     [-1, -0.5, -1], [0.7, 0.7, 0.7], shadow=shadow, scale=5, shadow_map_size=2048
        # )
        # self._scene.add_directional_light(
        #     [1, 1, -1], [0.7, 0.7, 0.7]
        # )
        # self._scene.add_directional_light(
        #     [0, 0, -1], [2, 2, 2]
        # )
        
        # 92% for rt-1-new-best standing coke can
        # self._scene.add_directional_light(
        #     [0, 0, -1], [2.5, 2.5, 2.5], shadow=shadow, scale=5, shadow_map_size=2048
        # )
        # self._scene.add_directional_light(
        #     [-1, -0.5, -1], [0.7, 0.7, 0.7]
        # )
        # self._scene.add_directional_light(
        #     [1, 1, -1], [0.7, 0.7, 0.7]
        # )
        
        # 80% for rt-1-new-best standing coke can 
        self._scene.add_directional_light(
            [0, 0, -1], [2.2, 2.2, 2.2], shadow=shadow, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light(
            [-1, -0.5, -1], [0.7, 0.7, 0.7]
        )
        self._scene.add_directional_light(
            [1, 1, -1], [0.7, 0.7, 0.7]
        )
        
        # self._scene.add_directional_light([0, -1, -1], [50, 50, 50])
        
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
    
    def _set_distractor_models(self, distractor_model_ids, distractor_model_scales):
        assert distractor_model_ids is not None
        
        self.selected_distractor_model_ids = distractor_model_ids
        
        if distractor_model_scales is None:
            distractor_model_scales = []
            for distractor_model_id in distractor_model_ids:
                model_scales = self.model_db[distractor_model_id].get("scales")
                if model_scales is None:
                    model_scale = 1.0
                else:
                    model_scale = random_choice(model_scales, self._episode_rng)
                distractor_model_scales.append(model_scale)
                
        self.selected_distractor_model_scales = distractor_model_scales

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
        self.obj.set_pose(sapien.Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

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
        
        if len(self.distractor_objs) > 0:
            for distractor_obj in self.distractor_objs:
                distractor_obj_init_options = self.distractor_obj_init_options.get(distractor_obj.name, None)
                
                distractor_init_xy = distractor_obj_init_options.get('init_xy', None)
                if distractor_init_xy is None:
                    while True:
                        distractor_init_xy = obj_init_xy + self._episode_rng.uniform(-0.3, 0.3, [2]) # hardcoded for now
                        distractor_init_xy = np.clip(distractor_init_xy, [-0.35, 0.0], [-0.1, 0.4])
                        if np.linalg.norm(distractor_init_xy - obj_init_xy) > 0.2:
                            break
                p = np.hstack([distractor_init_xy, obj_init_z]) # let distractor fall from the same height as the main object
                distractor_init_rot_quat = distractor_obj_init_options.get("init_rot_quat", None)
                q = obj_init_rot_quat if distractor_init_rot_quat is None else distractor_init_rot_quat
                distractor_obj.set_pose(sapien.Pose(p, q))

            # Move the robot far away to avoid collision
            # The robot should be initialized later
            self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))
            for distractor_obj in self.distractor_objs:
                # Lock rotation around x and y
                distractor_obj.lock_motion(0, 0, 0, 1, 1, 0)
            self._settle(0.5)
            for distractor_obj in self.distractor_objs:
                # Unlock motion
                distractor_obj.lock_motion(0, 0, 0, 0, 0, 0)
                distractor_obj.set_pose(distractor_obj.pose)
                distractor_obj.set_velocity(np.zeros(3))
                distractor_obj.set_angular_velocity(np.zeros(3))
            self._settle(0.5)
            lin_vel, ang_vel = 0.0, 0.0
            for distractor_obj in self.distractor_objs:
                lin_vel += np.linalg.norm(distractor_obj.velocity)
                ang_vel += np.linalg.norm(distractor_obj.angular_velocity)
            if lin_vel > 1e-3 or ang_vel > 1e-2:
                self._settle(1.5)
        
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


# ---------------------------------------------------------------------------- #
# YCB
# ---------------------------------------------------------------------------- #

@register_env("GraspSingleYCBInScene-v0", max_episode_steps=200)
class GraspSingleYCBInSceneEnv(GraspSingleInSceneEnv, CustomYCBInSceneEnv):

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)
        
        self.obj = self._build_actor_helper(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            physical_material=self._scene.create_physical_material(
                static_friction=self.obj_static_friction, dynamic_friction=self.obj_dynamic_friction, restitution=0.0
            ),
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id
        
        if self.selected_distractor_model_ids is not None:
            for distractor_model_id, distractor_model_scale in zip(self.selected_distractor_model_ids, self.selected_distractor_model_scales):
                distractor_obj = self._build_actor_helper(
                    distractor_model_id,
                    self._scene,
                    scale=distractor_model_scale,
                    density=self.model_db[distractor_model_id].get("density", 1000),
                    physical_material=self._scene.create_physical_material(
                        static_friction=self.obj_static_friction, dynamic_friction=self.obj_dynamic_friction, restitution=0.0
                    ),
                    root_dir=self.asset_root,
                )
                distractor_obj.name = distractor_model_id
                self.distractor_objs.append(distractor_obj)

    def _get_init_z(self):
        bbox_min = self.model_db[self.model_id]["bbox"]["min"]
        return -bbox_min[2] * self.model_scale + 0.05
        

@register_env("GraspSingleYCBCanInScene-v0", max_episode_steps=200)
class GraspSingleYCBCanInSceneEnv(GraspSingleYCBInSceneEnv):
    
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can", "010_potted_meat_can"]
        super().__init__(**kwargs)
    

@register_env("GraspSingleYCBTomatoCanInScene-v0", max_episode_steps=200)
class GraspSingleYCBTomatoCanInSceneEnv(GraspSingleYCBInSceneEnv):
    
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["005_tomato_soup_can"]
        super().__init__(**kwargs)
    
    
@register_env("GraspSingleYCBBoxInScene-v0", max_episode_steps=200)
class GraspSingleYCBBoxInSceneEnv(GraspSingleYCBInSceneEnv):
    
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["003_cracker_box", "004_sugar_box", "008_pudding_box", "009_gelatin_box"]
        super().__init__(**kwargs)
    
        
        
# ---------------------------------------------------------------------------- #
# Custom Assets
# ---------------------------------------------------------------------------- #

@register_env("GraspSingleCustomInScene-v0", max_episode_steps=200)
class GraspSingleCustomInSceneEnv(GraspSingleYCBInSceneEnv, CustomOtherObjectsInSceneEnv):
    pass
    

class GraspSingleCustomOrientationInSceneEnv(GraspSingleCustomInSceneEnv):
    def __init__(self, upright=False, laid_vertically=False, lr_switch=False, **kwargs):
        self.obj_upright = upright
        self.obj_laid_vertically = laid_vertically
        self.obj_lr_switch = lr_switch
        super().__init__(**kwargs)
        
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
            
        obj_init_options = options.get("obj_init_options", None)
        if obj_init_options is None:
            obj_init_options = dict()
            
        if self.obj_upright:
            obj_init_options['init_rot_quat'] = euler2quat(np.pi/2, 0, 0)
        elif self.obj_laid_vertically:
            obj_init_options['init_rot_quat'] = euler2quat(0, 0, np.pi/2)
        elif self.obj_lr_switch:
            obj_init_options['init_rot_quat'] = euler2quat(0, 0, np.pi)
            
        options['obj_init_options'] = obj_init_options
            
        return super().reset(seed=seed, options=options)

    
@register_env("GraspSingleCokeCanInScene-v0", max_episode_steps=200)
class GraspSingleCokeCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["coke_can"]
        super().__init__(**kwargs)

        
@register_env("GraspSingleOpenedCokeCanInScene-v0", max_episode_steps=200)
class GraspSingleOpenedCokeCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    """
    Opened cans are assumed to be empty, and therefore are (1) open, (2) have much lower density than unopened cans (50 vs 1000)
    """
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["opened_coke_can"]
        super().__init__(**kwargs)
    
    
@register_env("GraspSinglePepsiCanInScene-v0", max_episode_steps=200)
class GraspSinglePepsiCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["pepsi_can"]
        super().__init__(**kwargs)
        
@register_env("GraspSingleOpenedPepsiCanInScene-v0", max_episode_steps=200)
class GraspSingleOpenedPepsiCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["opened_pepsi_can"]
        super().__init__(**kwargs)
        

@register_env("GraspSingle7upCanInScene-v0", max_episode_steps=200)
class GraspSingle7upCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["7up_can"]
        super().__init__(**kwargs)
        

@register_env("GraspSingleOpened7upCanInScene-v0", max_episode_steps=200)
class GraspSingleOpened7upCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["opened_7up_can"]
        super().__init__(**kwargs)
    

@register_env("GraspSingleSpriteCanInScene-v0", max_episode_steps=200)
class GraspSingleSpriteCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["sprite_can"]
        super().__init__(**kwargs)
        

@register_env("GraspSingleOpenedSpriteCanInScene-v0", max_episode_steps=200)
class GraspSingleOpenedSpriteCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["opened_sprite_can"]
        super().__init__(**kwargs)
        

@register_env("GraspSingleFantaCanInScene-v0", max_episode_steps=200)
class GraspSingleFantaCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["fanta_can"]
        super().__init__(**kwargs)
        

@register_env("GraspSingleOpenedFantaCanInScene-v0", max_episode_steps=200)
class GraspSingleOpenedFantaCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["opened_fanta_can"]
        super().__init__(**kwargs)
        

@register_env("GraspSingleRedBullCanInScene-v0", max_episode_steps=200)
class GraspSingleRedBullCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["redbull_can"]
        super().__init__(**kwargs)
        

@register_env("GraspSingleOpenedRedBullCanInScene-v0", max_episode_steps=200)
class GraspSingleOpenedRedBullCanInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["opened_redbull_can"]
        super().__init__(**kwargs)
        

@register_env("GraspSingleBluePlasticBottleInScene-v0", max_episode_steps=200)
class GraspSingleBluePlasticBottleInSceneEnv(GraspSingleCustomOrientationInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["blue_plastic_bottle"]
        super().__init__(**kwargs)
        

@register_env("GraspSingleAppleInScene-v0", max_episode_steps=200)
class GraspSingleAppleInSceneEnv(GraspSingleCustomInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["apple"]
        super().__init__(**kwargs)
        

@register_env("GraspSingleOrangeInScene-v0", max_episode_steps=200)
class GraspSingleOrangeInSceneEnv(GraspSingleCustomInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["orange"]
        super().__init__(**kwargs)
        

@register_env("GraspSingleSpongeInScene-v0", max_episode_steps=200)
class GraspSingleSpongeInSceneEnv(GraspSingleCustomInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["sponge"]
        super().__init__(**kwargs)


@register_env("GraspSingleBridgeSpoonInScene-v0", max_episode_steps=200)
class GraspSingleBridgeSpoonInSceneEnv(GraspSingleCustomInSceneEnv):
    def __init__(self, **kwargs):
        kwargs.pop('model_ids', None)
        kwargs['model_ids'] = ["bridge_spoon_generated_modified"]
        super().__init__(**kwargs)
