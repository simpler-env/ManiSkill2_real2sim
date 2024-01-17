import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill2.utils.common import random_choice
from mani_skill2.utils.registration import register_env

from .base_env import CustomBridgeObjectsInSceneEnv
from .move_near_in_scene import MoveNearInSceneEnv

class PutOnInSceneEnv(MoveNearInSceneEnv):
    
    def _set_model(self, model_ids, model_scales):
        """Set the model id and scale. If not provided, choose one randomly."""
        
        if model_ids is None:
            src_model_id = random_choice(self.model_ids, self._episode_rng)
            tgt_model_id = (self.model_ids.index(src_model_id) + 1) % len(self.model_ids)
            model_ids = [src_model_id, tgt_model_id]
            
        return super()._set_model(model_ids, model_scales)
    
    def evaluate(self, **kwargs):
        source_obj_pose = self.source_obj_pose
        target_obj_pose = self.target_obj_pose
        
        source_obj_xy_move_dist = np.linalg.norm(self.episode_source_obj_xyz_after_settle[:2] - self.source_obj_pose.p[:2])
        other_obj_xy_move_dist = []
        for obj, obj_xyz_after_settle in zip(self.episode_objs, self.episode_obj_xyzs_after_settle):
            if obj.name == self.episode_source_obj.name:
                continue
            other_obj_xy_move_dist.append(np.linalg.norm(obj_xyz_after_settle[:2] - obj.pose.p[:2]))
        moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist]))
        moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any([x > source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        
        is_src_obj_grasped = self.agent.check_grasp(self.episode_source_obj)
        
        tgt_obj_half_length_bbox = self.episode_target_obj_bbox_world / 2 # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.episode_source_obj_bbox_world / 2 
        
        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        offset = pos_src - pos_tgt
        xy_flag = (
            np.linalg.norm(offset[:2]) <= np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.005
        )
        z_flag = (offset[2] > 0) and (offset[2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2] <= 0.02)
        src_on_target = (xy_flag and z_flag)
        
        success = src_on_target
        return dict(
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            is_src_obj_grasped=is_src_obj_grasped,
            src_on_target=src_on_target,
            success=success,
        )

    def get_language_instruction(self):
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        tgt_name = self._get_instruction_obj_name(self.episode_target_obj.name)
        return f"put {src_name} on {tgt_name}"
    

class PutOnBridgeInSceneEnv(PutOnInSceneEnv, CustomBridgeObjectsInSceneEnv):
    def __init__(
        self,
        source_obj_name=None,
        target_obj_name=None,
        xy_configs=[],
        quat_configs=[],
        **kwargs,
    ):
        self._source_obj_name = source_obj_name
        self._target_obj_name = target_obj_name
        self._xy_configs = xy_configs
        self._quat_configs = quat_configs
        super().__init__(**kwargs)
    
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        
        obj_init_options = options.pop("obj_init_options", {})
        episode_id = obj_init_options.get("episode_id", 0)
        xy_config = self._xy_configs[
            (episode_id % (len(self._xy_configs) * len(self._quat_configs))) // len(self._quat_configs)
        ]
        quat_config = self._quat_configs[
            episode_id % len(self._quat_configs)
        ]
        
        options['model_ids'] = [self._source_obj_name, self._target_obj_name]
        obj_init_options['source_obj_id'] = 0
        obj_init_options['target_obj_id'] = 1
        obj_init_options['init_xys'] = xy_config
        obj_init_options['init_rot_quats'] = quat_config
        options['obj_init_options'] = obj_init_options
        
        return super().reset(seed=seed, options=options)
    
    def _load_model(self):
        self.episode_objs = []
        for (model_id, model_scale) in zip(self.episode_model_ids, self.episode_model_scales):
            density = self.model_db[model_id].get("density", 1000)
            
            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction, dynamic_friction=self.obj_dynamic_friction, restitution=0.0
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id
            self.episode_objs.append(obj)
            
            
            
            
@register_env("PutSpoonOnTableClothInScene-v0", max_episode_steps=200)
class PutSpoonOnTableClothInScene(PutOnBridgeInSceneEnv):
    def __init__(
        self,
        **kwargs,
    ):
        source_obj_name = "bridge_spoon_generated_modified"
        target_obj_name = "table_cloth_generated"
        
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1)
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]
        
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append([grid_pos_1, grid_pos_2])
        
        quat_configs = [np.array([[1, 0, 0, 0], [1, 0, 0, 0]]), 
                        np.array([euler2quat(0, 0, np.pi/2), [1, 0, 0, 0]])]
        
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs
        )
    
    def get_language_instruction(self):
        return "put the spoon on the towel"
    
    
    
@register_env("PutCarrotOnPlateInScene-v0", max_episode_steps=200)
class PutCarrotOnPlateInScene(PutOnBridgeInSceneEnv):
    def __init__(
        self,
        **kwargs,
    ):
        source_obj_name = "bridge_carrot_generated"
        target_obj_name = "bridge_plate_objaverse"
        
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1)
        grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]
        
        xy_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xy_configs.append([grid_pos_1, grid_pos_2])
        
        quat_configs = [np.array([[1, 0, 0, 0], [1, 0, 0, 0]]), 
                        np.array([euler2quat(0, 0, np.pi/2), [1, 0, 0, 0]])]
        
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs
        )
        
    def get_language_instruction(self):
        return "put carrot on plate"
    
    

@register_env("StackGreenCubeOnYellowCubeInScene-v0", max_episode_steps=200)
class StackGreenCubeOnYellowCubeInScene(PutOnBridgeInSceneEnv):
    def __init__(
        self,
        **kwargs,
    ):
        source_obj_name = "green_cube_3cm"
        target_obj_name = "yellow_cube_3cm"
        
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_xs = [0.05, 0.1]
        half_edge_length_ys = [0.05, 0.1]
        xy_configs = []
        
        for (half_edge_length_x, half_edge_length_y) in zip(half_edge_length_xs, half_edge_length_ys):
            grid_pos = (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1)
            grid_pos = grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None] + xy_center[None]
            
            for i, grid_pos_1 in enumerate(grid_pos):
                for j, grid_pos_2 in enumerate(grid_pos):
                    if i != j:
                        xy_configs.append([grid_pos_1, grid_pos_2])
        
        quat_configs = [np.array([[1, 0, 0, 0], [1, 0, 0, 0]])]
        
        super().__init__(
            source_obj_name=source_obj_name,
            target_obj_name=target_obj_name,
            xy_configs=xy_configs,
            quat_configs=quat_configs,
            **kwargs
        )
        
    def get_language_instruction(self):
        return "stack the green block on the yellow block"