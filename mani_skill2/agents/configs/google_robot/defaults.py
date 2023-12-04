from copy import deepcopy
import numpy as np

from mani_skill2.agents.controllers import *
from mani_skill2.sensors.camera import CameraConfig


class GoogleRobotDefaultConfig:
    def __init__(self, mobile_base=False, base_arm_drive_mode='force') -> None:
        if mobile_base:
            self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/googlerobot_description/google_robot_meta_sim_fix_fingertip.urdf"
        else:
            self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/googlerobot_description/google_robot_meta_sim_fix_wheel_fix_fingertip.urdf"
        
        finger_min_patch_radius = 0.01
        finger_nail_min_patch_radius = 0.01
        # standard urdf does not support <contact> tag, so we manually define friction here
        self.urdf_config = dict(
            _materials=dict(
                finger_mat=dict(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
                finger_tip_mat=dict(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
                finger_nail_mat=dict(static_friction=0.1, dynamic_friction=0.1, restitution=0.0),
                base_mat=dict(static_friction=0.1, dynamic_friction=0.0, restitution=0.0),
                wheel_mat=dict(static_friction=1.0, dynamic_friction=0.0, restitution=0.0),
            ),
            link=dict(
                link_base=dict(
                    material="base_mat", patch_radius=0.1, min_patch_radius=0.1
                ),
                link_wheel_left=dict(
                    material="wheel_mat", patch_radius=0.1, min_patch_radius=0.1
                ),
                link_wheel_right=dict(
                    material="wheel_mat", patch_radius=0.1, min_patch_radius=0.1
                ),
                link_finger_left=dict(
                    material="finger_mat", patch_radius=finger_min_patch_radius, min_patch_radius=finger_min_patch_radius
                ),
                link_finger_right=dict(
                    material="finger_mat", patch_radius=finger_min_patch_radius, min_patch_radius=finger_min_patch_radius
                ),
                link_finger_tip_left=dict(
                    material="finger_tip_mat", patch_radius=finger_min_patch_radius, min_patch_radius=finger_min_patch_radius
                ),
                link_finger_tip_right=dict(
                    material="finger_tip_mat", patch_radius=finger_min_patch_radius, min_patch_radius=finger_min_patch_radius
                ),
                link_finger_nail_left=dict(
                    material="finger_nail_mat", patch_radius=finger_nail_min_patch_radius, min_patch_radius=finger_nail_min_patch_radius
                ),
                link_finger_nail_right=dict(
                    material="finger_nail_mat", patch_radius=finger_nail_min_patch_radius, min_patch_radius=finger_nail_min_patch_radius
                ),
            ),
        )
        
        self.base_joint_names = ['joint_wheel_left', 'joint_wheel_right']
        self.base_damping = 1e3
        self.base_force_limit = 500
        self.mobile_base = mobile_base # whether the robot base is mobile
        self.base_arm_drive_mode = base_arm_drive_mode # 'force' or 'acceleration'
        
        self.arm_joint_names = ['joint_torso', 'joint_shoulder', 'joint_bicep', 'joint_elbow', 
                                'joint_forearm', 'joint_wrist', 'joint_gripper', 'joint_head_pan', 'joint_head_tilt']
        self.gripper_joint_names = ['joint_finger_right', 'joint_finger_left']
        
        if self.base_arm_drive_mode == 'acceleration':
            self.arm_stiffness = [4000, 4000, 4000, 2000, 2000, 1000, 1000, 4000, 4000]
            self.arm_damping = 500
        elif self.base_arm_drive_mode == 'force':
            self.arm_stiffness = [4000, 4000, 4000, 2000, 2000, 1000, 1000, 4000, 4000]
            self.arm_damping = 500
        else:
            raise NotImplementedError()
        
        self.arm_friction = 0.0
        self.arm_force_limit = [1500, 1500, 300, 300, 300, 300, 300, 300, 300]

        self.gripper_stiffness = 200
        self.gripper_damping = 15
        self.gripper_force_limit = 60


        self.ee_link_name = "link_gripper_tcp"

    @property
    def controllers(self):
        _C = {}
        
        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        if self.mobile_base:
            _C["base"] = dict(
                # PD ego-centric joint velocity
                base_pd_joint_vel=PDBaseVelControllerConfig(
                    self.base_joint_names,
                    lower=[-0.5, -0.5],
                    upper=[0.5, 0.5],
                    damping=self.base_damping,
                    force_limit=self.base_force_limit,
                    drive_mode=self.base_arm_drive_mode,
                )
            )
        else:
            _C["base"] = [None]
        
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -1.0, # dummy limit, which is unused since normalize_action=False
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            frame="ee",
            normalize_action=False,
            drive_mode=self.base_arm_drive_mode,
        )
        arm_pd_ee_delta_pose_base = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            frame="base",
            normalize_action=False,
            drive_mode=self.base_arm_drive_mode,
        )
        arm_pd_ee_target_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            frame="ee",
            use_target=True,
            normalize_action=False,
            drive_mode=self.base_arm_drive_mode,
        )
        arm_pd_ee_target_delta_pose_base = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            frame="base",
            use_target=True,
            normalize_action=False,
            drive_mode=self.base_arm_drive_mode,
        )
        _C["arm"] = dict(
            arm_pd_ee_delta_pose=arm_pd_ee_delta_pose,
            arm_pd_ee_delta_pose_base=arm_pd_ee_delta_pose_base,
            arm_pd_ee_target_delta_pose=arm_pd_ee_target_delta_pose,
            arm_pd_ee_target_delta_pose_base=arm_pd_ee_target_delta_pose_base,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0.0, 
            1.3 + 0.01, # a trick to have force when grasping
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -1.3 - 0.01, 
            1.3 + 0.01, # a trick to have force when grasping
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pd_joint_target_delta_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -1.3 - 0.5, 
            1.3 + 0.5, # a trick to have force when grasping
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.5,
            normalize_action=True,
            drive_mode="force",
        )
        _C["gripper"] = dict(
            gripper_pd_joint_pos=gripper_pd_joint_pos,
            gripper_pd_joint_delta_pos=gripper_pd_joint_delta_pos,
            gripper_pd_joint_target_delta_pos=gripper_pd_joint_target_delta_pos,
        )

        controller_configs = {}
        for base_controller_name in _C["base"]:
            for arm_controller_name in _C["arm"]:
                for gripper_controller_name in _C["gripper"]:
                    c = {}
                    if base_controller_name is not None:
                        c = {"base": _C["base"][base_controller_name]}
                    c["arm"] = _C["arm"][arm_controller_name]
                    c["gripper"] = _C["gripper"][gripper_controller_name]
                    combined_name = arm_controller_name + "_" + gripper_controller_name
                    if base_controller_name is not None:
                        combined_name = base_controller_name + "_" + combined_name
                    controller_configs[combined_name] = c

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        return CameraConfig(
            uid="overhead_camera",
            p=[0, 0, 0],
            q=[0.5, 0.5, -0.5, 0.5], # SAPIEN uses ros camera convention; the rotation matrix of link_camera is in opencv convention, so we need to transform it to ros convention
            width=640,
            height=512,
            fov=1.5,
            near=0.01,
            far=10,
            actor_uid="link_camera",
            intrinsic=np.array([[425.0, 0, 320.0], [0, 425.0, 256.0], [0, 0, 1]]),
        )
        
        
class GoogleRobotStaticBaseConfig(GoogleRobotDefaultConfig):
    
    def __init__(self) -> None:
        super().__init__(mobile_base=False)
        
        
class GoogleRobotMobileBaseConfig(GoogleRobotDefaultConfig):
    
    def __init__(self) -> None:
        super().__init__(mobile_base=True)
        
        
        
        
        
"""
Debug and tuning


                # finger_mat=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
                # finger_tip_mat=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
                
                # finger_mat=dict(static_friction=4.0, dynamic_friction=4.0, restitution=0.0),
                # finger_tip_mat=dict(static_friction=4.0, dynamic_friction=4.0, restitution=0.0),
                # finger_nail_mat=dict(static_friction=0.4, dynamic_friction=0.4, restitution=0.0),
                

# self.arm_stiffness = [40, 40, 40, 20, 20, 10, 10, 40, 40] # TODO: arm and gripper both need system identification
# self.arm_damping = 10

if self.base_arm_drive_mode == 'acceleration':
            # Parameters obtained from https://github.com/google-deepmind/mujoco_menagerie/blob/main/google_robot/robot.xml
            self.arm_stiffness = [4000, 4000, 4000, 2000, 2000, 1000, 1000, 4000, 4000]
            self.arm_damping = 500
        elif self.base_arm_drive_mode == 'force':
            self.arm_stiffness = [4000, 4000, 4000, 2000, 2000, 1000, 1000, 4000, 4000]
            self.arm_damping = 500
            # self.arm_stiffness = [1000, 1000, 1000, 500, 500, 300, 300, 1000, 1000]
            # self.arm_damping = 125
            # self.arm_stiffness = [2000, 2000, 2000, 1000, 1000, 500, 500, 2000, 2000]
            # self.arm_damping = 250
            # self.arm_stiffness = [2000, 2000, 2000, 2000, 2000, 1000, 1000, 2000, 2000]
            # self.arm_damping = 500
            # self.arm_stiffness = [3000, 3000, 3000, 1500, 1500, 750, 750, 3000, 3000]
            # self.arm_damping = 375
            # self.arm_stiffness = [2000, 2000, 2000, 1000, 1000, 500, 500, 2000, 2000]
            # self.arm_damping = 250
            # self.arm_stiffness = [8000, 8000, 8000, 4000, 4000, 2000, 2000, 4000, 4000]
            # self.arm_damping = 1000
        else:
            raise NotImplementedError()
        
        self.arm_friction = 0.0 # 0.1
        
        # self.gripper_stiffness = 20 # TODO: arm and gripper both need system identification
        # self.gripper_damping = 2
        # self.gripper_stiffness = 2000
        # self.gripper_damping = 100 # 300
        self.gripper_stiffness = 1000
        self.gripper_damping = 50
        
        self.arm_force_limit = [150, 150, 30, 30, 30, 30, 30, 30, 30]
        self.gripper_force_limit = 30
        self.arm_force_limit = [x * 10 for x in self.arm_force_limit]
        self.gripper_force_limit = self.gripper_force_limit * 5
        
        
        # tuning draft
        # when decreasing damping, need to decrease stiffness as well (and in many cases, arm force limit)!!
        # good: damping45, v3: 48; v2: 41; v1: 46; v4: 43; v7: 33; v6: 41; v5: 39; v8: 47; v9: 49 (more near-robot success); v10: 41 (lower than v9, so j4&5 damping shouldn't be too low)
        # v11: 38; v12: 38; v13: 36
        
        # self.arm_stiffness = [400, 400, 400, 200, 200, 100, 100, 400, 400]
        # self.arm_damping = 68 # 100
        # self.arm_force_limit = [200, 200, 90, 90, 90, 90, 90, 90, 90]
        # self.arm_stiffness = [4000, 4000, 4000, 2000, 2000, 1000, 1000, 4000, 4000]
        # self.arm_stiffness = [4000, 4000, 2000, 2000, 2000, 1000, 1000, 4000, 4000] # damping45v3, v4
        # self.arm_stiffness = [3000, 3000, 1500, 1500, 1500, 1000, 1000, 4000, 4000] # damping45v5
        # self.arm_stiffness = [3000, 3000, 1500, 1000, 1000, 1000, 1000, 4000, 4000] # damping45v6, v7
        # self.arm_stiffness = [4000, 4000, 1500, 1500, 1500, 1000, 1000, 4000, 4000] # v8
        # self.arm_stiffness = [4000, 4000, 1500, 1000, 1000, 1000, 1000, 4000, 4000] # v9, v10
        # # self.arm_stiffness = [4000, 4000, 1500, 1000, 1000, 600, 600, 4000, 4000] # v11
        # # self.arm_stiffness = [2000, 2000, 1500, 1000, 1000, 600, 600, 4000, 4000] # v12
        # # self.arm_stiffness = [3000, 3000, 1500, 1000, 1000, 600, 600, 4000, 4000] # v13
        # # self.arm_damping = [300, 300, 250, 500, 500, 500, 500, 500, 500] # damping123
        # # self.arm_damping = [300, 300, 250, 300, 300, 500, 500, 500, 500] # damping45
        # self.arm_damping = [300, 300, 200, 200, 200, 500, 500, 500, 500] # damping45v2, v3, v5, v8, v9
        # # self.arm_damping = [300, 300, 200, 200, 200, 200, 200, 500, 500] # v11
        # # self.arm_damping = [200, 200, 200, 200, 200, 200, 200, 500, 500] # v12, v13
        # # self.arm_damping = [300, 300, 200, 120, 120, 500, 500, 500, 500] # damping45v4, v6, v10
        # # self.arm_force_limit = [300, 300, 300, 100, 100, 100, 100, 100, 100]
        # # self.arm_force_limit = [750, 750, 300, 300, 300, 300, 300, 300, 300] # lowerforcelimit
        # # self.arm_force_limit = [500, 500, 250, 300, 300, 300, 300, 300, 300] # evenlowerforcelimit
        # # self.arm_force_limit = [500, 500, 250, 250, 250, 300, 300, 300, 300] # damping45
        # self.arm_force_limit = [500, 500, 200, 200, 200, 300, 300, 300, 300] # damping45v2, v3, v4, v5, v6, v8, v9, v10
        # self.arm_force_limit = [500, 500, 200, 200, 200, 200, 200, 300, 300] # v11
        # self.arm_force_limit = [300, 300, 200, 200, 200, 200, 200, 300, 300] # v12, v13
        # self.arm_force_limit = [500, 500, 200, 150, 150, 300, 300, 300, 300] # damping45v7
        # self.arm_force_limit = 300
        
        # self.arm_stiffness = [4000, 4000, 4000, 2000, 2000, 1000, 1000, 4000, 4000]
        # self.arm_damping = [700, 700, 700, 500, 500, 500, 500, 500, 500] # tmp6
        # self.arm_damping = [700, 700, 500, 500, 500, 500, 500, 500, 500] # tmp7
        # self.arm_stiffness = [4000, 4000, 4000, 2000, 2000, 1000, 1000, 4000, 4000]
        # self.arm_damping = 200 # tmp8 # [500, 500, 500, 500, 500, 500, 500, 500, 500]
        # self.arm_force_limit = [1500, 1500, 300, 300, 300, 300, 300, 300, 300]
        
        # self.arm_stiffness = [40, 40, 40, 20, 20, 10, 10, 40, 40]
        # self.arm_damping = 10
        # self.arm_force_limit = [150, 150, 30, 30, 30, 30, 30, 30, 30]
        
        
        self.gripper_stiffness = 200
        self.gripper_damping = 15
        self.gripper_force_limit = 60
"""