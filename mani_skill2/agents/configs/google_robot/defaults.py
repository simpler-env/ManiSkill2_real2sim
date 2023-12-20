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
            raise NotImplementedError('Not yet tuned')
        elif self.base_arm_drive_mode == 'force':
            self.arm_stiffness = [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000]
            self.arm_damping = [850, 810, 500, 480, 460, 190, 250, 900, 900]
        else:
            raise NotImplementedError()
        
        self.arm_friction = 0.0
        self.arm_force_limit = [400, 400, 300, 300, 200, 200, 100, 100, 100]
        
        self.gripper_stiffness = 200
        self.gripper_damping = 80
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
        arm_pd_ee_delta_pose_align = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -1.0, # dummy limit, which is unused since normalize_action=False
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            frame="ee_align",
            normalize_action=False,
            drive_mode=self.base_arm_drive_mode,
        )
        arm_pd_ee_delta_pose_align_interpolate = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -1.0, # dummy limit, which is unused since normalize_action=False
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            frame="ee_align",
            interpolate=True,
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
        arm_pd_ee_target_delta_pose_align = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            frame="ee_align",
            use_target=True,
            normalize_action=False,
            drive_mode=self.base_arm_drive_mode,
        )
        arm_pd_ee_target_delta_pose_align_interpolate = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            frame="ee_align",
            use_target=True,
            interpolate=True,
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
            arm_pd_ee_delta_pose_align=arm_pd_ee_delta_pose_align,
            arm_pd_ee_delta_pose_align_interpolate=arm_pd_ee_delta_pose_align_interpolate,
            arm_pd_ee_delta_pose_base=arm_pd_ee_delta_pose_base,
            arm_pd_ee_target_delta_pose=arm_pd_ee_target_delta_pose,
            arm_pd_ee_target_delta_pose_align=arm_pd_ee_target_delta_pose_align,
            arm_pd_ee_target_delta_pose_align_interpolate=arm_pd_ee_target_delta_pose_align_interpolate,
            arm_pd_ee_target_delta_pose_base=arm_pd_ee_target_delta_pose_base,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -1.3 - 0.01,
            1.3 + 0.01, # a trick to have force when grasping
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=True, # action 0 maps to qpos=0, and action 1 maps to qpos=1.31
            drive_mode="force",
        )
        gripper_pd_joint_target_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -1.3 - 0.01,
            1.3 + 0.01, # a trick to have force when grasping
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.01,
            normalize_action=True, # action 0 maps to qpos=0, and action 1 maps to qpos=1.31
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
            -1.3 - 0.01, 
            1.3 + 0.01, # a trick to have force when grasping
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.01,
            normalize_action=True,
            drive_mode="force",
        )
        _C["gripper"] = dict(
            gripper_pd_joint_pos=gripper_pd_joint_pos,
            gripper_pd_joint_target_pos=gripper_pd_joint_target_pos,
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
                

self.arm_friction = 0.0
        self.arm_force_limit = [1500, 1500, 300, 300, 300, 300, 300, 300, 300]
        
        # self.arm_stiffness = [40, 40, 40, 20, 20, 10, 10, 40, 40] # TODO: arm and gripper both need system identification
        # self.arm_damping = [15, 15, 15, 7, 7, 5, 5, 15, 15]
        # self.arm_force_limit = [150, 150, 30, 30, 30, 30, 30, 30, 30]

        # self.gripper_stiffness = 200
        # self.gripper_damping = 15
        # self.gripper_force_limit = 60
        
        self.arm_stiffness = [4000, 4000, 4000, 2000, 2000, 1000, 1000, 4000, 4000] # TODO: arm and gripper both need system identification
        # self.arm_damping = [1600, 1600, 1500, 1000, 1200, 300, 550, 1800, 1800] # candidate1
        # self.arm_damping = [2000, 2000, 1600, 600, 1200, 350, 600, 1800, 1800]
        # self.arm_damping = [2000, 1600, 1600, 600, 1200, 350, 600, 1800, 1800] # candidate2
        # self.arm_damping = [1600, 1600, 1500, 600, 1000, 300, 550, 1800, 1800]
        # self.arm_damping = [1800, 1600, 1500, 800, 1200, 400, 650, 1800, 1800] # candidate3
        # self.arm_damping = [1800, 1600, 1500, 900, 1200, 380, 650, 1800, 1800] # candidate4
        # self.arm_damping = [1880, 1680, 1580, 980, 1200, 360, 650, 1800, 1800]
        # self.arm_damping = [2000, 1800, 1700, 1100, 1200, 360, 650, 1800, 1800]
        self.arm_force_limit = [450, 450, 150, 150, 150, 150, 150, 150, 150]
        
        # rotate gripper vert to horiz: [0][4] > [1],[2] > [3]
        # raise gripper when gripper horiz: [0] > [1][4] > [3]
        # move gripper horizontally outwards: [0][1] > [3][4] > [2]
        # move gripper vertically down: [0] > [3][2] > [1][4]
        self.arm_stiffness = [2000, 2000, 1500, 1000, 1000, 500, 500, 2000, 2000]
        # self.arm_damping = [900, 800, 600, 450, 640, 190, 325, 900, 900] # candidate1
        # self.arm_damping = [650, 900, 600, 450, 640, 190, 325, 900, 900]
        # self.arm_damping = [730, 900, 600, 580, 640, 190, 325, 900, 900]
        # self.arm_damping = [730, 900, 650, 580, 640, 190, 325, 900, 900]
        self.arm_damping = [730, 950, 650, 580, 580, 190, 325, 900, 900]
        self.arm_damping = [730, 950, 650, 530, 500, 190, 325, 900, 900]
        self.arm_stiffness = [2000, 2000, 1000, 1000, 1000, 500, 500, 2000, 2000] # candidate 2
        self.arm_damping = [730, 950, 650, 530, 500, 190, 325, 900, 900] # candidate 2; 730 -> 700 is also good
        self.arm_force_limit = [300, 300, 100, 100, 100, 100, 100, 100, 100]
        
        self.arm_stiffness = [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000]
        # self.arm_damping = [700, 950, 650, 530, 400, 190, 325, 900, 900] # decreasing damping[4] helps to raise arm; decrease damping[4] more than [2]
        # self.arm_damping = [770, 800, 560, 500, 430, 180, 350, 900, 900] # candidate 2 w/ [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000] [300, 300, 100, 100, 100, 100, 100, 100, 100]
        self.arm_damping = [770, 830, 560, 500, 460, 180, 350, 900, 900] 
        # gradient descent at [2000, 1800, 1200, 1000, 700, 500, 500, 2000, 2000] [700, 900, 650, 500, 380, 270, 325, 900, 900] [300, 300, 100, 100, 100, 100, 100, 100, 100]; 
        # current phenomenon: 805 too low and a bit outwards; 1257 fine (need a tiny bit deeper); 1495 overshoots to the right and second last link not horizontal, grasping fine
        # decreasing damping[0] makes horizontal gripper further outwards (also 1495); when gripper vertical, reaching object on the left overshoots;
        # decreasing damping[1] makes horizontal gripper further lower; when gripper vertical, the gripper is lower, too
        # decreasing damping[2] makes horizontal gripper slightly higher and further inwards; when gripper vertical, gripper also slightly higher and a bit inwards
        # decreasing damping[3] makes horizontal gripper doesn't change; 1495 gripper slightly higher but a bit less horizontal; 
        # vertical gripper 1257 "less distance to orange" but a tiny bit overshoot
        # decreasing damping[4] makes horizontal gripper higher (good) and a little bit inwards; vertical gripper 1257 a very slight overshoot; 
        # 1495 overshoots to the left and second last link significantly tiled and not horizontal
        # decreasing damping[5] makes horizontal gripper further outwards and same height; when gripper vertical, reaching object on the left overshoots; 
        # 1495 less horizontal and a bit higher
        # slightly increase 5, increase 4, increase 1, increase 2, 0
        self.arm_force_limit = [300, 300, 100, 100, 100, 100, 100, 100, 100]
        
        # 1495 almost perfect:[2000, 1800, 1000, 1000, 700, 500, 500, 2000, 2000] [800, 950, 550, 500, 380, 280, 325, 900, 900] [300, 300, 100, 100, 100, 100, 100, 100, 100]
        
        # gradient descent at [800, 900, 560, 480, 380, 230, 350, 900, 900]
        # decreasing damping[0] to 730 causes horizontal gripper to move slightly inwards; vertical gripper slightly overshoots; 1495 very similar?
        # decreasing damping[1] to 800 causes horizontal gripper to move lower and a bit outwards; 1495 gripper slightly lower and tiny bit outwards
        # decreasing damping[2] to 460 causes horizontal gripper to overshoot inwards and higher; when gripper vertical, also overshoots; 
        # 1495 overshoots to the left (but horizontal)
        # decreasing damping[3] to 400 causes horizontal gripper to a bit overshoot inwards; 1495 higher and a bit further?
        # decreasing damping[4] to 300 causes horizontal gripper to be higher and overshoot to the left; vertical gripper 1257 a bit overshoot; 
        # 1495 higher but overshoots to the left and not horizontal
        # decreasing damping[5] to 160 causes horizontal gripper to be higher and a bit further and second link in a qpos more similar to gt;
        # vertical gripper a bit further away from apple; 1495 up a lot and not horizontal
        
        self.arm_stiffness = [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000]
        # self.arm_damping = [750, 830, 540, 480, 460, 190, 350, 900, 900] # candidate 3 with [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000] [300, 300, 100, 100, 100, 100, 100, 100, 100]
        # damping[5] should be around 220
        # self.arm_damping = [750, 830, 500, 480, 460, 220, 350, 900, 900] # candidate 4
        self.arm_force_limit = [300, 300, 100, 100, 100, 100, 100, 100, 100]
        # gradient descent at [770, 830, 560, 500, 460, 180, 350, 900, 900] [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000] [300, 300, 100, 100, 100, 100, 100, 100, 100]
        # issue: 805 needs to be a bit higher and to the left; 1257 needs to be a bit deeper; 1495 a bit faster turn?
        # decreasing damping[0] to 700 causes 805 to be soft; 1495 a bit to the right; decreasing damping[0] to 750 might be a bit better
        # decreasing damping[1] to 760 causes 805 to be lower and undershoot; 1495 also a bit lower
        # decreasing damping[2] to 500 causes 805 to be higher and further to the left; 1495 overshoots to the left
        # decreasing damping[3] to 440 causes 805 to undershoot to the right; 1495 higher
        # decreasing damping[4] to 400 causes 805 to be a bit to the left; 1495 overshoots to the left and not horizontal
        # decreasing damping[5] to 130 causes 805 to be drifted a up and 1495 a lot up and doesn't turn?
        
        self.arm_stiffness = [2000, 1650, 1200, 1000, 650, 500, 500, 2000, 2000]
        self.arm_damping = [750, 830, 500, 480, 460, 220, 300, 900, 900]
        self.arm_force_limit = [250, 250, 100, 100, 100, 100, 100, 100, 100]
        # self.arm_stiffness = [1800, 1800, 1200, 1000, 650, 500, 500, 2000, 2000]
        self.arm_stiffness = [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000]
        # self.arm_damping = [750, 960, 450, 600, 460, 200, 450, 900, 900] # last tmp
        # self.arm_damping = [750, 1000, 640, 630, 460, 160, 350, 900, 900] # tmp too high
        self.arm_stiffness = [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000]
        # self.arm_damping = [730, 830, 400, 480, 480, 200, 450, 900, 900] # last tmp 2
        self.arm_damping = [850, 810, 500, 480, 460, 190, 450, 900, 900] # best candidate 1 w/ [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000] [400, 400, 300, 300, 200, 200, 100, 100, 100]
        # self.arm_damping = [880, 880, 500, 480, 460, 190, 450, 900, 900] # candidate 3
        # self.arm_damping = [880, 830, 500, 480, 460, 200, 450, 900, 900]
        # 1539 is fine w/ damping [850, 960, 440, 480, 460, 200, 350, 900, 900] and force limit [300, 300, 100, 100, 100, 100, 100, 100, 100]; increasing damping[0] causes 1539 to overshoot a bit
        # increasing damping[0] and decreasing damping[2] causes 1539 to reach closer towards coke can
        # increasing damping[3] causes 1539 to be further away from coke can, and when keeping damping[0] to be the same, horizontal moving distance is smaller
        # however decreasing damping[3] doesn't reach closer towards the can...
        self.arm_damping = [850, 810, 500, 480, 460, 190, 250, 900, 900]
        self.arm_force_limit = [400, 400, 300, 300, 200, 200, 100, 100, 100]
        # gradient descent at [750, 830, 500, 480, 460, 220, 350, 900, 900] [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000] [300, 300, 100, 100, 100, 100, 100, 100, 100]
        # decreasing damping[0] to 700 doesn't change?
        # decreasing damping[1] to 750 causes horizontal gripper to be lower in all horizontal cases (including 1539)
        # decreasing damping[2] to 430 causes horizontal gripper to be higher and overshoot; vertical gripper to overshoot; 1495 to overshoot to left; but 1539 is better and higher and deeper
        # decreasing damping[3] to 400 causes 805 to be a bit to the left and 1495 higher; 
        # decreasing damping[4] to 380 causes 805 second last link to be a bit incorrect and 1495 to be a bit overshoot to left
        # increasing stiffness[0] causes 1539 to undershoot and 1495 a bit to the right
"""