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
        
        # finger_min_patch_radius = 0.1
        # finger_nail_min_patch_radius = 0.05
        finger_min_patch_radius = 0.1
        # finger_min_patch_radius = 0.01
        finger_nail_min_patch_radius = 0.01
        # standard urdf does not support <contact> tag, so we manually define friction here
        self.urdf_config = dict(
            _materials=dict(
                finger_mat=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
                finger_tip_mat=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
                # finger_mat=dict(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
                # finger_tip_mat=dict(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
                finger_nail_mat=dict(static_friction=0.1, dynamic_friction=0.1, restitution=0.0),
                # finger_mat=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
                # finger_tip_mat=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0),
                # finger_nail_mat=dict(static_friction=0.4, dynamic_friction=0.4, restitution=0.0),
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
            """
            Controller 1: arm_pd_ee_delta_pose_align_interpolate
            """
            # self.arm_stiffness = [2000, 1800, 1200, 1000, 650, 500, 500, 2000, 2000]
            # self.arm_damping = [850, 810, 500, 480, 460, 190, 250, 900, 900]
            # self.arm_stiffness = [1932.2991678390808, 1826.4991049680966, 1172.273714250636, 882.4814756272485, 1397.5148682131537, 699.3489562744397, 660.0, 2000, 2000]
            # self.arm_damping = [884.6002482794984, 1000.0, 631.0539484239861, 509.9225285931856, 753.8467217080913, 329.60720242099455, 441.4206687847951, 900, 900]
            # self.arm_force_limit = [400, 400, 300, 300, 200, 200, 100, 100, 100]
            """
            Controller 2: arm_pd_ee_delta_pose_align; loss 0.29
            """
            self.arm_stiffness = [884.889847888136, 1191.9488617194804, 850.0, 888.5254575974557, 404.5410050640524, 380.0, 188.13744253666647, 1000, 1000]
            self.arm_damping = [662.1366513132282, 839.065246922699, 760.0, 660.0, 420.68394788419903, 280.0, 196.16997441864882, 700, 700]
            self.arm_force_limit = [300, 300, 100, 100, 100, 100, 100, 100, 100]
            """
            Controller 3: arm_pd_ee_target_delta_pose_align_interpolate_by_planner
            """
            # self.arm_stiffness = [1592.972328663337, 1400.0, 1497.4287567117276, 1164.2295013372682, 828.6693620938124, 642.8314490268634, 629.5110192592967, 2000, 2000]
            # self.arm_damping = [160.25729203049673, 217.68568199190813, 207.39601771854504, 209.00784945179217, 112.25359769139145, 74.27061800286, 61.75848806104408, 100, 100]
            # self.arm_stiffness = [1315.95463033549, 1600.0, 1600.0, 1280.0, 849.934627184539, 730.0, 638.0538482124314, 2000, 2000]
            # self.arm_damping = [50.2355742556141, 270.09306893587814, 168.01611248517094, 113.91288610862404, 75.854894309871, 56.18316986925525, 75.0, 100, 100]
            self.arm_force_limit = [300, 300, 100, 100, 100, 100, 100, 100, 100]
            
            self.arm_stiffness = [1932.2991678390808, 1826.4991049680966, 1172.273714250636, 882.4814756272485, 1397.5148682131537, 699.3489562744397, 660.0, 2000, 2000]
            self.arm_damping = [884.6002482794984, 1000.0, 631.0539484239861, 509.9225285931856, 753.8467217080913, 329.60720242099455, 441.4206687847951, 900, 900]
            # planning from current sensed qpos
            self.arm_stiffness = [1735.8948480824674, 1754.3342187522323, 1007.9762036720238, 872.5638913272953, 1277.700676022463, 608.0856938168192, 530.0, 2000, 2000]
            self.arm_damping = [1000.0, 1042.8696312830125, 606.8732757029185, 552.2718719738202, 528.0029778895791, 275.6999553621622, 530.0, 900, 900]
            
            """
            Controller 4: arm_pd_ee_delta_pose_align_interpolate_by_planner
            """
            # # continuous velocity but use last target as waypoint; loss 0.23 (vlim=1.5, alim=2)
            # self.arm_stiffness = [1803.294526822171, 2000.0, 1173.8928691477126, 849.4218867373878, 1200.0, 530.0, 553.7854875978537, 2000, 2000]
            # self.arm_damping = [1000.0, 1100.0, 748.6538324775375, 508.93417963876084, 655.8899700318738, 273.5057444692134, 360.0, 900, 900]
            # self.arm_stiffness = [1755.5802337759733, 1700.0, 1000.0, 896.1427073141074, 1181.0596023097614, 460.0, 518.7478307141772, 2000, 2000]
            # self.arm_damping = [1039.3004397057607, 997.7609238661106, 781.9120300040199, 533.1406757667885, 763.5690552485103, 247.37299930493683, 330.0, 900, 900]
            self.arm_stiffness = [1749.5298171431002, 1705.5726191663362, 973.703880069825, 880.2143611543981, 1150.557454707083, 468.8830266395885, 487.9445796329134, 2000, 2000]
            self.arm_damping = [1247.2693705772879, 1179.4662677725612, 827.9760463274691, 655.5257436519437, 698.1166353806758, 282.84185389851126, 363.80038823411655, 900, 900]
            # # continuous velocity but use current qpos as waypoint; loss 0.42
            # self.arm_stiffness = [1578.821511256692, 1950.0, 1430.0, 1258.064902497621, 721.684579284884, 714.0907339829357, 659.096253775289, 2000, 2000]
            # self.arm_damping = [380.0, 680.0, 603.9252159643679, 386.659479352307, 228.54468356490756, 156.10494372615977, 220.0]
            # # zero velocity simple replan; loss 0.49
            # self.arm_stiffness = [1522.6925826441493, 2158.4544756749015, 1400.1676094551071, 1142.6986700565294, 730.659637818336, 669.7021044436542, 628.821295716587, 2000, 2000]
            # self.arm_damping = [293.9747942850573, 103.83092695838668, 85.29843663304095, 40.0, 30.0, 30.0, 74.07034288138254, 900, 900]
            # self.arm_stiffness = [1522.6925826441493, 2158.4544756749015, 1400.1676094551071, 1142.6986700565294, 730.659637818336, 669.7021044436542, 628.821295716587, 2000, 2000]
            # self.arm_damping = [293.9747942850573, 103.83092695838668, 85.29843663304095, 40.0, 30.0, 30.0, 74.07034288138254, 900, 900]
            # self.arm_damping = [30, 30, 20, 20, 15, 15, 15, 900, 900]
            self.arm_force_limit = [300, 300, 100, 100, 100, 100, 100, 100, 100]
        else:
            raise NotImplementedError()
        
        self.arm_friction = 0.0
             
        self.gripper_stiffness = 200
        self.gripper_damping = 4
        self.gripper_force_limit = 60

        self.arm_vel_limit = 1.5
        self.arm_acc_limit = 2.0
        self.gripper_vel_limit = 1.0
        self.gripper_acc_limit = 7.0
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
        arm_common_args = [
            self.arm_joint_names,
            -1.0, # dummy limit, which is unused since normalize_action=False
            1.0,
            np.pi / 2,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
        ]
        arm_common_kwargs = dict(
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            normalize_action=False,
            drive_mode=self.base_arm_drive_mode,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee",
            **arm_common_kwargs
        )
        arm_pd_ee_delta_pose_align = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align",
            **arm_common_kwargs,
        )
        arm_pd_ee_delta_pose_align_interpolate = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align",
            interpolate=True,
            **arm_common_kwargs,
        )
        arm_pd_ee_delta_pose_align_interpolate_by_planner = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align",
            interpolate=True,
            interpolate_by_planner = True,
            interpolate_planner_vlim = self.arm_vel_limit,
            interpolate_planner_alim = self.arm_acc_limit,
            **arm_common_kwargs,
        )
        arm_pd_ee_delta_pose_align2_interpolate_by_planner = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align2",
            interpolate=True,
            interpolate_by_planner = True,
            interpolate_planner_vlim = self.arm_vel_limit,
            interpolate_planner_alim = self.arm_acc_limit,
            **arm_common_kwargs,
        )
        arm_pd_ee_delta_pose_base = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="base",
            **arm_common_kwargs,
        )
        arm_pd_ee_target_delta_pose = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee",
            use_target=True,
            **arm_common_kwargs,
        )
        arm_pd_ee_target_delta_pose_align = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align",
            use_target=True,
            **arm_common_kwargs,
        )
        arm_pd_ee_target_delta_pose_align_interpolate = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align",
            use_target=True,
            interpolate=True,
            **arm_common_kwargs,
        )
        arm_pd_ee_target_delta_pose_align_interpolate_by_planner = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align",
            use_target=True,
            interpolate=True,
            delta_target_from_last_drive_target=True,
            interpolate_by_planner = True,
            interpolate_planner_vlim = self.arm_vel_limit,
            interpolate_planner_alim = self.arm_acc_limit,
            **arm_common_kwargs,
        )
        arm_pd_ee_target_delta_pose_align2_interpolate_by_planner = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="ee_align2",
            use_target=True,
            interpolate=True,
            delta_target_from_last_drive_target=True,
            interpolate_by_planner = True,
            interpolate_planner_vlim = self.arm_vel_limit,
            interpolate_planner_alim = self.arm_acc_limit,
            **arm_common_kwargs,
        )
        arm_pd_ee_target_delta_pose_base = PDEEPoseControllerConfig(
            *arm_common_args,
            frame="base",
            use_target=True,
            **arm_common_kwargs,
        )
        _C["arm"] = dict(
            arm_pd_ee_delta_pose=arm_pd_ee_delta_pose,
            arm_pd_ee_delta_pose_align=arm_pd_ee_delta_pose_align,
            arm_pd_ee_delta_pose_align_interpolate=arm_pd_ee_delta_pose_align_interpolate,
            arm_pd_ee_delta_pose_align_interpolate_by_planner=arm_pd_ee_delta_pose_align_interpolate_by_planner,
            arm_pd_ee_delta_pose_align2_interpolate_by_planner=arm_pd_ee_delta_pose_align2_interpolate_by_planner,
            arm_pd_ee_delta_pose_base=arm_pd_ee_delta_pose_base,
            arm_pd_ee_target_delta_pose=arm_pd_ee_target_delta_pose,
            arm_pd_ee_target_delta_pose_align=arm_pd_ee_target_delta_pose_align,
            arm_pd_ee_target_delta_pose_align_interpolate=arm_pd_ee_target_delta_pose_align_interpolate,
            arm_pd_ee_target_delta_pose_align_interpolate_by_planner=arm_pd_ee_target_delta_pose_align_interpolate_by_planner,
            arm_pd_ee_target_delta_pose_align2_interpolate_by_planner=arm_pd_ee_target_delta_pose_align2_interpolate_by_planner,
            arm_pd_ee_target_delta_pose_base=arm_pd_ee_target_delta_pose_base,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_common_args = [
            self.gripper_joint_names,
            -1.3 - 0.01,
            1.3 + 0.01, # a trick to have force when grasping
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        ]
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            normalize_action=True, # action 0 maps to qpos=0, and action 1 maps to qpos=1.31
            drive_mode="force",
        )
        gripper_pd_joint_target_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.01,
            normalize_action=True, # action 0 maps to qpos=0, and action 1 maps to qpos=1.31
            drive_mode="force",
        )
        gripper_pd_joint_target_pos_interpolate_by_planner = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.01,
            normalize_action=True, # action 0 maps to qpos=0, and action 1 maps to qpos=1.31
            drive_mode="force",
            interpolate=True,
            interpolate_by_planner=True,
            interpolate_planner_init_no_vel=True,
            interpolate_planner_vlim=self.gripper_vel_limit,
            interpolate_planner_alim=self.gripper_acc_limit,
        )
        gripper_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_delta=True,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pd_joint_target_delta_pos = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_delta=True,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.01,
            normalize_action=True,
            drive_mode="force",
        )
        gripper_pd_joint_target_delta_pos_interpolate_by_planner = PDJointPosMimicControllerConfig(
            *gripper_common_args,
            use_delta=True,
            use_target=True,
            clip_target=True,
            clip_target_thres=0.01,
            normalize_action=True,
            drive_mode="force",
            interpolate=True,
            interpolate_by_planner=True,
            interpolate_planner_init_no_vel=True,
            interpolate_planner_vlim=self.gripper_vel_limit,
            interpolate_planner_alim=self.gripper_acc_limit,
        )
        _C["gripper"] = dict(
            gripper_pd_joint_pos=gripper_pd_joint_pos,
            gripper_pd_joint_target_pos=gripper_pd_joint_target_pos,
            gripper_pd_joint_target_pos_interpolate_by_planner=gripper_pd_joint_target_pos_interpolate_by_planner,
            gripper_pd_joint_delta_pos=gripper_pd_joint_delta_pos,
            gripper_pd_joint_target_delta_pos=gripper_pd_joint_target_delta_pos,
            gripper_pd_joint_target_delta_pos_interpolate_by_planner=gripper_pd_joint_target_delta_pos_interpolate_by_planner,
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