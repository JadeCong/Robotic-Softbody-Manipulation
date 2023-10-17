from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import re
from klampt.model import trajectory
import roboticstoolbox as rtb

from spatialmath import SE3

from robosuite.utils.transform_utils import convert_quat, quat2mat, mat2euler
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.models.base import MujocoModel

import robosuite.utils.transform_utils as T

from scan_models.objects import SoftTorsoObject, BoxObject, SoftBoxObject
from scan_models.tasks import ScanTask
from scan_models.arenas import ScanArena
from utils.quaternion import distance_quat, difference_quat

import math
from klampt.math import so3, se3, vectorops
from klampt import vis
# from utils.rtplot import RealtimePlotData
# from utils.dynplot import dynplot
from utils.qtplot import RealtimePlotWindow
# os.environ.update({"QT_QPA_PLATFORM_PLUGIN_PATH": "/home/jade/anaconda3/envs/rl_ultrasound/lib/python3.8/site-packages/PyQt5/Qt5/plugins/xcbglintegrations/libqxcb-glx-integration.so"})


class Ultrasound(SingleArmEnv):
    """
    This class corresponds to the ultrasound task for a single robot arm.
    
    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!
        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.
        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param
        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param
        table_full_size (3-tuple): x, y, and z dimensions of the table.
        table_friction (3-tuple): the three mujoco friction parameters for
            the table.
        use_camera_obs (bool): if True, every observation includes rendered image(s)
        use_object_obs (bool): if True, include object (cube) information in
            the observation.
        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering
        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse
        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).
        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.
        horizon (int): Every episode lasts for exactly @horizon timesteps.
        ignore_done (bool): True if never terminating the environment (ignore @horizon).
        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables
        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
            :Note: At least one camera must be specified if @use_camera_obs is True.
            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).
        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.
        early_termination (bool): If True, episode is allowed to finish early.
        save_data (bool): If True, data from the episode is collected and saved.
        deterministic_trajectory (bool): If True, chooses a deterministic trajectory which goes along the x-axis of the torso.
        torso_solref_randomization (bool): If True, randomize the stiffness and damping parameter of the torso. 
        initial_probe_pos_randomization (bool): If True, Gaussian noise will be added to the initial position of the probe.
        use_box_torso (bool): If True, use a box shaped soft body. Else, use a cylinder shaped soft body.
    Raises:
        AssertionError: [Invalid number of robots specified]
    """
    
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="ProbeGripper",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        # table_friction=100*(1., 5e-3, 1e-4),
        table_friction=10000*(1., 5e-1, 1e-1),  # changed from 100*(1., 5e-3, 1e-4) to 100*(1., 5e-1, 1e-3) (by JadeCong)
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=False,  # rerender the env upon a reset call, changed from True to False(by JadeCong)
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        early_termination=False,
        save_data=False,
        deterministic_trajectory=False,
        torso_solref_randomization=False,
        initial_probe_pos_randomization=False,
        use_box_torso=True,
    ):
        assert gripper_types == "ProbeGripper",\
            "Tried to specify gripper other than ProbeGripper in Ultrasound environment!"
        
        assert robots == "UR5e" or robots == "Panda", \
            "Robot must be UR5e or Panda!"
        
        assert "OSC" or "HMFC" in controller_configs["type"], \
            "The robot controller must be of type OSC or HMFC"
        
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        
        # settings for joint initialization noise (Gaussian)
        self.mu = 0
        self.sigma = 0.010
        
        # settings for contact force running mean
        self.alpha = 0.1    # decay factor (high alpha -> discounts older observations faster). Must be in (0, 1)
        
        # reward configuration 
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping
        
        # error multipliers
        self.pos_error_mul = 90
        self.ori_error_mul = 0.2  # changed from 0.2 to 20 (by JadeCong)
        self.vel_error_mul = 45
        self.force_error_mul = 0.7
        self.der_force_error_mul = 0.01
        
        # reward multipliers
        self.pos_reward_mul = 5
        self.ori_reward_mul = 1  # changed from 1 to 10 (by JadeCong)
        self.vel_reward_mul = 1
        self.force_reward_mul = 30
        self.der_force_reward_mul = 2
        
        # desired states
        self.goal_quat = np.array([-0.69192486,  0.72186726, -0.00514253, -0.01100909]) # Upright probe orientation found from experimenting (x,y,z,w)
        self.goal_velocity = 0.04                   # norm of velocity vector  # changed from 0.04 to 0.1
        self.goal_contact_z_force = 5               # (N)  # changed from 5 to 3
        self.goal_der_contact_z_force = 0           # derivative of contact force   
        
        # early termination configuration
        self.pos_error_threshold = 1.0
        self.ori_error_threshold = 0.10
        
        # examination trajectory
        self.top_torso_offset = 0.039 if use_box_torso else 0.065  # 0.041  # offset from z_center of torso to top of torso(by JadeCong)
        self.x_range = 0.15                                 #0.15  # how large the torso is from center to end in x-direction(by JadeCong)
        self.y_range = 0.09 if use_box_torso else 0.05      # 0.05  # how large the torso is from center to end in y-direction(by JadeCong)
        self.grid_pts = 50                                  # how many points in the grid
        
        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs
        
        # object placement initializer
        self.placement_initializer = placement_initializer
        
        # randomization settings
        self.torso_solref_randomization = torso_solref_randomization
        self.initial_probe_pos_randomization = initial_probe_pos_randomization
        
        # misc settings
        self.early_termination = early_termination
        self.save_data = save_data
        self.deterministic_trajectory = deterministic_trajectory
        self.use_box_torso = use_box_torso
        
        # for getting the custom trajectory given rows and columns(by JadeCong)
        self.grid_pts_x = 8
        self.grid_pts_y = 5
        
        # for getting the custom quat of scanning trajectory given torso start index form whole torso indexes.
        # self.torso_start_index = [176, 150, 124]  # for 3 rows
        # self.torso_start_index = [202, 176, 150, 124, 98]  # for 5 rows
        self.torso_start_index = [228, 202, 176, 150, 124, 98, 72]  # for 7 rows
        # self.torso_start_index = [272, 228, 202, 176, 150, 124, 98, 72, 46]  # for 9 rows
        
        # configure the settings for scanning trajectory
        # self.torso_scanning_waypoint_array = [[0 for i in range(self.grid_pts_x)] for j in range(self.grid_pts_y)]
        self.torso_scanning_waypoint_array = [[0 for i in range(9)] for j in range(len(self.torso_start_index))]
        self.torso_scanning_order_traj = {}
        self.scanning_index = 0
        self.scan_beginning = True
        self.traj_start = np.zeros(3)
        self.traj_end = np.zeros(3)
        self.scanning_waypoint_num = 10
        
        # for plotting data in real time # TODO: plot class
        # self.rtpd = RealtimePlotData(title="Ultrasound Scanning Sim Data", data_labels=['force', 'velocity'])
        # self.dplt = dynplot()
        self.qtplt = RealtimePlotWindow(win_title="Ultrasound Scanning Sim Data", ylabel=["Force", "Velocity", "Derivative_Force"],
                                        data_labels=["force", "vel", "der_force"], data_references=[self.goal_contact_z_force, self.goal_velocity, self.goal_der_contact_z_force],
                                        data_buffer_size=50, plot_together=False)
        # self.old_timestep = 0
        # self.old_force = 0
        # self.old_velocity = 0
        
        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=None,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )
    
    def reward(self, action=None):
        """
        Reward function for the task.
        
        Args:
            action (np array): [NOT USED]
        
        Returns:
            float: reward value
        """
        
        reward = 0.
        
        ee_current_ori = convert_quat(self._eef_xquat, to="wxyz")   # (w, x, y, z) quaternion
        # ee_desired_ori = convert_quat(self.goal_quat, to="wxyz")
        ee_desired_ori = convert_quat(T.axisangle2quat(self.traj_ori), to="wxyz")  # added(by JadeCong)
        
        # position
        self.pos_error = np.square(self.pos_error_mul * (self._eef_xpos[0:-1] - self.traj_pt[0:-1]))
        self.pos_reward = self.pos_reward_mul * np.exp(-1 * np.linalg.norm(self.pos_error))
        
        # orientation
        self.ori_error = self.ori_error_mul * distance_quat(ee_current_ori, ee_desired_ori)
        self.ori_reward = self.ori_reward_mul * np.exp(-1 * self.ori_error)
        
        # velocity
        self.vel_error =  np.square(self.vel_error_mul * (self.vel_running_mean - self.goal_velocity))
        self.vel_reward = self.vel_reward_mul * np.exp(-1 * np.linalg.norm(self.vel_error))
        
        # force
        self.force_error = np.square(self.force_error_mul * (self.z_contact_force_running_mean - self.goal_contact_z_force))
        self.force_reward = self.force_reward_mul * np.exp(-1 * self.force_error) if self._check_probe_contact_with_torso() else 0
        
        # derivative force
        self.der_force_error = np.square(self.der_force_error_mul * (self.der_z_contact_force - self.goal_der_contact_z_force))
        self.der_force_reward = self.der_force_reward_mul * np.exp(-1 * self.der_force_error) if self._check_probe_contact_with_torso() else 0
        
        # add rewards
        reward += (self.pos_reward + self.ori_reward + self.vel_reward + self.force_reward + self.der_force_reward)
        
        return reward
    
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        
        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        
        # Load model for table top workspace
        mujoco_arena = ScanArena()
        
        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])
        
        # Initialize torso object
        # self.torso = SoftBoxObject(name="torso") if self.use_box_torso else SoftTorsoObject(name="torso", joints=[{"name": "soft_human_torso_free_joint", "type": "slide", "axis": "0 0 1", "limited": "true", "range": "-0.000001 0.000001", "stiffness": "500"}])
        self.torso = SoftBoxObject(name="torso") if self.use_box_torso else SoftTorsoObject(name="human")
        
        if self.torso_solref_randomization:
            # Randomize torso's stiffness and damping (values are taken from my project thesis)
            stiffness = np.random.randint(1300, 1600)
            damping = np.random.randint(17, 41)
            
            self.torso.set_damping(damping)
            self.torso.set_stiffness(stiffness)
        
        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(self.torso)
        else:
            self.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[self.torso],
                x_range=[0, 0],  # [-0.12, 0.12],
                y_range=[0, 0],  # [-0.12, 0.12],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.005,
            )
        
        # task includes arena, robot, and objects of interest
        self.model = ScanTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots], 
            mujoco_objects=[self.torso]
        )
    
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()
        
        # additional object references from this env
        self.torso_body_id = self.sim.model.body_name2id(self.torso.root_body)
        self.probe_id = self.sim.model.body_name2id(self.robots[0].gripper.root_body)
    
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled
        
        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        
        pf = self.robots[0].robot_model.naming_prefix
        
        # Remove unnecessary observables
        del observables[pf + "joint_pos"]
        del observables[pf + "joint_pos_cos"]
        del observables[pf + "joint_pos_sin"]
        del observables[pf + "joint_vel"]
        del observables[pf + "gripper_qvel"]
        del observables[pf + "gripper_qpos"]
        del observables[pf + "eef_pos"]
        del observables[pf + "eef_quat"]
        
        sensors = []
        
        # probe information
        modality = f"{pf}proprio"  # Need to use this modality since proprio obs cannot be empty in GymWrapper
        
        @sensor(modality=modality)
        def eef_contact_force(obs_cache):
            return self.sim.data.cfrc_ext[self.probe_id][-3:]
        
        @sensor(modality=modality)
        def eef_torque(obs_cache):
            return self.robots[0].ee_torque
        
        @sensor(modality=modality)
        def eef_vel(obs_cache):
            return self.robots[0]._hand_vel
        
        @sensor(modality=modality)
        def eef_contact_force_z_diff(obs_cache):
            return self.z_contact_force_running_mean - self.goal_contact_z_force
        
        @sensor(modality=modality)
        def eef_contact_derivative_force_z_diff(obs_cache):
            return self.der_z_contact_force - self.goal_der_contact_z_force
        
        @sensor(modality=modality)
        def eef_vel_diff(obs_cache):
            return self.vel_running_mean - self.goal_velocity
        
        @sensor(modality=modality)
        def eef_pose_diff(obs_cache):
            pos_error = self._eef_xpos - self.traj_pt
            # quat_error = difference_quat(self._eef_xquat, self.goal_quat)
            quat_error = difference_quat(self._eef_xquat, T.axisangle2quat(self.traj_ori))  # added(by JadeCong)
            pose_error = np.concatenate((pos_error, quat_error))
            return pose_error
        
        sensors += [
            eef_contact_force,
            eef_torque, 
            eef_vel, 
            eef_contact_force_z_diff, 
            eef_contact_derivative_force_z_diff, 
            eef_vel_diff, 
            eef_pose_diff]
        
        names = [s.__name__ for s in sensors]
        
        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )
        
        return observables
    
    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        
        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()
            
            # Loop through all objects and reset their positions
            for obj_pos, _, obj in object_placements.values():
                print("obj: ", obj)
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array([0.5, 0.5, -0.5, -0.5])]))
                self.sim.forward()      # update sim states
                
        # says if probe has been in touch with torso
        self.has_touched_torso = False
        
        # initial position of end-effector
        self.ee_initial_pos = self._eef_xpos
        
        # Get the up surface of torso(by JadeCong)
        self.torso_up_surface_index = list()
        for idy in self.torso_start_index:
            for idx in iter(range(1, 10)):
                self.torso_up_surface_index.append(idy - idx)
        
        # Get the body_pos and body_quat(by JadeCong)
        # self.torso_mesh_body_pos_world_array, self.torso_mesh_body_quat_world_array = self.torso.get_body_world_pose_array(self._torso_xpos, np.array([0.5, 0.5, -0.5, -0.5]))
        self.torso_mesh_body_pos_world_array, _ = self.torso.get_body_world_pose_array(self._torso_xpos, np.array([0.5, 0.5, -0.5, -0.5]))
        self.torso_mesh_body_quat_world_array = self.torso.fit_body_world_quat_array(self.torso_start_index, self.torso_up_surface_index, self.torso_mesh_body_pos_world_array)
        
        # create trajectory
        # self.trajectory = self._get_sacanning_trajectory()  # get the scanning trajectory(by JadeCong)
        self.trajectory = self.get_sacanning_trajectory()  # get the scanning trajectory(by JadeCong)
        # vis.add("delta_traj", self.trajectory.discretize(0.001), color=(1, 0, 0, 1))
        # print("haha", self.trajectory.discretize(0.001))
        # vis.add("robot_eef", self.trajectory.discretize(0.001))
        # vis.animate("robot_eef", self.trajectory)
        # vis.spin(float('inf'))
        
        # initialize trajectory step
        self.initial_traj_step = 0  # np.random.default_rng().uniform(low=0, high=self.num_waypoints - 1)  # set to zero(by JadeCong)
        self.traj_step = self.initial_traj_step  # step at which to evaluate trajectory. Must be in interval [0, num_waypoints - 1]
        
        # set first trajectory point
        # self.traj_pt = self.trajectory.eval(self.traj_step)
        # self.traj_pt_vel = self.trajectory.deriv(self.traj_step)
        self.traj_pt = self.trajectory.eval(self.traj_step)[1]  # (by JadeCong)
        self.traj_ori = T.quat2axisangle(T.mat2quat(np.array(self.trajectory.eval(self.traj_step)[0]).reshape(3, 3)))  # (by JadeCong)
        self.traj_pt_vel = self.trajectory.deriv(self.traj_step)[1]  # (by JadeCong)
        
        # give controller access to robot (and its measurements)
        if self.robots[0].controller.name == "HMFC":
            self.robots[0].controller.set_robot(self.robots[0])
        
        # initialize controller's trajectory
        self.robots[0].controller.traj_pos = self.traj_pt
        # self.robots[0].controller.traj_ori = T.quat2axisangle(self.goal_quat)
        self.robots[0].controller.traj_ori = self.traj_ori  # (by JadeCong)
        
        # get initial joint positions for robot
        init_qpos = self._get_initial_qpos()
        # init_qpos = self.sim.data.qpos[self.robots[0]._ref_joint_pos_indexes]  # (by JadeCong)
        
        # override initial robot joint positions
        self.robots[0].set_robot_joint_positions(init_qpos)
        
        # update controller with new initial joints
        self.robots[0].controller.update_initial_joints(init_qpos)
        
        # initialize previously contact force measurement
        self.prev_z_contact_force = 0
        
        # intialize derivative of contact force
        self.der_z_contact_force = 0
        
        # initialize running mean of velocity 
        self.vel_running_mean = np.linalg.norm(self.robots[0]._hand_vel)
        
        # initialize running mean of contact force
        self.z_contact_force_running_mean = self.sim.data.cfrc_ext[self.probe_id][-1]
        
        # initialize data collection
        if self.save_data:
            # simulation data
            self.data_ee_pos = np.array(np.zeros((self.horizon, 3)))
            self.data_ee_goal_pos = np.array(np.zeros((self.horizon, 3)))
            self.data_ee_ori_diff = np.array(np.zeros(self.horizon))
            self.data_ee_vel = np.array(np.zeros((self.horizon, 3)))
            self.data_ee_goal_vel = np.array(np.zeros(self.horizon))
            self.data_ee_running_mean_vel = np.array(np.zeros(self.horizon))
            self.data_ee_quat = np.array(np.zeros((self.horizon, 4)))               # (x,y,z,w)
            self.data_ee_goal_quat = np.array(np.zeros((self.horizon, 4)))          # (x,y,z,w)
            self.data_ee_diff_quat = np.array(np.zeros(self.horizon))               # (x,y,z,w)
            self.data_ee_z_contact_force = np.array(np.zeros(self.horizon))
            self.data_ee_z_goal_contact_force = np.array(np.zeros(self.horizon))
            self.data_ee_z_running_mean_contact_force = np.array(np.zeros(self.horizon))
            self.data_ee_z_derivative_contact_force = np.array(np.zeros(self.horizon))
            self.data_ee_z_goal_derivative_contact_force = np.array(np.zeros(self.horizon))
            self.data_is_contact = np.array(np.zeros(self.horizon))
            self.data_q_pos = np.array(np.zeros((self.horizon, self.robots[0].dof)))
            self.data_q_torques = np.array(np.zeros((self.horizon, self.robots[0].dof)))
            self.data_time = np.array(np.zeros(self.horizon))
            
            # reward data
            self.data_pos_reward = np.array(np.zeros(self.horizon))
            self.data_ori_reward = np.array(np.zeros(self.horizon))
            self.data_vel_reward = np.array(np.zeros(self.horizon))
            self.data_force_reward = np.array(np.zeros(self.horizon))
            self.data_der_force_reward = np.array(np.zeros(self.horizon))
            
            # policy/controller data
            self.data_action = np.array(np.zeros((self.horizon, self.robots[0].action_dim)))
    
    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested
        
        Args:
            action (np.array): Action to execute within the environment
        
        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        reward, done, info = super()._post_action(action)
        
        # Set the flag of origin(by JadeCong)
        self._add_waypoint_site(self.viewer.viewer, "World_Origin", origin_size=[0.01, 0.01, 0.01], axis_size=[0.005, 0.005, 0.8],
                                pos=np.array([0, 0, 0]), quat=np.array([0, 0, 0, 1]))
        
        # Add the waypoint site of the trajectory(position and orientation) for every timestep given rows and columns(by JadeCong)
        # for idx in iter(range(self.grid_pts_x * self.grid_pts_y)):
        #     self._add_waypoint_site(self.viewer.viewer, str(idx), origin_size=[0.001, 0.001, 0.001], axis_size=[0.0005, 0.0005, 0.05],
        #                             pos=self.torso_scanning_order_traj['wp_{}'.format(idx)]['pos'], quat=self.torso_scanning_order_traj['wp_{}'.format(idx)]['ori'])
        #     if idx < (self.grid_pts_x * self.grid_pts_y - 1):
        #         self._add_waypoint_line(self.viewer.viewer, "", pos_start=self.torso_scanning_order_traj['wp_{}'.format(idx)]['pos'], pos_end=self.torso_scanning_order_traj['wp_{}'.format(idx + 1)]['pos'])
        
        # Add the waypoint site of the trajectory(position and orientation) for every timestep given torso_up_surface_index(by JadeCong)
        for idx in iter(range(len(self.torso_up_surface_index))):
            self._add_waypoint_site(self.viewer.viewer, str(idx), origin_size=[0.001, 0.001, 0.001], axis_size=[0.0005, 0.0005, 0.05],
                                    pos=self.torso_scanning_order_traj['wp_{}'.format(idx)]['pos'], quat=self.torso_scanning_order_traj['wp_{}'.format(idx)]['ori'])
            if idx < (len(self.torso_up_surface_index) - 1):
                self._add_waypoint_line(self.viewer.viewer, "", rgba=[0, 1, 0.7, 1], pos_start=self.torso_scanning_order_traj['wp_{}'.format(idx)]['pos'], pos_end=self.torso_scanning_order_traj['wp_{}'.format(idx + 1)]['pos'])
        
        # Convert to trajectory timstep
        normalizer = (self.horizon / (self.num_waypoints - 1))  # equally many timesteps to reach each waypoint
        self.traj_step = self.timestep / normalizer + self.initial_traj_step
        
        # Check the start flag of delta scanning trajectroy and the accomplish percent of delta scanning trajectory(by JadeCong)
        if self.traj_step <= 1 / normalizer:
            print("scanning_start_index: {}".format(self.scanning_index))
        print("traj_step: {}%".format(self.traj_step / (self.num_waypoints - 1) * 100))  # print the accomplish percent every step(by JadeCong)
        
        # Check the end flag of delta scanning trajectroy(by JadeCong)
        if self.traj_step == self.num_waypoints - 1:
            self.scanning_index += (self.scanning_waypoint_num - 1)  # for wp_num >= 2
            print("scanning_end_index: {}".format(self.scanning_index))
        
        # Check whether the all scanning trajectory gets accomplished and rerun the scanning demo(by JadeCong)
        # if self.scanning_index >= self.grid_pts_x * self.grid_pts_y - 1:
        if self.scanning_index >= len(self.torso_up_surface_index) - 1:
            self.scan_beginning = True
            self.scanning_index = 0
            print("<<" * 40 + "Let's rerun the torso scanning demo!" + ">>" * 40)
        
        # update trajectory point
        # self.traj_pt = self.trajectory.eval(self.traj_step)
        self.traj_pt = self.trajectory.eval(self.traj_step)[1]  # (by JadeCong)
        self.traj_ori = T.quat2axisangle(T.mat2quat(np.array(self.trajectory.eval(self.traj_step)[0]).reshape(3, 3)))  # (by JadeCong)
        
        # update controller's trajectory
        self.robots[0].controller.traj_pos = self.traj_pt
        self.robots[0].controller.traj_ori = self.traj_ori  # added(by JadeCong)
        
        # update velocity running mean (simple moving average)
        self.vel_running_mean += ((np.linalg.norm(self.robots[0]._hand_vel) - self.vel_running_mean) / self.timestep)
        
        # update derivative of contact force
        z_contact_force = self.sim.data.cfrc_ext[self.probe_id][-1]
        self.der_z_contact_force = (z_contact_force - self.prev_z_contact_force) / self.control_timestep
        self.prev_z_contact_force = z_contact_force
        
        # update contact force running mean (exponential moving average)
        self.z_contact_force_running_mean = self.alpha * z_contact_force + (1 - self.alpha) * self.z_contact_force_running_mean
        
        # check for early termination
        if self.early_termination:
            done = done or self._check_terminated()
        
        # plot the force and velocity # (by JadeCong) # TODO:add plot func
        # print("self.traj_step: ", self.timestep)
        # print("z_contact_force: ", z_contact_force)
        # print("self.vel_running_mean: ", self.vel_running_mean)
        # print("der_z_contact_force: ", self.der_z_contact_force)
        self.qtplt.update_plot(self.timestep, np.array([z_contact_force, self.vel_running_mean, self.der_z_contact_force]))
        # now_timestep = self.traj_step
        # now_force = z_contact_force
        # now_velocity = self.vel_running_mean
        # time = np.linspace(self.old_timestep, now_timestep, 2)
        # force = np.linspace(self.old_force, now_force, 2)
        # velocity = np.linspace(self.old_velocity, now_velocity, 2)
        # data = np.array([force, velocity])
        # self.rtpd.plot_data(time, data)
        # self.dplt.plot(time, data[0])
        # self.dplt.plot(time, data[1])
        # self.dplt.show()
        # update the old_timestep and old_force, old_velocity
        # self.old_timestep = self.traj_step
        # self.old_force = now_force
        # self.old_velocity = now_velocity
        # check whether end the plot
        # if self.scan_beginning == True:
        #     self.rtpd.end_plot()
        
        # collect data
        if self.save_data:
            # simulation data
            self.data_ee_pos[self.timestep - 1] = self._eef_xpos
            self.data_ee_goal_pos[self.timestep - 1] = self.traj_pt
            self.data_ee_vel[self.timestep - 1] = self.robots[0]._hand_vel
            self.data_ee_goal_vel[self.timestep - 1] = self.goal_velocity
            self.data_ee_running_mean_vel[self.timestep -1] = self.vel_running_mean
            self.data_ee_quat[self.timestep - 1] = self._eef_xquat
            # self.data_ee_goal_quat[self.timestep - 1] = self.goal_quat
            self.data_ee_goal_quat[self.timestep - 1] = T.axisangle2quat(self.traj_ori)  # (by JadeCong)
            # self.data_ee_diff_quat[self.timestep - 1] = distance_quat(convert_quat(self._eef_xquat, to="wxyz"), convert_quat(self.goal_quat, to="wxyz"))
            self.data_ee_diff_quat[self.timestep - 1] = distance_quat(convert_quat(self._eef_xquat, to="wxyz"), convert_quat(T.axisangle2quat(self.traj_ori), to="wxyz"))  # (by JadeCong)
            self.data_ee_z_contact_force[self.timestep - 1] = self.sim.data.cfrc_ext[self.probe_id][-1]
            self.data_ee_z_goal_contact_force[self.timestep - 1] = self.goal_contact_z_force
            self.data_ee_z_running_mean_contact_force[self.timestep - 1] = self.z_contact_force_running_mean
            self.data_ee_z_derivative_contact_force[self.timestep - 1] = self.der_z_contact_force
            self.data_ee_z_goal_derivative_contact_force[self.timestep - 1] = self.goal_der_contact_z_force
            self.data_is_contact[self.timestep - 1] = self._check_probe_contact_with_torso()
            self.data_q_pos[self.timestep - 1] = self.robots[0]._joint_positions
            self.data_q_torques[self.timestep - 1] = self.robots[0].torques
            self.data_time[self.timestep - 1] = (self.timestep - 1) / self.horizon * 100  # percentage of completed episode
            
            # reward data
            self.data_pos_reward[self.timestep - 1] = self.pos_reward
            self.data_ori_reward[self.timestep - 1] = self.ori_reward
            self.data_vel_reward[self.timestep - 1] = self.vel_reward
            self.data_force_reward[self.timestep - 1] = self.force_reward
            self.data_der_force_reward[self.timestep - 1] = self.der_force_reward
            
            # policy/controller data
            self.data_action[self.timestep - 1] = action
        
        # save data
        if done and self.save_data:
            # simulation data
            sim_data_fldr = "simulation_data"
            self._save_data(self.data_ee_pos, sim_data_fldr, "ee_pos")
            self._save_data(self.data_ee_goal_pos, sim_data_fldr, "ee_goal_pos")
            self._save_data(self.data_ee_vel, sim_data_fldr, "ee_vel")
            self._save_data(self.data_ee_goal_vel, sim_data_fldr, "ee_goal_vel")
            self._save_data(self.data_ee_running_mean_vel, sim_data_fldr, "ee_running_mean_vel")
            self._save_data(self.data_ee_quat, sim_data_fldr, "ee_quat")
            self._save_data(self.data_ee_goal_quat, sim_data_fldr, "ee_goal_quat")
            self._save_data(self.data_ee_diff_quat, sim_data_fldr, "ee_diff_quat")
            self._save_data(self.data_ee_z_contact_force, sim_data_fldr, "ee_z_contact_force")
            self._save_data(self.data_ee_z_goal_contact_force, sim_data_fldr, "ee_z_goal_contact_force")
            self._save_data(self.data_ee_z_running_mean_contact_force, sim_data_fldr, "ee_z_running_mean_contact_force")
            self._save_data(self.data_ee_z_derivative_contact_force, sim_data_fldr, "ee_z_derivative_contact_force")
            self._save_data(self.data_ee_z_goal_derivative_contact_force, sim_data_fldr, "ee_z_goal_derivative_contact_force")
            self._save_data(self.data_is_contact, sim_data_fldr, "is_contact")
            self._save_data(self.data_q_pos, sim_data_fldr, "q_pos")
            self._save_data(self.data_q_torques, sim_data_fldr, "q_torques")
            self._save_data(self.data_time, sim_data_fldr, "time")
            
            # reward data
            reward_data_fdlr = "reward_data"
            self._save_data(self.data_pos_reward, reward_data_fdlr, "pos")
            self._save_data(self.data_ori_reward, reward_data_fdlr, "ori")
            self._save_data(self.data_vel_reward, reward_data_fdlr, "vel")
            self._save_data(self.data_force_reward, reward_data_fdlr, "force")
            self._save_data(self.data_der_force_reward, reward_data_fdlr, "derivative_force")
            
            # policy/controller data
            self._save_data(self.data_action, "policy_data", "action")
        
        return reward, done, info
    
    def visualize(self, vis_settings):
        """
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)
    
    def _check_success(self):
        return False
    
    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:
            - Collision with table
            - Joint Limit reached
            - Deviates from trajectory position
            - Deviates from desired orientation when in contact with torso
            - Loses contact with torso
        
        Returns:
            bool: True if episode is terminated
        """
        
        terminated = False
        
        # Prematurely terminate if reaching joint limits
        if self.robots[0].check_q_limits():
            print(40 * '-' + " JOINT LIMIT " + 40 * '-')
            terminated = True
        
        # Prematurely terminate if probe deviates away from trajectory (represented by a low position reward)
        if np.linalg.norm(self.pos_error) > self.pos_error_threshold:
            print(40 * '-' + " DEVIATES FROM TRAJECTORY " + 40 * '-')
            terminated = True
        
        # Prematurely terminate if probe deviates from desired orientation when touching probe
        if self._check_probe_contact_with_torso() and self.ori_error > self.ori_error_threshold:
            print(40 * '-' + " (TOUCHING BODY) PROBE DEVIATES FROM DESIRED ORIENTATION " + 40 * '-')
            terminated = True
        
        # Prematurely terminate if probe loses contact with torso
        if self.has_touched_torso and not self._check_probe_contact_with_torso():
            print(40 * '-' + " LOST CONTACT WITH TORSO " + 40 * '-')
            terminated = True
        
        return terminated
    
    def _get_contacts_objects(self, model):
        """
        Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
        contact objects currently in contact with that model (excluding the geoms that are part of the model itself).
        
        Args:
            model (MujocoModel): Model to check contacts for.
        
        Returns:
            set: Unique contact objects containing information about contacts with this model.
        
        Raises:
            AssertionError: [Invalid input type]
        """
        # Make sure model is MujocoModel type
        assert isinstance(model, MujocoModel), \
            "Inputted model must be of type MujocoModel; got type {} instead!".format(type(model))
        contact_set = set()
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            # check contact geom in geoms; add to contact set if match is found
            g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)
            if g1 in model.contact_geoms or g2 in model.contact_geoms:
                contact_set.add(contact)
        
        return contact_set
    
    def _check_probe_contact_with_upper_part_torso(self):
        """
        Check if the probe is in contact with the upper/top part of torso. Touching the torso on the sides should not count as contact.
        
        Returns:
            bool: True if probe both is in contact with upper part of torso and inside distance threshold from the torso center.
        """     
        # check for contact only if probe is in contact with upper part and close to torso center
        if  self._eef_xpos[-1] >= self._torso_xpos[-1] and np.linalg.norm(self._eef_xpos[:2] - self._torso_xpos[:2]) < 0.14:
            return self._check_probe_contact_with_torso()
        
        return False
    
    def _check_probe_contact_with_torso(self):
        """
        Check if the probe is in contact with the torso.
        
        NOTE This method utilizes the autogenerated geom names for MuJoCo-native composite objects
        
        Returns:
            bool: True if probe is in contact with torso
        """     
        gripper_contacts = self._get_contacts_objects(self.robots[0].gripper)
        reg_ex = "[G]\d+[_]\d+[_]\d+$"
        
        # check contact with torso geoms based on autogenerated names
        for contact in gripper_contacts:
            g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2) 
            match1 = re.search(reg_ex, g1)
            match2 = re.search(reg_ex, g2)
            if match1 != None or match2 != None:
                contact_normal_axis = contact.frame[:3]
                self.has_touched_torso = True
                return True
        
        return False
    
    def _check_probe_contact_with_table(self):
        """
        Check if the probe is in contact with the tabletop.
        
        Returns:
            bool: True if probe is in contact with table
        """
        return self.check_contact(self.robots[0].gripper, "table_collision")
    
    def get_sacanning_trajectory(self):  #(by JadeCong)
        """
        Calculates a trajectory between two waypoints on the torso. The waypoints are extracted from a grid on the torso.
        The first waypoint is given at time t=0, and the second waypoint is given at t=1.
        If self.deterministic_trajectory is true, a deterministic trajectory along the x-axis of the torso is calculated.
        
        Args:
        
        Returns:
            (klampt.model.trajectory Object):  trajectory
        """
        self.torso_scanning_waypoint_array = self.get_torso_scanning_waypoint_array(self.torso_up_surface_index, self.torso_mesh_body_pos_world_array, self.torso_mesh_body_quat_world_array, self.goal_velocity, self.goal_contact_z_force)  # get the torso waypoint array(by JadeCong)
        self.torso_scanning_order_traj = self.get_torso_scanning_order_trajectory()  # get the torso scanning order trajectory(by JadeCong)
        
        if self.deterministic_trajectory:
            # start_point = [0.062, -0.020, 0.896]
            # end_point = [-0.032, -0.075, 0.896]
            # waypoints = np.array([start_point, end_point])
            start_point = np.array([0.062, -0.020,  0.896])
            end_point = np.array([-0.032, -0.075,  0.896])
            goal_quat = np.array([-0.69192486,  0.72186726, -0.00514253, -0.01100909])  # xyzw
            quat_array = quat2mat(goal_quat).flatten()
            waypoints = np.array([np.append(quat_array, start_point), np.append(quat_array, end_point)])  # R+t(9+3)
        else:
            waypoints = self.get_torso_scanning_delta_trajectory(self.scanning_waypoint_num)  # for wp_num >= 2
            print("===" * 50)
            print("Reset internal again!")
            print("start_point: {}".format(waypoints[0][9:]))
            print("end_point: {}".format(waypoints[-1][9:]))
        
        self.traj_start = waypoints[0][9:]
        self.traj_end = waypoints[-1][9:]
        milestones = np.array(waypoints)
        self.num_waypoints = np.size(milestones, 0)
        
        # return trajectory.Trajectory(milestones=milestones)
        return trajectory.SE3Trajectory(milestones=milestones)
    
    def get_torso_scanning_delta_trajectory(self, wp_num):  # (by JadeCong)
        """
        Get the delta trajectory for torso scanning.
        """
        wp_scanning = list()
        
        # Check whether torso scanning just begins(without ori, vel and force)
        if self.scan_beginning:
            for idx in iter(range(wp_num)):
                # wp_scanning.append(self.torso_scanning_order_traj['wp_{}'.format(idx)]['pos'])  # for wp_num >= 2
                
                wp_scanning.append(np.append(quat2mat(self.torso_scanning_order_traj['wp_{}'.format(idx)]['ori']).flatten(),
                                            self.torso_scanning_order_traj['wp_{}'.format(idx)]['pos']))  # for wp_num >= 2
            
            self.scan_beginning = False
        else:
            # check whether reach the end of scanning trajectory and rerun form the start
            if self.scanning_index <= len(self.torso_up_surface_index) - wp_num:
                for idx in iter(range(wp_num)):
                    # wp_scanning.append(self.torso_scanning_order_traj['wp_{}'.format(idx + self.scanning_index)]['pos'])  # for wp_num >= 2
                    
                    wp_scanning.append(np.append(quat2mat(self.torso_scanning_order_traj['wp_{}'.format(idx + self.scanning_index)]['ori']).flatten(),
                                                self.torso_scanning_order_traj['wp_{}'.format(idx + self.scanning_index)]['pos']))  # for wp_num >= 2
            
            elif (self.scanning_index > len(self.torso_up_surface_index) - wp_num) and (self.scanning_index < len(self.torso_up_surface_index) - 1):
                for idx in iter(range(len(self.torso_up_surface_index) - self.scanning_index)):
                    # wp_scanning.append(self.torso_scanning_order_traj['wp_{}'.format(idx + self.scanning_index)]['pos'])  # for wp_num >= 2
                    
                    wp_scanning.append(np.append(quat2mat(self.torso_scanning_order_traj['wp_{}'.format(idx + self.scanning_index)]['ori']).flatten(),
                                                self.torso_scanning_order_traj['wp_{}'.format(idx + self.scanning_index)]['pos']))  # for wp_num >= 2
        
        return wp_scanning  # for wp_num >= 2
    
    def get_torso_scanning_order_trajectory(self):  # (by JadeCong)
        """
        Get the torso waypoints in the specified S spline order and return a ordered array.
        """
        # Put the torso scanning waypoints in S spline order for a scanning trajectory
        torso_scanning_wp_array = self.torso_scanning_waypoint_array
        torso_scanning_order_traj = {}
        torso_scanning_row = len(self.torso_start_index)
        torso_scanning_column = int(len(self.torso_up_surface_index) / len(self.torso_start_index))
        
        for idx_y in iter(range(torso_scanning_row)):
            for idx_x in iter(range(torso_scanning_column)):
                if idx_y & 1 == 0:
                    torso_scanning_order_traj['wp_{}'.format(idx_y * torso_scanning_column + idx_x)] = torso_scanning_wp_array[idx_y][idx_x]
                else:
                    torso_scanning_order_traj['wp_{}'.format(idx_y * torso_scanning_column + idx_x)] = torso_scanning_wp_array[idx_y][torso_scanning_column - 1 - idx_x]
        
        return torso_scanning_order_traj
    
    def get_torso_scanning_waypoint_array(self, torso_scanning_index_list, torso_body_pos_world_array, torso_body_quat_world_array, goal_vel, goal_force):  # (by JadeCong)
        """
        Creates a torso waypoint array as the keypoints of scanning trajectory according to given points in x and y axis.
        A waypoint of trajectory contains position, orientation, velocity(linear and angular) and contact force(option).
        
        Args: given number of x axis(int) and y axis(int).
        
        Returns:
            (numpy.array): waypoint array of scanning trajectory([0:2]:position; [3:6]:orientation; [7:12]:velocity; [13:18]:contact force)
            (dict): waypoint array of scanning trajectory("position":np.array([3]), "orientation":np.array([4]), "velocity":np.array([6]), "contact_force":np.array([6]))
        """
        # Import the dependencies
        import numpy as np
        
        # Initialize the dict for waypoint of scanning trajectory
        torso_scanning_column = int(len(torso_scanning_index_list) / len(self.torso_start_index))
        print("torso_scanning_column", torso_scanning_column)
        torso_scanning_wp_array = [[0 for i in range(torso_scanning_column)] for j in range(len(self.torso_start_index))]  # j for row and i for column
        
        # Get the waypoint array(contain dicts)
        for idx_y in iter(range(len(self.torso_start_index))):
            for idx_x in iter(range(torso_scanning_column)):
                # position, unit: m
                pos = torso_body_pos_world_array[torso_scanning_index_list[idx_y * torso_scanning_column + idx_x]]
                
                # orientation(form mujoco(wxyz) to custom(xyzw)), unit: radians
                # ori = T.mat2quat(np.array(self.random_quat_sample(np.array([-0.69192486, 0.72186726, -0.00514253, -0.01100909]), 0.000001)).reshape(3, 3))  # xyzw
                # ori = convert_quat(torso_body_quat_world_array[torso_scanning_index_list[idx_y * torso_scanning_column + idx_x]], to="xyzw")  # xyzw
                # ori = torso_body_quat_world_array[idx_y * torso_scanning_column + idx_x]  # xyzw
                ori = T.mat2quat(np.array(self.random_quat_sample(torso_body_quat_world_array[idx_y * torso_scanning_column + idx_x], 0.000001)).reshape(3, 3))  # xyzw
                
                # velocity(linear and angular)
                # vel = np.array([0, 0.04, 0, 0, 0, 0])  # just limit the y axis linear velocity, unit:m/s,radians/s
                vel = np.array([0, goal_vel, 0, 0, 0, 0])
                
                # contact force(now given the custom value)
                # force = np.array([0, 0.5, 5, 0, 0, 0])  # just limit the z axis contact force and y axis scanning force, unit: N, N.m
                force = np.array([0, 0.5, goal_force, 0, 0, 0])
                
                # put the waypoint into list array
                waypoint = dict(zip(['pos', 'ori', 'vel', 'force'], [pos, ori, vel, force]))
                torso_scanning_wp_array[idx_y][idx_x] = waypoint
        
        return torso_scanning_wp_array
    
    def random_quat_sample(self, ref_quat, var):
        """Returns a uniformly distributed rotation matrix."""
        import random
        q = [random.gauss(ref_quat[0], var), random.gauss(ref_quat[1], var), random.gauss(ref_quat[2], var), random.gauss(ref_quat[3], var)]
        q = vectorops.unit(q)
        theta = math.acos(q[3]) * 2.0
        if abs(theta) < 1e-8:
            m = [0, 0, 0]
        else:
            m = vectorops.mul(vectorops.unit(q[0:3]), theta)
        return so3.from_rotation_vector(m)
    
    def _get_sacanning_trajectory(self):  #(by JadeCong)
        """
        Calculates a trajectory between two waypoints on the torso. The waypoints are extracted from a grid on the torso.
        The first waypoint is given at time t=0, and the second waypoint is given at t=1.
        If self.deterministic_trajectory is true, a deterministic trajectory along the x-axis of the torso is calculated.
        
        Args:
        
        Returns:
            (klampt.model.trajectory Object):  trajectory
        """
        self.torso_scanning_waypoint_array = self._get_torso_scanning_waypoint_array(self.grid_pts_x, self.grid_pts_y, self.goal_quat, self.goal_velocity, self.goal_contact_z_force)  # get the torso waypoint array(by JadeCong)
        self.torso_scanning_order_traj = self._get_torso_scanning_order_trajectory()  # get the torso scanning order trajectory(by JadeCong)
        
        if self.deterministic_trajectory:
            start_point = [0.062, -0.020,  0.896]
            end_point = [-0.032, -0.075,  0.896]
            waypoints = np.array([start_point, end_point])
        else:
            waypoints = self._get_torso_scanning_delta_trajectory(self.scanning_waypoint_num)  # for wp_num >= 2
            print("===" * 50)
            print("Reset internal again!")
            print("start_point: {}".format(waypoints[0]))
            print("end_point: {}".format(waypoints[-1]))
        
        self.traj_start = waypoints[0]
        self.traj_end = waypoints[-1]
        milestones = np.array(waypoints)
        self.num_waypoints = np.size(milestones, 0)
        
        return trajectory.Trajectory(milestones=milestones)
    
    def _get_torso_scanning_delta_trajectory(self, wp_num):  # (by JadeCong)
        """
        Get the delta trajectory for torso scanning.
        """
        wp_scanning = list()
        
        # Check whether torso scanning just begins(without ori, vel and force)
        if self.scan_beginning:
            for idx in iter(range(wp_num)):
                wp_scanning.append(self.torso_scanning_order_traj['wp_{}'.format(idx)]['pos'])  # for wp_num >= 2
            self.scan_beginning = False
        else:
            # check whether reach the end of scanning trajectory and rerun form the start
            if self.scanning_index <= self.grid_pts_x * self.grid_pts_y - wp_num:
                for idx in iter(range(wp_num)):
                    wp_scanning.append(self.torso_scanning_order_traj['wp_{}'.format(idx + self.scanning_index)]['pos'])  # for wp_num >= 2
            elif self.scanning_index > self.grid_pts_x * self.grid_pts_y - wp_num and self.scanning_index < self.grid_pts_x * self.grid_pts_y -1:
                for idx in iter(range(self.grid_pts_x * self.grid_pts_y - self.scanning_index)):
                    wp_scanning.append(self.torso_scanning_order_traj['wp_{}'.format(idx + self.scanning_index)]['pos'])  # for wp_num >= 2
        
        return wp_scanning  # for wp_num >= 2
    
    def _get_torso_scanning_order_trajectory(self):  # (by JadeCong)
        """
        Get the torso waypoints in the specified S spline order and return a ordered array.
        """
        # Put the torso scanning waypoints in S spline order for a scanning trajectory
        torso_scanning_wp_array = self.torso_scanning_waypoint_array
        torso_scanning_order_traj = {}
        for idx_y in iter(range(self.grid_pts_y)):
            for idx_x in iter(range(self.grid_pts_x)):
                if idx_y & 1 == 0:
                    torso_scanning_order_traj['wp_{}'.format(idx_y * self.grid_pts_x + idx_x)] = torso_scanning_wp_array[idx_y][idx_x]
                else:
                    torso_scanning_order_traj['wp_{}'.format(idx_y * self.grid_pts_x + idx_x)] = torso_scanning_wp_array[idx_y][self.grid_pts_x - 1 - idx_x]
        
        return torso_scanning_order_traj
    
    def _get_torso_scanning_waypoint_array(self, x_axis_points, y_axis_points, goal_ori, goal_vel, goal_force):  # (by JadeCong)
        """
        Creates a torso waypoint array as the keypoints of scanning trajectory according to given points in x and y axis.
        A waypoint of trajectory contains position, orientation, velocity(linear and angular) and contact force(option).
        
        Args: given number of x axis(int) and y axis(int).
        
        Returns:
            (numpy.array): waypoint array of scanning trajectory([0:2]:position; [3:6]:orientation; [7:12]:velocity; [13:18]:contact force)
            (dict): waypoint array of scanning trajectory("position":np.array([3]), "orientation":np.array([4]), "velocity":np.array([6]), "contact_force":np.array([6]))
        """
        # Import the dependencies
        import numpy as np
        
        # Get the torso xy grid and z axis coordinate for waypoint position
        x = np.linspace(-self.x_range + self._torso_xpos[0], self.x_range + self._torso_xpos[0],
                        num=x_axis_points)  # add offset in negative range due to weird robot angles close to robot base
        y = np.linspace(-self.y_range + self._torso_xpos[1], self.y_range + self._torso_xpos[1], num=y_axis_points)
        z_pos = self._torso_xpos[-1] + self.top_torso_offset
        
        # Initialize the dict for waypoint of scanning trajectory
        torso_scanning_wp_array = [[0 for i in range(x_axis_points)] for j in range(y_axis_points)]  # j for row and i for column
        
        # Get the waypoint array(contain dicts)
        for idx_y in iter(range(y_axis_points)):
            for idx_x in iter(range(x_axis_points)):
                # position, unit: m
                x_pos = x[idx_x]
                y_pos = y[idx_y]
                pos = np.array([x_pos, y_pos, z_pos])
                
                # orientation(now given the custom zero(wxyz)), unit: radians
                # ori = np.array([-0.01100909, -0.69192486,  0.72186726, -0.00514253])
                ori = goal_ori
                
                # velocity(linear and angular)
                # vel = np.array([0, 0.04, 0, 0, 0, 0])  # just limit the y axis linear velocity, unit:m/s,radians/s
                vel = np.array([0, goal_vel, 0, 0, 0, 0])
                
                # contact force(now given the custom value)
                # force = np.array([0, 0.5, 5, 0, 0, 0])  # just limit the z axis contact force and y axis scanning force, unit: N, N.m
                force = np.array([0, 0.5, goal_force, 0, 0, 0])
                
                # put the waypoint into list array
                waypoint = dict(zip(['pos', 'ori', 'vel', 'force'], [pos, ori, vel, force]))
                torso_scanning_wp_array[idx_y][idx_x] = waypoint
        
        return torso_scanning_wp_array
    
    def _add_marker(self, viewer, name="mkr", origin_size=[0.006, 0.006, 0.006], pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        # Import the dependencies
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        # Add the origin of the marker
        viewer.add_marker(type=2,  # type=const.GEOM_SPHERE,  # 2 for sphere
                          size=origin_size,
                          pos=np.array(pos),
                          mat=R.from_quat(quat[[0, 1, 2, 3]]).as_matrix().flatten(),
                          # transfer the index from wxyz to xyzw
                          rgba=[1, 1, 0, 1],
                          label=name)
    
    def _add_waypoint_site(self, viewer, name="wp", origin_size=[0.006, 0.006, 0.006], axis_size=[0.002, 0.002, 0.2],
                            pos=[0, 0, 0], quat=[1, 0, 0, 0]):
        # Import the dependencies
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        # Add the origin of the site
        viewer.add_marker(type=2,  # type=const.GEOM_SPHERE,  # 2 for sphere
                          size=origin_size,
                          pos=np.array(pos),
                          mat=R.from_quat(quat[[0, 1, 2, 3]]).as_matrix().flatten(),  # transfer the index from wxyz to xyzw
                          rgba=[1, 1, 0, 1],
                          label=name)
        
        # Add the XYZ axes of the site
        viewer.add_marker(type=100,  # type=const.GEOM_ARROW,  # 5 for cylinder and 100 for arrow
                          size=axis_size,
                          pos=np.array(pos),
                          mat=(R.from_quat(quat[[0, 1, 2, 3]]) * R.from_euler('xyz', [0, 90, 0], degrees=True)).as_matrix().flatten(),  # transfer the index from wxyz to xyzw
                          rgba=[1, 0, 0, 1],
                          label="")
        viewer.add_marker(type=100,  # type=const.GEOM_ARROW,  # 5 for cylinder and 100 for arrow
                          size=axis_size,
                          pos=np.array(pos),
                          mat=(R.from_quat(quat[[0, 1, 2, 3]]) * R.from_euler('xyz', [-90, 0, 0], degrees=True)).as_matrix().flatten(),  # transfer the index from wxyz to xyzw
                          rgba=[0, 1, 0, 1],
                          label="")
        viewer.add_marker(type=100,  # type=const.GEOM_ARROW,  # 5 for cylinder and 100 for arrow
                          size=axis_size,
                          pos=np.array(pos),
                          mat=(R.from_quat(quat[[0, 1, 2, 3]]) * R.from_euler('xyz', [0, 0, 0], degrees=True)).as_matrix().flatten(),  # transfer the index from wxyz to xyzw
                          rgba=[0, 0, 1, 1],
                          label="")
    
    def _add_waypoint_line(self, viewer, name="line", rgba=[0, 1, 0.5, 1], pos_start=[0, 0, 0], pos_end=[1, 1, 1]):
        # Import the dependencies
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        # Function for getting the rotation vector between two vectors
        def get_rotation_vector(vector1, vector2):  # rotation from vector1 to vector2
            # get the rotation axis
            rot_axis = np.cross(vector1, vector2)
            # get the rotation angle
            rot_angle = np.arccos(vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
            # get the rotation vector
            rot_vector = rot_axis / np.linalg.norm(rot_axis) * rot_angle
            
            return rot_vector
        
        # Add the line between two waypoints
        viewer.add_marker(type=103,  # type=const.GEOM_LINE,  # 103 for line
                          size=[1, 1, np.linalg.norm(pos_end - pos_start)],
                          pos=pos_start,  # the pos of start waypoint
                          mat=R.from_rotvec(get_rotation_vector(np.array([0, 0, 1]), (pos_end - pos_start))).as_matrix().flatten(),  # transfer the index from wxyz to xyzw
                          rgba=rgba,
                          label=name)
    
    def _get_torso_grid(self):
        """
        Creates a 2D grid in the xy-plane on the top of the torso.
        
        Args:
        
        Returns:
            (numpy.array):  grid. First row contains x-coordinates and the second row contains y-coordinates.
        """
        x = np.linspace(-self.x_range + self._torso_xpos[0] + 0.03, self.x_range + self._torso_xpos[0], num=self.grid_pts)  # add offset in negative range due to weird robot angles close to robot base
        y = np.linspace(-self.y_range + self._torso_xpos[1], self.y_range + self._torso_xpos[1], num=self.grid_pts)
        
        x = np.array([x])
        y = np.array([y])
        
        return np.concatenate((x, y))
    
    def _get_waypoint(self, grid):
        """
        Extracts a random waypoint from the grid.
        
        Args:
        
        Returns:
            (numpy.array):  waypoint
        """
        x_pos = np.random.choice(grid[0])
        y_pos = np.random.choice(grid[1])
        z_pos = self._torso_xpos[-1] + self.top_torso_offset
        
        return np.array([x_pos, y_pos, z_pos])
    
    def _get_initial_qpos(self):
        """
        Calculates the initial joint position for the robot based on the initial desired pose (self.traj_pt, self.goal_quat).
        If self.initial_probe_pos_randomization is True, Guassian noise is added to the initial position of the probe.
        
        Args:
        
        Returns:
            (np.array): n joint positions 
        """
        pos = np.array(self.traj_pt)
        if self.initial_probe_pos_randomization:
            pos = self._add_noise_to_pos(pos)
        
        pos = self._convert_robosuite_to_toolbox_xpos(pos)
        # ori_euler = mat2euler(quat2mat(self.goal_quat))
        ori_euler = mat2euler(np.array(self.trajectory.eval(self.traj_step)[0]).reshape(3, 3))  # (by JadeCong)
        
        # desired pose
        T = SE3(pos) * SE3.RPY(ori_euler)
        
        # find initial joint positions
        if self.robots[0].name == "UR5e":
            robot = rtb.models.DH.UR5()
            sol = robot.ikine_min(T, q0=self.robots[0].init_qpos)
            # sol = robot.ikine_min(T, q0=self.robots[0]._joint_positions)  # by JadeCong
            
            # flip last joint around (pi)
            sol.q[-1] -= np.pi
            return sol.q
        
        elif self.robots[0].name == "Panda":
            robot = rtb.models.DH.Panda()
            sol = robot.ikine_min(T, q0=self.robots[0].init_qpos)
            # sol = robot.ikine_min(T, q0=self.robots[0]._joint_positions)  # by JadeCong
            return sol.q
    
    def _convert_robosuite_to_toolbox_xpos(self, pos):
        """
        Converts origin used in robosuite to origin used for robotics toolbox. Also transforms robosuite world frame (vectors x, y, z) to
        to correspond to world frame used in toolbox.
        
        Args:
            pos (np.array): position (x,y,z) given in robosuite coordinates and frame
        
        Returns:
            (np.array):  position (x,y,z) given in robotics toolbox coordinates and frame
        """
        xpos_offset = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])[0]
        zpos_offset = self.robots[0].robot_model.top_offset[-1] - 0.016
        
        # the numeric offset values have been found empirically, where they are chosen so that 
        # self._eef_xpos matches the desired position.
        if self.robots[0].name == "UR5e":
            return np.array([-pos[0] + xpos_offset + 0.08, -pos[1] + 0.025, pos[2] - zpos_offset + 0.15]) 
        
        if self.robots[0].name == "Panda":
            return np.array([pos[0] - xpos_offset - 0.06, pos[1], pos[2] - zpos_offset + 0.111])
    
    def _add_noise_to_pos(self, init_pos):
        """
        Adds Gaussian noise (variance) to the position.
        
        Args:
            init_pos (np.array): initial probe position 
        
        Returns:
            (np.array):  position with added noise
        """
        z_noise = np.random.normal(self.mu, self.sigma, 1)
        xy_noise = np.random.normal(self.mu, self.sigma/4, 2)
        
        x = init_pos[0] + xy_noise[0]
        y = init_pos[1] + xy_noise[1]
        z = init_pos[2] + z_noise[0]
        
        return np.array([x, y, z])
    
    def _save_data(self, data, fldr, filename):
        """
        Saves data to desired path.
        
        Args:
            data (np.array): Data to be saved 
            fldr (string): Name of destination folder
            filename (string): Name of file
        
        Returns:
        """
        os.makedirs(fldr, exist_ok=True)
        
        idx = 1
        path = os.path.join(fldr, filename + "_" + str(idx) + ".csv")
        
        while os.path.exists(path):
            idx += 1
            path = os.path.join(fldr, filename + "_" + str(idx) + ".csv")
        
        pd.DataFrame(data).to_csv(path, header=None, index=None)
    
    @property
    def _torso_xpos(self):
        """
        Grabs torso center position
        
        Returns:
            np.array: torso pos (x,y,z)
        """
        return np.array(self.sim.data.body_xpos[self.torso_body_id])
