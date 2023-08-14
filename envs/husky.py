# ===== add python path ===== #
import glob
import sys
import os
ROOT_PATH = os.getcwd()
for dir_idx, dir_name in enumerate(ROOT_PATH.split('/')):
    dir_path = '/'.join(ROOT_PATH.split('/')[:(dir_idx+1)])
    file_list = [os.path.basename(sub_dir) for sub_dir in glob.glob(f"{dir_path}/.*")]
    if '.git_package' in file_list:
        ROOT_PATH = dir_path
        break
if not ROOT_PATH in sys.path:
    sys.path.append(ROOT_PATH)
# =========================== #

# Isaac Gym
from isaacgym import gymtorch
from isaacgym import gymapi

# Isaac Gym Envs
from isaacgymenvs.tasks.base.vec_task import VecTask

# Others
import numpy as np
import pickle
import torch
import yaml
import time
import os

ABS_PATH = os.path.dirname(__file__)

class Husky(VecTask):
    def __init__(self, args, virtual_screen_capture=False, force_render=False):
        # load environmental configurations
        with open(args.cfg_env, 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # ==== should be defined for BaseTask ==== #
        self.cfg["env"]["numObservations"] = 6 + 32
        self.cfg["env"]["numActions"] = 2
        self.cfg["sim"]["use_gpu_pipeline"] = (args.pipeline.lower() == "gpu")
        self.cfg["sim"]["physx"]["num_threads"] = args.num_threads
        # ======================================== #

        # environmental parameters
        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        # set PD coefficient for Jackal velocity controller
        self.Kp = 100.0
        self.Kd = 100.0
        # Kinematics
        self.wheel_radius = 0.17775
        self.chasis_width = 0.2854*2.0
        self.action_scale = 1.0

        # call parent's __init__
        super().__init__(
            config=self.cfg, rl_device=args.sim_device, sim_device=args.sim_device, 
            graphics_device_id=args.graphics_device_id, headless=args.headless,
            virtual_screen_capture=virtual_screen_capture, force_render=force_render,
        )

        # reset camera pose
        if self.viewer != None:
            cam_pos = gymapi.Vec3(*[10.0, 10.0, 10.0])
            cam_target = gymapi.Vec3(*[-1.0, -1.0, -1.0])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # for observation
        self.num_actors = 1
        self.actor_robot_idx = 0
        root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(root_state).view(self.num_envs, self.num_actors, 13)
        # last dimension: 13 = pos (3) + quat (4) + lin_vel (3) + ang_vel (3)
        self.robot_state_tensor = self.root_state_tensor[:, self.actor_robot_idx, :]
        self.robot_pos =self.robot_state_tensor[:, :3] 
        self.robot_quat = self.robot_state_tensor[:, 3:7]
        self.robot_lin_vel = self.robot_state_tensor[:, 7:10]
        self.robot_ang_vel = self.robot_state_tensor[:, 10:13]

        # for initialization
        self.all_actor_indices = torch.arange(self.num_actors * self.num_envs, dtype=torch.long, 
                                                device=self.device).view(self.num_envs, self.num_actors)
        self.init_robot_pose = torch.tensor(
            [0.0, 0.0, 0.06344, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            device=self.device, dtype=torch.float)

        # for action
        self.action_transfrom = torch.tensor([
            [1.0/self.wheel_radius,                     1.0/self.wheel_radius], 
            [-0.5*self.chasis_width/self.wheel_radius,  0.5*self.chasis_width/self.wheel_radius]
        ], device=self.device, dtype=torch.float)

        # reset at first
        env_ids = torch.linspace(0, self.num_envs-1, self.num_envs, device=self.device, dtype=torch.long)
        self.reset(env_ids)


    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.linspace(0, self.num_envs-1, self.num_envs, device=self.device, dtype=torch.long)
        self.robot_state_tensor[env_ids, :] = self.init_robot_pose
        actor_indices = self.all_actor_indices[env_ids, :10].flatten()
        actor_indices_int32 = actor_indices.to(device=self.device, dtype=torch.int32)
        result = self.gym.set_actor_root_state_tensor_indexed(self.sim, 
            gymtorch.unwrap_tensor(self.root_state_tensor), 
            gymtorch.unwrap_tensor(actor_indices_int32), 
            len(actor_indices_int32))
        if not result:
            print("reset fail.")

    def create_sim(self):
        '''called from super().__init__()
        '''
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
    def pre_physics_step(self, actions):
        actions_tensor = actions.clone().to(self.device)
        actions_tensor[:, 0] = torch.clamp(actions_tensor[:, 0], 0.0, 1.0)
        actions_tensor[:, 1] = torch.clamp(actions_tensor[:, 1], -1.0, 1.0)
        actions_tensor = self.action_scale*torch.matmul(actions_tensor, self.action_transfrom)
        self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(actions_tensor))

    def post_physics_step(self):
        self.compute_observations()

    def _create_ground_plane(self):
        ''' called from create_sim()
        '''
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        ''' called from create_sim()
        '''
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # get asset info from configuration file
        asset_root = f"{ROOT_PATH}/assets/husky"
        asset_file = "husky.urdf"
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # husky asset
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.use_mesh_materials = True
        asset_options.flip_visual_attachments = True
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)

        # change friction property
        shape_names = self.gym.get_asset_rigid_body_names(robot_asset)
        shape_indices = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        front_ball_idx = 0
        rear_ball_idx = 0
        for i, idx in enumerate(shape_indices):
            if shape_names[i] == "front_ball":
                front_ball_idx = idx.start
            elif shape_names[i] == "rear_ball":
                rear_ball_idx = idx.start
        shape_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        shape_props[front_ball_idx].friction = 0.0
        shape_props[front_ball_idx].rolling_friction = 0.0
        shape_props[front_ball_idx].torsion_friction = 0.0
        shape_props[rear_ball_idx].friction = 0.0
        shape_props[rear_ball_idx].rolling_friction = 0.0
        shape_props[rear_ball_idx].torsion_friction = 0.0
        self.gym.set_asset_rigid_shape_properties(robot_asset, shape_props)

        # robot spawn pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 0.06344)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # set collision filter 
        # (if two objects have same values, not collide)
        collision_filter_robot = 1
        collision_filter_wall = 2

        self.robot_handles = []
        self.env_handles = []
        for i in range(num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            collision_group = i

            # get jackal actor handler
            robot_handle = self.gym.create_actor(env_handle, robot_asset, pose, "robot", collision_group, collision_filter_robot)

            # change jackal actor property
            dof_props = self.gym.get_actor_dof_properties(env_handle, robot_handle)
            dof_props['driveMode'][:] = gymapi.DOF_MODE_VEL
            dof_props['stiffness'][:] = self.Kp
            dof_props['damping'][:] = self.Kd
            dof_props['hasLimits'][:] = False
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)

            self.env_handles.append(env_handle)
            self.robot_handles.append(robot_handle)
