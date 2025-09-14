import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import torch
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.control.action_scale = 0.3
    env_cfg.curriculum.pull_force = False
    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, env_cfg=env_cfg, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    logger = Logger(env.dt)
    for i in range(10*int(env.max_episode_length)):

        result = env.gym.fetch_results(env.sim, True)
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())


if __name__ == '__main__':
    args = get_args()
    play(args)
# import sys
# from legged_gym import LEGGED_GYM_ROOT_DIR
# import os

# import isaacgym
# from isaacgym import gymapi
# from legged_gym.envs import *
# from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
# import imageio

# import torch
# import time

# import matplotlib.pyplot as plt
# import numpy as np
# from collections import defaultdict
# from multiprocessing import Process, Value


# def play(args):
#     env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
#     env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
#     env_cfg.terrain.num_rows = 4
#     env_cfg.terrain.num_cols = 4
#     env_cfg.terrain.curriculum = False
#     env_cfg.noise.add_noise = False
#     env_cfg.control.action_scale = 0.3
#     env_cfg.curriculum.pull_force = False
#     env_cfg.env.test = True

#     # prepare environment
#     env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#     obs = env.get_observations()

#     W, H = 1280, 720
#     cam_props = gymapi.CameraProperties()
#     cam_props.width  = W
#     cam_props.height = H
#     cam_props.enable_tensors = False  # we’ll read CPU images

#     cam_props.use_collision_geometry = False
#     cam_props.near_plane = 0.01
#     cam_props.far_plane  = 200.0

#     first_env = env.envs[0]
#     cam_handle = env.gym.create_camera_sensor(first_env, cam_props)

#     # Find the robot (it's named "t1" per your printout)
#     robot_handle = env.gym.find_actor_handle(first_env, "t1")
#     assert robot_handle != -1, "Actor 't1' not found in first_env"

#     body_names = env.gym.get_actor_rigid_body_names(first_env, robot_handle)
#     candidates = ["base", "trunk", "Trunk", "torso", "pelvis", "base_link", "root"]
#     base_name = next((n for n in candidates if n in body_names), body_names[0])

#     # Aim the camera at the robot (don’t rely on raw transform quats)
#     def look_at(gym, envh, camh, eye, target):
#         gym.set_camera_location(camh, envh,
#             gymapi.Vec3(*eye), gymapi.Vec3(*target))

#     root_rb = env.gym.get_actor_rigid_body_handle(first_env, robot_handle, root_name)
#     root_tf = env.gym.get_rigid_transform(first_env, root_rb)  # <- this exists
#     eye    = (root_tf.p.x + 6.0, root_tf.p.y + 6.0, root_tf.p.z + 3.0)
#     target = (root_tf.p.x,        root_tf.p.y,        root_tf.p.z + 0.5)
#     look_at(env.gym, first_env, cam_handle, eye, target)

#     # (Optional but helpful) Add light so a dark mesh isn’t invisible
#     env.gym.set_light_parameters(env.sim, 0,
#         ambient=gymapi.Vec3(0.6,0.6,0.6),
#         diffuse=gymapi.Vec3(0.8,0.8,0.8),
#         specular=gymapi.Vec3(1,1,1),
#         direction=gymapi.Vec3(5,5,10))

#     for i in range(10*int(env.max_episode_length)):

#         result = env.gym.fetch_results(env.sim, True)
#         actions = policy(obs.detach())
#         obs, _, rews, dones, infos = env.step(actions.detach())

#         env.gym.render_all_camera_sensors(env.sim)
#         rgba = env.gym.get_camera_image(env.sim, first_env, cam_handle, gymapi.IMAGE_COLOR)
#         # IMAGE_COLOR returns HxWx4 uint8 (RGBA). Convert to RGB for imageio:
#         frame = rgba.reshape(H, W, 4)[..., :3]
#         writer.append_data(frame)
#         # print("Recording frame: ", i)


# if __name__ == '__main__':
#     args = get_args()
#     play(args)
