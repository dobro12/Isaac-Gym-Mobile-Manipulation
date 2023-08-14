# Isaac Gym
from isaacgym import gymapi, gymutil

# Isaac Gym Envs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from envs.franka_husky import Husky
# from envs.husky import Husky

# Others
from collections import deque
import numpy as np
import torch
import time
import yaml
import sys
import os

''' existing parameters
    'sim_device': 'cuda:0',
    'pipeline': 'gpu',
    'graphics_device_id': 0,
    'flex': False,
    'physx': False,
    'num_threads': 0,
    'subscenes': 0,
    'slices': 0,
'''
custom_parameters = [
    {"name": "--headless", "action": "store_true", "help": "Force display off at all times."},
    {"name": "--test", "action": "store_true", "help": "For test."},
    {"name": "--cfg_env", "type": str, "default": "env.yaml", "help": "Configuration file for environment."},
    {"name": "--cfg_train", "type": str, "default": "train.yaml", "help": "Configuration file for training."},
    {"name": "--save_freq", "type": int, "default": int(2e5), "help": "Agent save frequency."},
    {"name": "--total_steps", "type": int, "default": int(1e7), "help": "Total number of environmental steps."},
    {"name": "--update_steps", "type": int, "default": int(1e4), "help": "Number of environmental steps for updates."},
    {"name": "--seed", "type": int, "default": 1, "help": "Seed."},
]

def train(args):
    # default
    seed = args.seed
    set_seed(seed)

    # define environment
    env = Husky(args, force_render=(not args.headless))

    env.reset()
    step = 0
    while step < 1000:
        step += 1
        print(step)
        actions = torch.zeros(100, 11)
        if step > 30:
            actions[:, 3] = 1.0
        _, rewards, dones, infos = env.step(actions)
        if not args.headless: env.render()
 


def test(args):
    pass



if __name__ == "__main__":
    set_np_formatting()
    args = gymutil.parse_arguments(
        description="IsaacGym",
        custom_parameters=custom_parameters,
    )
    if args.test:
        test(args)
    else:
        train(args)
