'''
Custom Gym environment
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import v0_warehouse_robot as wr
import numpy as np

register(
    id='WarehouseRobot-v0',
    entry_point='v0_warehouse_robot_env:WarehouseRobotEnv',
)

class WarehouseRobotEnv(gym.Env):
    metadata={"render_modes":["human"],'render_fps':1}
    
    def __init__(self,grid_rows=4,grid_cols=5,render_mode=None):
        self.grid_rows=grid_rows
        self.grid_cols=grid_cols
        self.render_mode=render_mode
        