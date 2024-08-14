import gym
from gym import spaces
import numpy as np
import pygame
from pygame.locals import QUIT

class BuildingEnergyImgEnv(gym.Env):
    def __init__(self):
        super(BuildingEnergyImgEnv, self).__init__()
        # 定义动作空间，假设移动范围是 -1 到 1
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        # 定义观测空间，包含agent和目标点的横纵向坐标
        self.observation_space = spaces.Box(low=0, high=255,shape=(200,200,3), dtype=np.uint8)
        
        # 初始化状态
        self.reset()

        # 初始化Pygame
        pygame.init()
        self.screen_size = 200
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption('Custom Environment')
        self.clock = pygame.time.Clock()
        self.target_radius=0.3
    
    def reset(self):
        # 初始化agent的位置和目标位置
        self.agent_pos = np.random.uniform(low=-9, high=9, size=(2,))
        self.goal_pos = np.random.uniform(low=-9, high=9, size=(2,))
        
        # 返回初始观测
        return np.concatenate([self.agent_pos, self.goal_pos])
    
    def step(self, action):
        # 更新agent位置
        self.agent_pos += action
        
        # 检查边界
        self.agent_pos = np.clip(self.agent_pos, -10, 10)
        
        # 计算reward，假设目标是最小化agent与目标的距离
        distance_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -distance_to_goal
        
        done=False
        if distance_to_goal<self.target_radius:
            reward+=100
            done=True
        if np.any(self.agent_pos <= -10) or np.any(self.agent_pos >= 10):
            # reward +=-200
            done=True
        
        # 检查是否到达目标（设定一个阈值）
        # done = distance_to_goal < self.target_radius or np.any(self.agent_pos <= -10) or np.any(self.agent_pos >= 10)
        
        # 返回新状态、奖励、是否结束以及额外信息
        state = np.concatenate([self.agent_pos, self.goal_pos])
        info = {}
        
        return state, reward, done, info
    
    def render(self, mode='human'):
        # 绘制背景
        self.screen.fill((255, 255, 255))
        
        # 坐标转换，适应屏幕尺寸
        def convert_to_screen_coordinates(pos):
            return (int((pos[0] + 10) / 20 * self.screen_size), 
                    int((pos[1] + 10) / 20 * self.screen_size))
        
        agent_screen_pos = convert_to_screen_coordinates(self.agent_pos)
        goal_screen_pos = convert_to_screen_coordinates(self.goal_pos)
        
        # 绘制agent和目标点
        pygame.draw.circle(self.screen, (0, 0, 255), agent_screen_pos, 10)
        pygame.draw.circle(self.screen, (255, 0, 0), goal_screen_pos, 10)
        pygame.time.wait(50)
        # 刷新屏幕
        pygame.display.flip()
        
        # 检查退出事件
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
    
    def close(self):
        pygame.quit()
