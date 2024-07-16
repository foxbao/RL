import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, size=10, obstacles=None):
        self.size = size
        self.obstacles = obstacles if obstacles else []
        self.reset()
        
    def reset(self, start=None, end=None):
        self.grid = np.zeros((self.size, self.size))
        for (x, y) in self.obstacles:
            self.grid[x, y] = -1  # obstacle
        self.start = start if start else (0, 0)
        self.end = end if end else (self.size - 1, self.size - 1)
        self.agent_position = self.start
        return self._get_state()
    
    def _get_state(self):
        state = np.zeros((self.size, self.size))
        state[self.agent_position] = 1
        state[self.end] = 2
        for (x, y) in self.obstacles:
            state[x, y] = -1
        return state.flatten()
    
    def step(self, action):
        if action == 0:  # up
            next_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif action == 1:  # down
            next_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif action == 2:  # left
            next_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif action == 3:  # right
            next_position = (self.agent_position[0], self.agent_position[1] + 1)
        
        if (next_position[0] < 0 or next_position[0] >= self.size or
            next_position[1] < 0 or next_position[1] >= self.size or
            next_position in self.obstacles):
            next_position = self.agent_position  # hit the wall or obstacle
        
        self.agent_position = next_position
        reward = -1
        done = False
        if self.agent_position == self.end:
            reward = 100
            done = True
        return self._get_state(), reward, done

    def render(self):
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) == self.agent_position:
                    print("A", end=" ")
                elif (r, c) == self.end:
                    print("E", end=" ")
                elif (r, c) in self.obstacles:
                    print("X", end=" ")
                else:
                    print(".", end=" ")
            print()
        print()


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.device = device
        self.model = DQN(state_size, action_size).to(device)
        self.target_model = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state.unsqueeze(0))[0]).item()
            target_f = self.model(state.unsqueeze(0))
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = F.mse_loss(target_f, self.model(state.unsqueeze(0)))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_state_dict(torch.load(name))
    
    def save(self, name):
        torch.save(self.model.state_dict(), name)

def train_dqn(episodes, grid_size, obstacles, device):
    env = GridWorld(size=grid_size, obstacles=obstacles)
    state_size = grid_size * grid_size
    action_size = 4
    agent = DQNAgent(state_size, action_size, device)
    batch_size = 32
    
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
    
    return agent

