
from rl import *
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_size = 10
    obstacles = [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)]
    episodes = 1000
    agent = train_dqn(episodes, grid_size, obstacles, device)
    test_dqn(agent, grid_size, obstacles, device)