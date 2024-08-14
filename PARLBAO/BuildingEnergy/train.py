# 检查版本
import gym
import parl
import paddle
assert paddle.__version__ == "2.2.0", "[Version WARNING] please try `pip install paddlepaddle==2.2.0`"
assert parl.__version__ == "2.0.3", "[Version WARNING] please try `pip install parl==2.0.3`"
# assert gym.__version__ == "0.18.0", "[Version WARNING] please try `pip install gym==0.18.0`"

import numpy as np
from parl.utils import logger


from model import Model
from algorithm import DDPG  # from parl.algorithms import DDPG
from agent import Agent
from env import BuildingEnergyEnv
from replay_memory import ReplayMemory

ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate
GAMMA = 0.99  # reward 的衰减因子
TAU = 0.001  # 软更新的系数
MEMORY_SIZE = int(1e6)  # 经验池大小
# MEMORY_SIZE = int(1000)  # 经验池大小
MEMORY_WARMUP_SIZE = MEMORY_SIZE // 20  # 预存一部分经验之后再开始训练
BATCH_SIZE = 512
REWARD_SCALE = 0.1  # reward 缩放系数
NOISE = 0.1  # 动作噪声方差
TRAIN_EPISODE = int(10e4)  # 训练的总episode数


# 训练一个episode
def run_train_episode(agent:Agent, env:BuildingEnergyEnv, rpm:ReplayMemory):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.sample(batch_obs.astype('float32'))
        # action = action[0]  # ContinuousCartPoleE输入的action为一个实数
        next_obs, reward, done, info = env.step(action)
        # action = [action]  # 方便存入replaymemory
        rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))
        if len(rpm) > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)
        obs = next_obs
        total_reward += reward
        if done or steps >= 200:
            break
    return total_reward

# 评估 agent, 跑 5 个episode，总reward求平均
def run_evaluate_episodes(agent:Agent, env:BuildingEnergyEnv, render=False):
    eval_reward = []
    for i in range(2):
        obs=env.reset()
        total_reward = 0
        steps = 0
        steps_limit=50
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            steps += 1
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            total_reward += reward
            if render:
                env.render()
            if done:
                print("testing target hit")
            if steps>=steps_limit:
                print("testing",steps_limit, "steps hit")
            if done or steps >= steps_limit:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)
            
def main():
    env = BuildingEnergyEnv()                                      
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.reset()
    
    model = Model(act_dim=act_dim, obs_dim=obs_dim)
    algorithm = DDPG(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, act_dim, expl_noise=NOISE)
    
    rpm = ReplayMemory(MEMORY_SIZE)
    # 根据parl框架构建agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    
    # run_evaluate_episodes(agent, env, render=True)
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)
    
    episode = 0
    
    while episode < TRAIN_EPISODE:
        print("training")
        for i in range(100):
            total_reward = run_train_episode(agent, env, rpm)
            episode += 1
            
        print("testing")
        eval_reward = run_evaluate_episodes(agent, env, render=True)
        logger.info('episode:{}    Test reward:{}'.format(
            episode, eval_reward))
    # 训练结束，保存模型
    save_path = './dqn_model.ckpt'
    agent.save(save_path)

if __name__ == '__main__':
    main()