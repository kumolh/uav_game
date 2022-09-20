from gym import Env
from stable_baselines3 import A2C, TD3, PPO, DQN, HerReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from settings import TILESIZE
from uav_game import UAV_Env
from RNN import RNN
import numpy as np

def train(env: Env):
    # model = PPO('MlpPolicy', env, verbose=1)
    
    # tmp_path = "./tmp/sb3_log/"
    # # set up logger
    # new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    model = A2C('MlpPolicy', env, tensorboard_log="./tmp/sb3_log/sparse_reward", verbose=1)
    # model.set_logger(new_logger)
    model.learn(total_timesteps=150000)
    model.save('A2C_model_sparse_reward')

def test(env: Env):
    # model = A2C('MlpPolicy', env, verbose=1)
    # model = PPO.load('PPO_model_normalized')
    model = A2C.load('A2C_model_sparse_reward')  
    # evaluate_policy(model, env, 5, deterministic=True)
    obs = env.reset()
    for _ in range(10000):
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        env.render()

if __name__=='__main__':
    env = UAV_Env(goal=0, record=False)
    train(env)
    test(env)
    # model = PPO('MlpPolicy', env, verbose=1)
    # model = PPO.load('PPO_model_normalized') 
    # lst = evaluate_policy(model, env, 10, deterministic=True, warn=False)
    # print(lst)
