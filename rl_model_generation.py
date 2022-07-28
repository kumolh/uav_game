from gym import Env
from stable_baselines3 import A2C, TD3, PPO, DQN
from settings import TILESIZE
from uav_game import UAV_Env
from RNN import RNN
import numpy as np

def train(env: Env):
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=300000)
    model.save('PPO_model_normalized')

def test(env: Env):
    model = PPO('MlpPolicy', env, verbose=1)
    obs = env.reset()
    for _ in range(10000):
        action, _state = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        env.render()

if __name__=='__main__':
    env = UAV_Env(goal=0, record=False)
    # train(env)
    test(env)
    
