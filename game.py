from uav_game import *
from RNN import RNN
from transformer import *
from torch.nn import Transformer

def main():
    env = UAV_Env(goal=-1, record=False)
    obs = env.reset()
    while True:
        action = env.player.input()
        obs, reward, done, info = env.step(action)
        if action != 0:
            env.remember(obs, action)
        if done:
            obs = env.reset()
        env.render()

if __name__ == '__main__':
    main()