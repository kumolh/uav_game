from stable_baselines3 import A2C, TD3, PPO, DQN
from settings import TILESIZE
from uav_game import UAV_Env
from RNN import RNN
import numpy as np

def save_demo(name, record, memory, goal):
    if not record: return
    file = open(name + '.csv', 'w')
    for tuple in memory:
        lst = list(tuple) + [goal] 
        np.savetxt(file, [lst], delimiter=', ')
    file.close()

def main():
    model = PPO.load('PPO_model_normalized')  
    for goal in range(4, 5):
        for num in range(1):
                env_mask = UAV_Env(goal=goal, record=True)
                if goal == 4: # random action, no goal actually
                    model = PPO.load('PPO_model')
                obs = env_mask.reset()
                for steps in range(200):
                    action, _state = model.predict(obs, deterministic=False)
                    obs, reward, done, info = env_mask.step(action)
                    action = info['action']
                    env_mask.remember(obs, action)
                    if done: break
                    env_mask.render()
                save_demo('raw_data/goal{}-{}'.format(goal, num), env_mask.record, env_mask.memory, env_mask.goal)

if __name__=='__main__':
    main()



