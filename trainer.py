from stable_baselines3 import A2C, TD3, PPO, DQN
from uav_game import UAV_Env
import numpy as np

env = UAV_Env(goal=0, record=False)
model = PPO('MlpPolicy', env, verbose=1)
# model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)
# model.save('A2C_model')
# model = A2C.load('A2C_model')
model.learn(total_timesteps=300000)
# model.save('PPO_model')
# model = PPO.load('PPO_model')  
obs = env.reset()
# num_of_steps = 1e4
# action_dim = 15
# obs_dim = 6
def save_demo(record, memory, goal):
    if not record: return
    file = open('goal2.csv', 'w')
    for tuple in memory:
        lst = list(tuple[0]) + [tuple[1], goal] 
        np.savetxt(file, [lst], delimiter=', ')
    file.close()

for i in range(10000):
    action, _state = model.predict(obs, deterministic=False)
    # if store:
    #     exp_action[i] = action
    #     exp_obs[i] = obs
    obs, reward, done, info = env.step(action)
    env.remember(obs, action)
    # tuple = env.memory[0]
    # lst = list(tuple[0]) + [tuple[1], env.goal] 
    env.render()
    if done:
      save_demo(env.record, env.memory, env.goal)
      obs = env.reset()


