from stable_baselines3 import A2C, TD3, PPO, DQN
from uav_game import UAV_Env
import numpy as np

def save_demo(name, record, memory, goal):
    if not record: return
    file = open(name + '.csv', 'w')
    for tuple in memory:
        lst = list(tuple[0]) + [tuple[1], goal] 
        np.savetxt(file, [lst], delimiter=', ')
    file.close()

env = UAV_Env(goal=-1, record=False)
# # model = PPO('MlpPolicy', env, verbose=1)
# # model.learn(total_timesteps=300000)
# # model.save('PPO_model_normalized')
model = PPO.load('PPO_model_normalized')  
# obs = env.reset()

for i in range(10000):
    # action, _state = model.predict(obs, deterministic=False)
    action = 0
    obs, reward, done, info = env.step(action)
    env.remember(obs, action)
    env.render()
    if done:
      save_demo('goal{}'.format(env.goal), env.record, env.memory, env.goal)
      obs = env.reset()

# for goal in range(4):
#       for num in range(10):
#             env_mask = UAV_Env(goal=goal, record=True)
#             obs = env_mask.reset()
#             for steps in range(200):
#                 action, _state = model.predict(obs, deterministic=False)
#                 obs, reward, done, info = env_mask.step(action)
#                 action = info['action']
#                 env_mask.remember(obs, action)
#                 if done: break
#                 # env_mask.render()
#             save_demo('raw_data/goal{}-{}'.format(goal, num), env_mask.record, env_mask.memory, env_mask.goal)


