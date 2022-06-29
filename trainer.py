from stable_baselines3 import A2C, TD3, PPO
from uav_game import UAV_Env

env = UAV_Env()
# model = A2C('MlpPolicy', env, verbose=1)
model = PPO('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=3000000)
# model.save('A2C_model')
# model = A2C.load('A2C_model')
model.learn(total_timesteps=30000)
# model = PPO.load('PPO_model')
obs = env.reset()
for i in range(10000):
    action, _state = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()