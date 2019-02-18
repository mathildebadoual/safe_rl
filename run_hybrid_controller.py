import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO1
from cartpole import CartPoleEnv

# multiprocess environment
n_cpu = 4
# env = SubprocVecEnv([lambda: CartPoleEnv() for i in range(n_cpu)])
env = DummyVecEnv([lambda: CartPoleEnv()])

model = PPO1(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=100000)
model.save("a2c_cartpole_hybrid")

del model

env = CartPoleEnv()

model = PPO1.load("a2c_cartpole_hybrid")

obs = env.reset()
max_steps = 1000
x_list = []
xdot_list = []
theta_list = []
thetadot_list = []
action_list = [0]
env.render()

for i in range(max_steps):
    state = obs
    x_list.append(state[0])
    xdot_list.append(state[1])
    theta_list.append(state[2])
    thetadot_list.append(state[3])
    action, _states = model.predict(obs)
    action_list.append(action)
    try:
        obs, r, done, info = env.step(action)
    except AssertionError:
        done = True
    env.render()
    if done:
        break

env.close()

plt.figure(figsize=(10, 7))

plt.plot(x_list, label='x')
plt.plot(xdot_list, label='x dot')
plt.plot(theta_list, label='theta')
plt.plot(thetadot_list, label='theta dot')
plt.plot(action_list, label='action')
plt.grid()
plt.legend()
plt.savefig('figures/state.png')