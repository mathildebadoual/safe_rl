import gym
import numpy as np
import matplotlib.pyplot as plt
import control

from cartpole import CartPoleEnv

g = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5  # actually half the pole's length
J = 1 / 3 * masspole * length ** 2

A = np.array(
    [[0, 1, 0, 0], [0, 0, -masspole * g / masscart, 0], [0, 0, 0, 1], [0, 0, total_mass * g / (length * masscart), 0]])
B = np.array([[0], [1 / masscart], [0], [-1 / (length * masscart)]])
Q = np.eye(4)
R = 1

K, _, _ = control.lqr(A, B, Q, R)

env = CartPoleEnv()

max_steps = 1000

obs = env.reset(init_state=[-3, 0.1, 5, -1])
print('initial observation:', obs)
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
    action = - np.dot(K, state)
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