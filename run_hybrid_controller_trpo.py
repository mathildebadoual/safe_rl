from sys import platform as sys_pf

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
from cartpole import CartPoleEnv
import os
import spinup.algos.trpo.core as core
from spinup import trpo
from spinup.utils.test_policy import load_policy, run_policy
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.plot import make_plots
from spinup.utils.mpi_tools import mpi_fork
import numpy as np
import control
import math
from scipy.integrate import odeint


def test_with_rl(env, init_state, get_action):
    obs = env.reset(init_state=init_state)
    max_steps = 1000
    x_list = []
    xdot_list = []
    theta_list = []
    thetadot_list = []
    action_list = [0]
    action_lqr = [0]
    reward_list = [0]
    # env.render()

    for i in range(max_steps):
        state = obs
        x_list.append(state[0])
        xdot_list.append(state[1])
        theta_list.append(state[2])
        thetadot_list.append(state[3])
        # action, _states = model.predict(obs)
        action = get_action(obs)
        action_list.append(action)
        try:
            obs, r, done, info = env.step(action)
        except AssertionError:
            done = True
        action_lqr.append(info['action_lqr'])
        reward_list.append(r)
        # env.render()
        if done:
            break

    env.close()

    plt.figure(figsize=(10, 7))
    plt.plot(x_list, label='x')
    plt.plot(xdot_list, label='x dot')
    plt.plot(theta_list, label='theta')
    plt.plot(thetadot_list, label='theta dot')
    plt.plot(action_list, label='action rl')
    plt.plot(action_lqr, label='action lqr')
    plt.plot(reward_list, label='reward')
    plt.grid()
    plt.legend()
    plt.savefig('figures/state_rl_trpo.png')


def test_with_lqr(env, init_state):
    obs = env.reset(init_state=init_state)
    max_steps = 1000
    x_list = []
    xdot_list = []
    theta_list = []
    thetadot_list = []
    action_list = [0]
    action_lqr = [0]
    reward_list = [0]
    # env.render()

    for i in range(max_steps):
        state = obs
        x_list.append(state[0])
        xdot_list.append(state[1])
        theta_list.append(state[2])
        thetadot_list.append(state[3])
        action = 0
        action_list.append(action)
        try:
            obs, r, done, info = env.step(action)
        except AssertionError:
            done = True
        action_lqr.append(info['action_lqr'])
        reward_list.append(r)
        # env.render()
        if done:
            break

    env.close()

    plt.figure(figsize=(10, 7))
    plt.plot(x_list, label='x')
    plt.plot(xdot_list, label='x dot')
    plt.plot(theta_list, label='theta')
    plt.plot(thetadot_list, label='theta dot')
    plt.plot(action_list, label='action rl')
    plt.plot(action_lqr, label='action lqr')
    plt.plot(reward_list, label='reward')
    plt.grid()
    plt.legend()
    plt.savefig('figures/state_lqr_trpo.png')


def test_system_difference(env, get_action, init_state):
    max_steps = 1000
    gravity = 9.8
    masscart = 1.0
    masspole = 0.01
    total_mass = (masspole + masscart)
    length = 0.5  # actually half the pole's length
    polemass_length = (masspole * length)
    tau = 0.02
    A = np.array(
        [[0, 1, 0, 0], [0, 0, -masspole * gravity / masscart, 0], [0, 0, 0, 1],
         [0, 0, total_mass * gravity / (length * masscart), 0]])
    B = np.array([[0], [1 / masscart], [0], [-1 / (length * masscart)]])
    Q = np.eye(4)
    R = 0.1
    K, _, _ = control.lqr(A, B, Q, R)
    x0 = np.array(init_state)

    def linear_model(x, t):
        dydt = np.dot(A, x) + np.dot(B, - np.dot(K, x))
        return dydt

    def nonlinear_model_rl(x, t):
        dydt = np.dot(A, x) + np.dot(B, - np.dot(K, x) + get_action(x))
        return dydt

    def nonlinear_model(x, t):
        dydt = np.array([0, 0, 0, 0])
        costheta = math.cos(x[2])
        sintheta = math.sin(x[2])
        temp = (get_action(x) + polemass_length * x[3] * x[
            3] * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (
                    4.0 / 3.0 - masspole * costheta * costheta / total_mass))
        xacc = temp - polemass_length * thetaacc * costheta / total_mass
        dydt[0] = x[0] + tau * x[1]
        dydt[1] = x[1] + tau * xacc
        dydt[2] = x[2] + tau * x[3]
        dydt[3] = x[3] + tau * thetaacc
        return dydt

    # time points
    t = np.linspace(0, 1000)

    # solve ODE
    x_linear = odeint(linear_model, x0, t)
    x_nonlinear_rl = odeint(nonlinear_model_rl, x0, t)
    x_nonlinear = odeint(nonlinear_model, x0, t)

    plt.figure(figsize=(10, 7))
    plt.plot([x_linear[i][0] for i in range(len(x_linear))],
             [x_linear[i][1] for i in range(len(x_linear))],
             color='blue', label='linear')
    plt.plot(x_linear[0][0], x_linear[0][1], color='blue', marker='o')
    plt.plot([x_nonlinear_rl[i][0] for i in range(len(x_nonlinear_rl))],
             [x_nonlinear_rl[i][1] for i in range(len(x_nonlinear_rl))],
             color='green', label='non linear')
    plt.plot(x_nonlinear_rl[0][0], x_nonlinear_rl[0][1], color='green',
             marker='o')
    plt.grid()
    plt.legend()
    plt.savefig('figures/phase_models.png')

    plt.figure(figsize=(10, 7))
    plt.plot([x_linear[i][0] for i in range(len(x_linear))], color='blue',
             label='linear')
    plt.plot([x_nonlinear_rl[i][0] for i in range(len(x_nonlinear_rl))],
             color='orange', label='non linear rl')
    plt.plot([x_nonlinear[i][0] for i in range(len(x_nonlinear))],
             label='non linear true')
    plt.grid()
    plt.legend()
    plt.savefig('figures/models.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='trpo')
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # init_state = [0, 2.87, 0, 2.87]
    init_state = [0.1, 0, 0, 0]

    if args.plot:
        directory = [os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  args.exp_name)]
        make_plots(directory, args.legend, args.xaxis, args.value, args.count,
                   smooth=args.smooth, select=args.select, exclude=args.exclude,
                   estimator=args.est)
    else:
        if not args.test:
            mpi_fork(args.cpu)  # run parallel code with mpi
            trpo(env_fn=CartPoleEnv, actor_critic=core.mlp_actor_critic,
                 ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                 gamma=args.gamma,
                 seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
                 logger_kwargs=logger_kwargs)
        else:
            env = CartPoleEnv()
            _, get_action = load_policy(
                args.exp_name + '/' + args.exp_name + '_s' + str(args.seed))
            test_with_rl(env, init_state, get_action)
            test_with_lqr(env, init_state)
            test_system_difference(env, get_action, init_state)
