import os
from sys import platform as sys_pf

from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.test_policy import load_policy

from agents.controllers import PidController

if sys_pf == 'darwin':
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
from envs.cartpole import CartPoleEnvContinous
import spinup.algos.trpo.core as core
from spinup import trpo, vpg
from spinup.utils.plot import make_plots
from spinup.utils.mpi_tools import mpi_fork
import numpy as np
import control
import math
from scipy.integrate import odeint


def test_with_controller(env, init_state, feedback_controller):
    obs = env.reset(init_state=init_state)
    max_steps = 1000
    x_list = []
    xdot_list = []
    theta_list = []
    thetadot_list = []
    action_list = [0]
    reward_list = [0]
    # env.render()

    for i in range(max_steps):
        state = obs
        x_list.append(state[0])
        xdot_list.append(state[1])
        theta_list.append(state[2])
        thetadot_list.append(state[3])
        t = env.tau * i
        action = feedback_controller(t, state)
        action_list.append(action)
        try:
            obs, r, done = env.step(action)
            reward_list.append(r)
        except AssertionError:
            done = True
        # env.render()
        if done:
            break

    env.close()

    return {
        'x': x_list,
        'xdot': xdot_list,
        'theta': theta_list,
        'thetadot': thetadot_list,
        'action': action_list,
        'reward': reward_list,
    }


def plot_figures(saved_dict, exp_name, controller_description):
    fig, ax = plt.subplots(2, 1)
    plt.figure(figsize=(10, 7))
    ax[0].plot(saved_dict['x'], label='x')
    ax[0].plot(saved_dict['xdot'], label='x dot')
    ax[0].plot(saved_dict['theta'], label='theta')
    ax[0].plot(saved_dict['thetadot'], label='theta dot')
    ax[0].plot(saved_dict['action'], label='action %s' % controller_description)
    ax[1].plot(saved_dict['reward'], label='reward')
    ax[0].grid()
    ax[0].legend()
    ax[1].grid()
    ax[1].legend()
    fig.savefig('figures/state_%s_%s.png' % (controller_description, exp_name))


def test_with_pid(env, init_state, pid):
    return test_with_controller(env, init_state, pid.get_action)

def test_with_rl(env, init_state, get_action):
    return test_with_controller(env, init_state, lambda t, state: get_action(state))


def test_with_lqr(env, init_state):
    return test_with_controller(env, init_state, lambda t, state: 0)


def test_system_difference(env, get_action, pid, init_state):
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

    def nonlinear_model_pid(x, t):
        pid_val = pid.get_action(t, x)
        dydt = np.dot(A, x) + np.dot(B, (- np.dot(K, x) + pid_val))
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
    x_nonlinear_pid = odeint(nonlinear_model_pid, x0, t)
    x_nonlinear = odeint(nonlinear_model, x0, t)

    plt.figure(figsize=(10, 7))

    fig_phase, ax_phase = plt.subplots()
    fig_time, ax_time = plt.subplots()

    for label, (x, color) in {
        'linear': (x_linear, 'blue'),
        'non linear true': (x_nonlinear, 'black'),
        'non linear rl': (x_nonlinear_rl, 'orange'),
        'non linear + PID': (x_nonlinear_pid, 'purple')
    }.items():
        ax_phase.plot([x[i][0] for i in range(len(x))],
                 [x[i][1] for i in range(len(x))],
                 color=color, label=label)
        ax_phase.scatter(x[0][0], x[0][1], color=color, marker='o')
        ax_phase.grid()
        ax_phase.legend()

        ax_time.plot([x[i][0] for i in range(len(x))], color=color, label=label)
        ax_time.grid()
        ax_time.legend()

    fig_phase.set_size_inches(10, 7)
    fig_time.set_size_inches(10, 7)
    fig_phase.savefig('figures/phase_models.png')
    fig_time.savefig('figures/models.png')


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
    parser.add_argument('--exp_name', type=str)
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
    parser.add_argument('--data-dir')
    parser.add_argument('--vpg', action='store_true', default=False)

    args = parser.parse_args()

    if args.exp_name is not None:
        exp_name = args.exp_name
    else:
        exp_name = 'vpg' if args.vpg else 'trpo'

    logger_kwargs = setup_logger_kwargs(exp_name, args.seed, data_dir=args.data_dir)

    # init_state = [0, 2.87, 0, 2.87]
    init_state = [0.1, 0, 0, 0]

    if args.plot:
        directory = [os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  args.exp_name)]
        make_plots(directory, args.legend, args.xaxis, args.value, args.count,
                   smooth=args.smooth, select=args.select, exclude=args.exclude,
                   estimator=args.est)
    else:
        env_fn = CartPoleEnvContinous
        if not args.test:
            mpi_fork(args.cpu)  # run parallel code with mpi
            if args.vpg:
                print("Using vpg for training.")
                vpg(env_fn=env_fn, actor_critic=core.mlp_actor_critic,
                    ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
                    seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
                    # pi_lr=3e-6, vf_lr=1e-4,
                    logger_kwargs=logger_kwargs)
            else:
                print("Using trpo for training.")
                trpo(env_fn=env_fn, actor_critic=core.mlp_actor_critic,
                     ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                     gamma=args.gamma,
                     seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
                     logger_kwargs=logger_kwargs)
        else:
            env = env_fn()
            pid = PidController()
            _, rl_action = load_policy(logger_kwargs['output_dir'])
            d = test_with_rl(env, init_state, rl_action)
            plot_figures(d, exp_name, 'rl')

            d = test_with_lqr(env, init_state)
            plot_figures(d, exp_name, 'lqr')

            d = test_with_pid(env, init_state, pid)
            plot_figures(d, exp_name, 'pid')

            test_system_difference(env, rl_action, pid, init_state)
