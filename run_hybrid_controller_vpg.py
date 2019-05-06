import matplotlib.pyplot as plt
from cartpole import CartPoleEnv
import os
import spinup.algos.vpg.core as core
from spinup import vpg
from spinup.utils.test_policy import load_policy, run_policy
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.plot import make_plots
from spinup.utils.mpi_tools import mpi_fork


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
    plt.savefig('figures/state_rl_vpg.png')


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
    plt.savefig('figures/state_lqr_vpg.png')


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
    parser.add_argument('--exp_name', type=str, default='vpg')
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

    mpi_fork(args.cpu)  # run parallel code with mpi

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    init_state = [0, 0, 0, 0.1]

    if args.plot:
        directory = [os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  args.exp_name)]
        make_plots(directory, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est)
    else:
        if not args.test:
            vpg(env_fn=CartPoleEnv, actor_critic=core.mlp_actor_critic,
                ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                gamma=args.gamma,
                seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
                # pi_lr=3e-6, vf_lr=1e-4,
                logger_kwargs=logger_kwargs)
        else:
            env = CartPoleEnv()
            print('testing')
            _, get_action = load_policy(args.exp_name + '/' + args.exp_name +
                                        '_s' + str(args.seed))
            test_with_rl(env, init_state, get_action)
            test_with_lqr(env, init_state)
