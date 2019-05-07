import argparse

import matplotlib.pyplot as plt

from agents.controllers import LqrController, PidController, HybridController
from envs.cartpole import CartPoleEnvContinous


def run_controller(env, controller, max_steps_simulation=1000, render=False):
    obs = env.reset()
    x_list = []
    xdot_list = []
    theta_list = []
    thetadot_list = []
    action_list = []
    if render:
        env.render()

    for i in range(max_steps_simulation):

        state = obs
        x_list.append(state[0])
        xdot_list.append(state[1])
        theta_list.append(state[2])
        thetadot_list.append(state[3])
        t = i * env.tau
        print(state)
        action = controller.get_action(state, t=t)
        action_list.append(action)
        try:
            obs, r, done = env.step(action)
        except AssertionError:
            done = True
        if done:
            break
        if render:
            env.render()

    env.close()

    saved_dict = {
        'x': x_list,
        'xdot': xdot_list,
        'theta': theta_list,
        'thetadot': thetadot_list,
        'action': action_list,
    }

    return saved_dict


def plot_results(saved_dict, figure_name):
    plt.figure(figsize=(10, 7))
    plt.plot(saved_dict['x'], label='x')
    plt.plot(saved_dict['xdot'], label='x dot')
    plt.plot(saved_dict['theta'], label='theta')
    plt.plot(saved_dict['thetadot'], label='theta dot')
    plt.plot(saved_dict['action'], label='action')
    plt.grid()
    plt.legend()
    plt.savefig('figures/%s.png' % figure_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', '-c', type=str,
                        default='lqr_controller')
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--max_steps', type=int, default=1000)
    args = parser.parse_args()

    env = CartPoleEnvContinous()

    if args.controller == 'lqr_controller':
        controller = LqrController(env)
    elif args.controller == 'pid_controller':
        controller = PidController()
    elif args.controller == 'hybrid_controller':
        controller = HybridController(env)
    else:
        raise ValueError("Invalid controller name given. Was %s"
                         % args.controller)

    saved_dict = run_controller(
        env=env,
        render=args.render,
        max_steps_simulation=args.max_steps,
        controller=controller,
    )
    plot_results(saved_dict, args.controller)
