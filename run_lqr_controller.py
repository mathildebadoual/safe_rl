import argparse

import imageio
import matplotlib.pyplot as plt

from agents.controllers import LqrController, PidController
from envs.cartpole import CartPoleEnvContinous


def run_controller(env, controller, max_steps=1000, render=False):
    obs = env.reset()
    print('initial observation:', obs)
    x_list = []
    xdot_list = []
    theta_list = []
    thetadot_list = []
    action_list = []
    if render:
        env.render()

    for i in range(max_steps):

        state = obs
        x_list.append(state[0])
        xdot_list.append(state[1])
        theta_list.append(state[2])
        thetadot_list.append(state[3])
        t = i * env.tau
        action = controller.get_action(t, state)
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
    # plt.plot(saved_dict['action'], label='action')
    plt.grid()
    plt.legend()
    plt.savefig('figures/%s.png' % figure_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='lqr_controller')
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--max_steps', type=int, default=1000)
    args = parser.parse_args()

    if args.save_video and not args.render:
        raise ValueError('Cant have save-video true but render false.')

    env = CartPoleEnvContinous(store_rendered_frames=args.save_video)

    if args.name == 'lqr_controller':
        controller = LqrController(env)
    elif args.name == 'pid_controller':
        controller = PidController()
    else:
        raise ValueError("Invalid controller name given. Was %s" % args.name)

    saved_dict = run_controller(
        env=env,
        render=args.render,
        max_steps=args.max_steps,
        controller=controller,
    )

    plot_results(saved_dict, args.name)
    if len(env.get_rendered_frames()) > 0:
        writer = imageio.get_writer('figures/%s.mp4' % args.name, fps=120)
        for i, frame in enumerate(env.get_rendered_frames()):
            writer.append_data(frame)
        writer.close()

