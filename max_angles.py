import argparse
import math

import imageio
import matplotlib.pyplot as plt
import numpy as np

from agents.controllers import LqrController, PidController, HybridController
from envs.cartpole import CartPoleEnvContinous


def run_controller(env, controller, init_state, max_steps_simulation=200, render=False):
    obs = env.reset(init_state=init_state)
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

    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(10, 7)
    ax[0].plot(saved_dict['x'], label='x')
    ax[0].plot(saved_dict['xdot'], label='x dot')
    ax[0].plot(saved_dict['theta'], label='theta')
    ax[0].plot(saved_dict['thetadot'], label='theta dot')
    ax[1].plot(saved_dict['action'], label='action')

    ax[0].grid()
    ax[0].legend()
    ax[1].legend()
    ax[1].grid()
    plt.savefig('figures/%s.png' % figure_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=1000)
    args = parser.parse_args()

    angles = np.linspace(5 * math.pi / 180, 80 * math.pi / 180, 50)
    # angles = (40 * math.pi/180,)

    render = False
    for name in ('lqr_controller', 'pid_controller', 'hybrid_controller'):
        for angle in angles:
            env = CartPoleEnvContinous(store_rendered_frames=False)

            if name == 'lqr_controller':
                controller = LqrController(env)
            elif name == 'pid_controller':
                controller = PidController()
            elif name == 'hybrid_controller':
                controller = HybridController(env, load_model=True, train_model=False)
            else:
                raise ValueError("Invalid controller name given. Was %s"
                                 % name)

            init_state = np.array([0, 0, angle, 0])
            saved_dict = run_controller(
                init_state=init_state,
                env=env,
                render=render,
                max_steps_simulation=args.max_steps,
                controller=controller,
            )

            threshold_ts = {}
            num_time_steps = len(saved_dict['x'])

            indices = np.repeat(True, num_time_steps)
            for (var, threshold) in {
                'x': 0.01,
                'xdot': 1e-3,
                'theta': 1 * math.pi / 180,
                'thetadot': 1e-3,
            }.items():
                indices = indices & (np.abs(np.array(saved_dict[var])) < threshold)

            subthreshold_times = np.argwhere(indices)
            print("[%s] Angle %.4f\ttf=%.4f\tx(tf)=%.4f\tangle=%.4f\tintu2=%.4f\tthreshold_t=%.4f" % (
                name,
                angle * 180 / math.pi,
                num_time_steps * env.tau,
                saved_dict['x'][-1],
                (180 / math.pi) * saved_dict['theta'][-1],
                np.sqrt(np.sum(np.power(saved_dict['action'][:], 2)))*env.tau,
                -1 if len(subthreshold_times) == 0 else env.tau * np.min(subthreshold_times)))
            if render:
                plot_results(saved_dict, name)

            if len(env.get_rendered_frames()) > 0:
                writer = imageio.get_writer('figures/%s.mp4' % name, fps=120)

                for i, frame in enumerate(env.get_rendered_frames()):
                    if frame.shape != (400, 600, 3):
                        continue
                    writer.append_data(frame)
                writer.close()
