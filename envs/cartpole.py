import math
import random

import numpy as np
from gym import logger, spaces
from gym.utils import seeding


class CartPoleEnv():
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.01
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None
        self.previous_state = None
        self.previous_action = None

        self.steps_beyond_done = None

    def step(self):
        raise NotImplementedError

    def dynamics_step(self, state, action):
        x, x_dot, theta, theta_dot = state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (action + self.polemass_length * theta_dot * theta_dot *
                sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta *
                           costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / \
            self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array((x, x_dot, theta, theta_dot))
        done = x < -self.x_threshold \
            or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = float(
                self.compute_cost(
                    self.state,
                    action))
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment "
                    "has already returned done = True. You should always call "
                    "'reset()' once you receive 'done = True' -- any further "
                    "steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = -100

        return done, reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, init_state=None):
        if init_state is None:
            # self.state = self.observation_space.sample()
            self.state = np.array([0, 0, random.uniform(-0.1, 0.1), 0])
        else:
            self.state = init_state
        self.previous_state = self.state
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2 / 1000
        # world_width = 2.4 * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0 / 10
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, \
                         -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - \
                polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()

    def compute_cost(self, state, action):
        Q = 10 * np.eye(4)
        R = 0.01
        return - np.dot(state.T, np.dot(Q, state)) - R * action ** 2


class CartPoleEnvContinous(CartPoleEnv):
    def __init__(self):
        super(CartPoleEnvContinous, self).__init__()
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,),
                                       dtype=np.float32)

    def step(self, action):
        state = self.state
        if self.previous_action is None:
            self.previous_action = action

        done, reward = self.dynamics_step(state, action)

        self.previous_state = self.state
        self.previous_action = action

        return np.array(self.state), reward, done


class CartPoleEnvDiscrete(CartPoleEnv):
    def __init__(self):
        super(CartPoleEnvDiscrete, self).__init__()
        self.force_mag = 1
        self.action_space = spaces.Discrete(2)

    def step(self, action):
        state = self.state
        if self.previous_action is None:
            self.previous_action = action
        force = self.force_mag if action == 1 else -self.force_mag

        done, reward = self.dynamics_step(state, force)

        self.previous_state = self.state
        self.previous_action = action

        return np.array(self.state), reward, done
