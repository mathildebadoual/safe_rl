import abc
import os
import control
import sys
import pylab
import numpy as np

from agents.virtual_pid import PID
from agents.actor_critic import ActorCriticAgent


class Controller(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state, t=None):
        raise NotImplementedError("get_action")


class LqrController(Controller):
    def __init__(self, env):
        self.env = env

    def get_action(self, state, t=None):
        A = np.array(
            [[0, 1, 0, 0],
             [0, 0, -self.env.masspole * self.env.gravity / self.env.masscart,
              0],
             [0, 0, 0, 1],
             [0, 0, self.env.total_mass * self.env.gravity /
              (self.env.length * self.env.masscart), 0]
             ])
        B = np.array(
            [[0],
             [1 / self.env.masscart],
             [0],
             [-1 / (self.env.length * self.env.masscart)]])
        Q = 10 * np.eye(4)
        R = 0.01
        K, _, _ = control.lqr(A, B, Q, R)
        action = - np.dot(K[0], state.T)
        return action


class PidController(Controller):
    def __init__(self):
        self.pid = PID(1, 0.1, 0.05, setpoint=0)

    def get_action(self, state, t=None):
        if t is None:
            print('t should be given for PID')
        return self.pid(t, state[0])


# hybrid controller = LQR + ActorCritic
class HybridController(Controller):
    def __init__(self, env):
        self.env = env
        self.lqr_controller = LqrController(self.env)
        if os.path.isfile('./agents/save_model/model_actor.h5') and \
                os.path.isfile('./agents/save_model/model_critic.h5'):
            print("True")
            self.actor_critic = ActorCriticAgent(self.env, load_model=True)
        else:
            self.actor_critic = ActorCriticAgent(self.env)
            self.run_training()

    def get_action(self, state, t=None):
        action_lqr = self.lqr_controller.get_action(state)
        action_ac = self.actor_critic.get_action(state)
        action = action_ac + action_lqr
        return action

    def run_training(self, num_episodes=100, max_score=100,
                     max_iteration=100):
        scores, episodes = [], []

        for e in range(num_episodes):
            done = False
            score = 0
            state = self.env.reset()
            state = np.reshape(state, [1, self.actor_critic.state_size])

            iteration = 0
            while not done:
                iteration += 1
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                next_state = np.reshape(next_state,
                                        [1, self.actor_critic.state_size])
                # if an action make the episode end, then gives penalty of -100
                reward = reward if not done or score == 499 else -100

                self.actor_critic.train_model(state, action, reward,
                                              next_state, done)

                score += reward
                state = next_state

                if iteration > max_iteration:
                    done = True

                if done:
                    # every episode, plot the play time
                    score = score if score == 500.0 else score + 100
                    scores.append(score)
                    episodes.append(e)
                    pylab.plot(episodes, scores, 'b')
                    pylab.savefig("./figures/cartpole_a2c.png")
                    print("episode:", e, "  score:", score)

                    # if np.mean(scores[-min(10, len(scores)):]) > max_score:
                    #     sys.exit()

            # save the model
            if e % 50 == 0:
                self.actor_critic.actor.save_weights(
                    "./agents/save_model/cartpole_actor.h5")
                self.actor_critic.critic.save_weights(
                    "./agents/save_model/cartpole_critic.h5")
