import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, Lambda,
                          MaxPooling2D, Multiply)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, Adamax, RMSprop


def categorical_crossentropy(target, output):
    _epsilon = tf.convert_to_tensor(10e-8, dtype=output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
    return (- target * tf.log(output))


class actor_critic():
    def __init__(self, env, actor_learning_rate=1e-3,
                 critic_learning_rate=1e-3, gamma=0.9):
        self.env = env
        self.actions_space_dim = env.action_space.n
        self.observation_dim = env.observation_space.shape[0]
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.log_path = './actor_critic.log'
        self.dummy_act_picked = np.zeros((1, self.actions_space_dim))

        # Actor nn
        input = Input(shape=(self.observation_dim,))
        hidden = Dense(20, activation='relu')(input)
        actions_prob = Dense(self.actions_space_dim,
                             activation='softmax')(hidden)
        action_picked = Input(shape=(self.actions_space_dim,))
        action = Multiply()([actions_prob, action_picked])
        action = Lambda(lambda x: K.sum(
            x, axis=-1, keepdims=True), output_shape=(1,))(action)
        model = Model(inputs=[input, action_picked],
                      outputs=[actions_prob, action])
        opt = Adam(lr=self.actor_learning_rate)
        model.compile(loss=['mse', categorical_crossentropy],
                      loss_weights=[0.0, 1.0], optimizer=opt)
        self.actor = model

        # Critic nn
        model = Sequential()
        model.add(Dense(20, activation='relu',
                        input_shape=(self.observation_dim,)))
        model.add(Dense(1))
        opt = Adam(lr=self.critic_learning_rate)
        model.compile(loss='mse', optimizer=opt)
        self.critic = model

    def train(self):
        log = open(self.log_path, 'w')
        log.write('reward, avg_reward \n')
        batch_size = 1
        frames, prob_actions, dlogps, drs = [], [], [], []
        tr_x, tr_y = [], []
        reward_record = []
        avg_reward = []
        reward_sum = 0
        ep_number = 0
        ep_step = 0
        observation = self.env.reset()

        while True:
            act = np.random.choice(np.arange(
                self.actions_space_dim),
                p=self.actor.predict([np.expand_dims(observation, axis=0),
                                      self.dummy_act_picked])[0].flatten())

            act_one_hot = np.zeros((1, self.actions_space_dim))
            act_one_hot[0, act] = 1.0
            next_observation, reward, done, info = self.env.step(act)
            if done:
                reward = -20

            reward_sum += reward
            predict_reward = self.critic.predict(
                np.expand_dims(observation, axis=0))
            predict_next_reward = self.critic.predict(
                np.expand_dims(next_observation, axis=0))

            td_target = np.expand_dims(
                reward, axis=0) + self.gamma * predict_next_reward
            td_error = td_target - predict_reward

            self.critic.train_on_batch(
                np.expand_dims(observation, axis=0), td_target)
            self.actor.train_on_batch([np.expand_dims(observation, axis=0),
                                       act_one_hot],
                                      [self.dummy_act_picked, td_error])

            observation = next_observation

            self.t += 1
            ep_step += 1

            if done or ep_step > MAX_TIMESTEP:
                ep_number += 1

                avg_reward.append(float(reward_sum))
                if len(avg_reward) > 30:
                    avg_reward.pop(0)

                print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / REWARD: {2:5d} / AVG_REWARD: {3:2.3f} '.format(
                    ep_number, self.t, int(reward_sum), np.mean(avg_reward)))
                print('{:.4f},{:.4f}'.format(reward_sum, np.mean(
                    avg_reward)), end='\n', file=log, flush=True)

                observation = self.env.reset()
                reward_sum = 0.0
                ep_step = 0

            if ep_number >= MAX_EP:
                self.actor.save('actor.h5')
                self.critic.save('critictor.h5')
                break

    def get_action(self, observation):
        pass
