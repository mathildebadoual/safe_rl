import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


class ActorCriticAgent:
    def __init__(self, env, load_model=False, actor_lr=0.001,
                 critic_lr=0.005,
                 discount_factor=0.99):

        self.env = env
        self.load_model = load_model

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.value_size = 1
        self.num_action_discrete = 100

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = discount_factor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        # for discretize_action
        min_action = self.env.action_space.low
        max_action = self.env.action_space.high
        self.action_discrete = np.linspace(min_action, max_action,
                                           self.num_action_discrete)

        if self.load_model:
            self.actor.load_weights("./agents/save_model/model_actor.h5")
            self.critic.load_weights("./agents/save_model/model_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor(self):
        actor = Sequential()
        actor.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        actor.add(Dense(self.num_action_discrete, activation='softmax',
                        kernel_initializer='he_uniform'))
        actor.summary()
        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=self.state_size, activation='relu',
                         kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size, activation='linear',
                         kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        policy = self.actor.predict(state, batch_size=1).flatten()
        idx = np.argmax(policy)
        action = self.action_discrete[idx]
        return action

    def train_model(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.num_action_discrete))

        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        action_index = self.discretize_action(action)

        if done:
            advantages[0][action_index] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action_index] = reward + self.discount_factor * \
                (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(state, advantages, epochs=1, verbose=0)
        self.critic.fit(state, target, epochs=1, verbose=0)

    def discretize_action(self, action):
        idx = (np.abs(self.action_discrete - action)).argmin()
        return idx
