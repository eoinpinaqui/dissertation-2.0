# Library imports
import collections
from tensorflow import keras
import numpy as np
import random

'''
Factory functions for building neural networks for discrete action spaces
'''


# Base function to create a network
def _build_model(observation_shape, action_space, learning_rate, linear: bool):
    if linear:
        return _build_linear_model(observation_shape, action_space, learning_rate)
    return _build_conv_model(observation_shape, action_space, learning_rate)


# Function to create a network with a linear observation space
def _build_linear_model(observation_shape, action_space, learning_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(24, input_dim=observation_shape, activation='relu'))
    model.add(keras.layers.Dense(24, activation='relu'))
    model.add(keras.layers.Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model


# Function to create a network with a 2D observation space
def _build_conv_model(observation_shape, action_space, learning_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, 8, strides=(4, 4), padding='valid', activation='relu', input_shape=observation_shape))
    model.add(keras.layers.Conv2D(64, 4, strides=(2, 2), padding='valid', activation='relu'))
    model.add(keras.layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation='relu'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model


# The DQNAgent holds all the logic for a DQN model
class DQNAgent:
    def __init__(self, observation_shape, action_space, epsilon_decay, learning_rate, linear=False):
        self.observation_shape = observation_shape
        self.action_space = action_space
        self.memory = collections.deque(maxlen=100000)
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = _build_model(self.observation_shape, self.action_space, self.learning_rate, linear)
        self.model.summary()

    def memorize(self, state, action: int, reward: int, next_state, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space, size=1)[0]
        action_values = self.model.predict(state, verbose=0)[0]
        return np.argmax(action_values)

    def replay(self, batch_size):
        if batch_size > len(self.memory):
            return -1

        losses = []
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_function = self.model.predict(state, verbose=0)
            target_function[0][action] = target
            loss = self.model.fit(state, target_function, epochs=1, verbose=0)
            losses.append(loss.history['loss'])
        self.decay_epsilon()
        return np.mean(losses)

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save_weights(path)
