# Library imports
import gym
import numpy as np

# Local imports
from model.dqn import DQNAgent

env = gym.make('CartPole-v1')
observation_shape = env.observation_space.shape[0]
action_space = env.action_space.n
agent = DQNAgent(observation_shape, action_space, epsilon_decay=0.995, learning_rate=0.01, linear=True)
batch_size = 32

max_episodes = 50
max_episode_length = 500
solved = False
for e in range(max_episodes):
    if solved: break
    state = env.reset()[0]
    state = np.reshape(state, (1, observation_shape))
    for time in range(max_episode_length):
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state, [1, observation_shape])
        reward = reward if not done else -10
        agent.memorize(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f'Episode {e}/{max_episodes}, Score: {time}, e: {agent.epsilon}')
            if time > max_episode_length // 2:
                solved = True
            break
        agent.replay(batch_size)
