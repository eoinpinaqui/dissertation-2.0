# Library imports
from pynput.keyboard import Listener

# Local imports
from game.single_player import SinglePlayerGame

action = 0


def press(key):
    global action
    if 'char' in dir(key):
        if key.char == 'w': action = 1
        if key.char == 's': action = 2
        if key.char == 'a': action = 3
        if key.char == 'd': action = 4
        if key.char == 'm': action = 5


def release(key):
    global action
    action = 0


Listener(on_press=press, on_release=release).start()

env = SinglePlayerGame()
env.reset()
done = False
total_reward = 0
while not done:
    next_state, reward, _done, _ = env.step(action)
    total_reward += reward
    done = _done
    env.render()

env.close()
print(total_reward)
