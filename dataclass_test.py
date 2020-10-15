from utils import write_with_folder
from MeleeEnv import GameMemory, Trajectory, Experience, Transition
from StateManager import StateManager
import numpy as np

x = GameMemory()

sm = StateManager(x)

tt = Transition(x, 1, False)

t = Trajectory()

for i in range(30):
    t.states[i] = tt

indexes = [0, 3, 5, 9]


def mapper(state):
    return state.state


states_map = np.vectorize(mapper)

print(states_map(t.states[indexes]))
