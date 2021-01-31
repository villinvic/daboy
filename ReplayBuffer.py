import numpy as np
from MeleeEnv import Transition, Trajectory, MeleeEnv
from typing import *


# Not used
class ReplayBuffer:
    def __init__(self, max_size, traj_length):
        self.mem_size = max_size
        self.traj_length = traj_length
        self.mem_cntr = 0
        self.last_cntr = 0

        self.trajectories = np.array([{
            'state': np.array([np.zeros(MeleeEnv.observation_space, ) for _ in range(traj_length)], dtype=np.ndarray),
            'action': np.zeros((traj_length,), dtype=np.int32),
            'rew': np.zeros((traj_length,), dtype=np.float32),
        } for _ in range(self.mem_size)])

    def add(self, trajectory):
        index = self.mem_cntr % self.mem_size
        self.trajectories[index] = trajectory
        self.mem_cntr += 1

    def get_stored_size(self):
        return min([self.mem_cntr, self.mem_size])

    def sample_traj(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        traj = np.random.choice(self.trajectories[:max_mem])

        return traj

    def get_stored_delta(self):
        delta = self.mem_cntr - self.last_cntr
        self.last_cntr = self.mem_cntr

        return delta

