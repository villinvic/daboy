import numpy as np

# Not used
class MemoryBuffer:

    def __init__(self, obs_shape):
        self.obs_shape = obs_shape
        self.index = 0
        self.max_mem = 3000

        self.transitions = {
            'rew': np.zeros((self.max_mem,), dtype=np.float32),
            'obs': np.array([np.zeros(obs_shape, dtype=np.float32) for _ in range(self.max_mem)]),
            'done': np.zeros((self.max_mem,), dtype=np.bool),
            'act': np.zeros((self.max_mem,), dtype=np.float32)
        }

    def add(self, transition):
        index = self.index % self.max_mem

        self.transitions['rew'][index] = transition['rew']
        self.transitions['obs'][index] = transition['obs']
        self.transitions['done'][index] = transition['done']
        self.transitions['act'][index] = transition['act']

        self.index += 1

    def get_stored_size(self):
        return min([self.index, self.max_mem])

    def sample_traj(self, length):
        if length > self.get_stored_size() * 5:
            return None
        start = np.random.randint(0, self.get_stored_size())

        end = (start + length) % self.max_mem

        traj = dict()
        if end < start:
            traj['rew'] = np.append(self.transitions['rew'][:end], self.transitions['rew'][start:])
            traj['done'] = np.append(self.transitions['done'][:end], self.transitions['done'][start:])
            traj['obs'] = np.concatenate((self.transitions['obs'][:end], self.transitions['obs'][start:]))
            traj['act'] = np.append(self.transitions['act'][:end], self.transitions['act'][start:])
        else:
            traj['rew'] = self.transitions['rew'][start: end]
            traj['obs'] = self.transitions['obs'][start: end]
            traj['done'] = self.transitions['done'][start: end]
            traj['act'] = self.transitions['act'][start: end]
        return traj
