from RERPI import *
import gym
from Memory import MemoryBuffer
import datetime

"""
Testing RL algos with gym api
"""

class Test:

    @staticmethod
    def get_obs():
        return np.array([
            np.random.normal() for _ in range(5)
        ], dtype=np.float32)

    @staticmethod
    def get_r():
        return np.random.normal()

    @staticmethod
    def is_done():
        return np.random.random() < 0.1

    def __init__(self):
        dummy = self.get_obs()

        action_space = [i for i in range(10)]

        x = RERPI(dummy.shape, len(action_space), 0.03, 0.01, 0.99, 10.0)

        batch_size = 30

        for i in range(30):
            actions = np.array(np.random.choice(action_space, batch_size - 1), dtype=np.int32)
            dones = np.array([self.is_done() for _ in range(batch_size - 1)], dtype=np.float32)
            rewards = np.array([self.get_r() for _ in range(batch_size - 1)], dtype=np.float32)
            states = np.array([self.get_obs() for _ in range(batch_size)], dtype=np.float32)

            policy_loss, std_error = x.train(states, actions, rewards, dones)

            tf.print(policy_loss, std_error)


if __name__ == '__main__':

    logdir = 'rerpilogs\\' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()
    tf.summary.experimental.set_step(0)

    # Test()

    env = gym.make('LunarLander-v2')
    mem = MemoryBuffer(env.observation_space.shape)
    # state_shape, action_dim, epsilon_greedy, lr, gamma, nu, target_update_freq=250
    """
    agent = RERPI(env.observation_space.shape, 4,
                  epsilon_greedy=0.05, lr=0.0003, gamma=0.99, nu=2.0, gpu=0, target_update_freq=250)
    rs = []
    n_warmup = 300
    n = 0
    policy_loss = None
    std_error = None
    mean_entropy = None
    traj_length = 512
    actions = [0, 0, 0, 0]
    for i in range(10000):
        done = False
        obs = env.reset()
        r = 0
        while not done:
            if i > 10000:
                env.render()
            if n < n_warmup:
                act = env.action_space.sample()
            else:
                act = agent.policy.get_action(obs)
                actions[act] += 1
            n += 1

            next_obs, rew, done, _ = env.step(act)

            r += rew
            t = {
                'rew': rew,
                'obs': obs,
                'done': done,
                'act': act,
            }
            mem.add(t)
            obs = next_obs

            traj = mem.sample_traj(traj_length)
            tf.summary.experimental.set_step(n)
            if traj is not None:
                policy_loss, std_error, mean_entropy = agent.train(traj['obs'], traj['act'][:-1], traj['rew'][:-1],
                                                                   traj['done'][:-1])

        rs.append(r)
        tf.summary.scalar(name="main/reward", data=r)

        print('Episode %d, Mean reward:%.3f' % (i, np.mean(rs[-20:])))
        print('actions', actions)
        actions = [0, 0, 0, 0]

        if i > 0 and i % 40 == 0:
            percent_actions = np.array(actions, dtype=np.float32) / (1.0 + sum(actions))
            print('\npolicy loss:%.3f, q loss:%.3f, entropy:%.3f\n' % (policy_loss, std_error, mean_entropy))
            
    """

    agent = AC(env.observation_space.shape, 4,
               epsilon_greedy=0.05, lr=0.0003, gamma=0.99, entropy_scale=0.001, lamb=0.95, gpu=0)
    rs = []
    n_warmup = 300
    n = 0
    policy_loss = None
    std_error = None
    mean_entropy = None
    traj_length = 1024
    traj = {
        'rew': np.zeros((traj_length,), dtype=np.float32),
        'obs': np.zeros((traj_length, env.observation_space.shape[0]), dtype=np.float32),
        'done': np.zeros((traj_length,), dtype=np.bool),
        'act': np.zeros((traj_length,), dtype=np.int32),
    }
    actions = [0, 0, 0, 0]
    for i in range(10000):
        done = False
        obs = env.reset()
        r = 0
        traj_index = 0

        while not done:
            if i > 10000:
                env.render()
            if n < n_warmup:
                act = env.action_space.sample()
            else:
                act = agent.policy.get_action(obs)
                actions[act] += 1
            n += 1

            next_obs, rew, done, _ = env.step(act)

            r += rew
            traj['rew'][traj_index] = rew
            traj['done'][traj_index] = done
            traj['act'][traj_index] = act
            traj['obs'][traj_index] = obs
            traj_index += 1
            obs = next_obs

            tf.summary.experimental.set_step(n)

        policy_loss, std_error, mean_entropy = agent.train(traj['obs'][:traj_index], traj['act'][:traj_index - 1],
                                                           traj['rew'][1:traj_index], traj['done'][1:traj_index])

        rs.append(r)
        tf.summary.scalar(name="main/reward", data=r)

        print('Episode %d, Mean reward:%.3f' % (i, np.mean(rs[-20:])))
        print('actions', actions)
        actions = [0, 0, 0, 0]

        if i > 0 and i % 40 == 0:
            percent_actions = np.array(actions, dtype=np.float32) / (1.0 + sum(actions))
            print('\npolicy loss:%.3f, q loss:%.3f, entropy:%.3f\n' % (policy_loss, std_error, mean_entropy))
