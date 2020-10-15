import numpy as np
from time import sleep, time
import datetime
import tensorflow as tf
import os
import zmq
import utils
import rewards


class ACLearner:
    def __init__(self, policy, check_point_manager, traj_length):

        self.AC = policy
        self.model_file = "\\models\\model.ckpt"
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/' + current_time + '/train'
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()
        tf.summary.experimental.set_step(0)
        self.checkpoint_manager = check_point_manager
        self.flush_freq = 100
        self.save_ckpt_freq = 1000
        self.write_summary_freq = 3

        self.setup_context()
        self.param_sharing_freq = 100

        self.rcv_amount = 120
        self.rcved = 0

        self.time = 0

        self.dones = np.zeros((traj_length - 1,), dtype=np.float32)
        self.traj_length = traj_length
        self.dones[-1] = 1.0
        self.dones = self.dones[:, np.newaxis]

        self.exp = []
        self.ent_hist = [np.nan for _ in range(20)]


    def learn(self):
        self.cntr += 1
        trajectory = self.exp.pop(0)

        # states, actions, rews = self.compute_traj(trajectory)
        # print('etc  ', time() - self.time)  # 0.14
        with tf.summary.record_if(self.cntr % self.write_summary_freq == 0):
            # t = time()
            mean_ent = self.AC.train(np.float32(trajectory['state']), np.float32(trajectory['action'][:-1][:, np.newaxis]),
                          np.float32(trajectory['rew'][1:][:, np.newaxis]), self.dones)
            # print('train', time() - t)  # 0.0
            self.ent_hist.append(mean_ent)
            self.ent_hist.pop(0)
            
            tf.summary.scalar(name=self.AC.name + "/mean_entropy", data=np.nanmean(self.ent_hist))

        # self.time = time()

        try:
            r = self.eval_socket.recv_pyobj(zmq.NOBLOCK)
            tf.summary.scalar(name="misc/reward", data=r)
        except zmq.ZMQError:
            pass

        dt = time() - self.time
        if self.cntr % (self.write_summary_freq * 5) == 0:
            self.time = time()
            if dt < 3600:
                fps = float(self.traj_length * self.rcved) / dt
                tf.summary.scalar(name="misc/TPS", data=fps)
                self.rcved = 0
                print('exp waiting : ', len(self.exp))

        tf.summary.experimental.set_step(self.cntr - 1)

    def setup_context(self):
        context = zmq.Context()
        self.exp_socket = context.socket(zmq.PULL)
        self.exp_socket.bind("tcp://192.168.32.1:5557")
        self.blob_socket = context.socket(zmq.PUB)
        self.blob_socket.bind("tcp://192.168.32.1:5555")
        self.topic = b''
        self.eval_socket = context.socket(zmq.PULL)
        self.eval_socket.bind("tcp://192.168.32.1:5556")


    def empty_sockets(self):
        try:
            while True:
                self.exp_socket.recv_pyobj(zmq.NOBLOCK)
        except zmq.ZMQError:
            pass
        try:
            while True:
                self.eval_socket.recv_pyobj(zmq.NOBLOCK)
        except zmq.ZMQError:
            pass


    def rcv_exp(self):
        if len(self.exp) < 1000:
            try:
                received = 0
                while received < self.rcv_amount:
                    traj = self.exp_socket.recv_pyobj(zmq.NOBLOCK)
                    self.exp.append(traj)
                    received += 1
            except zmq.ZMQError:
                pass

        self.rcved += received

    def send_params(self):
        params = self.AC.policy.get_params()
        self.blob_socket.send_pyobj(params, flags=zmq.NOBLOCK)

    def run(self):
        try:
            timer = time()
            # self.send_params()
            self.cntr = 1
            self.empty_sockets()
            while True:
                self.rcv_exp()

                if self.exp:
                    self.learn()
                    if self.cntr % self.flush_freq == 0:
                        self.writer.flush()

                    if time() - timer > 10:
                        self.send_params()
                        timer = time()

                    if self.cntr % self.save_ckpt_freq == 0:
                        self.checkpoint_manager.save()
                        print("saved model")

        except KeyboardInterrupt:
            print("Caught SIGINT")
            pass
        self.writer.close()
        print("Learner Exited")


class Learner:
    def __init__(self, memory, policy, check_point_manager, traj_length):

        self.n_loops = 300000
        self.memory = memory
        self.SAC = policy
        self.action_space = [i for i in range(4)]
        self.model_file = "\\models\\model.ckpt"
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/' + current_time + '/train'
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()
        tf.summary.experimental.set_step(0)
        self.last_stock_rew = 0
        self.checkpoint_manager = check_point_manager
        self.flush_freq = 600
        self.save_ckpt_freq = 8000
        self.write_summary_freq = 30

        self.setup_context()
        self.param_sharing_freq = 100
        self.mem_cntr = 0

        self.rcv_amount = 120

        self.time = 0

        self.dones = np.zeros((traj_length - 1,), dtype=np.float32)
        self.dones[-1] = 1.0
        self.dones = self.dones[:, np.newaxis]

    def learn(self):
        self.cntr += 1
        trajectory = self.memory.sample_traj()

        # states, actions, rews = self.compute_traj(trajectory)
        #print('etc  ', time() - self.time)  # 0.14
        with tf.summary.record_if(self.cntr % self.write_summary_freq == 0):
            t = time()
            self.SAC.train(np.float32(trajectory['state']), np.float32(trajectory['action'][:-1][:, np.newaxis]),
                           np.float32(trajectory['rew'][1:][:, np.newaxis]), self.dones)
            #print('train', time() - t)  # 0.0


        #self.time = time()

        try:
            r = self.eval_socket.recv_pyobj(zmq.NOBLOCK)
            tf.summary.scalar(name="misc/reward", data=r)
        except zmq.ZMQError:
            pass

        dt = time() - self.time
        if self.cntr % (self.write_summary_freq * 5) == 0:
            self.time = time()
            if dt < 3600:
                fps = self.memory.traj_length * np.float32(self.memory.get_stored_delta()) / dt
                tf.summary.scalar(name="misc/TPS", data=fps)

        tf.summary.experimental.set_step(self.cntr - 1)

    def setup_context(self):
        context = zmq.Context()
        self.exp_socket = context.socket(zmq.PULL)
        self.exp_socket.bind("tcp://192.168.32.1:5557")
        self.blob_socket = context.socket(zmq.PUB)
        self.blob_socket.bind("tcp://192.168.32.1:5555")
        self.topic = b''
        self.eval_socket = context.socket(zmq.PULL)
        self.eval_socket.bind("tcp://192.168.32.1:5556")

    def compute_traj(self, traj):
        # todo Compute before storing ?
        all_states = utils.map_np_state(traj.states)
        actions = utils.map_action(traj.states[:-1])
        # dones = np.array(utils.map_done( traj.states[:-1]), dtype=np.float32)
        rews = utils.map_rew(traj.states[1:])

        return all_states, actions[:, np.newaxis], rews[:, np.newaxis]

    def rcv_exp(self):
        try:
            received = 0
            while received < self.rcv_amount:
                traj = self.exp_socket.recv_pyobj(zmq.NOBLOCK)
                self.memory.add(traj)
                self.mem_cntr += 1
                received += 1
        except zmq.ZMQError:
            pass

    def send_params(self):
        params = self.SAC.actor.get_params()
        self.blob_socket.send_pyobj(params, flags=zmq.NOBLOCK)

    def run(self):
        try:
            timer = time()
            last_mem_index = 0
            # self.send_params()
            self.cntr = 1
            while True:
                self.rcv_exp()

                if self.memory.get_stored_size() > 10:
                    self.learn()
                    if self.cntr % self.flush_freq == 0:
                        self.writer.flush()

                    if time() - timer > 10:
                        self.send_params()
                        timer = time()

                    if self.cntr % self.save_ckpt_freq == 0:
                        self.checkpoint_manager.save()
                        print("saved model")

        except KeyboardInterrupt:
            print("Caught SIGINT")
            pass
        self.writer.close()
        print("Learner Exited")
