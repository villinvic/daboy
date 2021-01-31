import numpy as np
from time import time
import datetime
import tensorflow as tf
import zmq
import gc
import pandas as pd

class ACLearner:
    def __init__(self, net, check_point_manager, traj_length, params, batch_size):

        self.params = params
        self.AC = net
        self.model_file = "\\models\\model.ckpt"
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/' + current_time + '/train'
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()
        tf.summary.experimental.set_step(0)
        self.checkpoint_manager = check_point_manager
        self.flush_freq = 100
        self.save_ckpt_freq = int(1000/float(batch_size))
        self.gc_freq = 3000
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
        
        self.batch_size = batch_size


    def learn(self):
        self.cntr += 1
        trajectory = pd.DataFrame(self.exp[:self.batch_size]).values
        self.exp = self.exp[self.batch_size:]

        #print(np.stack(trajectory[:, -1], axis = 0)[:, 1:])
        states = np.float32(np.stack(trajectory[:, 0], axis = 0))
        actions = np.float32(np.stack(trajectory[:, 1], axis = 0)[:, :-1])
        rews = np.float32(np.stack(trajectory[:, 2], axis = 0)[:, 1:])
        # states, actions, rews = self.compute_traj(trajectory)
        # print('etc  ', time() - self.time)  # 0.14
        with tf.summary.record_if(self.cntr % self.write_summary_freq == 0):
            # t = time()
            self.AC.train( states, actions, rews, self.dones)
            # print('train', time() - t)  # 0.0

        # self.time = time()
        
        
        try:
            r = 0
            received = 0
            n_collect = 6 * self.batch_size
            for _ in range(n_collect):
                r += self.eval_socket.recv_pyobj(zmq.NOBLOCK)
                received += 1
        except zmq.ZMQError:
            pass
        if received != 0:
            tf.summary.scalar(name="misc/reward", data=r/float(received))

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
        self.exp_socket.bind("tcp://127.0.0.1:5557")
        self.blob_socket = context.socket(zmq.PUB)
        self.blob_socket.bind("tcp://127.0.0.1:5555")
        self.topic = b''
        self.eval_socket = context.socket(zmq.PULL)
        self.eval_socket.bind("tcp://127.0.0.1:5556")


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
                    # sym_traj = self.gen_sym_traj(traj) # off policy
                    self.exp.append(traj)
                    # self.exp.append(sym_traj)
                    received += 1
            except zmq.ZMQError:
                pass

        self.rcved += received

    def send_params(self):
        params = self.AC.policy.get_params()
        params.update(self.params)
        self.blob_socket.send_pyobj(params, flags=zmq.NOBLOCK)

    def run(self):
        try:
            timer = time()
            dummy_states = np.zeros((self.batch_size, self.traj_length, 847))
            self.AC.policy.get_probs(dummy_states)
            self.cntr = 1
            self.empty_sockets()
            while True:
                self.rcv_exp()

                if len(self.exp)>= self.batch_size:
                    self.learn()
                    if self.cntr % self.flush_freq == 0:
                        self.writer.flush()

                    if self.cntr % self.save_ckpt_freq == 0:
                        self.checkpoint_manager.save()
                        print("saved model")
                      
                    if self.cntr % self.gc_freq == 0:
                        gc.collect()
                        
                if time() - timer > 10:
                        self.send_params()
                        timer = time()

        except KeyboardInterrupt:
            print("Caught SIGINT")
            pass
        self.writer.close()
        print("Learner Exited")

