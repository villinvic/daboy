import numpy as np
from time import time
import datetime
import tensorflow as tf
import zmq
import gc
import pandas as pd
import socket


class ACLearner:
    def __init__(self, net, checkpoint_manager, traj_length, params, batch_size, is_localhost):
        '''
        Learner class
        accepts data sent from SSBM_ENV classes in an experience queue
        uses the data in a training loop.
        Regularly send to the environments the new network weights.

        net : base network class.
        checkpoint_manager: checkpointManager class used for saving the network.
        traj_length : trajectory length, must be the same as the one specified in SSBM_ENV.
        params : some params to send to the environments, such as the distance reward scale.
        batch_size :  batch size is a trade of between convergence speed and stability.
            Higher batch size = higher stability
        is_localhost : must be true if testing or training on only one machine.

        '''

        self.ip = '127.0.0.1' if is_localhost else socket.gethostbyname(socket.gethostname())
        self.params = params
        self.AC = net
        self.model_file = "\\models\\model.ckpt"
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/' + current_time + '/train'
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()
        tf.summary.experimental.set_step(0)
        self.checkpoint_manager = checkpoint_manager
        self.flush_freq = 100
        self.save_ckpt_freq = int(10000/float(batch_size))
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

        # Get experience from the queue
        trajectory = pd.DataFrame(self.exp[:self.batch_size]).values
        self.exp = self.exp[self.batch_size:]

        # Cook data
        states = np.float32(np.stack(trajectory[:, 0], axis = 0))
        actions = np.float32(np.stack(trajectory[:, 1], axis = 0)[:, :-1])
        rews = np.float32(np.stack(trajectory[:, 2], axis = 0)[:, 1:])
        r_states = np.stack(trajectory[:, 3], axis = 0)[:, 0,:]


        # Train
        with tf.summary.record_if(self.cntr % self.write_summary_freq == 0):
            self.AC.train( states, actions, rews, r_states)
        
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
        self.exp_socket.bind("tcp://%s:5557" % self.ip)
        self.blob_socket = context.socket(zmq.PUB)
        self.blob_socket.bind("tcp://%s:5558" % self.ip)
        self.topic = b''
        self.eval_socket = context.socket(zmq.PULL)
        self.eval_socket.bind("tcp://%s:5556" % self.ip)


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
        params.update(self.params)
        self.blob_socket.send_pyobj(params, flags=zmq.NOBLOCK)

    def run(self):
        try:
            timer = time()
            dummy_states = np.zeros((self.batch_size, self.traj_length, self.AC.state_shape[0]), dtype=np.float32)
            self.AC.policy.get_probs(dummy_states)
            self.AC.V(dummy_states)
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

