from Default import Default, Option
from Learner import Learner, ACLearner
from Actor import Actor
from ReplayBuffer import ReplayBuffer
import RERPI
import Pad


from threading import Event
from MeleeEnv import MeleeEnv
import os
import time
import signal
from argparse import ArgumentParser
import tensorflow as tf
import pickle
from subprocess import Popen
from sys import stdout
import psutil


#Cudnn fix ...
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# TODO
# Move state and reward computation to actors ( convert trajectories to dicts)
# Concat trajectories for bigger batch size

# action embedding
# higher stock reward if high percent
# light l press
# twitch bot, listener on learner side
# reset actor every n mins : reset sockets ?
# separate actor and learner threads (find new way to track eval (socket ?))
# logger ?
# gui
# gae ?
# longer episodes ?
# increase reward if hit when in hitlag (combo reward)
# speedhack ?


class Main(Default):
    _options = [
        Option('learning_rate', type=float, default=1e-4),
        Option('discount_factor', type=float, default=0.994),
        Option('batch_size', type=int, default=256),
        Option('file_name', type=str, default='models\\model.h5',
               help="path to which the checkpoint will be loaded/saved"),
        Option('optimizer', type=str, default="GradientDescent", help="which tf optimizer to use"),
        Option('mem_size', type=int, default=150000, help='amount of experience stockable'),
        Option('replace_target', type=int, default=8000, help='frequency at which the target q_value is updated'),
        Option('load_checkpoint', action="store_true"),
        Option('mw_port', type=int, default=5559),
        # port 5556 is buggy ?
        Option('pad_port', type=int, default=5560),
        Option('n_actors', type=int, default=1),
        Option('loss', type=str, default='mse'),
        Option('layer_size', type=int, default=256),
        Option('dolphin_dir', type=str, default='../dolphin/'),
        Option('iso_path', type=str, default=r'../isos/melee.iso'),
        Option('video_backend', type=str, default='D3D'),
        Option('cuda', type=bool, default=False),  # TODO
        Option('n_loops', type=int, default=10 ** 6),
        Option('self_play', action="store_true"),
        Option('output_dir', default="checkpoints", help="checkpoint dir"),
        Option('dump_file', default="memory_dump"),
        Option('n_warmup', type=int, default=20 * 60),
        Option('render_all', action="store_true"),
        Option('render_none', action="store_true"),
        Option('load_memory', action="store_true"),
        Option('epsilon', type=float, default=0.01),
        Option('gae_lambda', type=float, default=1.0),
        Option('alpha', type=float, default=7e-4),
        Option('ep_length', type=int, default=20 * 5),
        Option('generate_exp', action='store_true'),
        Option('mode', choices=['learner', 'actor', 'both'], default='both'),
        Option('eval', action='store_true'),
        Option('char', type=str, default='ganon'),
    ]

    _members = [
    ]

    def __init__(self, **kwargs):
        Default.__init__(self, init_members=False, **kwargs)

        if self.mode == 'actor':
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.mode = {
            'learner': self.mode == 'learner' or self.mode == 'both',
            'actor': self.mode == 'actor' or self.mode == 'both'
        }

        self.pad_path = [self.dolphin_dir + 'User/Pipes/' for i in range(self.n_actors)]
        self.mw_path = self.dolphin_dir + 'User/MemoryWatcher/'
        self.dolphin_exe = self.dolphin_dir + 'dolphin-emu-nogui'

        if self.mode['learner']:
            dummy_env = MeleeEnv(self.char)
            self.action_space = dummy_env.action_space
            state_shape = dummy_env.observation_space
            print(state_shape)

            if self.load_checkpoint or self.load_memory:
                self.n_warmup = 0

            self.net = RERPI.AC(state_shape=state_shape, action_dim=self.action_space.len, epsilon_greedy=self.epsilon,
                                lr=self.learning_rate, gamma=self.discount_factor, entropy_scale=self.alpha, gae_lambda=self.gae_lambda, gpu=0,
                                traj_length=self.ep_length)


            # Save and restore model
            self.checkpoint = tf.train.Checkpoint(policy=self.net, actor=self.net.policy)
            self.checkpoint_manager = tf.train.CheckpointManager(
                self.checkpoint, directory=self.output_dir, max_to_keep=10)

            if self.load_checkpoint:
                assert os.path.isdir(self.output_dir)
                path_ckpt = tf.train.latest_checkpoint(self.output_dir)
                status = self.checkpoint.restore(path_ckpt).expect_partial()
                status.assert_existing_objects_matched()
                # status.assert_consumed()
                print("Restored {}".format(path_ckpt))
                self.net.policy.epsilon.assign(self.epsilon)
                self.net.entropy_scale.assign(self.alpha)

            # if self.load_memory or self.load_checkpoint:
            #    self.load_mem()
            self.learner = ACLearner(self.net, self.checkpoint_manager, self.ep_length)

        if self.mode['actor']:
            if self.render_none:
                self.video_backend = "Null"
            self.actors = self.make_actors()

    def make_actors(self, first=True):
        if not first:
            self.n_warmup = 0
        return [Actor(self.self_play and not (i == self.n_actors - 1 and self.eval), self.output_dir,
                      self.ep_length, self.dolphin_exe, self.dolphin_dir, self.iso_path,
                      self.video_backend if (i == 0 or (i == self.n_actors - 1 and self.eval) or self.render_all) else "Null", self.mw_path,
                      self.pad_path[i], self.pad_port + 4 * i, i, self.n_actors, self.epsilon, self.n_warmup) for i in
                range(self.n_actors)]

    def load_mem(self):
        try:
            print('...Loading memory dump from checkpoint...', end=" ")
            stdout.flush()

            with open(self.dump_path, mode="rb") as f:
                self.memory = pickle.load(f)
            print('done.')
        except Exception as e:
            print("Couldn't load memory dump:", e)
            pass

    def dump_mem(self):
        try:
            print("...Dumping memory...", end=" ")
            stdout.flush()
            with open(self.dump_path, mode="wb") as f:
                pickle.dump(self.memory, f)

            print("done.")
        except Exception as e:
            print("Failed dumping memory:", e)

    def __call__(self):
        """main loop"""
        # Boot actor(s)
        print("...Starting...")
        if self.mode['actor']:
            for a in self.actors:
                a.start()

        if self.mode['learner']:
            time.sleep(3)
            # Wait for SIGINT
            # --------------------------- #
            self.learner.run()
            # --------------------------- #

        if self.mode['actor'] and not self.mode['learner']:
            restart_timer = 60*30 # restart actors every 30 mins
            cntr = 0
            try:
                while True:
                    time.sleep(1.0)
                    cntr += 1
                    if cntr % restart_timer == 0:
                        print('Restarting actors...')
                        for a in self.actors:
                            a.exit()
                        time.sleep(1.0)
                        self.actors = self.make_actors(first=False)
                        for a in self.actors:
                            a.start()
                    
            except KeyboardInterrupt:
                pass

    def exit(self):
        print('...Exiting daboy...')
        if self.mode['actor']:
            for a in self.actors:
                a.exit()

            time.sleep(2.0)

        if self.mode['learner']:
            time.sleep(0.1)
            # self.dump_mem()
        print('Exited.')


def find_procs_by_name(name):
    "Return a list of processes matching 'name'."
    ls = []
    for p in psutil.process_iter(["name", "exe", "cmdline"]):
        if name == p.info['name'] or \
                p.info['exe'] and os.path.basename(p.info['exe']) == name or \
                p.info['cmdline'] and p.info['cmdline'][0] == name:
            ls.append(p)
    return ls


def kill_dolphin_instances():
    for instance in find_procs_by_name("Dolphin.exe"):
        instance.terminate()


if __name__ == '__main__':

    parser = ArgumentParser()
    for opt in Main.full_opts():
        opt.update_parser(parser)
    args = parser.parse_args()

    main = Main(**args.__dict__)

    try:
        main()
        main.exit()
    except EOFError as e:
        print(e)
        kill_dolphin_instances()
        exit()
