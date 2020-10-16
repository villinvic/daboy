from threading import Thread, Event
import numpy as np
import tensorflow as tf
from DolphinInitializer import DolphinInitializer
from MemoryWatcher import MemoryWatcher, MemoryWatcherZMQ
from State import State
from StateManager import StateManager
from MenuManager import MenuManager
import State as S
import Pad as P
from utils import convertState2Array, make_async
from rewards import compute_rewards
from MeleeEnv import *
from RERPI import CategoricalActor
import signal

from os import walk
from copy import deepcopy
import time
import zmq
import sys


class ACAgent(CategoricalActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_action(self, state):
        assert isinstance(state, np.ndarray)
        is_single_state = len(state.shape) == self.state_ndim

        state = state[np.newaxis].astype(
            np.float32) if is_single_state else state
        action = self._get_action_body(tf.constant(state))

        return action.numpy()[0] if is_single_state else action

    @tf.function
    def _get_action_body(self, state):
        param = self._compute_dist(state)
        return tf.squeeze(self.dist.sample(param), axis=1)


class Actor:

    def __init__(self, self_play, ckpt_dir, ep_length,
                 dolphin_dir, iso_path, video_backend,
                 actor_id, n_actors, epsilon, n_warmup,
                 char):
                 
        signal.signal(signal.SIGINT, self.exit)
        super().__init__()
        
        self.self_play = self_play
        self.ckpt_dir = ckpt_dir

        self.mw_path = dolphin_dir + 'User/MemoryWatcher/MemoryWatcher'
        self.locations = dolphin_dir + 'User/MemoryWatcher/Locations.txt'
        self.pad_path = dolphin_dir + 'User/Pipes/daboy'
        dolphin_exe = dolphin_dir + 'dolphin-emu-nogui'
        print(dolphin_exe, dolphin_dir, self.mw_path, self.pad_path)

        #self.setDaemon(True)
        self.id = actor_id
        self.n_actors = n_actors
        self.dolphin_proc = DolphinInitializer(self.id, dolphin_exe, iso_path, video_backend)
        self.dolphin_dir = dolphin_dir
        self.action_space = P.Action_Space(char=char)
        self.discrete_action_space = [i for i in range(self.action_space.len)]
        self.shutdown_flag = Event()
        self.steps = 0
        self.epsilon = 0.
        self.current_episode_length = 0
        self.opp_current_episode_length = 0 if self.self_play else None
        env = MeleeEnv(char)
        self.game_start = True
        # self.policy = Agent(MeleeEnv.observation_space, MeleeEnv.action_space.len, epsilon, units=[512, 256])
        # self.opp = Agent(MeleeEnv.observation_space, MeleeEnv.action_space.len, epsilon,
        #                 units=[512, 256]) if self.self_play else None
        self.policy = ACAgent(env.observation_space, env.action_space.len, epsilon)
        
        self.opp = ACAgent(env.observation_space, env.action_space.len, epsilon) if self.self_play else None
        if self.self_play:
            self.checkpoint = tf.train.Checkpoint(actor=self.opp)

        self.n_warmup = n_warmup

        self.mw = MemoryWatcherZMQ(self.mw_path)
        self.state = GameMemory()
        self.sm = StateManager(self.state)
        self.pad = [P.Pad(self.pad_path + "_enemy"), P.Pad(self.pad_path)]
        make_async(self.pad)
        self.mm = MenuManager()
        self.mm.setup_move(self.pad, 1, cpu=not self.self_play, char=char)
        self.activated = False if actor_id == 0 else True

        self.write_locations()
        self.init_state_queue(self.state)

        self.action_list = []
        self.opp_action_list = [] if self.self_play else None
        self.last_action = 0
        self.opp_last_action = 0 if self.self_play else None
        self.last_action_id = 0
        self.opp_last_id = 0 if self.self_play else None
        self.last_action_wait = 3
        self.opp_last_action_wait = 3 if self.self_play else None
        self.reward_latency = 0
        self.status_detect_freq = 60 + 15
        self.action_queue = [0 for _ in range(5)]
        self.status_hist = [0, 0, 0, 0]
        self.opp_status_hist = [0, 0, 0, 0] if self.self_play else None

        self.scores = [np.nan for _ in range(60 * 15)]
        self.scores_hist = []
        self.actions = [0 for _ in range(4)]

        # test
        self.setup_context()
        self.frame_start = -1
        self.episode_score = -1
        self.episode_length = ep_length
        self.check_cntr = 0
        self.gru_reset_freq = 2

        self.bench = [[], []]

        self.communicate_freq = 60 * 10
        # self.trajectory = Trajectory(ep_length, 1)
        self.trajectory = {
            'state': np.array(
                [np.zeros(env.observation_space, dtype=np.float32) for _ in range(self.episode_length)],
                dtype=np.ndarray),
            'action': np.zeros((self.episode_length,), dtype=np.int32),
            'rew': np.zeros((self.episode_length,), dtype=np.float32),
            # 'mode': 1,
        }
        if self.self_play:
            # self.opp_trajectory = Trajectory(ep_length, -1)
            self.opp_trajectory = {
                'state': np.array([np.zeros(env.observation_space, ) for _ in range(self.episode_length)],
                                  dtype=np.ndarray),
                'action': np.zeros((self.episode_length,), dtype=np.int32),
                'rew': np.zeros((self.episode_length,), dtype=np.float32),
                # 'mode': -1,
            }
        else:
            self.opp_trajectory = None


        self.randomize_opp_freq = 60 * 60

        self.restart_freq = 60 * 60 * 60  # every hour

    def restart(self):
        print('Restarting Actor', self.id)
        self.dolphin_proc.close()
        self.state = GameMemory()
        self.sm = StateManager(self.state)
        self.mm.setup_move(self.pad, 1, cpu=not self.self_play)
        self.dolphin_proc.run()

    def write_locations(self):
        locations = self.sm.locations()
        """Writes out the locations list to the appropriate place under dolphin_dir."""
        with open(self.locations, 'w') as f:
            f.write('\n'.join(locations))

    def init_state_queue(self, dummy):
        self.state_queue = [deepcopy(dummy) for _ in range(5)]
        # self.opp_state_queue = [deepcopy(dummy) for _ in range(5)] if self.self_play else None
        if self.self_play:
            self.opp_state_queue = [deepcopy(dummy) for _ in range(5)]
        else:
            self.opp_state_queue = None

    def setup_context(self):
        context = zmq.Context()
        self.exp_socket = context.socket(zmq.PUSH)
        self.exp_socket.connect("tcp://127.0.0.1:5557")
        self.blob_socket = context.socket(zmq.SUB)
        self.blob_socket.connect("tcp://127.0.0.1:5555")
        self.blob_socket.subscribe(b'')
        if not self.self_play:
            self.eval_socket = context.socket(zmq.PUSH)
            self.eval_socket.connect("tcp://127.0.0.1:5556")
        else:
            self.eval_socket = None

    def queue_state(self, players=[0, 1], mode="normal"):
        if mode == "normal":
            del self.state_queue[0]
            self.state_queue.append(deepcopy(self.state))
            frame_dif = self.state.frame - self.state_queue[-2].frame

            # Update status history
            for p_num in players:
                self.update_status_hist(p_num, frame_dif)

        else:
            del self.opp_state_queue[0]
            self.opp_state_queue.append(deepcopy(self.state))

            frame_dif = self.state.frame - self.opp_state_queue[-2].frame

            # Update status history
            for p_num in players:
                self.update_status_hist(p_num, frame_dif, mode=mode)

    def queue_action(self):
        del self.action_queue[0]
        self.action_queue.append(self.last_action_id)

    def update_status_hist(self, p_num, frame_dif, mode="normal"):
        if mode == "normal":
            if self.status_hist[p_num] > 0:
                self.status_hist[p_num] -= frame_dif
                if self.status_hist[p_num] < 0:
                    self.status_hist[p_num] = 0
        else:
            if self.opp_status_hist[p_num] > 0:
                self.opp_status_hist[p_num] -= frame_dif
                if self.opp_status_hist[p_num] < 0:
                    self.opp_status_hist[p_num] = 0

    def isDead(self, p_num, mode="normal"):
        if mode == "normal":
            if self.status_hist[p_num] == 0:
                for state in self.state_queue:
                    if state.players[p_num].action_state.value <= 0xA:
                        if self.id == 0:
                            print('player', p_num, 'is dead.')
                        if state.players[p_num].action_state.value == 0x4:
                            self.status_hist[p_num] = self.status_detect_freq * 3.4
                        else:
                            self.status_hist[p_num] = self.status_detect_freq * 2
                        return True
            return False
        else:
            if self.opp_status_hist[p_num] == 0:
                for state in self.opp_state_queue:
                    if state.players[p_num].action_state.value <= 0xA:
                        if self.id == 0:
                            print('[opp] player', p_num, 'is dead.')
                        if state.players[p_num].action_state.value == 0x4:
                            self.opp_status_hist[p_num] = self.status_detect_freq * 3.4
                        else:
                            self.opp_status_hist[p_num] = self.status_detect_freq * 2
                        return True
            return False

    def isTeching(self, p_num):
        if self.status_hist[p_num] == 0:
            for state in self.state_queue:
                if 0x00C7 <= state.players[p_num].action_state.value <= 0x00CC:
                    if self.id == 0:
                        print('player', p_num, 'techd.')
                    self.status_hist[p_num] = 55
                    return True
        return False

    def isRespawning(self, p_num):
        for state in self.state_queue:
            if state.players[p_num].action_state.value <= 0x0C:
                return True
        return False

    def is_done(self, players=[0, 1]):
        for p_num in players:
            if self.isDead(p_num):
                if self.id == 0:
                    print('Done.')
                return True

        return False

    def isSpecialFalling(self, p_num):

        return (0x0023 <= self.state.players[p_num].action_state.value <= 0x0025) and abs(
            self.state.players[p_num].pos_x) < 60

    def update_params(self, block=False):
        flag = 0 if block else zmq.NOBLOCK
        try:
            params = self.blob_socket.recv_pyobj(flag)
            self.policy.load_params(params)
            if self.self_play:
                self.opp.load_params(params)
        except zmq.ZMQError:
            pass

    def randomize_opp(self):
        checkpoints = []
        try:
            for (_, _, filenames) in walk(self.ckpt_dir):
                for filename in filenames:
                    if "index" in filename:
                        checkpoints.append(self.ckpt_dir + "\\" + filename)
                break
            random_ckpt = np.random.choice(checkpoints)
            status = self.checkpoint.restore(random_ckpt).expect_partial()
        except Exception as e:
            print("Couldn't randomize opp:", e)

    def check_episode(self, mode='normal'):
        if mode == 'normal':
            if self.current_episode_length == self.episode_length:
                if not self.self_play:
                    self.eval_socket.send_pyobj(self.episode_score)
                    self.episode_score = 0
                self.current_episode_length = 0

                # get last params
                self.update_params()
                # send trajectory
                if self.self_play:
                    self.exp_socket.send_pyobj(self.trajectory)

                self.check_cntr += 1
        else:
            if self.opp_current_episode_length == self.episode_length:
                self.opp_current_episode_length = 0

                # send trajectory
                if self.self_play:
                    self.exp_socket.send_pyobj(self.opp_trajectory)

        if (self.current_episode_length + 1) == self.episode_length:
            return True
        return False

    def evaluate(self, death_loss=1.0, p1=0, p2=1, mode="normal"):
        is_off_stage = abs(self.state_queue[-1].players[p2].pos_x) > 60  # fd  60# bf
        off_stage_kill_bonus = 1.4 if is_off_stage else 1.0
        done = self.isDead(p2, mode=mode)
        if done:
            p = self.state_queue[-1].players[p2].percent
            if p > 220 :
                death_loss = 0.2
            elif p > 100:
                death_loss *= (250 - p) / 150.0
                
        r = int(self.isDead(p1, mode=mode)) * off_stage_kill_bonus - death_loss * int(done)  # Death
        # r += int(self.isTeching( 1)) * 0.3
        # r -= int(self.isSpecialFalling( 1)) * 0.03F

        if mode == "normal":
            states = self.state_queue[-2:]
        else:
            states = self.opp_state_queue[-2:]

        r += compute_rewards(states, p1=p1, p2=p2, damage_ratio=0.01, distance_ratio=0.001,
                             loss_intensity=0.98)

        return np.float32(r)

    def toNumpyArray(self, state, p1=0, p2=1, last_action_id=0):
        array = np.array(convertState2Array(state, p1=p1, p2=p2, last_action_id=last_action_id), dtype=np.float32)
        return array

    def choose_action(self, state=None, mode="normal", random=False):
        rand = 0.0 if random else np.random.random()
        self.steps += 1
        if self.n_warmup > self.steps or rand < self.epsilon:
            #  print("random action")
            return np.random.choice(self.action_space.len)

        if mode == "normal":
            return self.policy.get_action(state)
        else:
            return self.opp.get_action(state)

    def store_transition(self, action, np_obs, mode='normal'):
        done = self.check_episode(mode=mode)
        if mode == 'opp':
            r = self.evaluate(p1=1, p2=0, mode=mode)
            # self.opp_trajectory.states[self.opp_current_episode_length] = Transition(np_obs, action, done, r)
            self.opp_trajectory['state'][self.opp_current_episode_length] = np_obs
            self.opp_trajectory['action'][self.opp_current_episode_length] = action
            self.opp_trajectory['rew'][self.opp_current_episode_length] = r
            self.opp_current_episode_length += 1
        else:
            r = self.evaluate( mode=mode)
            # self.trajectory.states[self.current_episode_length] = Transition(np_obs, action, done, r)
            self.trajectory['state'][self.current_episode_length] = np_obs
            self.trajectory['action'][self.current_episode_length] = action
            self.trajectory['rew'][self.current_episode_length] = r
            self.current_episode_length += 1
            if not self.self_play:
                self.episode_score += r


    def mark_state(self, player):
        self.state_queue[-1].mark[player] = 1

    def get_obs(self, player, delay=0, last_action_id=0):
        if player == 1:
            return self.toNumpyArray(self.state, p1=0, p2=1, last_action_id=last_action_id)
        else:
            return self.toNumpyArray(self.state, p1=1, p2=0, last_action_id=last_action_id)

    def advance(self, p1, p2):
        if self.frame_start == -1:
            self.frame_start = self.state.frame - 10 - self.id * 60 * 3

        # if self.self_play and (self.state.frame - self.frame_start) % self.randomize_opp_freq == 0:
        #    self.randomize_opp()

        if p1 == 0:
            t = time.time()
            if not self.action_list:
                if self.state.frame - self.last_action >= self.last_action_wait:
                    """DECISION MAKING"""
                    self.queue_state()

                    # AI
                    last_action_id = self.trajectory['action'][self.current_episode_length % self.episode_length]
                    observation = self.get_obs(p2, last_action_id=last_action_id)
                    action_id = self.choose_action(observation)
                    action = self.action_space[action_id]

                    self.store_transition(action_id, observation)

                    # if wavedashing in air, choose random action
                    # if isinstance(action, list):
                    #    if not self.state.players[1].on_ground and (action[1]['L'][0] == 0.8 or action[1]['L'][0] == 0.2):
                    #        action_id = self.choose_action( random=True)
                    #        action = self.action_space[action_id]

                    if isinstance(action, list):  # action chain
                        self.action_list.extend(action)
                    else:
                        self.action_list.append(
                            action
                        )

            if self.self_play and not self.opp_action_list:
                if self.state.frame - self.opp_last_action >= self.opp_last_action_wait:
                    """OPP DECISION MAKING"""
                    self.queue_state(mode="opp")

                    last_action_id = self.opp_trajectory['action'][self.opp_current_episode_length % self.episode_length]
                    observation = self.get_obs(p1, last_action_id=last_action_id)

                    opp_action_id = self.choose_action(observation, mode="opp")
                    opp_action = self.action_space[opp_action_id]

                    self.store_transition(opp_action_id, observation, mode='opp')

                    # if wavedashing in air, choose random action
                    # if isinstance(action, list):
                    #    if not self.state.players[1].on_ground and (action[1]['L'][0] == 0.8 or action[1]['L'][0] == 0.2):
                    #        action_id = self.choose_action( random=True)
                    #        action = self.action_space[action_id]

                    if isinstance(opp_action, list):  # action chain
                        self.opp_action_list.extend(
                            opp_action
                        )
                    else:
                        self.opp_action_list.append(
                            opp_action
                        )
            t1 = time.time()
            if self.action_list and self.state.frame - self.last_action >= self.last_action_wait:
                """ACTING"""
                action = self.action_list[0]

                self.last_action_wait = action['duration']
                self.action_list.pop(0)
                # ban dodges when recovering
                # if (not self.state.players[p2].on_ground) and abs(self.state.players[p2].pos_x) > 59 and action['L'] == 1:
                #    P.neutralPad.send_controller(self.pad[p2])
                # elif abs(self.state.players[p2].pos_x) > 60 and action['B'] == 1 and action['stick'][0] != 0.5 \
                #    and self.state.players[p2].jumps_used > 1:
                #    P.neutralPad.send_controller(self.pad[p2])

                action.send_controller(self.pad[p2])

                self.last_action = self.state.frame
                self.last_action_id = action['id']

            if self.self_play and self.opp_action_list and self.state.frame - self.opp_last_action >= self.opp_last_action_wait:
                """OPP ACTING"""
                opp_action = self.opp_action_list[0]

                self.opp_last_action_wait = opp_action['duration']
                self.opp_action_list.pop(0)
                # ban dodges when recovering
                # if (not self.state.players[p1].on_ground) and abs(self.state.players[p1].pos_x) > 59 and opp_action['L'] == 1:
                #    P.neutralPad.send_controller(self.pad[p1])
                # elif abs(self.state.players[p1].pos_x) > 60 and opp_action['B'] == 1 and opp_action['stick'][0] != 0.5 \
                #    and self.state.players[p1].jumps_used > 1:
                #    P.neutralPad.send_controller(self.pad[p1])

                # else:
                opp_action.send_controller(self.pad[p1])

                self.opp_last_action = self.state.frame
                self.opp_last_action_id = opp_action['id']

            t2 = time.time()
            dt1 = t1 - t
            dt2 = t2 - t1

            # if self.id == 0 and dt1 + dt2 > 0.0:
            #    self.bench[0].append(dt1)
            #    self.bench[1].append(dt2)

    def make_action(self, p1, p2):
        if self.state.menu == S.Menu.Game:
            if self.game_start:
                self.game_start = False
                self.state.frame_start = self.state.frame
                print('Actor', self.id, 'is now playing.')

            self.advance(p1, p2)

        elif self.state.menu == S.Menu.Characters or self.state.menu == S.Menu.Stages:
            self.mm.pick_char(self.state, self.pad)
        # elif self.state.menu == S.Menu.Stages:
        #    # Handle this once we know where the cursor position is in memory.
        #    if p1 == 0:
        #        self.pad[1].tilt_stick(P.Stick.C, 0.5, 0.5)
        #    elif p1 == 1:
        #        pass

        elif self.state.menu == S.Menu.PostGame:
            if p1 == 0:
                self.mm.press_start_lots(self.state, self.pad[1])
            elif p1 == 1:
                pass

    def exit(self, signal=signal.SIGINT, frame=None):
        self.shutdown_flag.set()
        self.dolphin_proc.close()
        print("Actor", self.id, "Exited.")
        sys.exit()
        self.mw.unbind()
        for p in self.pad:
            if p is not None:
                p.unbind()


    def main_loop(self):
        last_frame = self.state.frame
        res = next(self.mw)
        if res is not None:
            self.sm.handle(res)


        if self.state.frame > last_frame:
            if self.state.frame - last_frame > 1:
                print('Actor', self.id, 'skipped', self.state.frame - last_frame, 'frames.')
            self.make_action(0, 1)

        self.mw.advance()

    def run(self):
        # Copying nn from learner
        # self.update_params(block=True)
        # if self.self_play:
        #    self.randomize_opp()
        #    act2 = self.opp.get_action(self.get_obs(0))
        # test
        act = self.policy.get_action(self.get_obs(1))
        print(act)
        print(act)
        if self.self_play:
            act = self.opp.get_action(self.get_obs(1))
        # Start bounded dolphin process
        self.dolphin_proc.run()

        # wait dolphin
        time.sleep(2)

        while not self.shutdown_flag.is_set():
            # print("Actor", self.id)
            self.main_loop()


        # if self.id == 0:
        #    print(np.mean(self.bench[0]), np.mean(self.bench[1]))
