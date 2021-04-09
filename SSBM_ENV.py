
from DolphinInitializer import DolphinInitializer
from MemoryWatcher import MemoryWatcherZMQ
from State import State
from StateManager import StateManager, GameMemory
from MenuManager import MenuManager
import State as S
import Pad as P
from utils import update_observation, make_async
from rewards import compute_rewards
from Spaces import *
from RL import CategoricalActor

import signal
from os import walk
from copy import deepcopy
import time
import zmq
import sys
from threading import Event
import numpy as np
import tensorflow as tf
import platform

class SSBM_ENV:

    '''
    SSBM environment:
        handles a dolphin instance, emulates the game.
        At least one learning player is attached to the class : cf. SSBM_ENV.policy attribute.

        Once the class is created, run the emulation with the run function :
            it will execute the main_loop function until a SIGINT is received.

        # TODO :
            make it compatible with 2vs2
    '''

    def __init__(self, learner_ip, self_play, enemy_path, ep_length,
                 dolphin_dir, iso_path, video_backend,
                 actor_id, epsilon,
                 char, test):
        self.id = actor_id

        self.windows = platform.system() == 'Windows'

        # Signal handler
        signal.signal(signal.SIGINT, self.exit)
        self.shutdown_flag = Event()

        sep = '\\' if self.windows else '/'
        self.mw_path = sep.join([dolphin_dir, 'User', 'MemoryWatcher', 'MemoryWatcher'])
        self.locations = sep.join([dolphin_dir, 'User', 'MemoryWatcher', 'Locations.txt'])
        self.pad_path = sep.join([dolphin_dir, 'User', 'Pipes', 'daboy'])
        dolphin_exe = sep.join([dolphin_dir, 'dolphin-emu-nogui'])

        # Prepare the dolphin process...
        print(dolphin_exe, dolphin_dir, self.mw_path, self.pad_path)
        self.dolphin_proc = DolphinInitializer(self.id, dolphin_exe, iso_path, video_backend, test)

        self.self_play = self_play
        self.test = test
        self.enemy_path = enemy_path
        self.learner_ip = learner_ip
        self.game_start = True

        self.epsilon = epsilon

        self.steps = 0
        self.current_episode_length = [0] * 2

        self.char = [char] * 2
        if not self_play:
            ckpt = self.randomize_opp()
            self.is_cpu = (ckpt == 'cpu')
            if self.is_cpu:
                self.char[0]= None
        else :
            self.is_cpu = False

        self.action_space = [P.Action_Space(char=self.char[i]) for i in range(2)]
        spaces = [Spaces(self.char[i]) for i in range(2)]

        # Here are built NNs for the players
        self.policy = [CategoricalActor(spaces[i].observation_space, 1, ep_length-1, self.action_space[i].len, epsilon)
                       for i in range(2)]
        if not self_play and not self.is_cpu:
            # restore the values of the NN for the opponent if loaded from the enemy folder
            self.checkpoint = tf.train.Checkpoint(actor=self.policy[0])
            self.checkpoint.restore(ckpt).expect_partial()


        self.mw = MemoryWatcherZMQ(self.mw_path, self.id)
        self.state = GameMemory()
        self.sm = StateManager(self.state, test or video_backend != "Null")
        self.mm = MenuManager()
        if self.windows:
            self.pad = [P.Pad(self.pad_path + "_enemy", 5559), P.Pad(self.pad_path, 5560)]
        else:
            self.pad = [P.Pad(self.pad_path + "_enemy"), P.Pad(self.pad_path)]
        # Start the piping with dolphin
        make_async(self.pad)

        self.write_locations()
        self.init_state_queue(self.state)

        self.action_list = [[], []]
        self.last_action = [0] * 2
        self.last_action_id = [0] * 2
        self.last_action_wait = [3] * 2
        self.status_detect_freq = 60 + 15

        self.status_hist = [[0, 0, 0, 0], [0, 0, 0, 0]]

        self.setup_context()
        self.frame_start = -1
        self.init_frame = 0
        self.episode_score = 0
        self.episode_length = ep_length
        self.check_cntr = 0

        self.observation = [np.zeros(spaces[i].observation_space, dtype=np.float32) for i in range(2)]

        self.trajectory = [{
            'state': np.array(
                [np.zeros(spaces[i].observation_space, dtype=np.float32) for _ in range(self.episode_length)],
                dtype=np.ndarray),
            'action': np.zeros((self.episode_length,), dtype=np.int32),
            'rew': np.zeros((self.episode_length,), dtype=np.float32),
            # 'mode': 1,
            'r_state': None,
        } for i in range(2)]

        self.recurrent_state = [None, None]

        self.neg_scale = 0.9
        self.dist_scale = 0.
        self.dmg_scale = 0.01

        self.max_reward_bug = 20 if not self.self_play else 50
        self.reward_bug_counter = 0

        self.ttt = 0  # debugging counter

    def write_locations(self):
        locations = self.sm.locations()
        """Writes out the locations list to the appropriate place under dolphin_dir."""
        with open(self.locations, 'w') as f:
            f.write('\n'.join(locations))

    def init_state_queue(self, dummy):
        self.state_queue = [[deepcopy(dummy) for _ in range(5)] for _ in range(2)]

    def setup_context(self):
        '''
        Connects to the Learner process,
        and launcher process (in order to signal if encountered a bug)
        '''
        context = zmq.Context()
        self.alert_socket = context.socket(zmq.PUSH)
        self.alert_socket.connect("tcp://127.0.0.1:7555")
        self.exp_socket = context.socket(zmq.PUSH)
        self.exp_socket.connect("tcp://%s:5557" % self.learner_ip) # 157.16.63.57
        self.blob_socket = context.socket(zmq.SUB)
        self.blob_socket.connect("tcp://%s:5558" % self.learner_ip)
        self.blob_socket.subscribe(b'')
        if not self.self_play:
            self.eval_socket = context.socket(zmq.PUSH)
            self.eval_socket.connect("tcp://%s:5556" % self.learner_ip)
        else:
            self.eval_socket = None

    def queue_state(self, player_num):
        '''
        update the state queue
        '''

        del self.state_queue[player_num][0]
        self.state_queue[player_num].append(deepcopy(self.state))
        frame_dif = self.state.frame - self.state_queue[player_num][-2].frame

        # Update status history
        for p_num in range(2):
            self.update_status_hist(player_num, p_num, frame_dif)

    def update_status_hist(self, player, p_num, frame_dif):
        if self.status_hist[player][p_num] > 0:
            self.status_hist[player][p_num] -= frame_dif
            if self.status_hist[player][p_num] < 0:
                self.status_hist[player][p_num] = 0

    def is_dead(self, player, p_num):
        if self.status_hist[player][p_num] == 0:
            for state in self.state_queue[player]:
                if state.players[p_num].action_state.value <= 0xA:
                    if self.id == 0:
                        print('player', p_num, 'is dead.')
                    if state.players[p_num].action_state.value == 0x4:
                        self.status_hist[player][p_num] = self.status_detect_freq * 3.4
                    else:
                        self.status_hist[player][p_num] = self.status_detect_freq * 2
                    return True
        return False

    def update_params(self, block=False):
        '''
        communicate with the Learner to update the NN's version
        '''
        flag = 0 if block else zmq.NOBLOCK
        try:
            params = self.blob_socket.recv_pyobj(flag)
            self.policy[1].load_params(params['weights'])
            if self.self_play:
                self.policy[0].load_params(params['weights'])
            self.neg_scale = params['neg_scale']
            self.dist_scale = params['dist_scale']
            self.dmg_scale = params['dmg_scale']
        except zmq.ZMQError:
            return False
        return True

    def randomize_opp(self):
        '''
        Checks in the enemy path (daboy/enemies is the default),
        and picks a random enemy, if the cpu folder is chosen, an in-game cpu is the opponent
        '''

        checkpoints = []
        chars = []
        try:
            for (_, dirs, _) in walk(self.enemy_path):
                for dir in dirs:
                    checkpoints.append(dir)
                    chars.append(dir.split('--')[0])
                break
            random_index = np.random.randint(0, len(checkpoints))
            random_ckpt = checkpoints[random_index]
            print(random_ckpt)
            if random_ckpt == 'cpu':
                self.opp_char = None
                return 'cpu'
            self.char[0] = chars[random_index]
            for (_, _, files) in walk(self.enemy_path + "/" + random_ckpt):
                for file in files:
                    return self.enemy_path + "/" + random_ckpt + "/" + file.split('.')[0]

        except Exception as e:
            print("Couldn't load opp:", e)

        return None

    def check_episode(self, player):
        '''
        Check if the episode is finished,
        if it is, send the experience and update the NN
        '''

        if self.current_episode_length[player] == self.episode_length:
            if not self.self_play:
                self.eval_socket.send_pyobj(self.episode_score)
                print(self.episode_score)
            if self.episode_score == 0:
                self.reward_bug_counter += 1
            else:
                self.reward_bug_counter = 0
            if self.reward_bug_counter == self.max_reward_bug:
                print("BUG????, Restarting...")
                self.alert_socket.send_pyobj(self.id)

            self.episode_score = 0
            self.current_episode_length[player] = 0

            # get last params
            if player == 1 :
                self.update_params()
            # send trajectory
            if self.steps > 20:
                self.exp_socket.send_pyobj(self.trajectory[player])

            self.check_cntr += 1

    def evaluate(self, death_loss=1.0, p1=0, p2=1):
        '''
        computes the reward between the last two frames
        '''

        is_off_stage = abs(self.state_queue[p2][-1].players[p2].pos_x) > 60 or self.state_queue[p2][-1].players[
            p2].pos_y < -1  # fd  60# bf
        off_stage_kill_bonus = 1.2 if is_off_stage else 1.0
        done = self.is_dead(p2, p2)
                
        r = int(self.is_dead(p2, p1)) * off_stage_kill_bonus - int(done) * death_loss  # Death

        states = self.state_queue[p2][-2:]

        r += compute_rewards(states, p1=p1, p2=p2, damage_ratio=self.dmg_scale, distance_ratio=self.dist_scale,
                             loss_intensity=self.neg_scale)

        return np.nan_to_num(np.float32(r), True)

    def choose_action(self, player, state=None):
        if player == 1:
            self.steps += 1

        return self.policy[player].get_action(state)

    def store_transition(self, player, action, np_obs):
        other = (player + 1) % 2

        done = self.check_episode(player)
        r = self.evaluate(p1=other, p2=player)

        if self.current_episode_length[player] == 0:
            self.trajectory[player]['r_state'] = self.recurrent_state[player]
        self.trajectory[player]['state'][self.current_episode_length[player]] = np_obs
        self.trajectory[player]['action'][self.current_episode_length[player]] = action
        self.trajectory[player]['rew'][self.current_episode_length[player]] = r
        self.current_episode_length[player] += 1

        if player == 1:
            self.episode_score += r

    def update_obs(self, player, last_action_id=0):
        '''
        Update the observation of the game
        '''
        other = (player + 1) % 2
        update_observation(self.state, self.observation[player], p1=other, p2=player, last_action_id=last_action_id,
                           action_space_len=self.action_space[player].len)

    def advance(self):
        if self.frame_start == -1:
            self.frame_start = self.state.frame - 10 - self.id * 60 * 3

        for player_num in range(2):
            if player_num == 0 and self.is_cpu:
                pass
            else:
                if not self.action_list[player_num] and self.state.frame - self.last_action[player_num] \
                        >= self.last_action_wait[player_num]:

                    """DECISION MAKING"""
                    self.queue_state(player_num)

                    # AI
                    self.update_obs(player_num, last_action_id=self.last_action_id[player_num])
                    action_id, r_state = self.choose_action(player_num, self.observation[player_num])

                    if action_id == self.action_space[player_num].len:
                        print('error action nan ? ', self.id)
                        action_id = 0
                        self.policy[player_num].l1.reset_states()

                    a = self.action_space[player_num][action_id]

                    if not (player_num == 0 and not self.self_play):
                        self.store_transition(player_num, action_id, deepcopy(self.observation[player_num]))
                    self.recurrent_state[player_num] = r_state

                    if isinstance(a, list):  # action chain
                         self.action_list[player_num].extend(a)
                    else:
                        self.action_list[player_num].append(
                            a
                        )

                if self.action_list[player_num] and self.state.frame - self.last_action[player_num] >=\
                        self.last_action_wait[player_num]:
                    """ACTING"""
                    action = self.action_list[player_num].pop(0)

                    self.last_action_wait[player_num] = action['duration']
                    if not action.no_op:
                        action.send_controller(self.pad[player_num])

                    self.last_action[player_num] = self.state.frame
                    self.last_action_id[player_num] = action['id']

    def make_action(self):
        if self.state.menu == S.Menu.Game or self.state.menu == S.Menu.Stages:
            if self.game_start:
                self.game_start = False
                self.state.frame_start = self.state.frame
                print('SSBM_ENV', self.id, 'is now playing.')

            self.advance()

        elif self.state.menu == S.Menu.Characters or self.state.menu == S.Menu.Stages:
            self.mm.press_start_lots(self.state, self.pad[1])

        elif self.state.menu == S.Menu.PostGame:
            self.mm.press_start_lots(self.state, self.pad[1])

    def exit(self, signal=signal.SIGINT, frame=None):
        self.shutdown_flag.set()
        self.dolphin_proc.close()
        print("SSBM_ENV", self.id, "Exited.")
        sys.exit()

    def main_loop(self):
        last_frame = self.state.frame

        res = next(self.mw)
        if res is not None:
            # Update the state
            self.sm.handle(res)
        if (self.test and self.init_frame < 120) or self.init_frame < 240:
                self.init_frame += 1
        elif self.state.frame > last_frame:
            self.ttt = 0
            if last_frame % (60 * 5 * 30) == 0:
                print(self.id, 'running')
            if self.state.frame - last_frame > 1:
                print('SSBM_ENV', self.id, 'skipped', self.state.frame - last_frame, 'frames.')
            self.make_action()
        else :
                self.ttt += 1
                if self.ttt > 100:
                        print('stuck')
        

        self.mw.advance()

    def run(self):

        # Start the dolphin proc
        self.dolphin_proc.run(*self.char)

        tries = 15
        # Init values, needed because of eager execution
        act, self.recurrent_state[1] = self.policy[1].get_action(self.observation[1])
        if not self.is_cpu:
            act, self.recurrent_state[0] = self.policy[0].get_action(self.observation[0])

        # Copying nn from Learner
        if self.id == 0:
                print('getting params from learner...')
        for _ in range(tries):
                if self.update_params():
                        break
                time.sleep(1)

        # wait dolphin
        time.sleep(2)

        while not self.shutdown_flag.is_set():
            # print("SSBM_ENV", self.id)
            self.main_loop()
