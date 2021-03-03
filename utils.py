import itertools as IT
import tempfile
import os
import errno
import numpy as np
from threading import Thread


def make_async(xs):
        threads = len(xs) * [None]
        for i,x in enumerate(xs):
                threads[i] = Thread(target=x.connect)
                threads[i].start()
        
 

def uniquify(path, sep=''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s=sep, n=next(count))

    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir=dirname, prefix=filename, suffix=ext)
        tempfile._name_sequence = orig
    return filename

def get_np_state( transition):
    return transition.state

def get_action( transition):
    return transition.action

def get_done( transition):
    return transition.done

def get_rew( transition):
    return transition.rew
    
    
def set_0( value):
        return 0


'''
def sym_action_b( action):
        return action_space.sym[action]


def sym_state_b( state):
        last_action = state[-MeleeEnv.action_space.len-1:-1]
        last_action_id = np.argwhere(last_action==1)
        last_action[last_action_id] = 0.0
        last_action[action_space.sym[last_action_id[0][0]]] = 1.0
        state[-MeleeEnv.action_space.len-1:-1] = last_action
        
        state[385] = -state[385]
        state[388] = -state[388]
        state[395] = -state[395]
        state[397] = -state[397]
        state[399] = -state[399]
        
        state[385+399] = -state[385+399]
        state[388+399] = -state[388+399]
        state[395+399] = -state[395+399]
        state[397+399] = -state[397+399]
        state[399+399] = -state[399+399]
        
        return state
'''
        
# sym_action = np.vectorize( sym_action_b)
reset_0 = np.vectorize( set_0)
map_np_state = np.vectorize( get_np_state)
map_action = np.vectorize( get_action)
map_done = np.vectorize( get_done)
map_rew = np.vectorize( get_rew)


def loadplayerinfo(s, observation, player_num, order , sym=False):
    # array = []
    p = s.players[player_num]
    if sym:
        mirror =  -1.0
    else:
     mirror = 1.0
    if order == 0:
        start_index = 0
    else:
        start_index = 401
    observation[start_index] = np.clip(np.float32(p.action_frame) / 50.0, 0.0, 1.0)
    observation[start_index + 1: start_index + 1 + (383+1)] = 0.0
    observation[start_index+1 + p.action_state.value] = 1.0
    observation[start_index + 1 + (383+1)] = np.clip(np.float32(mirror*p.attack_vel_x)/2.0, -5.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 1] = np.clip(np.float32(p.attack_vel_y) / 2.0, -5.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 2] = np.clip(np.float32(p.body_state.value) / 2.0, 0.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 3] = np.clip(np.float32(mirror*p.facing), -1.0, 1.0)
    observation[start_index + 1 + (383 + 1) + 4] = np.clip(np.float32(p.hitlag) / 10.0, 0.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 5] = np.clip(np.float32(p.hitstun) / 10.0, 0.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 6] = np.clip(np.float32(p.jumps_used), 0.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 7] = np.clip(np.float32(p.on_ground), 0.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 8] = np.clip(np.float32(p.shield_size) / 100.0, 0.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 9] = np.clip(np.float32(p.percent) / 100.0, 0.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 10] = np.clip(np.float32(mirror*p.pos_x) / 100.0, -5.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 11] = np.clip(np.float32(p.pos_y) / 100.0, -5.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 12] = np.clip(np.float32(mirror*p.self_air_vel_x) / 2.0, -5.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 13] = np.clip(np.float32(p.self_air_vel_y) / 2.0, -5.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 14] = np.clip(np.float32(mirror*p.speed_ground_x_self) / 2.0, -5.0, 5.0)
    observation[start_index + 1 + (383 + 1) + 15] = np.clip(np.float32(p.charging_smash) / 2.0, 0.0, 5.0)

def state_info2list(s):
    info = [
        # state.frame,
        # state.menu.value,
        # state.stage.value
    ]
    array = []
    for item in info:
        array.append(np.float32(item))

    return array

def write_with_folder(path, text):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(path, "w") as f:
        f.write(text)

def convertState2Array(state, observation, p1=0, p2=1, last_action_id=0, action_space_len=1, sym=False):
    if sym:
        if state.players[p2].pos_x >= 0:
                sym = False

    dist = np.clip(np.sqrt(np.square((state.players[1].pos_x - state.players[0].pos_x)/100.0) + np.square((state.players[1].pos_y - state.players[0].pos_y)/100.0)), 0.0, 5.0)
    loadplayerinfo(state, observation, player_num=p1, order=0, sym=sym)
    loadplayerinfo(state, observation, player_num=p2, order=1, sym=sym)
    
    observation[802: 802 + action_space_len] = 0.0
    observation[802 + last_action_id] = 1.0
    observation[802 + action_space_len] = dist
    
    np.nan_to_num(observation, False)
