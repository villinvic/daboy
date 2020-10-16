import itertools as IT
import tempfile
import os
import errno
import numpy as np
from State import UsefulActionState
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

map_np_state = np.vectorize( get_np_state)
map_action = np.vectorize( get_action)
map_done = np.vectorize( get_done)
map_rew = np.vectorize( get_rew)



def player2list(s, player_num):
    array = []
    p = s.players[player_num]
    fields = [
        np.float32(p.action_frame) / 50.0,
        [(1.0 if i == p.action_state.value else 0.0) for i in range(383 + 1)],
        np.float32(p.attack_vel_x),
        np.float32(p.attack_vel_y),
        np.float32(p.body_state.value) / 2.0,
        # p.character.value,
        np.float32(p.facing),
        np.float32(p.hitlag) / 1.0,
        np.float32(p.hitstun) / 1.0,
        np.float32(p.jumps_used),
        np.float32(p.on_ground),
        np.float32(p.shield_size) / 100.0,
        np.float32(p.percent) / 100.0,
        np.float32(p.pos_x) / 10.0,
        np.float32(p.pos_y) / 10.0,
        np.float32(p.self_air_vel_x) / 1.0,
        np.float32(p.self_air_vel_y) / 1.0,
        np.float32(p.speed_ground_x_self) / 1.0,
    ]

    for f in fields:
        # onehots
        if isinstance(f, list):
            array.extend(f)
        else:
            array.append(np.clip(f, -10, 10))

    return array


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

from MeleeEnv import MeleeEnv
def convertState2Array(state, p1=0, p2=1, last_action_id=0):
    #last_action = [(np.float32(1.0) if i == last_action_id else np.float32(0.0)) for i in range(MeleeEnv.action_space.len)]
    #dist = np.sqrt(np.square(state.players[1].pos_x - state.players[0].pos_x) + np.square((state.players[1].pos_y - state.players[0].pos_y)**2))

    array = player2list(state, player_num=p1) + player2list(state, player_num=p2) + state_info2list( state)
    #print(array)
    return array
