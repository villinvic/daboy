import Pad
import tensorflow as tf
import numpy as np
from dataclasses import dataclass, field
from typing import *
from State import *

# traj_length = 30
ep_length = 60 * 4


class MeleeEnv:

        def __init__(self, char='mario'):
                self.action_space = Pad.Action_Space(char=char)
                self.observation_space = (800,)


@dataclass
class PlayerMemory:
    percent: np.int32
    facing: np.float32
    pos_x: np.float32
    pos_y: np.float32
    action_state: np.int32
    action_counter: np.int32
    action_frame: np.float32
    character: np.int32
    body_state: np.bool
    hitlag: np.float32
    hitstun: np.float32
    jumps_used: np.int32
    charging_smash: np.bool
    on_ground: np.bool
    self_air_vel_x: np.float32
    speed_ground_x_self: np.float32
    self_air_vel_y: np.float32
    attack_vel_x: np.float32
    attack_vel_y: np.float32
    shield_size: np.float32
    cursor_x: np.float32
    cursor_y: np.float32

    def __init__(self):
        return


def player_fac():
    return [PlayerMemory(), PlayerMemory()]


@dataclass
class GameMemory:
    frame: np.int32
    menu: np.int32
    stage: np.int32
    players: List[PlayerMemory] = field(default_factory=player_fac())

    # stage select screen
    # sss_cursor_x: np.float32
    # sss_cursor_y: np.float32

    def __init__(self):
        self.players = player_fac()
        return


@dataclass
class Transition:
    state: np.array  # GameMemory
    action: np.int32
    done: np.bool
    rew: np.float32


def traj_fac(traj_length):
    return np.array([Transition(GameMemory(), 0, False, 0) for _ in range(traj_length)])


@dataclass
class Trajectory:
    states: np.ndarray
    type: np.int32

    def __init__(self, traj_length, mode):
        self.states = traj_fac(traj_length)
        self.type = mode


def exp_fac():
    return [Trajectory() for _ in range(3000)]


@dataclass
class Experience:
    exp: List[Trajectory] = field(default_factory=exp_fac)

    def __init__(self):
        self.exp = exp_fac()
