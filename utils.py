import numpy as np
from threading import Thread
import platform

def make_async(xs):
    '''
    used to start pad pipes in parallel,
    should improve dolphins side to avoid using that.
    '''
    threads = len(xs) * [None]
    for i,x in enumerate(xs):
            threads[i] = Thread(target=x.connect)
            threads[i].start()


def load_player_info(s, observation, player_num, offset=0):
    '''
    Updates the observation space with the state s. player_num is the current player.
    '''

    p = s.players[player_num]
    
    observation[offset] = np.clip(np.float32(p.action_frame) / 50.0, 0.0, 1.0)
    observation[offset + 1: offset + 1 + (383+1)] = 0.0

    if p.action_state.value > 383 :
        observation[offset+1 + 383] = 1.0
    else :
        observation[offset+1 + p.action_state.value] = 1.0

    # move this to Player object ?
    observation[offset + 1 + (383+1)] = np.clip(np.float32(p.attack_vel_x)/2.0, -5.0, 5.0)
    observation[offset + 1 + (383 + 1) + 1] = np.clip(np.float32(p.attack_vel_y) / 2.0, -5.0, 5.0)
    observation[offset + 1 + (383 + 1) + 2] = np.clip(np.float32(p.body_state.value) / 2.0, 0.0, 5.0)
    observation[offset + 1 + (383 + 1) + 3] = np.clip(np.float32(p.facing), -1.0, 1.0)
    observation[offset + 1 + (383 + 1) + 4] = np.clip(np.float32(p.hitlag) / 10.0, 0.0, 5.0)
    observation[offset + 1 + (383 + 1) + 5] = np.clip(np.float32(p.hitstun) / 10.0, 0.0, 5.0)
    observation[offset + 1 + (383 + 1) + 6] = np.clip(np.float32(p.jumps_used), 0.0, 5.0)
    observation[offset + 1 + (383 + 1) + 7] = np.clip(np.float32(p.on_ground), 0.0, 5.0)
    observation[offset + 1 + (383 + 1) + 8] = np.clip(np.float32(p.shield_size) / 100.0, 0.0, 5.0)
    observation[offset + 1 + (383 + 1) + 9] = np.clip(np.float32(p.percent) / 100.0, 0.0, 5.0)
    observation[offset + 1 + (383 + 1) + 10] = np.clip(np.float32(p.pos_x) / 100.0, -5.0, 5.0)
    observation[offset + 1 + (383 + 1) + 11] = np.clip(np.float32(p.pos_y) / 100.0, -5.0, 5.0)
    observation[offset + 1 + (383 + 1) + 12] = np.clip(np.float32(p.self_air_vel_x) / 2.0, -5.0, 5.0)
    observation[offset + 1 + (383 + 1) + 13] = np.clip(np.float32(p.self_air_vel_y) / 2.0, -5.0, 5.0)
    observation[offset + 1 + (383 + 1) + 14] = np.clip(np.float32(p.speed_ground_x_self) / 2.0, -5.0, 5.0)
    observation[offset + 1 + (383 + 1) + 15] = np.clip(np.float32(p.charging_smash) / 2.0, 0.0, 5.0)


def update_observation(state, observation, p1=0, p2=1, last_action_id=0, action_space_len=1):

    # Euclidian distance between p1 and p2
    dist = np.clip(np.sqrt(np.square((state.players[1].pos_x - state.players[0].pos_x)/100.0) +
                           np.square((state.players[1].pos_y - state.players[0].pos_y)/100.0)), 0.0, 5.0)
    
    load_player_info(state, observation, player_num=p1, offset=0)
    load_player_info(state, observation, player_num=p2, offset=401)
    
    observation[802: 802 + action_space_len] = 0.0
    observation[802 + last_action_id] = 1.0
    observation[802 + action_space_len] = dist

    # Filter bad values
    np.nan_to_num(observation, False)
