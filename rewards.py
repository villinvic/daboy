import numpy as np
import tensorflow as tf

sign = lambda x: (x > 0) - (x < 0)

def compute_damages(player, states):
    dmg = states[-1].players[player].percent - states[0].players[player].percent
    return np.clip(dmg, 0.0, 20.0) # reward more for small rewards


def is_dead(p_num, states):  # 2 states
    return float(
        (states[-1].players[p_num].action_state.value <= 0xA) and (states[0].players[p_num].action_state.value > 0xA))


"""
def isTeching(p_num):
    if .status_hist[p_num] == 0:
        for state in self.state_queue:
            if 0x00C7 <= state.players[p_num].action_state.value <= 0x00CC:
                if self.id == 0:
                    print('player', p_num, 'techd.')
                self.status_hist[p_num] = 55
                return True
    return False


def isRespawning(p_num):
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

"""


def compute_all_rewards(states, mode, damage_ratio=0.01, distance_ratio=0.0003, loss_intensity=1, death_scale=1.0):
    if mode == 1:
        p1 = 0
        p2 = 1
    else:
        p1 = 1
        p2 = 0

    rews = np.zeros((states.shape[0] - 1,), dtype=np.float32)
    for i in range(states.shape[0] - 1):
        rews[i] = compute_rewards(states[i: i + 2], p1, p2, damage_ratio=damage_ratio, distance_ratio=distance_ratio,
                                  loss_intensity=loss_intensity)

        rews[i] += death_scale * (is_dead(p1, states[i: i + 2]) - is_dead(p2, states[i: i + 2]))

    return rews

no_dmg_counter = [0,0]

def compute_rewards(states, p1, p2, damage_ratio, distance_ratio, loss_intensity):
    global no_dmg_counter
    """Computes rewards from a list of state transitions.

    Args:
      states: current states
      enemies: The list of pids on the enemy team.
      allies: The list of pids on our team.
      damage_ratio: How much damage (percent) counts relative to stocks.
      distance_ratio: How much distance counts
      loss_intensity: does loosing suck ?
      velocity_ratio: make if flashy
      recovery_ratio: reward if goes up when low on y axis
    Returns:
      A length T numpy array with the rewards on each transition.
    """

    if distance_ratio == 0.0:
        distance_rwd = 0
        d2 = distance(states[-1])
    else:
        dx = sign(states[-1].players[p2].pos_x - states[-1].players[p1].pos_x)
        dy = sign(states[-1].players[p2].pos_y - states[-1].players[p1].pos_y)
        momentum_x = states[-1].players[p2].self_air_vel_x+states[-1].players[p2].speed_ground_x_self
        momentum_y = states[-1].players[p2].self_air_vel_y
        dist = np.sqrt(np.square(states[-1].players[p2].pos_x-states[0].players[p2].pos_x) + np.square(states[-1].players[p2].pos_y-states[0].players[p2].pos_y))
        dist = np.clip(dist, 0.0, 10.0)
        if dx * momentum_x + dy * momentum_y < 0 :
                distance_rwd = dist
        else:
                #print('away')
                distance_rwd = -dist

    # elif dy * players[1][-1].self_air_vel_y > 0:
    #    distance_rwd = 0
    # self_dx = abs(players[1][-1].pos_x - players[1][0].pos_x)
    # self_dy = abs(players[1][-1].pos_y - players[1][0].pos_y)
    # if self_dx + self_dy == 0:
    #    distance_rwd = 0

    # falling_loss = falling_penalty( players[1]) * recovery_ratio
    dmg_reward = compute_damages(p1, states) - compute_damages(p2, states)
    total = distance_rwd*distance_ratio + dmg_reward*damage_ratio
    # total = distance_rwd + sum(losses[p] for p in enemies) - (sum(losses[p] for p in allies))
    #if total < 0:
    #    total *= loss_intensity
    return total


def falling_penalty(player):
    if player[-1].pos_y > -9:
        return 0.0
    else:
        # print("Has to recover !")
        return abs(player[-1].pos_y) / 110.0


def distance(state):
    players = state.players
    x0 = players[0].pos_x
    y0 = players[0].pos_y
    x1 = players[1].pos_x
    y1 = players[1].pos_y

    dx = x1 - x0
    dy = y1 - y0

    return np.sqrt(np.square(dx) + np.square(dy))
