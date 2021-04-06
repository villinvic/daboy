import numpy as np
import tensorflow as tf

sign = lambda x: (x > 0) - (x < 0)


def compute_damages(player, states):
    dmg = states[-1].players[player].percent - states[0].players[player].percent
    return np.clip(dmg, 0.0, 20.0)  # reward more for small rewards


def is_dead(p_num, states):  # 2 states
    return float(
        (states[-1].players[p_num].action_state.value <= 0xA) and (states[0].players[p_num].action_state.value > 0xA))


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


no_dmg_counter = [0, 0]


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

    momentum_x = states[-1].players[p2].self_air_vel_x + states[-1].players[p2].speed_ground_x_self
    if distance_ratio == 0.0:
        distance_rwd = 0
    else:
        dx = sign(states[-1].players[p2].pos_x - states[-1].players[p1].pos_x)
        dy = sign(states[-1].players[p2].pos_y - states[-1].players[p1].pos_y)
        momentum_y = states[-1].players[p2].self_air_vel_y
        dist = np.sqrt(np.square(states[-1].players[p2].pos_x - states[0].players[p2].pos_x) + np.square(
            states[-1].players[p2].pos_y - states[0].players[p2].pos_y))
        dist = np.clip(dist, 0.0, 10.0)
        if dx * momentum_x + dy * momentum_y < 0:
            distance_rwd = dist
        else:
            # print('away')
            distance_rwd = -dist

    edge_guard_boost = 1.0
    combo_bonus = 1.0
    damaged_bonus = 1.0
    dist_from_center = np.abs(states[-1].players[p2].pos_x)
    dist_below_center = - (states[-1].players[p2].pos_y + 1)
    if dist_from_center > 59 or dist_below_center > 0 and not states[-1].players[p2].on_ground:
        edge_guard_boost = 1.2  # np.clip((dist_from_center - 57) / 40.0, 0, 2)
    if states[0].players[p1].hitstun >= 1:
        combo_bonus = 1.25
    p = states[-1].players[p2].percent
    if p > 250:
        damaged_bonus *= 2.0
    elif p > 80:
        damaged_bonus *= 1 + (p - 80.0) / 170.0
        
    
    if  0x0023 <= states[-1].players[p2].action_state.value <= 0x0025 :
        fall_special_penalty = 0.02
    
    else :
        fall_special_penalty = 0
    
    dmg_reward = compute_damages(p1, states) * edge_guard_boost * combo_bonus * damaged_bonus - compute_damages(p2, states)
    total = distance_rwd * distance_ratio + dmg_reward * damage_ratio
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
