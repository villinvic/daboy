import math
import Pad
from time import sleep
from movie import Movie, tapA, endless_play, stages, lvl9, color
from State import Character

characters = dict(
    fox=(-23.5, 11.5),
    falco=(-30, 11),
    falcon=(18, 18),
    roy=(18, 5),
    marth=(11, 5),
    zelda=(11, 11),
    sheik=(11, 11),
    mewtwo=(-2, 5),
    luigi=(-16, 18),
    mario=(-23.5, 17),
    doc=(-30.5, 17),
    puff=(-10, 5),
    kirby=(-2, 11),
    peach=(-2, 18),
    ganon=(23, 18),
    samus=(3, 11),
    bowser=(-9, 19),
    yoshi=(5, 18),
    dk=(11, 18),
)

settings = (0, 24)
p1_cpu = [-30, -1.8]
p1_level = (-30, -12)
p1_char = characters['fox']


def locateCSSCursor(pid):
    def locate(state):
        player = state.players[pid]
        return (player.cursor_x, player.cursor_y)

    return locate


class Action:
    def __init__(self, action):
        self.action = action
        self.acted = False

    def done(self):
        return self.acted

    def move(self, state):
        self.action()
        self.acted = True


class Sequential:
    def __init__(self, *actions):
        self.actions = actions
        self.index = 0

    def move(self, state):
        if not self.done():
            action = self.actions[self.index]
            if action.done():
                self.index += 1
            else:
                action.move(state)

    def done(self):
        return self.index == len(self.actions)


class Parallel:
    def __init__(self, *actions):
        self.actions = actions
        self.complete = False

    def move(self, state):
        self.complete = True
        for action in self.actions:
            if not action.done():
                action.move(state)
                self.complete = False

    def done(self):
        return self.complete


class MoveTo:
    def __init__(self, target, locator, pad, relative=False):
        self.target = target
        self.locator = locator
        self.pad = pad
        self.reached = False
        self.relative = relative

    def move(self, state):
        x, y = self.locator(state)

        if self.relative:
            self.target[0] += x
            self.target[1] += y
            self.relative = False

        dx = self.target[0] - x
        dy = self.target[1] - y
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 0.6:
            self.pad.tilt_stick(Pad.Stick.MAIN, 0.5, 0.5)
            self.reached = True
        else:
            self.pad.tilt_stick(Pad.Stick.MAIN, 0.4 * (dx / (mag + 2)) + 0.5, 0.4 * (dy / (mag + 2)) + 0.5)
            self.reached = False

    def done(self):
        return self.reached


class MenuManager:
    def __init__(self):
        self.cntr = 0
        self.locator = None
        self.actions = Sequential([])

    def setup_move(self, pad, player, char='mario', cpu=True):
        opp = 0
        locator = locateCSSCursor(player)
        opp_locator = locateCSSCursor(opp)
        pick_chars = []

        actions = []
        if cpu:
            actions.append(MoveTo([0, 20], opp_locator, pad[0], True))
            actions.append(Movie(tapA, pad[0]))
            actions.append(Movie(tapA, pad[0]))
            actions.append(MoveTo([0, -14], opp_locator, pad[0], True))
            actions.append(Movie(tapA, pad[0]))
            actions.append(Movie(lvl9, pad[0]))
            actions.append(Movie(tapA, pad[0]))

            actions.append(MoveTo(characters[char], opp_locator, pad[0]))
            actions.append(Movie(tapA, pad[0]))

        else:
            actions.append(MoveTo(characters[char], opp_locator, pad[0]))
            actions.append(Movie(tapA, pad[0]))

        pick_chars.append(Sequential(*actions))

        actions = []

        actions.append(MoveTo(characters[char], locator, pad[1]))
        actions.append(Movie(tapA, pad[1]))
        actions.append(Movie(color, pad[1]))

        pick_chars.append(Sequential(*actions))

        pick_chars = Parallel(*pick_chars)

        enter_settings = Sequential(
            MoveTo(settings, opp_locator, pad[0]),
            Movie(tapA, pad[0])
        )

        start_game = Movie(endless_play + stages["battlefield"], pad[0])

        all_actions = [pick_chars, enter_settings, start_game]
        self.actions = Sequential(*all_actions)

    def pick_char(self, state, pad):
        self.actions.move(state)

    def press_start_lots(self, state, pad):
        if state.frame % 2 == 0:
            pad.press_button(Pad.Button.START)
        else:
            pad.release_button(Pad.Button.START)
