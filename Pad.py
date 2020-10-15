import enum
import zmq
from utils import write_with_folder


@enum.unique
class UsefullButton(enum.Enum):
    A = 0
    B = 1
    X = 2
    Z = 4
    L = 6


@enum.unique
class Button(enum.Enum):
    A = 0
    B = 1
    X = 2
    Y = 3
    Z = 4
    START = 5
    L = 6
    R = 7
    D_UP = 8
    D_DOWN = 9
    D_LEFT = 10
    D_RIGHT = 11


@enum.unique
class Trigger(enum.Enum):
    L = 0
    R = 1


@enum.unique
class Stick(enum.Enum):
    MAIN = 0
    C = 1


class ControllerState(dict):
    def __init__(self, button=None, stick=(0.5, 0.5), c_stick=(0.5, 0.5), duration=3, ac=False):
        dict.__init__(self)
        for item in UsefullButton:
            if button and item.name == button:
                self[button] = 1
            else:
                self[item.name] = 0

        self['stick'] = stick
        self['c_stick'] = c_stick
        self['duration'] = duration
        self['id'] = None
        self['ac'] = ac

    def __str__(self):

        string = ""
        for item in UsefullButton:
            string += "%s(%d) " % (item.name, self[item.name])
        string += "S(%s,%s) " % self['stick']
        string += "C(%s,%s)" % self['c_stick']

        return string

    def make_state(self, button=None, stick=(0.5, 0.5), c_stick=(0.5, 0.5)):  # deprecated
        for item in UsefullButton:
            if button and item.name == button:
                self[button] = 1
            else:
                self[item.name] = 0

        self['stick'] = stick
        self['c_stick'] = c_stick

    def send_controller(self, pad):

        for item in UsefullButton:
            if self[item.name] == 0:
                pad.release_button(item, buffering=True)
            else:
                pad.press_button(item, buffering=True)
        x, y = self['stick']
        pad.tilt_stick(Stick.MAIN, x, y, buffering=True)
        x, y = self['c_stick']
        pad.tilt_stick(Stick.C, x, y, buffering=True)

        pad.flush()


neutralPad = ControllerState()


class Action_Space(dict):

    def add(self, cs):
        if isinstance(cs, list):
            for ccs in cs:
                ccs['id'] = self.len
        else:
            cs['id'] = self.len
        self[self.len] = cs
        self.len += 1

    def __init__(self):
        dict.__init__(self)
        self.stick_states = [
            (0.5, 0.5),
            (1.0, 0.5),
            # (1.0, 0.0),
            # (0.0, 0.0),
            (0.5, 1.0),
            (0.5, 0.0),
            (0.0, 0.5)
        ]

        self.stick_states_upB = [
            (1.0, 1.0),
            # (0.5, 1.0),
            (0.0, 1.0)
        ]

        self.smash_states = [
            # (0.5, 0.5),
            (0.5, 1.0),
            # (0.5, 0.0),
            # (0.0, 0.5),
            # (1.0, 0.5)
        ]

        self.tilt_stick_states = [
            (0.3, 0.5),
            (0.7, 0.5),
            (0.5, 0.3),
            (0.5, 0.7)
        ]

        self.len = 0

        for s_state in self.stick_states:
            for item in UsefullButton:
                if not item.name == 'Z' and not item.name == 'X' and not (item.name == 'B' and s_state[1] != 0.5) \
                        and not (item.name == 'L' and s_state[1] > 0.5):
                    self.add(ControllerState(button=item.name, stick=s_state))

            # for sc_state in self.smash_states:
            #    if s_state != (0.5, 1.0):
            #        self.add(ControllerState(stick=s_state, c_stick=sc_state))

            # no button
            self.add(ControllerState(stick=s_state))

        # Up_air no jump
        self.add(ControllerState(c_stick=(0.5, 1.0)))

        # simple x
        self.add(ControllerState(button='X', duration=4))

        for s_state in self.stick_states_upB:
            self.add(ControllerState(button='B', stick=s_state))

        # Specific techs:

        # tilt

        for s_state in self.tilt_stick_states:
            self.add(ControllerState(button='A', stick=s_state))

        # WAVE LAND

        self.add(ControllerState(button='L', stick=(0.2, 0.3), duration=4))
        self.add(ControllerState(button='L', stick=(0.8, 0.3), duration=4))

        # shield grab
        sg = ControllerState(button='L')
        sg['A'] = 1
        self.add(sg)

        # jumpgrab
        jump1 = ControllerState(button='X', duration=1)
        grab = ControllerState(button='Z', ac=True)
        jgrab = [jump1, grab]
        self.add(jgrab)

        # PERFECT DOWN B
        down_b_start = ControllerState(button='B', stick=(0.5, 0.0), duration=6)
        down_b_left = ControllerState(button='B', stick=(0.0, 0.5), duration=2)
        down_b_right = ControllerState(button='B', stick=(1.0, 0.5), duration=2)
        down_b_left_interrupt = ControllerState(stick=(0.0, 0.5), duration=2)
        down_b_right_interrupt = ControllerState(stick=(1.0, 0.5), duration=2)
        end = ControllerState(duration=1)

        perfect_down_b_left = [down_b_start] + [down_b_left_interrupt, down_b_left] * 10 + [end]
        perfect_down_b_right = [down_b_start] + [down_b_right_interrupt, down_b_right] * 10 + [end]

        # self.add( perfect_down_b_left)
        # self.add( perfect_down_b_right)

        # for i in range(self.len):
        #    if isinstance(self[i], list):
        #        print(i, self[i][0])
        #    else:
        #        print(i, self[i])
        print("Action_space dim:", self.len)


class Pad:
    """Writes out controller inputs."""

    def __init__(self, path, port):
        """Create, but do not open the fifo."""
        self.pipe = None
        self.path = path
        self.context = zmq.Context()
        self.port = port
        self.message = ""
        self.action_space = []

        write_with_folder(self.path, str(self.port))

        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind("tcp://127.0.0.1:%d" % self.port)

    def __exit__(self, *args):
        pass
        
    def unbind(self):
        self.socket.unbind("tcp://127.0.0.1:%d" % self.port)

    def flush(self):
        self.socket.send_string(self.message)
        self.message = ""

    def write(self, command, buffering=False):
        self.message += command
        if not buffering:
            self.flush()

    def press_button(self, button, buffering=False):
        """Press a button."""
        assert button in Button or button in UsefullButton
        self.write('PRESS {}\n'.format(button.name), buffering)

    def release_button(self, button, buffering=False):
        """Release a button."""
        assert button in Button or button in UsefullButton
        self.write('RELEASE {}\n'.format(button.name), buffering)

    def press_trigger(self, trigger, amount, buffering=False):
        """Press a trigger. Amount is in [0, 1], with 0 as released."""
        assert trigger in Trigger or trigger in UsefullButton
        assert 0 <= amount <= 1
        self.write('SET {} {:.2f}\n'.format(trigger.name, amount), buffering)

    def tilt_stick(self, stick, x, y, buffering=False):
        """Tilt a stick. x and y are in [0, 1], with 0.5 as neutral."""
        assert stick in Stick
        assert 0 <= x <= 1 and 0 <= y <= 1
        self.write('SET {} {:.2f} {:.2f}\n'.format(stick.name, x, y), buffering)

    def reset(self):
        for button in Button:
            self.release_button(button)
        for trigger in Trigger:
            self.press_trigger(trigger, 0)
        for stick in Stick:
            self.tilt_stick(stick, 0.5, 0.5)
