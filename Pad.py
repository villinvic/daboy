import enum
import zmq
import os
import platform

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
    
sh_dict = {
        'mario':3,
        'luigi':3,
        'ylink':3,
        'ganon':3,
}


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
        self.no_op = False
        
    def copy(self, other):
        for item in UsefullButton:
            self[item.name] = other[item.name]

        self['stick'] = other['stick']
        self['c_stick'] = other['c_stick']
        self['duration'] = other['duration']
        self['id'] = None
        self['ac'] = other['ac']
        
    def egals(self,  other):

        for item in UsefullButton:
            if self[item.name] != other[item.name]:
                return False
                
        
        return self['stick'][1] == other['stick'][1] and\
        abs(self['stick'][0] - other['stick'][0]) < 0.001 and\
        self['c_stick'] == other['c_stick'] and\
        self['duration'] == other['duration'] and\
        self['ac'] == other['ac']
        
    def sym_state(self):
        sym = ControllerState()
        sym.copy(self)
        sym['stick'] = (abs(self['stick'][0] - 1.0), self['stick'][1])
        sym['c_stick'] = (abs(self['c_stick'][0] - 1.0), self['c_stick'][1])
        return sym

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
        
    def build_sym(self):
        self.sym = dict()
        for i in range(self.len):
            action = self[i]
            if isinstance(action, list): # suppose chain action has no x component
                self.sym[i] = i
            
            else:
                sym = action.sym_state()
                sym['id'] = i
                
                if sym.egals(action):
                    self.sym[i] = i
                else:
                    #print(sym)
                    
                    for j in range(self.len):
                        if j != i:
                            action2 = self[j]
                            if not isinstance(action2, list):
                                if sym.egals(action2):
                                    self.sym[i] = j
                                    break
                                    

    def __init__(self, char='ganon'):
        dict.__init__(self)
        self.len = 0

        if char is None:
            return

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

        if char == 'ylink':
            # regular
            for s_state in self.stick_states:
                # no button
                self.add(ControllerState(stick=s_state))

                for button in ['A', 'B', 'L']:
                    self.add(ControllerState(button=button, stick=s_state))

            # tilt
            for s_state in self.tilt_stick_states:
                self.add(ControllerState(button='A', stick=s_state))
            # c stick
            # add all directions ??
            for sc_state in self.smash_states:
                self.add(ControllerState(c_stick=sc_state))

            # short hop, full jump
            sh = [ControllerState(button='X', duration=sh_dict[char]), ControllerState(duration=1)]
            self.add(sh)
            self.add(ControllerState(button='X', duration=sh_dict[char] + 1))

            # shield drop
            self.add([ControllerState(button='L', stick=(0.5, 0.5)),
                      ControllerState(button='L', stick=(0.13, 0.5), duration=1),
                      ])
            # jump grab, shield grab, neutral z
            sg = ControllerState(button='L')
            sg['A'] = 1
            self.add(sg)
            self.add([
                ControllerState(button='X', duration=1),
                ControllerState(button='Z', duration=2)
            ])
            # useful for L cancel
            self.add(ControllerState(button='Z'))

            # wave land
            self.add(ControllerState(button='L', stick=(0.023, 0.353), duration=3))
            self.add(ControllerState(button='L', stick=(0.097, 0.353), duration=3))
            # wave dash
            self.add(sh + [ControllerState(button='L', stick=(0.023, 0.353), duration=1)])
            self.add(sh + [ControllerState(button='L', stick=(0.097, 0.353), duration=1)])

            # no op
            no_op = ControllerState()
            no_op.no_op = True
            self.add(no_op)

        else :

            for s_state in self.stick_states:
                for item in UsefullButton:
                    if item.name == 'X' :
                        self.add(ControllerState(button=item.name, stick=s_state, duration=sh_dict[char] + 1))

                    elif not (char in ['mario', 'luigi'] and item.name=='b' and (s_state[1] == 0.0 or s_state[0] != 0.5)): # down b requires side b
                        self.add(ControllerState(button=item.name, stick=s_state))

                for sc_state in self.smash_states:
                    self.add(ControllerState(stick=s_state, c_stick=sc_state))

                # no button
                self.add(ControllerState(stick=s_state))

            # Specific techs:

            # tilt

            for s_state in self.tilt_stick_states:
                self.add(ControllerState(button='A', stick=s_state))

             # WAVE LAND
            sh = [ControllerState(button='X', duration=sh_dict[char]), ControllerState(duration=1)]
            self.add(sh+[ControllerState(button='L', stick=(0.05, 0.3), duration=1)])
            self.add(sh+[ControllerState(button='L', stick=(0.95, 0.3), duration=1)])

            # shield grab
            sg = ControllerState(button='L')
            sg['A'] = 1
            self.add(sg)

            # jumpgrab
            jump1 = ControllerState(button='X', duration=1)
            grab = ControllerState(button='Z', ac=True)
            jgrab = [jump1, grab]
            self.add(jgrab)

            if char in  ['mario', 'luigi']:
                leftb = ControllerState(button='B', stick=(0.0, 0.5), duration=1)
                rightb = ControllerState(button='B', stick=(1.0, 0.5), duration=1)
                left = ControllerState(stick=(0.0, 0.5), duration=1)
                right = ControllerState(stick=(1.0, 0.5), duration=1)
                downb = ControllerState(button='B', stick=(0.5, 0.0), duration=1)
                down = ControllerState(stick=(0.5, 0.0), duration=1)
                self.add([leftb, left])
                self.add([rightb, right])
                self.add([downb, down])

                self.add(sh) # short hop



            no_op = ControllerState()
            no_op.no_op = True
            self.add(no_op)


            #self.build_sym()


            # for i in range(self.len):
            #    if isinstance(self[i], list):
            #        print(i, self[i][0])
            #    else:
            #        print(i, self[i])
            print("Action_space :", self.len)
            # print(self.sym)
        
        


class Pad:
    """Writes out controller inputs."""

    def __init__(self, path, port=None):
        """Create, but do not open the fifo."""
        self.pipe = None
        self.path = path
        self.windows = port is not None
        self.port = port
        self.message = ""
        self.action_space = []

        
    def connect(self):
        if self.windows:
            context = zmq.Context()
            with open(self.path, 'w') as f:
                f.write(str(self.port))

            self.pipe = context.socket(zmq.PUSH)
            address = "tcp://127.0.0.1:%d" % self.port
            print("Binding pad %s to address %s" % (self.path, address))
            self.pipe.bind(address)
        else:
            try:
                    os.unlink(self.path)
            except:
                    pass

            os.mkfifo(self.path)

            self.pipe = open(self.path, 'w', buffering=1)

    def __exit__(self, *args):
        pass
        
    def unbind(self):
        if not self.windows:
            self.pipe.close()

    def flush(self):
        if self.windows:
            self.pipe.send_string(self.message)
        else:
            self.pipe.write(self.message)
        self.message = ""

    def write(self, command, buffering=False):
        self.message += command + '\n'
        if not buffering:
            self.flush()

    def press_button(self, button, buffering=False):
        """Press a button."""
        assert button in Button or button in UsefullButton
        self.write('PRESS {}'.format(button.name), buffering)

    def release_button(self, button, buffering=False):
        """Release a button."""
        assert button in Button or button in UsefullButton
        self.write('RELEASE {}'.format(button.name), buffering)

    def press_trigger(self, trigger, amount, buffering=False):
        """Press a trigger. Amount is in [0, 1], with 0 as released."""
        assert trigger in Trigger or trigger in UsefullButton
        assert 0 <= amount <= 1
        self.write('SET {} {:.2f}'.format(trigger.name, amount), buffering)

    def tilt_stick(self, stick, x, y, buffering=False):
        """Tilt a stick. x and y are in [0, 1], with 0.5 as neutral."""
        assert stick in Stick
        assert 0 <= x <= 1 and 0 <= y <= 1
        self.write('SET {} {:.2f} {:.2f}'.format(stick.name, x, y), buffering)

    def reset(self):
        for button in Button:
            self.release_button(button)
        for trigger in Trigger:
            self.press_trigger(trigger, 0)
        for stick in Stick:
            self.tilt_stick(stick, 0.5, 0.5)
