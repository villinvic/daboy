import Pad


def pushButton(button):
    return lambda pad: pad.press_button(button)


def releaseButton(button):
    return lambda pad: pad.release_button(button)


def tiltStick(stick, x, y):
    return lambda pad: pad.tilt_stick(stick, x, y)


neutral = tiltStick(Pad.Stick.MAIN, 0.5, 0.5)
left = tiltStick(Pad.Stick.MAIN, 0, 0.5)
down = tiltStick(Pad.Stick.MAIN, 0.5, 0)
up = tiltStick(Pad.Stick.MAIN, 0.5, 1)
right = tiltStick(Pad.Stick.MAIN, 1, 0.5)

endless_play = [
    # time
    (0, left),

    # infinite time
    (26, down),
    (19, left),
    (25, neutral),

    # exit settings
    (1, pushButton(Pad.Button.START)),
    (1, releaseButton(Pad.Button.START)),

    # enter stage select
    (28, pushButton(Pad.Button.START)),
    (1, releaseButton(Pad.Button.START)),

    (45, neutral)
]

stages = dict(
    battlefield=[
        (0, up),
        (2, neutral),

        # (60 * 60, neutral),

        # start game
        (20, pushButton(Pad.Button.START)),
        (1, releaseButton(Pad.Button.START)),
    ],

    final_destination=[
        (0, tiltStick(Pad.Stick.MAIN, 1, 0.8)),
        (5, neutral),

        # (60 * 60, neutral),

        # start game
        (20, pushButton(Pad.Button.START)),
        (1, releaseButton(Pad.Button.START)),
    ]
)

tapA = [
    (0, pushButton(Pad.Button.A)),
    (0, releaseButton(Pad.Button.A)),
]

lvl9 = [
    (0, right),
    (25, neutral),
]

color = [
    (0, pushButton(Pad.Button.X)),
    (0, releaseButton(Pad.Button.X)),
    (0, pushButton(Pad.Button.X)),
    (0, releaseButton(Pad.Button.X)),
    (0, pushButton(Pad.Button.X)),
    (0, releaseButton(Pad.Button.X)),
]


class Movie:
    def __init__(self, actions, pad):
        self.actions = actions
        self.frame = 0
        self.index = 0
        self.pad = pad

    def move(self, state):
        if not self.done():
            frame, action = self.actions[self.index]
            if self.frame == frame:
                action(self.pad)
                self.index += 1
                self.frame = 0
            else:
                self.frame += 1

    def done(self):
        return self.index == len(self.actions)
