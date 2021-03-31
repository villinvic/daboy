import Pad


class Spaces:
        def __init__(self, char, embed_action=True, embed_char=False):

                self.action_space = Pad.Action_Space(char)

                dim = 803
                if embed_char:
                    dim += 25  # 25 chars in the game
                if embed_action:
                    dim += self.action_space.len

                self.observation_space = (dim,)
