import binascii
import zmq
from utils import write_with_folder


def chunk(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def parseMessage(message):
    lines = message.splitlines()

    assert (len(lines) % 2 == 0)

    diffs = chunk(lines, 2)

    for diff in diffs:
        diff[1] = binascii.unhexlify(diff[1].zfill(8))

    return diffs


class MemoryWatcher:
    """Reads and parses game memory changes.

    Pass the location of the socket to the constructor, then either manually
    call next() on this class to get a single change, or else use it like a
    normal iterator.
    """

    def __init__(self, path, port):
        self.path = path
        self.messages = None
        self.port = port

        write_with_folder(self.path, str(self.port))

        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)
        self.socket.bind("tcp://127.0.0.1:%d" % self.port)

    def __exit__(self, *args):
        """Closes the socket."""
        pass
        
    def unbind(self):
        self.socket.unbind("tcp://127.0.0.1:%d" % self.port)

    def __iter__(self):
        """Iterate over this class in the usual way to get memory changes."""
        return self

    def __next__(self):
        """Returns the next (address, value) tuple, or None on timeout.

        address is the string provided by dolphin, set in Locations.txt.
        value is a four-byte string suitable for interpretation with struct.
        """

        return self.get_messages()

    def get_messages(self):
        if self.messages is None:
            message = self.socket.recv()
            message = message.decode('utf-8')
            self.messages = parseMessage(message)

        return self.messages

    def advance(self):
        # self.socket.send(b'')
        self.messages = None
