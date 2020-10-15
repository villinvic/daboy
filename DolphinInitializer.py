from threading import Thread, Event
from subprocess import Popen
from time import sleep
import signal
import sys


class DolphinInitializer:

    def __init__(self, actor, exe, iso_path, video, port):
        self.actor = actor
        self.exe = exe
        self.iso_path = iso_path
        self.video = video
        self.port = port

    def run(self):
        cmd = r'%s -b --exec="%s" --video_backend="%s" --mw_port="%s" -C None' \
              % (self.exe, self.iso_path, self.video, self.port)

        print('dolphin', self.actor, ' started:', cmd)
        self.proc = Popen(cmd)

    def close(self):
        try:
            self.proc.send_signal(signal.SIGTERM)
            print('Sent signal to %d th Dolphin process ' % self.actor)
        except Exception as e:
            pass
