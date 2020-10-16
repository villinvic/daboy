from threading import Thread, Event
from subprocess import Popen
from time import sleep
import signal
import sys


class DolphinInitializer:

    def __init__(self, actor, exe, iso_path, video):
        self.actor = actor
        self.exe = exe
        self.iso_path = iso_path
        self.video = video
        self.n = 0

    def run(self):
        cmd = r'%s --exec=%s --video_backend=%s' \
              % (self.exe, self.iso_path, self.video)
              
        if self.actor > self.n:
                cmd += ' --platform=headless'

        print('dolphin', self.actor, ' started:', cmd)
        self.proc = Popen(cmd.split())

    def close(self):
        try:
            self.proc.send_signal(signal.SIGINT)
            sleep(1.5)
            self.proc.send_signal(signal.SIGINT)
            if self.actor > self.n:
                sleep(1.5)
                self.proc.send_signal(signal.SIGINT)
            print('Sent signal to %d th Dolphin process ' % self.actor)
        except Exception as e:
            pass
