from threading import Thread, Event
from subprocess import Popen
from time import sleep
import signal
import configparser
import sys
from speed_hack import *
import numpy as np
import pprint


class DolphinInitializer:

    def __init__(self, actor, exe, iso_path, video, test):
        self.actor = actor
        self.exe = exe
        self.iso_path = iso_path
        self.video = video
        self.n =-1
        self.test = test
        
    def run(self, char1=None, char2=None):
        config_path = '../dolphin/User/Config/Dolphin.ini'
        game_config_path = '../dolphin/User/GameSettings/GALE01.ini'
        if self.actor <= self.n or self.video != "Null":
            speed = 1.0
            speed_hack = "Speedhack"
        else:
            speed = 5.0
            speed_hack = "Speedhack no render"
        config = configparser.ConfigParser()
        config.read(config_path)
        config['Core']['EmulationSpeed'] = str(speed)
        config['Core']['GFXBackend'] = self.video
        if self.test:
            config['Core']['sidevice0'] = '12'
        else:
            config['Core']['sidevice0'] = '6'
            
        if self.video != "Null":
            config['DSP']['volume'] = '10'
        else:
            config['DSP']['volume'] = '0'

        with open(config_path, 'w') as configfile:
            config.write(configfile)

        with open(game_config_path, 'r') as gconfigfile:
            gconfig = gconfigfile.read()

        gconfig = gconfig.split('$Setup match')

        if self.test:
            print("List of available chars:")
            pprint.pprint(char_ids.keys())
            print("Human's character : ", end="")
            char1 = input()
            assert char1 in char_ids.keys()
                
        playertype = PlayerStatus.CPU if char1 is None else PlayerStatus.HUMAN
        if char1 is None :
            char1 = np.random.choice(list(char_ids.keys()))
        gconfig = gconfig[0] + '$Setup match' + setup_match_code('battlefield', player1=playertype, char1=char1, char2=char2)
        if self.test or self.video != "Null":
            enabled = """
[Gecko_Enabled]
$Skip Memcard Prompt
$Boot to match
$Setup match
$Flash White on Successful L-Cancel
                    """
        else:
            enabled = """
[Gecko_Enabled]
$Skip Memcard Prompt
$%s
$Boot to match
$Setup match
$DMA Read Before Poll
$Flash White on Successful L-Cancel
        """ % speed_hack
        gconfig += enabled
        with open(game_config_path, 'w') as gconfigfile:
            gconfigfile.write( gconfig)
          

        cmd = r'%s --exec=%s' \
              % (self.exe, self.iso_path)
              
        if self.actor > self.n and self.video == "Null":
                cmd += ' --platform=headless'

        print('dolphin', self.actor, ' started:', cmd)
        self.proc = Popen(cmd.split())

    def close(self):
        try:
            sleep(0.3)
            self.proc.send_signal(signal.SIGINT)
            sleep(1.5)
            self.proc.send_signal(signal.SIGINT)
            sleep(1.5)
            self.proc.send_signal(signal.SIGINT)
            print('Sent signal to %d th Dolphin process ' % self.actor)
        except Exception as e:
            pass
