from fire import Fire
from SSBM_ENV import SSBM_ENV
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def run(learner_ip='127.0.0.1', self_play=True, enemy_path='enemies',
        ep_length=201, dolphin_dir='../dolphin', iso_path='../isos/melee.iso',
        video_backend='D3D', i=0, epsilon=0.01, char='ganon',
        test=False):

        '''
        Runs an environment.
        Cf actor_runner.py for training.

        test: if --test is specified, the port number 1 will be set to be controlled by a human.
        '''

        if test:
            epsilon = 0

        if i != 0:
                video_backend='Null'
        ENV = SSBM_ENV(learner_ip, self_play, enemy_path,
                         ep_length, dolphin_dir, iso_path,
                         video_backend, i, epsilon, char, test)
                       
        ENV.run()
        
                
                
if __name__ == '__main__':

        sys.exit(Fire(run))
