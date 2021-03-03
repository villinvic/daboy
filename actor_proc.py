from fire import Fire
from Actor import Actor
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def run(learner_ip='127.0.0.1', self_play=True, enemy_path='enemies',
        ep_length=201, dolphin_dir='../dolphin/', iso_path='../isos/melee.iso',
        video_backend='D3D', i=0, n_actors=1, epsilon=0.01, n_warmup=0, char='ganon',
        test=False):
        if test:
            epsilon = 0

        if i != 0:
                video_backend='Null'
        ACTOR = Actor(learner_ip, self_play, enemy_path,
                      ep_length, dolphin_dir, iso_path,
                      video_backend, i, n_actors, epsilon, n_warmup, char, test)
                       
        ACTOR.run()
        
                
                
if __name__ == '__main__':

        sys.exit(Fire(run))
