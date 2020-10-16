from fire import Fire
from Actor import Actor
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run(self_play=True, output_dir='checkpoints',
        ep_length=20*5, dolphin_dir='../dolphin/', iso_path='../isos/melee.iso',
        video_backend='D3D', i=0, n_actors=1, epsilon=0.01, n_warmup=20*15, char='mario'):

        ACTOR = Actor(self_play, output_dir,
                      ep_length, dolphin_dir, iso_path,
                      video_backend, i, n_actors, epsilon, n_warmup, char)
                       
        ACTOR.run()
        
                
                
if __name__ == '__main__':

        sys.exit(Fire(run))
