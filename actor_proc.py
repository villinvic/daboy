from fire import Fire
from Actor import Actor
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def run(self_play=True, output_dir='checkpoints',
        ep_length=401, dolphin_dir='../dolphin/', iso_path='../isos/melee.iso',
        video_backend='D3D', i=0, n_actors=1, epsilon=0.02, n_warmup=0, char='mario',
        test=False):
        if test:
            epsilon = 0

        if i != 0:
                video_backend='Null'
        ACTOR = Actor(self_play, output_dir,
                      ep_length, dolphin_dir, iso_path,
                      video_backend, i, n_actors, epsilon, n_warmup, char, test)
                       
        ACTOR.run()
        
                
                
if __name__ == '__main__':

        sys.exit(Fire(run))
