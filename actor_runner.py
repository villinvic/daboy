from fire import Fire
from subprocess import Popen, PIPE
import time
import signal
import sys
            

def run(self_play=True, n_actors=1, eval=False, output_dir='checkpoints',
        ep_length=20*5, dolphin_dir='../dolphin/', iso_path='../isos/melee.iso',
        video_backend='D3D', epsilon=0.01, n_warmup=20*15, render_all=False, char='ganon'):

        cmd = [f'python3 actor_proc.py {self_play and not (i == n_actors - 1 and eval)} {output_dir} {ep_length} {dolphin_dir} {iso_path} {video_backend if (i == 0 or (i == n_actors - 1 and eval) or render_all) else "Null"} {i} {n_actors} {epsilon} {n_warmup} {char}' for i in range(n_actors)]
        
        procs = n_actors * [None]
        
        for i in range(n_actors):
                
                procs[i] = Popen(cmd[i].split())
                time.sleep(3)
        try:
                while True:
                        time.sleep(1)
        except KeyboardInterrupt:
                pass
        for i in range(n_actors):
                procs[i].send_signal(signal.SIGINT)
            
           
        time.sleep(6)
        
        for i in range(n_actors):
                procs[i].send_signal(signal.SIGTERM)
                
                
                
if __name__ == '__main__':

        sys.exit(Fire(run))
