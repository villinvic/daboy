from fire import Fire
from subprocess import Popen
import time
import signal
import sys
import gc
import zmq


def run(ip='127.0.0.1', self_play=True, n_actors=1, eval=False, enemy_dir='enemies',
        ep_length=20*10+1, dolphin_dir='../dolphin', iso_path='../isos/melee.iso',
        video_backend='D3D', epsilon=0.01, render_all=False, char='ganon', restart_freq=30*60):

        '''
        Run multiple actors processes,
        terminated with a ^C

        self_play : whether to train the ai playing against itself or not
        eval : the last process will let the ai play against a fixed opponent (regardless of self_play), to get an idea
               progress.
        enemy_dir : folder where fixed opponents are saved
        ep_length : episode length in number of actions (1 action = 3 frames in general)
        dolphin_dir : directory of the dolphin executable
        iso_path: iso path
        video_backend : video backend to use, set to "Null" when training
        epsilon : probability of random move while playing.
        render_all : set the video backend to video_backend for all dolphin instances, not only the first.
        char : the character selected by the ai.
        restart_freq : Used to restart the dolphin instances regularly, to prevent crashes. In seconds.

        '''

        cmd = [f'python3 actor_proc.py {ip} {self_play and not (i == n_actors-1 and eval)} {enemy_dir} {ep_length} {dolphin_dir} {iso_path} {video_backend if (i == 0 or (i == n_actors - 1 and eval) or render_all) else "Null"} {i} {epsilon} {char}' for i in range(n_actors)]
        
        alert_socket = zmq.Context().socket(zmq.PULL)
        alert_socket.bind("tcp://127.0.0.1:7555")
        def close(p):

                for i in range(n_actors):
                        p[i].send_signal(signal.SIGINT)
                            
                           
                time.sleep(6)
                
                for i in range(n_actors):
                        p[i].send_signal(signal.SIGTERM)
                
        
        try:
                while True:
                        procs = n_actors * [None]
                        timer = restart_freq
                        
                        for i in range(n_actors):
                                
                                procs[i] = Popen(cmd[i].split())
                                time.sleep(3)
                                
                        while timer>0:
                                time.sleep(1)
                                timer -= 1
                                try:
                                        i = alert_socket.recv_pyobj(zmq.NOBLOCK)
                                        print(i, "restarting")
                                        procs[i].send_signal(signal.SIGINT)
                                        time.sleep(6)
                                        procs[i].send_signal(signal.SIGTERM)
                                        procs[i] = Popen(cmd[i].split())
                                        
                                except zmq.ZMQError:
                                        pass

                        print('restarting...')
                        close(procs)
                        gc.collect()
         
        except KeyboardInterrupt:
                close(procs)
                
                
if __name__ == '__main__':
        sys.exit(Fire(run))
