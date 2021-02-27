from fire import Fire
from subprocess import Popen, PIPE
import time
import signal
import sys
import gc
import zmq


def run(ip='127.0.0.1', self_play=True, n_actors=1, eval=False, output_dir='checkpoints',
        ep_length=20*10+1, dolphin_dir='../dolphin/', iso_path='../isos/melee.iso',
        video_backend='OGL', epsilon=0.01, n_warmup=0, render_all=False, char='ganon', restart_freq=45*60):

        cmd = [f'python3 actor_proc.py {ip, self_play and not (i == n_actors-1 and eval)} {output_dir} {ep_length} {dolphin_dir} {iso_path} {video_backend if (i == 0 or (i == n_actors - 1 and eval) or render_all) else "Null"} {i} {n_actors} {epsilon} {n_warmup} {char}' for i in range(n_actors)]
        
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
