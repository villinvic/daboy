

cd "C:\Users\Victor\Desktop\daboy\daboy\SAC2\daboy"
set pyexe="python37"
set logdir="logs"

if not "%1"=="" (set alpha=%1) else (set alpha=0.0002)

if not "%2"=="" (set n_actors=%2) else (set n_actors=5)

if not "%3"=="" (if %%3==0 (goto noeval) else (goto eval)) else (goto noeval)

:noeval
wt %pyexe% main.py --mode learner --load_checkpoint --alpha %alpha%; split-pane %pyexe% main.py --mode actor --n_warmup 0 --n_actors %n_actors% --self_play
goto end

:eval
wt %pyexe% main.py --mode learner --load_checkpoint --alpha %alpha%; split-pane %pyexe% main.py --mode actor --n_warmup 0 --n_actors %n_actors% --self_play --eval
goto end


:end