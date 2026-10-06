#!/bin/bash
# Ship the FROST bench environment to board 0614's local disk (no NFS mounts there).
set -u
H=w2u1g-lc-0614; R=/tmp/vagarwalla; SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
SP=/home/scratch.vagarwalla_libs_1/venv_frost/lib/python3.10/site-packages
PYDIR=$(dirname $(dirname $(ls -d /home/vagarwalla/.local/share/uv/python/cpython-3.10.*-linux-x86_64-gnu/bin/python3.10 | tail -1)))
WT=/home/scratch.vagarwalla_gpu/cudnn-fe-github/.claude/worktrees/rubin-ladder-bench
echo "start $(date +%T) rsync local: $(which rsync) remote: $($SSH $H 'which rsync' 2>&1)"
$SSH $H "mkdir -p $R/py310 $R/venv_sp $R/cudnn926 $R/wt $R/ladder"
t0=$(date +%s)
rsync -a --delete -e "$SSH" $PYDIR/ $H:$R/py310/ && echo "py310 done $(( $(date +%s) - t0 ))s"
rsync -a -e "$SSH" --exclude '.git' --exclude 'out' --exclude 'build' --exclude '__pycache__' $WT/ $H:$R/wt/ && echo "worktree done $(( $(date +%s) - t0 ))s"
rsync -a -e "$SSH" /home/scratch.vagarwalla_gpu/cudnn_9.26.0.51/lib/ $H:$R/cudnn926/lib/ && echo "cudnn926 done $(( $(date +%s) - t0 ))s"
rsync -a -e "$SSH" --exclude 'triton' --exclude 'triton-*' --exclude '__pycache__' $SP/ $H:$R/venv_sp/ && echo "site-packages done $(( $(date +%s) - t0 ))s"
$SSH $H "du -sh $R/* 2>/dev/null; $R/py310/bin/python3.10 --version"
echo SHIP_DONE
