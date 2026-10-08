#!/bin/bash
# Ship the K1 worktree (python + test + benchmark, incl. the pybind .so) and the K1 helper scripts to board 0614
# (/tmp/vagarwalla/wt_k1 and /tmp/vagarwalla/k1; the board has no NFS mounts).
set -u
H=w2u1g-lc-0614; R=/tmp/vagarwalla/wt_k1; SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
WT=${WT:-/home/scratch.vagarwalla_gpu/cudnn-fe-github/.claude/worktrees/mxb-k1-d128}
$SSH $H "mkdir -p $R /tmp/vagarwalla/k1 /tmp/vagarwalla/ladder /tmp/vagarwalla/k1/out"
t0=$(date +%s)
rsync -a -e "$SSH" --exclude '__pycache__' --exclude '*.pyc' $WT/python $WT/test $WT/benchmark $H:$R/ && echo "worktree synced $(( $(date +%s) - t0 ))s"
rsync -a -e "$SSH" --exclude '__pycache__' $WT/benchmark/rubin_mxfp8_paged/k1_d128/ $H:/tmp/vagarwalla/k1/ && echo "k1 scripts synced $(( $(date +%s) - t0 ))s"
$SSH $H "ls -la $R/python/cudnn/*.so; ls $R/python/cudnn/sdpa/fwd/kernels/sm107/ | grep bench; ls /tmp/vagarwalla/k1"
echo SYNC_DONE
