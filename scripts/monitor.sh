#!/usr/bin/env bash
set +e

echo "== tmux sessions =="
tmux ls
echo
echo "== nvidia-smi =="
nvidia-smi
echo
echo "== recent logs =="
tail -n 40 logs/*.log
