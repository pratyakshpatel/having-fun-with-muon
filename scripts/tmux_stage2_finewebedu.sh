#!/usr/bin/env bash
set -euo pipefail

SESSION="muon_stage2"
LOG="logs/stage2_finewebedu.log"
mkdir -p logs

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash -lc 'set -o pipefail; { source scripts/_env.sh && echo Starting stage 2 at \$(date) && bash scripts/_stage2_inner.sh && echo Finished stage 2 at \$(date); } 2>&1 | tee $LOG'"
echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Watch log with: tail -f $LOG"
