#!/usr/bin/env bash
set -euo pipefail

SESSION="muon_analyze"
LOG="logs/analyze.log"
mkdir -p logs

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already exists."
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash -lc 'set -o pipefail; { source scripts/_env.sh && echo Starting analysis at \$(date) && python analyze_muon_atlas.py --runs_dir runs --report_dir report && python generate_report_assets.py --runs_dir runs --report_dir report && echo Finished analysis at \$(date); } 2>&1 | tee $LOG'"
echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Watch log with: tail -f $LOG"
